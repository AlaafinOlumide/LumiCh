from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional

from dotenv import load_dotenv

from .config import Config
from .data import TwelveDataClient, TwelveDataQuotaError, TwelveDataError
from .high_impact_news import check_high_impact_news
from .sessions import now_in_sessions_utc, parse_sessions, session_label
from .strategy import (
    detect_trend,
    m15_confirms,
    risk_tag_from_context,
    score_entry_m5,
    Signal,
    atr as atr_value,
    compute_tp_sl_from_atr,
    confidence_score,
)
from .telegram import TelegramClient


cfg_global: Config


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _log_dedupe(log: logging.Logger, key: str, message: str, dedupe_state: dict, every_seconds: int = 300) -> None:
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    last_t = dedupe_state.get(key, 0.0)
    if (now - last_t) >= every_seconds:
        log.info(message)
        dedupe_state[key] = now


def _is_weekend_utc(now: dt.datetime) -> bool:
    # Monday=0 ... Sunday=6
    return now.weekday() >= 5


def _safe_float(x: Optional[float], fallback: float) -> float:
    try:
        if x is None:
            return fallback
        v = float(x)
        if v != v:
            return fallback
        return v
    except Exception:
        return fallback


def fmt_signal(
    sig: Signal,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    rr: float,
    confidence: int,
    emoji: str,
    entry_source: str,
    entry_zone_low: float,
    entry_zone_high: float,
    atr_m15: float,
) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []

    lines.append(f"Xauusd: {sig.direction}")
    lines.append(f"ENTRY: {entry:.2f}  ({entry_source})")
    lines.append(f"ENTRY ZONE: {entry_zone_low:.2f} - {entry_zone_high:.2f}  (ATR15={atr_m15:.2f})")
    lines.append(f"TP1: {tp1:.2f}")
    lines.append(f"TP2: {tp2:.2f} (TP2 may not be reached — take profit when convenient)")
    lines.append(f"SL: {sl:.2f}")
    lines.append(f"RR (to TP2): {rr:.2f}")
    lines.append(f"Confidence: {confidence}% {emoji}")
    lines.append("")

    lines.append(f"{ts} UTC | Session: *{sig.session_label}*")
    lines.append(f"Signal: *{sig.direction}* | Risk: *{sig.risk_tag}*")
    lines.append(f"Mode: `{sig.timeframe_mode}`")
    lines.append("")

    t = sig.trend_state
    lines.append("*Trend (HTF)*")
    lines.append(f"- TF: `{t.timeframe}` | Dir: *{t.direction}* | Close: {t.close:.2f}")
    lines.append(f"- EMA{cfg_global.ema_fast}: {t.ema_fast:.2f} | EMA{cfg_global.ema_slow}: {t.ema_slow:.2f} | Slope: {t.slope}")
    lines.append("")

    lines.append("*Filters & Confirmations*")
    joined = ", ".join(sig.confirmations) if sig.confirmations else "None"
    lines.append(f"- Confirmations: *{sig.confirmations_passed}/{sig.confirmations_required}* ({joined})")
    lines.append(f"- ADX: {sig.adx_value:.1f} (min {sig.adx_min:.1f})")
    lines.append(f"- News: {sig.news_status}")
    lines.append("")

    lines.append("*Reason for Trade*")
    for b in sig.reason_bullets[:8]:
        lines.append(f"- {b}")

    return "\n".join(lines)


def main() -> None:
    global cfg_global

    load_dotenv()
    setup_logging()

    cfg = Config.from_env()
    cfg_global = cfg

    log = logging.getLogger("xauusd_bot")

    sessions = parse_sessions(cfg.trading_sessions)
    if not sessions:
        raise RuntimeError("No TRADING_SESSIONS configured")

    td = TwelveDataClient(cfg.twelvedata_api_key, timeout=20, max_retries=2, backoff_seconds=1.0)
    tg = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)

    last_signal_time: dt.datetime | None = None
    last_direction: str | None = None
    dedupe_state: dict[str, float] = {}

    # Optional: calibrate closer to your MT5 broker price feed.
    # If your MT5 price is usually (MT5 - TwelveData) = -18.0, set BROKER_PRICE_OFFSET=-18.0
    broker_offset = float(getattr(cfg, "broker_price_offset", 0.0)) if hasattr(cfg, "broker_price_offset") else 0.0

    log.info(
        "Bot started. Sessions=%s | Symbol=%s | min_confirmations=%s",
        cfg.trading_sessions,
        cfg.symbol,
        cfg.min_confirmations,
    )

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)

            # ✅ Block weekends
            if _is_weekend_utc(now):
                _log_dedupe(
                    log,
                    key="weekend_block",
                    message=f"Weekend detected (UTC). Bot paused. Sleeping {cfg.poll_seconds}s...",
                    dedupe_state=dedupe_state,
                    every_seconds=900,
                )
                time.sleep(cfg.poll_seconds)
                continue

            if not now_in_sessions_utc(sessions, now):
                _log_dedupe(
                    log,
                    key="outside_sessions",
                    message=f"Outside trading sessions. Sleeping {cfg.poll_seconds}s...",
                    dedupe_state=dedupe_state,
                    every_seconds=600,
                )
                time.sleep(cfg.poll_seconds)
                continue

            s_label = session_label(sessions, now)

            # News check (soft-fail + cached inside)
            news = check_high_impact_news(
                provider=cfg.news_api_provider or "fmp",
                api_key=cfg.news_api_key or "",
                base_url=getattr(cfg, "news_base_url", None),
                lookahead_min=getattr(cfg, "news_lookahead_min", 60),
                cooldown_after_min=getattr(cfg, "news_cooldown_after_min", 30),
                now_utc=now,
                ttl_seconds=300,
            )

            if news.is_high_impact and cfg.news_mode == "BLOCK":
                _log_dedupe(
                    log,
                    key="blocked_news",
                    message=f"Signals blocked due to news: {news.message}",
                    dedupe_state=dedupe_state,
                    every_seconds=300,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # Fetch candles (cached)
            try:
                trend_tf = "1h"
                trend_df = td.fetch_time_series_cached(cfg.symbol, "1h", outputsize=200, ttl_seconds=3600, now_utc=now).df
                if len(trend_df) < 100:
                    raise RuntimeError("Insufficient H1 candles")
            except Exception as e:
                log.warning("H1 unavailable (%s). Falling back to M15 as HTF.", e)
                trend_tf = "15min"
                trend_df = td.fetch_time_series_cached(cfg.symbol, "15min", outputsize=200, ttl_seconds=900, now_utc=now).df

            df_m15 = td.fetch_time_series_cached(cfg.symbol, "15min", outputsize=200, ttl_seconds=900, now_utc=now).df
            df_m5 = td.fetch_time_series_cached(cfg.symbol, "5min", outputsize=200, ttl_seconds=300, now_utc=now).df

            if len(df_m5) < 100 or len(df_m15) < 100 or len(trend_df) < 100:
                _log_dedupe(
                    log,
                    key="insufficient_data",
                    message="Insufficient data. Sleeping.",
                    dedupe_state=dedupe_state,
                    every_seconds=300,
                )
                time.sleep(cfg.poll_seconds)
                continue

            trend = detect_trend(trend_df, trend_tf, cfg.ema_fast, cfg.ema_slow, cfg.ema_slope_bars)
            if trend.direction == "NEUTRAL":
                _log_dedupe(
                    log,
                    key="trend_neutral",
                    message="HTF trend neutral. No signals.",
                    dedupe_state=dedupe_state,
                    every_seconds=300,
                )
                time.sleep(cfg.poll_seconds)
                continue

            ok_confirm, confirm_reason = m15_confirms(df_m15, trend.direction, cfg.ema_fast, cfg.rsi_period)
            if not ok_confirm:
                _log_dedupe(
                    log,
                    key="m15_not_confirm",
                    message=f"M15 did not confirm. {confirm_reason}",
                    dedupe_state=dedupe_state,
                    every_seconds=180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            direction = "BUY" if trend.direction == "BULL" else "SELL"

            # Cooldown logic
            if last_signal_time is not None:
                mins_since = (now - last_signal_time).total_seconds() / 60
                if mins_since < cfg.cooldown_minutes and last_direction == direction:
                    _log_dedupe(
                        log,
                        key="cooldown",
                        message=f"Cooldown active ({mins_since:.1f} min). Skipping.",
                        dedupe_state=dedupe_state,
                        every_seconds=180,
                    )
                    time.sleep(cfg.poll_seconds)
                    continue

            confirmations, adx_val, _adx_pass, reason_bullets = score_entry_m5(df_m5, direction, cfg)

            # Confirmations gate
            if len(confirmations) < cfg.min_confirmations:
                _log_dedupe(
                    log,
                    key="not_enough_confirm",
                    message=f"Not enough confirmations: {len(confirmations)} (need {cfg.min_confirmations})",
                    dedupe_state=dedupe_state,
                    every_seconds=180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # ADX gate (simple + strict)
            if adx_val < float(cfg.adx_min):
                _log_dedupe(
                    log,
                    key="adx_failed",
                    message=f"ADX filter failed: {adx_val:.1f} < {float(cfg.adx_min):.1f}",
                    dedupe_state=dedupe_state,
                    every_seconds=180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # Reasons
            reasons: list[str] = []
            reasons.append(f"HTF {trend.timeframe} trend {trend.direction} (EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})")
            reasons.append(confirm_reason)
            reasons.extend(reason_bullets)

            tf_mode = "H1→M15→M5" if trend_tf == "1h" else "M15→M5 (fallback)"
            risk_tag = risk_tag_from_context(trend_tf)
            news_status = ("⚠️ " + news.message) if news.is_high_impact else "Normal"

            # ✅ Entry: use quote, fallback to M5 close if quote fails
            entry_source = "QUOTE"
            try:
                q = td.fetch_quote_cached(cfg.symbol, ttl_seconds=2, now_utc=now)
                entry_price = float(q.price) + broker_offset
            except (TwelveDataError, Exception) as e:
                log.warning("Quote unavailable (%s). Falling back to M5 close.", e)
                entry_source = "M5_CLOSE"
                entry_price = float(df_m5["close"].iloc[-1])

            # ✅ Risk model uses M15 ATR (more stable)
            atr_m15 = atr_value(df_m15, cfg.atr_period)
            tp2, sl, rr = compute_tp_sl_from_atr(
                entry=entry_price,
                direction=direction,
                atr_value=atr_m15,
                sl_mult=cfg.atr_sl_mult,
                tp_mult=cfg.atr_tp_mult,
            )

            # ✅ TP1 is half of TP2 distance
            if direction == "BUY":
                tp1 = entry_price + (tp2 - entry_price) * 0.5
            else:
                tp1 = entry_price - (entry_price - tp2) * 0.5

            # ✅ Entry zone based on ATR15 (prevents “wrong entry” feeling)
            zone_half = max(atr_m15 * 0.25, 2.0)  # at least $2 width
            entry_zone_low = entry_price - zone_half
            entry_zone_high = entry_price + zone_half

            conf, emoji = confidence_score(
                confirmations_passed=len(confirmations),
                confirmations_required=cfg.min_confirmations,
                adx_value=adx_val,
                adx_min=float(cfg.adx_min),
                news_is_normal=not news.is_high_impact,
            )

            sig = Signal(
                symbol=cfg.symbol,
                timestamp_utc=now,
                direction=direction,
                risk_tag=risk_tag,
                timeframe_mode=tf_mode,
                session_label=s_label,
                trend_state=trend,
                confirmations=confirmations,
                confirmations_passed=len(confirmations),
                confirmations_required=cfg.min_confirmations,
                adx_value=adx_val,
                adx_min=float(cfg.adx_min),
                news_status=news_status,
                reason_bullets=reasons,
                entry_price=entry_price,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                rr=rr,
                confidence=conf,
                confidence_emoji=emoji,
            )

            tg.send_message(
                fmt_signal(
                    sig=sig,
                    entry=entry_price,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    rr=rr,
                    confidence=conf,
                    emoji=emoji,
                    entry_source=entry_source,
                    entry_zone_low=entry_zone_low,
                    entry_zone_high=entry_zone_high,
                    atr_m15=atr_m15,
                )
            )

            last_signal_time = now
            last_direction = direction
            log.info("Signal sent: %s %s | entry=%.2f (%s)", direction, cfg.symbol, entry_price, entry_source)

        except TwelveDataQuotaError:
            log.error("TwelveData daily credits exhausted. Sleeping 3600s.")
            time.sleep(3600)
            continue

        except Exception as e:
            log.exception("Loop error: %s", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()