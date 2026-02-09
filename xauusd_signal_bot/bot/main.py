from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from .config import Config
from .data import TwelveDataClient, TwelveDataQuotaError
from .high_impact_news import check_high_impact_news
from .sessions import now_in_sessions_utc, parse_sessions, session_label
from .strategy import (
    detect_trend,
    m15_confirms,
    risk_tag_from_context,
    score_entry_m5,
    atr,
    compute_sl_tp2_tp1_from_atr,
    rr_from_sl_tp,
    confidence_score,
    Signal,
)
from .telegram import TelegramClient


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _seconds_until_monday_utc(now: dt.datetime) -> int:
    # weekday: Mon=0 ... Sun=6
    wd = now.weekday()
    if wd < 5:
        return 0
    days_ahead = 7 - wd  # Sat(5)->2 days, Sun(6)->1 day
    monday = (now + dt.timedelta(days=days_ahead)).replace(hour=0, minute=0, second=5, microsecond=0)
    return max(60, int((monday - now).total_seconds()))


def _log_dedupe(log: logging.Logger, key: str, message: str, dedupe_state: dict, every_seconds: int = 300) -> None:
    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    last_t = dedupe_state.get(key, 0.0)
    if (now_ts - last_t) >= every_seconds:
        log.info(message)
        dedupe_state[key] = now_ts


def _confidence_tag(passed: int, required: int, score: int) -> str:
    if required <= 0:
        return "MEDIUM"
    if score >= 75 and passed >= required:
        return "HIGH"
    if score >= 55:
        return "MEDIUM"
    return "LOW"


def fmt_signal(
    sig: Signal,
    cfg: Config,
    entry_quote: float,
    m5_close: float,
    tp1: Optional[float],
    tp2: Optional[float],
    sl: Optional[float],
    conf_score: int,
    conf_emoji: str,
) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    dev = abs(entry_quote - m5_close)

    tp1_txt = f"{tp1:.2f}" if tp1 is not None else "TBD"
    tp2_txt = f"{tp2:.2f}" if tp2 is not None else "TBD"
    sl_txt = f"{sl:.2f}" if sl is not None else "TBD"

    conf_tag = _confidence_tag(sig.confirmations_passed, sig.confirmations_required, conf_score)

    lines: list[str] = []
    lines.append(f"Xauusd: {sig.direction}")
    lines.append(f"ENTRY PRICE: {entry_quote:.2f}")
    lines.append(f"M5 Close: {m5_close:.2f} | Dev: {dev:.2f}")
    lines.append(f"TP1: {tp1_txt}")
    lines.append(f"TP2: {tp2_txt} (TP2 may not be reached — take profit when convenient)")
    lines.append(f"SL: {sl_txt}")
    lines.append(f"Confidence: {conf_tag} ({conf_score}%) {conf_emoji}")
    lines.append("")

    lines.append(f"{ts} UTC | Session: *{sig.session_label}*")
    lines.append(f"Signal: *{sig.direction}* | Risk: *{sig.risk_tag}*")
    lines.append(f"Mode: `{sig.timeframe_mode}`")
    lines.append("")

    t = sig.trend_state
    lines.append("*Trend (HTF)*")
    lines.append(f"- TF: `{t.timeframe}` | Dir: *{t.direction}* | Close: {t.close:.2f}")
    lines.append(f"- EMA{cfg.ema_fast}: {t.ema_fast:.2f} | EMA{cfg.ema_slow}: {t.ema_slow:.2f} | Slope: {t.slope}")
    lines.append("")

    lines.append("*Filters & Confirmations*")
    joined = ", ".join(sig.confirmations) if sig.confirmations else "None"
    lines.append(f"- Confirmations: *{sig.confirmations_passed}/{sig.confirmations_required}* ({joined})")
    lines.append(f"- ADX: {sig.adx_value:.1f} (min {sig.adx_min:.1f})")
    lines.append(f"- News: {sig.news_status}")
    lines.append("")

    lines.append("*Reason for Trade*")
    for b in sig.reason_bullets[:6]:
        lines.append(f"- {b}")

    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    setup_logging()

    cfg = Config.from_env()
    log = logging.getLogger("xauusd_bot")

    sessions = parse_sessions(cfg.trading_sessions)
    if not sessions:
        raise RuntimeError("No TRADING_SESSIONS configured")

    td = TwelveDataClient(cfg.twelvedata_api_key)
    tg = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)

    last_signal_time: dt.datetime | None = None
    last_direction: str | None = None

    dedupe_state: dict[str, float] = {}

    log.info(
        "Bot started. Sessions=%s | Symbol=%s | min_confirmations=%s",
        cfg.trading_sessions,
        cfg.symbol,
        cfg.min_confirmations,
    )

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)

            # ---------
            # Weekend block (UTC)
            # ---------
            if now.weekday() >= 5:
                secs = _seconds_until_monday_utc(now)
                _log_dedupe(
                    log,
                    "weekend_block",
                    f"Weekend detected (UTC). Bot paused until Monday. Sleeping {secs}s.",
                    dedupe_state,
                    every_seconds=600,
                )
                time.sleep(secs)
                continue

            # ---------
            # Session block
            # ---------
            if not now_in_sessions_utc(sessions, now):
                _log_dedupe(
                    log,
                    "outside_sessions",
                    f"Outside trading sessions. Sleeping {cfg.poll_seconds}s...",
                    dedupe_state,
                    every_seconds=600,
                )
                time.sleep(cfg.poll_seconds)
                continue

            s_label = session_label(sessions, now)

            # ---------
            # News check (soft fail)
            # ---------
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
                    "blocked_news",
                    f"Signals blocked due to news: {news.message}",
                    dedupe_state,
                    every_seconds=300,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # ---------
            # Fetch candles (cached)
            # ---------
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
                _log_dedupe(log, "insufficient_data", "Insufficient data. Sleeping.", dedupe_state, 300)
                time.sleep(cfg.poll_seconds)
                continue

            # ---------
            # Freshness check (avoid stale cached candles)
            # ---------
            if "time" in df_m5.columns and isinstance(df_m5["time"].iloc[-1], pd.Timestamp):
                last_m5_time = df_m5["time"].iloc[-1].to_pydatetime()
                age_sec = (now - last_m5_time).total_seconds()
                if age_sec > 180:
                    _log_dedupe(log, "stale_m5", f"M5 data stale ({age_sec:.0f}s). Skipping.", dedupe_state, 180)
                    time.sleep(cfg.poll_seconds)
                    continue

            trend = detect_trend(trend_df, trend_tf, cfg.ema_fast, cfg.ema_slow, cfg.ema_slope_bars)
            if trend.direction == "NEUTRAL":
                _log_dedupe(log, "trend_neutral", "HTF trend neutral. No signals.", dedupe_state, 300)
                time.sleep(cfg.poll_seconds)
                continue

            ok_confirm, confirm_reason = m15_confirms(df_m15, trend.direction, cfg.ema_fast, cfg.rsi_period)
            if not ok_confirm:
                _log_dedupe(log, "m15_not_confirm", f"M15 did not confirm. {confirm_reason}", dedupe_state, 180)
                time.sleep(cfg.poll_seconds)
                continue

            direction = "BUY" if trend.direction == "BULL" else "SELL"

            # Cooldown logic
            if last_signal_time is not None:
                mins_since = (now - last_signal_time).total_seconds() / 60
                if mins_since < cfg.cooldown_minutes and last_direction == direction:
                    _log_dedupe(log, "cooldown", f"Cooldown active ({mins_since:.1f} min). Skipping.", dedupe_state, 180)
                    time.sleep(cfg.poll_seconds)
                    continue

            confirmations, adx_val, _adx_pass, reason_bullets = score_entry_m5(df_m5, direction, cfg)

            # ---------
            # Entry price: live QUOTE (preferred) + fallback M5 close
            # ---------
            m5_close = float(df_m5["close"].iloc[-1])
            entry_quote: float | None = None
            try:
                entry_quote = td.fetch_quote(cfg.symbol)  # <-- requires data.py fetch_quote()
            except Exception:
                entry_quote = None

            entry = float(entry_quote) if entry_quote is not None else m5_close

            # ---------
            # Quote/M5 mismatch guard (prevents crazy entries)
            # threshold = max(1.5 * M15_ATR, 4.0)
            # ---------
            m15_atr = atr(df_m15, period=cfg.atr_period)
            max_dev = max(1.5 * m15_atr, 4.0)

            if entry_quote is not None:
                dev = abs(entry_quote - m5_close)
                if dev > max_dev:
                    _log_dedupe(
                        log,
                        "quote_m5_mismatch",
                        f"Quote/M5 mismatch too large: |{entry_quote:.2f}-{m5_close:.2f}|={dev:.2f} > {max_dev:.2f}. Skipping.",
                        dedupe_state,
                        180,
                    )
                    time.sleep(cfg.poll_seconds)
                    continue

            # ---------
            # ADX gate (slightly relaxed if confirmations are strong)
            # ---------
            strict_min = float(cfg.adx_min)
            relaxed_min = max(10.0, strict_min - 3.0)
            strong_conf = len(confirmations) >= (cfg.min_confirmations)

            effective_adx_min = relaxed_min if strong_conf else strict_min
            if adx_val < effective_adx_min:
                _log_dedupe(
                    log,
                    "adx_failed",
                    f"ADX filter failed: {adx_val:.1f} < {effective_adx_min:.1f} (strict={strict_min:.1f}, strong_conf={strong_conf})",
                    dedupe_state,
                    180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # ---------
            # Confirmations gate (recommended)
            # Allow 1 less confirmation ONLY if ADX is very strong
            # ---------
            required = int(cfg.min_confirmations)
            allow_minus_one = adx_val >= (strict_min + 10)
            effective_required = required - 1 if allow_minus_one else required

            if len(confirmations) < effective_required:
                _log_dedupe(
                    log,
                    "not_enough_confirm",
                    f"Not enough confirmations: {len(confirmations)} (need {effective_required})",
                    dedupe_state,
                    180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            reasons: list[str] = []
            reasons.append(f"HTF {trend.timeframe} trend {trend.direction} (EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})")
            reasons.append(confirm_reason)
            reasons.extend(reason_bullets)

            tf_mode = "H1→M15→M5" if trend_tf == "1h" else "M15→M5 (fallback)"
            risk_tag = risk_tag_from_context(trend_tf)
            news_status = ("⚠️ " + news.message) if news.is_high_impact else "Normal"

            # ---------
            # M15 ATR-based SL/TP
            # TP2 = original target, TP1 = half-way to TP2
            # ---------
            sl, tp2, tp1 = compute_sl_tp2_tp1_from_atr(
                entry=entry,
                direction=direction,
                atr_value=m15_atr,
                sl_mult=cfg.atr_sl_mult,
                tp_mult=cfg.atr_tp_mult,
            )

            rr_tp1 = rr_from_sl_tp(entry, direction, sl, tp1)
            rr_tp2 = rr_from_sl_tp(entry, direction, sl, tp2)

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
                confirmations_required=required,
                adx_value=adx_val,
                adx_min=effective_adx_min,
                news_status=news_status,
                reason_bullets=reasons,
                entry_price=entry,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                rr=rr_tp2,
            )

            # confidence score (numeric + emoji)
            news_is_normal = (not news.is_high_impact)
            conf_score, conf_emoji = confidence_score(
                confirmations_passed=sig.confirmations_passed,
                confirmations_required=sig.confirmations_required,
                adx_value=adx_val,
                adx_min=effective_adx_min,
                news_is_normal=news_is_normal,
            )

            msg = fmt_signal(
                sig=sig,
                cfg=cfg,
                entry_quote=float(entry),
                m5_close=m5_close,
                tp1=tp1,
                tp2=tp2,
                sl=sl,
                conf_score=conf_score,
                conf_emoji=conf_emoji,
            )

            tg.send_message(msg)

            last_signal_time = now
            last_direction = direction
            log.info(
                "Signal sent: %s %s | entry=%.2f | M15_ATR=%.2f | SL=%.2f | TP1=%.2f (RR=%.2f) | TP2=%.2f (RR=%.2f)",
                direction,
                cfg.symbol,
                entry,
                m15_atr,
                sl,
                tp1,
                rr_tp1,
                tp2,
                rr_tp2,
            )

        except TwelveDataQuotaError:
            log.error("TwelveData daily credits exhausted. Sleeping 3600s.")
            time.sleep(3600)
            continue

        except Exception as e:
            msg = str(e).lower()
            if "run out of api credits" in msg or "out of api credits" in msg:
                log.error("API credits exhausted (generic). Sleeping 3600s.")
                time.sleep(3600)
                continue

            log.exception("Loop error: %s", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()