# bot/main.py
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
from .strategy import detect_trend, m15_confirms, risk_tag_from_context, score_entry_m5, Signal
from .telegram import TelegramClient


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _confidence_from_confirmations(passed: int, required: int, adx_val: float, adx_min: float) -> str:
    if required <= 0:
        return "MEDIUM"

    ratio = passed / required
    adx_strong = adx_val >= (adx_min + 10)
    adx_ok = adx_val >= adx_min

    if ratio >= 1.0 and adx_strong:
        return "HIGH"
    if ratio >= 1.0 and adx_ok:
        return "MEDIUM"
    if ratio >= 0.8 and adx_ok:
        return "MEDIUM"
    return "LOW"


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Lightweight ATR for risk model. Uses SMA of True Range.
    Expects columns: high, low, close.
    """
    if df is None or len(df) < period + 2:
        return 0.0

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_s = tr.rolling(period).mean()
    v = float(atr_s.iloc[-1])
    return v if v == v else 0.0


def _compute_tp1_tp2_sl_from_atr(
    entry: float,
    direction: str,
    atr_value: float,
    sl_mult: float,
    tp2_mult: float,
) -> tuple[float, float, float]:
    """
    TP2 uses tp2_mult * ATR.
    TP1 is halfway to TP2.
    """
    if atr_value <= 0:
        # fallback: 0.1% of price or at least 1.0
        atr_value = max(entry * 0.001, 1.0)

    sl_dist = atr_value * sl_mult
    tp2_dist = atr_value * tp2_mult

    if direction.upper() == "BUY":
        sl = entry - sl_dist
        tp2 = entry + tp2_dist
        tp1 = entry + (tp2 - entry) * 0.5
    else:
        sl = entry + sl_dist
        tp2 = entry - tp2_dist
        tp1 = entry - (entry - tp2) * 0.5

    return float(tp1), float(tp2), float(sl)


def fmt_signal(
    cfg: Config,
    sig: Signal,
    entry_price: float,
    tp1: Optional[float],
    tp2: Optional[float],
    sl: Optional[float],
) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    confidence = _confidence_from_confirmations(
        passed=sig.confirmations_passed,
        required=sig.confirmations_required,
        adx_val=sig.adx_value,
        adx_min=sig.adx_min,
    )

    tp1_txt = f"{tp1:.2f}" if tp1 is not None else "TBD"
    tp2_txt = f"{tp2:.2f}" if tp2 is not None else "TBD"
    sl_txt = f"{sl:.2f}" if sl is not None else "TBD"

    lines: list[str] = []
    lines.append(f"Xauusd: {sig.direction}")
    lines.append(f"ENTRY PRICE: {entry_price:.2f}")
    lines.append(f"TP1: {tp1_txt}")
    lines.append(f"TP2: {tp2_txt} (TP2 may not be reached — take profit when convenient)")
    lines.append(f"SL: {sl_txt}")
    lines.append(f"Confidence: {confidence}")
    lines.append("")

    lines.append(f"{ts} UTC | Session: *{sig.session_label}*")
    lines.append(f"Signal: *{sig.direction}* | Risk: *{sig.risk_tag}*")
    lines.append(f"Mode: `{sig.timeframe_mode}`")
    lines.append("")

    t = sig.trend_state
    lines.append("*Trend (HTF)*")
    lines.append(f"- TF: `{t.timeframe}` | Dir: *{t.direction}* | Close: {t.close:.2f}")
    lines.append(
        f"- EMA{cfg.ema_fast}: {t.ema_fast:.2f} | EMA{cfg.ema_slow}: {t.ema_slow:.2f} | Slope: {t.slope}"
    )
    lines.append("")

    lines.append("*Filters & Confirmations*")
    joined = ", ".join(sig.confirmations) if sig.confirmations else "None"
    lines.append(f"- Confirmations: *{sig.confirmations_passed}/{sig.confirmations_required}* ({joined})")
    lines.append(f"- ADX: {sig.adx_value:.1f} (min {sig.adx_min})")
    lines.append(f"- News: {sig.news_status}")
    lines.append("")

    lines.append("*Reason for Trade*")
    for b in sig.reason_bullets[:6]:
        lines.append(f"- {b}")

    return "\n".join(lines)


def _log_dedupe(log: logging.Logger, key: str, message: str, dedupe_state: dict, every_seconds: int = 300) -> None:
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    last_t = dedupe_state.get(key, 0.0)
    if (now - last_t) >= every_seconds:
        log.info(message)
        dedupe_state[key] = now


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

            # =========================
            # Block weekends
            # =========================
            if now.weekday() >= 5:  # 5=Sat, 6=Sun
                _log_dedupe(
                    log,
                    key="weekend_block",
                    message=f"Weekend detected (UTC). Bot paused. Sleeping {cfg.poll_seconds}s...",
                    dedupe_state=dedupe_state,
                    every_seconds=900,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # =========================
            # Sessions gate
            # =========================
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

            # =========================
            # News (soft-fail + cached inside)
            # =========================
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

            # =========================
            # Fetch candles (cached)
            # =========================
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

            # =========================
            # HTF trend
            # =========================
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

            # =========================
            # Cooldown
            # =========================
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

            # =========================
            # ADX gate (keep your existing logic)
            # =========================
            strict_min = cfg.adx_min
            relaxed_min = max(10.0, cfg.adx_min - 3.0)
            strong_conf = len(confirmations) >= (cfg.min_confirmations + 1)

            effective_min = relaxed_min if strong_conf else strict_min
            adx_ok = adx_val >= effective_min

            if not adx_ok:
                _log_dedupe(
                    log,
                    key="adx_failed",
                    message=f"ADX filter failed: {adx_val:.4f} < {effective_min:.4f} (strict={strict_min:.4f}, strong_conf={strong_conf})",
                    dedupe_state=dedupe_state,
                    every_seconds=180,
                )
                time.sleep(cfg.poll_seconds)
                continue

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

            reasons: list[str] = []
            reasons.append(
                f"HTF {trend.timeframe} trend {trend.direction} (EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})"
            )
            reasons.append(confirm_reason)
            reasons.extend(reason_bullets)

            tf_mode = "H1→M15→M5" if trend_tf == "1h" else "M15→M5 (fallback)"
            risk_tag = risk_tag_from_context(trend_tf)
            news_status = ("⚠️ " + news.message) if news.is_high_impact else "Normal"

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
                adx_min=effective_min,
                news_status=news_status,
                reason_bullets=reasons,
            )

            # =========================
            # ENTRY: LIVE QUOTE (fixes discrepancy)
            # =========================
            entry_price = td.fetch_quote_cached(cfg.symbol, ttl_seconds=2, now_utc=now).price

            # =========================
            # Risk model: M15 ATR -> SL, TP1, TP2
            # =========================
            atr_val = _atr(df_m15, period=cfg.atr_period)
            tp1, tp2, sl = _compute_tp1_tp2_sl_from_atr(
                entry=float(entry_price),
                direction=direction,
                atr_value=atr_val,
                sl_mult=cfg.atr_sl_mult,
                tp2_mult=cfg.atr_tp_mult,
            )

            tg.send_message(
                fmt_signal(
                    cfg=cfg,
                    sig=sig,
                    entry_price=float(entry_price),
                    tp1=tp1,
                    tp2=tp2,
                    sl=sl,
                )
            )

            last_signal_time = now
            last_direction = direction
            log.info("Signal sent: %s %s | entry=%.2f | TP1=%.2f | TP2=%.2f | SL=%.2f", direction, cfg.symbol, entry_price, tp1, tp2, sl)

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