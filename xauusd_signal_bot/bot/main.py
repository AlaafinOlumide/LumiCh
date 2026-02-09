from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Optional

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
    Signal,
    atr_value,
    is_value_zone_m15,
    is_range_blocked_m15,
    compute_tp_sl_from_atr,
)
from .telegram import TelegramClient


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _ensure_chronological(df):
    """
    Make sure DF is oldest->newest so iloc[-1] is ALWAYS the latest candle.
    Handles both datetime index and 'datetime' column.
    """
    if df is None or len(df) < 2:
        return df

    if "datetime" in df.columns:
        try:
            df = df.copy()
            df["datetime"] = df["datetime"].astype(str)
            df = df.sort_values("datetime")
            df = df.reset_index(drop=True)
            return df
        except Exception:
            return df

    # index sort (common)
    try:
        return df.sort_index()
    except Exception:
        return df


def _confidence_from_confirmations(passed: int, required: int, adx_val: float, adx_min: float) -> str:
    if required <= 0:
        return "MEDIUM"
    ratio = passed / required
    if ratio >= 1.0 and adx_val >= (adx_min + 10):
        return "HIGH"
    if ratio >= 1.0 and adx_val >= adx_min:
        return "MEDIUM"
    if ratio >= 0.8 and adx_val >= adx_min:
        return "MEDIUM"
    return "LOW"


def fmt_signal(
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
    lines.append(f"- EMA{cfg_global.ema_fast}: {t.ema_fast:.2f} | EMA{cfg_global.ema_slow}: {t.ema_slow:.2f} | Slope: {t.slope}")
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


cfg_global: Config


def _log_dedupe(log: logging.Logger, key: str, message: str, dedupe_state: dict, every_seconds: int = 300) -> None:
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    last_t = dedupe_state.get(key, 0.0)
    if (now - last_t) >= every_seconds:
        log.info(message)
        dedupe_state[key] = now


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

    td = TwelveDataClient(cfg.twelvedata_api_key)
    tg = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)

    last_signal_time: dt.datetime | None = None
    last_direction: str | None = None
    dedupe_state: dict[str, float] = {}

    # Defaults (safe if not in config)
    atr_period = int(getattr(cfg, "atr_period", 14))
    sl_atr_mult = float(getattr(cfg, "sl_atr_mult", 1.2))
    tp_atr_mult = float(getattr(cfg, "tp_atr_mult", 1.8))
    value_zone_atr_frac = float(getattr(cfg, "value_zone_atr_frac", 0.25))   # your requested logic
    ema_sep_atr_frac = float(getattr(cfg, "ema_sep_atr_frac", 0.15))

    log.info(
        "Bot started. Sessions=%s | Symbol=%s | min_confirmations=%s",
        cfg.trading_sessions,
        cfg.symbol,
        cfg.min_confirmations,
    )

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)

            # ✅ block weekends
            if now.weekday() >= 5:  # 5=Sat,6=Sun
                _log_dedupe(log, "weekend_block", f"Weekend block active. Sleeping {cfg.poll_seconds}s...", dedupe_state, 900)
                time.sleep(cfg.poll_seconds)
                continue

            if not now_in_sessions_utc(sessions, now):
                _log_dedupe(log, "outside_sessions", f"Outside trading sessions. Sleeping {cfg.poll_seconds}s...", dedupe_state, 600)
                time.sleep(cfg.poll_seconds)
                continue

            s_label = session_label(sessions, now)

            # News check (soft-fail)
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
                _log_dedupe(log, "blocked_news", f"Signals blocked due to news: {news.message}", dedupe_state, 300)
                time.sleep(cfg.poll_seconds)
                continue

            # Fetch (cached) + force chronological order ✅
            try:
                trend_tf = "1h"
                trend_df = td.fetch_time_series_cached(cfg.symbol, "1h", outputsize=200, ttl_seconds=3600, now_utc=now).df
                trend_df = _ensure_chronological(trend_df)
                if len(trend_df) < 100:
                    raise RuntimeError("Insufficient H1 candles")
            except Exception as e:
                log.warning("H1 unavailable (%s). Falling back to M15 as HTF.", e)
                trend_tf = "15min"
                trend_df = td.fetch_time_series_cached(cfg.symbol, "15min", outputsize=200, ttl_seconds=900, now_utc=now).df
                trend_df = _ensure_chronological(trend_df)

            df_m15 = td.fetch_time_series_cached(cfg.symbol, "15min", outputsize=200, ttl_seconds=900, now_utc=now).df
            df_m5 = td.fetch_time_series_cached(cfg.symbol, "5min", outputsize=200, ttl_seconds=300, now_utc=now).df
            df_m15 = _ensure_chronological(df_m15)
            df_m5 = _ensure_chronological(df_m5)

            if len(df_m5) < 100 or len(df_m15) < 100 or len(trend_df) < 100:
                _log_dedupe(log, "insufficient_data", "Insufficient data. Sleeping.", dedupe_state, 300)
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

            # ✅ M15 ATR-based range filter (blocks chop)
            blocked, range_reason = is_range_blocked_m15(df_m15, cfg.ema_fast, cfg.ema_slow, atr_period, ema_sep_atr_frac)
            if blocked:
                _log_dedupe(log, "range_block", f"Range blocked. {range_reason}", dedupe_state, 180)
                time.sleep(cfg.poll_seconds)
                continue

            # ✅ M15 ATR-based value zone (prevents chasing)
            vz_ok, vz_reason = is_value_zone_m15(df_m15, cfg.ema_fast, atr_period, value_zone_atr_frac)
            if not vz_ok:
                _log_dedupe(log, "value_zone_fail", f"Value zone failed. {vz_reason}", dedupe_state, 180)
                time.sleep(cfg.poll_seconds)
                continue

            # Cooldown
            if last_signal_time is not None:
                mins_since = (now - last_signal_time).total_seconds() / 60
                if mins_since < cfg.cooldown_minutes and last_direction == direction:
                    _log_dedupe(log, "cooldown", f"Cooldown active ({mins_since:.1f} min). Skipping.", dedupe_state, 180)
                    time.sleep(cfg.poll_seconds)
                    continue

            confirmations, adx_val, _, reason_bullets = score_entry_m5(df_m5, direction, cfg)

            # Gate confirmations (keep your approach)
            effective_min_conf = int(getattr(cfg, "effective_min_confirmations", max(1, cfg.min_confirmations - 1)))
            if len(confirmations) < effective_min_conf:
                _log_dedupe(log, "not_enough_confirm", f"Not enough confirmations: {len(confirmations)} (need {effective_min_conf})", dedupe_state, 180)
                time.sleep(cfg.poll_seconds)
                continue

            # ADX gate (strict)
            if adx_val < float(cfg.adx_min):
                _log_dedupe(log, "adx_failed", f"ADX filter failed: {adx_val:.4f} < {float(cfg.adx_min):.4f}", dedupe_state, 180)
                time.sleep(cfg.poll_seconds)
                continue

            # ✅ entry = latest M5 close (now correct after sorting)
            entry_price = float(df_m5["close"].astype(float).iloc[-1])

            # ✅ M15 ATR for TP/SL
            a15 = atr_value(df_m15, atr_period)
            tp2, sl, rr = compute_tp_sl_from_atr(entry_price, direction, a15, sl_atr_mult, tp_atr_mult)

            # ✅ TP1 = half distance to TP2
            if direction == "BUY":
                tp1 = entry_price + (tp2 - entry_price) * 0.5
            else:
                tp1 = entry_price - (entry_price - tp2) * 0.5

            reasons: list[str] = []
            reasons.append(f"HTF {trend.timeframe} trend {trend.direction} (EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})")
            reasons.append(confirm_reason)
            reasons.append(range_reason)
            reasons.append(vz_reason)
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
                confirmations_required=effective_min_conf,
                adx_value=adx_val,
                adx_min=float(cfg.adx_min),
                news_status=news_status,
                reason_bullets=reasons,
                entry_price=entry_price,
                tp=tp2,
                sl=sl,
                rr=rr,
            )

            tg.send_message(fmt_signal(sig, entry_price=entry_price, tp1=tp1, tp2=tp2, sl=sl))

            last_signal_time = now
            last_direction = direction
            log.info("Signal sent: %s %s | entry=%.2f | tp1=%.2f | tp2=%.2f | sl=%.2f", direction, cfg.symbol, entry_price, tp1, tp2, sl)

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