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


def fmt_signal(sig: Signal, entry_price: float, tp: Optional[float] = None, sl: Optional[float] = None) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    confidence = _confidence_from_confirmations(
        passed=sig.confirmations_passed,
        required=sig.confirmations_required,
        adx_val=sig.adx_value,
        adx_min=sig.adx_min,
    )

    tp_txt = f"{tp:.2f}" if tp is not None else "TBD"
    sl_txt = f"{sl:.2f}" if sl is not None else "TBD"

    lines: list[str] = []
    lines.append(f"Xauusd: {sig.direction}")
    lines.append(f"ENTRY PRICE: {entry_price:.2f}")
    lines.append(f"TP: {tp_txt}")
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
        f"- EMA{cfg_global.ema_fast}: {t.ema_fast:.2f} | EMA{cfg_global.ema_slow}: {t.ema_slow:.2f} | Slope: {t.slope}"
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


cfg_global: Config


def _log_dedupe(log: logging.Logger, key: str, message: str, dedupe_state: dict, every_seconds: int = 300) -> None:
    """
    Rate-limit repeated logs of the same reason.
    """
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

    # log dedupe state
    dedupe_state: dict[str, float] = {}

    log.info("Bot started. Sessions=%s | Symbol=%s", cfg.trading_sessions, cfg.symbol)

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)

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

            # Fetch candles (cached in client)
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

            confirmations, adx_val, adx_pass, reason_bullets = score_entry_m5(df_m5, direction, cfg)

            # ---------
            # Smarter ADX gate to increase signals safely:
            # If confirmations are strong, allow slightly lower ADX.
            # ---------
            strict_min = cfg.adx_min
            relaxed_min = max(10.0, cfg.adx_min - 3.0)  # e.g., 20 -> 17
            strong_conf = len(confirmations) >= (cfg.min_confirmations + 1)

            effective_min = relaxed_min if strong_conf else strict_min
            adx_ok = adx_val >= effective_min

            if not adx_ok:
                _log_dedupe(
                    log,
                    key="adx_failed",
                    message=f"ADX filter failed: {adx_val:.1f} < {effective_min:.1f} (strict={strict_min:.1f}, strong_conf={strong_conf})",
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
                adx_min=effective_min,  # show effective min in the message
                news_status=news_status,
                reason_bullets=reasons,
            )

            # ENTRY = latest M5 close
            try:
                entry_price = float(df_m5["close"].iloc[-1])
            except Exception:
                entry_price = float(trend.close)

            tp = None
            sl = None

            tg.send_message(fmt_signal(sig, entry_price=entry_price, tp=tp, sl=sl))

            last_signal_time = now
            last_direction = direction
            log.info("Signal sent: %s %s | entry=%.2f", direction, cfg.symbol, entry_price)

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