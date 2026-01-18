from __future__ import annotations

import datetime as dt
import logging
import time

from dotenv import load_dotenv

from .config import Config
from .data import TwelveDataClient
from .high_impact_news import check_high_impact_news
from .sessions import now_in_sessions_utc, parse_sessions, session_label
from .strategy import detect_trend, m15_confirms, risk_tag_from_context, score_entry_m5, Signal
from .telegram import TelegramClient


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def fmt_signal(sig: Signal) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"*XAUUSD* | {ts} UTC | Session: *{sig.session_label}*")
    lines.append(f"Signal: *{sig.direction}* | Risk: *{sig.risk_tag}*")
    lines.append(f"Mode: `{sig.timeframe_mode}`")
    lines.append("")

    t = sig.trend_state
    lines.append("*Trend (HTF)*")
    lines.append(f"- TF: `{t.timeframe}` | Dir: *{t.direction}* | Close: {t.close:.2f}")
    lines.append(f"- EMA{cfg_global.ema_fast}: {t.ema_fast:.2f} | EMA{cfg_global.ema_slow}: {t.ema_slow:.2f} | Slope: {t.slope}")
    lines.append("")

    lines.append("*Filters & Confirmations*")
    lines.append(f"- Confirmations: *{sig.confirmations_passed}/{sig.confirmations_required}* (" + ", ".join(sig.confirmations) + ")")
    lines.append(f"- ADX: {sig.adx_value:.1f} (min {sig.adx_min})")
    lines.append(f"- News: {sig.news_status}")
    lines.append("")

    lines.append("*Reason for Trade*")
    for b in sig.reason_bullets[:6]:
        lines.append(f"- {b}")

    return "\n".join(lines)


cfg_global: Config


def main() -> None:
    global cfg_global
    load_dotenv()
    setup_logging()

    cfg = Config.load()
    cfg_global = cfg

    log = logging.getLogger("xauusd_bot")

    sessions = parse_sessions(cfg.trading_sessions)
    if not sessions:
        raise RuntimeError("No TRADING_SESSIONS configured")

    td = TwelveDataClient(cfg.twelvedata_api_key)
    tg = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)

    last_signal_time: dt.datetime | None = None
    last_direction: str | None = None

    log.info("Bot started. Sessions=%s", cfg.trading_sessions)

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)
            if not now_in_sessions_utc(sessions, now):
                log.info("Outside trading sessions. Sleeping %ss...", cfg.poll_seconds)
                time.sleep(cfg.poll_seconds)
                continue

            s_label = session_label(sessions, now)

            # News check
            news = check_high_impact_news(
                provider=cfg.news_api_provider or "fmp",
                api_key=cfg.news_api_key or "",
                base_url=cfg.news_base_url,
                lookahead_min=cfg.news_lookahead_min,
                cooldown_after_min=cfg.news_cooldown_after_min,
            )
            if news.is_high_impact and cfg.news_mode == "BLOCK":
                log.info("Signals blocked due to news: %s", news.message)
                time.sleep(cfg.poll_seconds)
                continue

            # Fetch candles
            # H1 trend (fallback to M15)
            trend_df = None
            trend_tf = "1h"
            try:
                trend_df = td.fetch_time_series(cfg.symbol, "1h", outputsize=200).df
                if len(trend_df) < 100:
                    raise RuntimeError("Insufficient H1 candles")
            except Exception as e:
                log.warning("H1 unavailable (%s). Falling back to M15 as HTF.", e)
                trend_tf = "15min"
                trend_df = td.fetch_time_series(cfg.symbol, "15min", outputsize=200).df

            df_m15 = td.fetch_time_series(cfg.symbol, "15min", outputsize=200).df
            df_m5 = td.fetch_time_series(cfg.symbol, "5min", outputsize=200).df

            if len(df_m5) < 100 or len(df_m15) < 100 or len(trend_df) < 100:
                log.info("Insufficient data. Sleeping.")
                time.sleep(cfg.poll_seconds)
                continue

            trend = detect_trend(trend_df, trend_tf, cfg.ema_fast, cfg.ema_slow, cfg.ema_slope_bars)
            if trend.direction == "NEUTRAL":
                log.info("HTF trend neutral. No signals.")
                time.sleep(cfg.poll_seconds)
                continue

            ok_confirm, confirm_reason = m15_confirms(df_m15, trend.direction, cfg.ema_fast, cfg.rsi_period)
            if not ok_confirm:
                log.info("M15 did not confirm. %s", confirm_reason)
                time.sleep(cfg.poll_seconds)
                continue

            direction = "BUY" if trend.direction == "BULL" else "SELL"

            # Cooldown logic
            if last_signal_time is not None:
                mins_since = (now - last_signal_time).total_seconds() / 60
                if mins_since < cfg.cooldown_minutes and last_direction == direction:
                    log.info("Cooldown active (%.1f min). Skipping.", mins_since)
                    time.sleep(cfg.poll_seconds)
                    continue

            confirmations, adx_val, adx_pass, reason_bullets = score_entry_m5(df_m5, direction, cfg)

            if not adx_pass:
                log.info("ADX filter failed: %.1f < %.1f", adx_val, cfg.adx_min)
                time.sleep(cfg.poll_seconds)
                continue

            if len(confirmations) < cfg.min_confirmations:
                log.info("Not enough confirmations: %s (need %s)", len(confirmations), cfg.min_confirmations)
                time.sleep(cfg.poll_seconds)
                continue

            # Build reasons - include HTF & M15 reasons
            reasons = []
            reasons.append(f"HTF {trend.timeframe} trend {trend.direction} (EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})")
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
                adx_min=cfg.adx_min,
                news_status=news_status,
                reason_bullets=reasons,
            )

            msg = fmt_signal(sig)
            tg.send_message(msg)

            last_signal_time = now
            last_direction = direction
            log.info("Signal sent: %s %s", direction, cfg.symbol)

        except Exception as e:
            logging.getLogger("xauusd_bot").exception("Loop error: %s", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
