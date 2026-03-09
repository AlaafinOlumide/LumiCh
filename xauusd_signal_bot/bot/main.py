from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
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
    trigger_entry_m1,
    Signal,
    atr as atr_value,
    compute_tp_sl_from_atr,
    confidence_score,
)
from .telegram import TelegramClient


cfg_global: Config


@dataclass
class SetupState:
    created_utc: dt.datetime
    expires_utc: dt.datetime
    direction: str
    trend_tf: str
    session_label: str
    risk_tag: str
    timeframe_mode: str
    zone_low: float
    zone_high: float
    atr_m15: float
    confirmations: list[str]
    adx_val: float
    reasons: list[str]
    news_status: str
    trend_state: object


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
    return now.weekday() >= 5


def _safe_entry_from_quote_or_close(
    td: TwelveDataClient,
    symbol: str,
    df_m5,
    now: dt.datetime,
    offset: float,
) -> tuple[float, str]:
    try:
        q = td.fetch_quote_cached(symbol, ttl_seconds=2, now_utc=now)
        return float(q.price) + offset, "QUOTE"
    except (TwelveDataError, Exception):
        return float(df_m5["close"].iloc[-1]) + offset, "M5_CLOSE"


def fmt_setup(setup: SetupState, confidence: int, emoji: str, effective_setup_min: int) -> str:
    ts = setup.created_utc.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"Xauusd: {setup.direction} (SETUP)")
    lines.append(f"ENTRY ZONE: {setup.zone_low:.2f} - {setup.zone_high:.2f}  (ATR15={setup.atr_m15:.2f})")
    lines.append("Trigger rule: waits for price to enter zone + M1 rejection candle")
    lines.append(f"Confidence: {confidence}% {emoji}")
    lines.append("")
    lines.append(f"{ts} UTC | Session: *{setup.session_label}*")
    lines.append(f"Mode: `{setup.timeframe_mode}` | Risk: *{setup.risk_tag}*")
    lines.append(f"Expires: {setup.expires_utc.strftime('%H:%M')} UTC")
    lines.append("")
    lines.append("*Filters & Confirmations (Setup Quality)*")
    joined = ", ".join(setup.confirmations) if setup.confirmations else "None"
    lines.append(f"- Confirmations: *{len(setup.confirmations)}/{effective_setup_min}* ({joined})")
    lines.append(f"- ADX: {setup.adx_val:.1f} (min {cfg_global.adx_min:.1f})")
    lines.append(f"- News: {setup.news_status}")
    lines.append("")
    lines.append("*Reason for Setup*")
    for b in setup.reasons[:8]:
        lines.append(f"- {b}")
    return "\n".join(lines)


def fmt_entry(sig: Signal, entry_source: str, trigger_reason: str) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"Xauusd: {sig.direction} (ENTRY TRIGGERED)")
    lines.append(f"ENTRY: {sig.entry_price:.2f}  ({entry_source})")
    lines.append(f"TP1: {sig.tp1:.2f}")
    lines.append(f"TP2: {sig.tp2:.2f} (TP2 may not be reached — take profit when convenient)")
    lines.append(f"SL: {sig.sl:.2f}")
    lines.append(f"RR (to TP2): {sig.rr:.2f}")
    lines.append(f"Confidence: {sig.confidence}% {sig.confidence_emoji}")
    lines.append("")
    lines.append(f"{ts} UTC | Session: *{sig.session_label}*")
    lines.append(f"Signal: *{sig.direction}* | Risk: *{sig.risk_tag}*")
    lines.append(f"Mode: `{sig.timeframe_mode}`")
    lines.append("")
    lines.append("*Trigger*")
    lines.append(f"- {trigger_reason}")
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

    td = TwelveDataClient(
        cfg.twelvedata_api_key,
        timeout=cfg.http_timeout,
        max_retries=cfg.http_max_retries,
        backoff_seconds=cfg.http_backoff_seconds,
    )
    tg = TelegramClient(cfg.telegram_bot_token, cfg.telegram_chat_id)

    dedupe_state: dict[str, float] = {}
    last_entry_time: dt.datetime | None = None
    last_entry_direction: str | None = None
    setup_state: SetupState | None = None

    broker_offset = float(cfg.broker_price_offset)
    effective_setup_min = max(3, int(cfg.min_confirmations) - 1)

    log.info(
        "Bot started. Sessions=%s | Symbol=%s | setup_min=%s | entry_min=%s",
        cfg.trading_sessions,
        cfg.symbol,
        effective_setup_min,
        cfg.min_confirmations,
    )

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)

            if cfg.block_weekends and _is_weekend_utc(now):
                _log_dedupe(
                    log,
                    key="weekend_block",
                    message=f"Weekend (UTC). Bot paused. Sleeping {cfg.poll_seconds}s...",
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

            news = check_high_impact_news(
                provider=cfg.news_api_provider or "fmp",
                api_key=cfg.news_api_key or "",
                base_url=cfg.news_base_url,
                lookahead_min=cfg.news_lookahead_min,
                cooldown_after_min=cfg.news_cooldown_after_min,
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

            trend = detect_trend(trend_df, trend_tf, cfg.ema_fast, cfg.ema_slow, cfg.ema_slope_bars)
            if trend.direction == "NEUTRAL":
                _log_dedupe(log, "trend_neutral", "HTF trend neutral. No setups.", dedupe_state, 300)
                setup_state = None
                time.sleep(cfg.poll_seconds)
                continue

            ok_confirm, confirm_reason = m15_confirms(df_m15, trend.direction, cfg.ema_fast, cfg.rsi_period)
            if not ok_confirm:
                _log_dedupe(log, "m15_not_confirm", f"M15 did not confirm. {confirm_reason}", dedupe_state, 180)
                setup_state = None
                time.sleep(cfg.poll_seconds)
                continue

            direction = "BUY" if trend.direction == "BULL" else "SELL"

            if last_entry_time is not None:
                mins_since = (now - last_entry_time).total_seconds() / 60
                if mins_since < cfg.cooldown_minutes and last_entry_direction == direction:
                    _log_dedupe(log, "cooldown", f"Cooldown active ({mins_since:.1f} min).", dedupe_state, 180)
                    time.sleep(cfg.poll_seconds)
                    continue

            if setup_state is not None:
                if now >= setup_state.expires_utc:
                    _log_dedupe(log, "setup_expired", "Setup expired. Waiting for new setup.", dedupe_state, 120)
                    setup_state = None
                elif setup_state.direction != direction:
                    _log_dedupe(log, "setup_flip", "Trend direction flipped. Resetting setup.", dedupe_state, 120)
                    setup_state = None
                else:
                    entry_ref, entry_src = _safe_entry_from_quote_or_close(
                        td, cfg.symbol, df_m5, now, broker_offset
                    )

                    trigger_ttl = 60 if cfg.trigger_tf == "1min" else 300
                    df_trigger = td.fetch_time_series_cached(
                        cfg.symbol,
                        cfg.trigger_tf,
                        outputsize=200,
                        ttl_seconds=trigger_ttl,
                        now_utc=now,
                    ).df

                    trig_ok, trig_reason = trigger_entry_m1(
                        df_m1=df_trigger,
                        direction=setup_state.direction,
                        ema_period=cfg.trigger_ema_period,
                        rsi_period=cfg.rsi_period,
                        rsi_min_buy=cfg.trigger_rsi_min_buy,
                        rsi_max_sell=cfg.trigger_rsi_max_sell,
                        zone_low=setup_state.zone_low,
                        zone_high=setup_state.zone_high,
                        live_price=entry_ref,
                    )

                    if trig_ok:
                        atr_m15 = setup_state.atr_m15
                        tp2, sl, rr = compute_tp_sl_from_atr(
                            entry=entry_ref,
                            direction=setup_state.direction,
                            atr_value=atr_m15,
                            sl_mult=cfg.atr_sl_mult,
                            tp_mult=cfg.atr_tp_mult,
                        )
                        if setup_state.direction == "BUY":
                            tp1 = entry_ref + (tp2 - entry_ref) * 0.5
                        else:
                            tp1 = entry_ref - (entry_ref - tp2) * 0.5

                        conf, emoji = confidence_score(
                            confirmations_passed=len(setup_state.confirmations),
                            confirmations_required=effective_setup_min,
                            adx_value=setup_state.adx_val,
                            adx_min=float(cfg.adx_min),
                            news_is_normal=not news.is_high_impact,
                        )

                        sig = Signal(
                            symbol=cfg.symbol,
                            timestamp_utc=now,
                            direction=setup_state.direction,
                            risk_tag=setup_state.risk_tag,
                            timeframe_mode=setup_state.timeframe_mode,
                            session_label=setup_state.session_label,
                            trend_state=setup_state.trend_state,
                            confirmations=setup_state.confirmations,
                            confirmations_passed=len(setup_state.confirmations),
                            confirmations_required=effective_setup_min,
                            adx_value=setup_state.adx_val,
                            adx_min=float(cfg.adx_min),
                            news_status=setup_state.news_status,
                            reason_bullets=setup_state.reasons,
                            entry_price=entry_ref,
                            sl=sl,
                            tp1=tp1,
                            tp2=tp2,
                            rr=rr,
                            confidence=conf,
                            confidence_emoji=emoji,
                        )

                        tg.send_message(fmt_entry(sig, entry_source=entry_src, trigger_reason=trig_reason))
                        last_entry_time = now
                        last_entry_direction = setup_state.direction
                        setup_state = None
                        log.info("ENTRY triggered: %s %s @ %.2f", sig.direction, cfg.symbol, entry_ref)
                    else:
                        _log_dedupe(
                            log,
                            key="waiting_trigger",
                            message=f"Waiting trigger: {trig_reason}",
                            dedupe_state=dedupe_state,
                            every_seconds=180,
                        )

                    time.sleep(cfg.poll_seconds)
                    continue

            confirmations, adx_val, _adx_pass, reason_bullets = score_entry_m5(df_m5, direction, cfg)

            if len(confirmations) < effective_setup_min:
                _log_dedupe(
                    log,
                    "not_enough_setup_confirm",
                    f"Not enough setup confirmations: {len(confirmations)} (need {effective_setup_min})",
                    dedupe_state,
                    180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            relaxed_adx_min = max(18.0, float(cfg.adx_min) - 2.0)
            if adx_val < relaxed_adx_min:
                _log_dedupe(
                    log,
                    "adx_failed_setup",
                    f"ADX setup filter failed: {adx_val:.1f} < {relaxed_adx_min:.1f}",
                    dedupe_state,
                    180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            atr_m15 = atr_value(df_m15, cfg.atr_period)
            entry_ref, entry_src = _safe_entry_from_quote_or_close(td, cfg.symbol, df_m5, now, broker_offset)

            zone_half = max(
                atr_m15 * float(cfg.entry_zone_atr_mult),
                float(cfg.entry_zone_min_width),
            )
            zone_low = entry_ref - zone_half
            zone_high = entry_ref + zone_half

            tf_mode = "H1→M15→M5" if trend_tf == "1h" else "M15→M5 (fallback)"
            risk_tag = risk_tag_from_context(trend_tf)
            news_status = ("⚠️ " + news.message) if news.is_high_impact else "Normal"

            reasons: list[str] = []
            reasons.append(
                f"HTF {trend.timeframe} trend {trend.direction} (EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})"
            )
            reasons.append(confirm_reason)
            reasons.extend(reason_bullets)

            expires = now + dt.timedelta(minutes=int(cfg.setup_ttl_minutes))
            setup_state = SetupState(
                created_utc=now,
                expires_utc=expires,
                direction=direction,
                trend_tf=trend_tf,
                session_label=s_label,
                risk_tag=risk_tag,
                timeframe_mode=tf_mode,
                zone_low=zone_low,
                zone_high=zone_high,
                atr_m15=atr_m15,
                confirmations=confirmations,
                adx_val=adx_val,
                reasons=reasons,
                news_status=news_status,
                trend_state=trend,
            )

            conf, emoji = confidence_score(
                confirmations_passed=len(confirmations),
                confirmations_required=effective_setup_min,
                adx_value=adx_val,
                adx_min=relaxed_adx_min,
                news_is_normal=not news.is_high_impact,
            )

            tg.send_message(fmt_setup(setup_state, confidence=conf, emoji=emoji, effective_setup_min=effective_setup_min))
            log.info(
                "SETUP sent: %s %s zone=[%.2f..%.2f] ref=%.2f(%s)",
                direction,
                cfg.symbol,
                zone_low,
                zone_high,
                entry_ref,
                entry_src,
            )

        except TwelveDataQuotaError:
            log.error("TwelveData daily credits exhausted. Sleeping 3600s.")
            time.sleep(3600)
            continue
        except Exception as e:
            log.exception("Loop error: %s", e)

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
