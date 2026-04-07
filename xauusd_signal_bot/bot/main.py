from __future__ import annotations

"""
main.py — LumiCh bot entry point.

Key improvements vs prior version:
- Zone bias corrected: BUY zones pull below current price, SELL zones push above
- Pullback trend states (BULL_PULLBACK / BEAR_PULLBACK) generate signals
- M15 trigger added for strong H1 trends
- Session window extended to 22:00 UTC (catches NY close)
- Direction-specific cooldown: shorter in high-ADX environments
- ADX minimum enforced without a "relaxed" bypass
- setup_ttl reduced guidance added (env var respected, suggested 90 min)
"""

import datetime as dt
import logging
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from .config import Config
from .data import TwelveDataClient, TwelveDataQuotaError, TwelveDataError
from .high_impact_news import check_high_impact_news
from .sessions import now_in_sessions_utc, parse_sessions, session_label
from .strategy import (
    detect_trend,
    trend_is_tradeable,
    m15_confirms,
    risk_tag_from_context,
    score_entry_m5,
    trigger_entry_m1_confirmed,
    trigger_entry_m5_confirmed,
    trigger_entry_m15_confirmed,
    is_market_too_compressed,
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
    zone_mid: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _log_dedupe(
    log: logging.Logger,
    key: str,
    message: str,
    dedupe_state: dict,
    every_seconds: int = 300,
) -> None:
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    if (now - dedupe_state.get(key, 0.0)) >= every_seconds:
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


def _zone_too_close(
    new_mid: float,
    old_mid: float,
    atr_m15: float,
    cfg: Config,
) -> bool:
    buffer_size = max(atr_m15 * cfg.zone_reentry_buffer_atr, 2.0)
    return abs(new_mid - old_mid) < buffer_size


def _build_entry_zone(
    entry_ref: float,
    direction: str,
    atr_m15: float,
    entry_zone_atr_mult: float,
    entry_zone_min_width: float,
) -> tuple[float, float, float]:
    """
    Direction-biased entry zone:
    - BUY  → zone extends BELOW current price (expecting pullback into support)
    - SELL → zone extends ABOVE current price (expecting pullback into resistance)

    Returns (zone_low, zone_high, zone_mid).
    """
    half = max(atr_m15 * entry_zone_atr_mult, entry_zone_min_width)

    if direction == "BUY":
        # Zone sits beneath current price: from (price - full_width) to price
        zone_low = entry_ref - half * 2
        zone_high = entry_ref
    else:
        # Zone sits above current price: from price to (price + full_width)
        zone_low = entry_ref
        zone_high = entry_ref + half * 2

    zone_mid = (zone_low + zone_high) / 2.0
    return zone_low, zone_high, zone_mid


def _effective_cooldown(adx_val: float, adx_min: float, base_cooldown_minutes: int) -> int:
    """
    Shorter cooldown when ADX is strong (trending strongly) — allows
    continuation entries.  Longer cooldown in weak / range environments.
    """
    if adx_val >= adx_min + 8:
        return max(10, base_cooldown_minutes // 3)
    if adx_val >= adx_min:
        return max(15, base_cooldown_minutes // 2)
    return base_cooldown_minutes


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def fmt_setup(
    setup: SetupState,
    confidence: int,
    emoji: str,
    effective_setup_min: int,
) -> str:
    ts = setup.created_utc.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"Xauusd: {setup.direction} (SETUP)",
        f"ENTRY ZONE: {setup.zone_low:.2f} – {setup.zone_high:.2f}",
        f"(ATR15={setup.atr_m15:.2f})",
        "Trigger: price enters zone + M1 / M5 / M15 confirmation",
        f"Confidence: {confidence}% {emoji}",
        "",
        f"{ts} UTC | Session: *{setup.session_label}*",
        f"Mode: `{setup.timeframe_mode}` | Risk: *{setup.risk_tag}*",
        f"Expires: {setup.expires_utc.strftime('%H:%M')} UTC",
        "",
        "*Filters & Confirmations (Setup Quality)*",
        f"- Confirmations: *{len(setup.confirmations)}/{effective_setup_min}* "
        f"({', '.join(setup.confirmations) or 'None'})",
        f"- ADX: {setup.adx_val:.1f} (min {cfg_global.adx_min:.1f})",
        f"- News: {setup.news_status}",
        "",
        "*Reason for Setup*",
    ]
    for b in setup.reasons[:8]:
        lines.append(f"- {b}")
    return "\n".join(lines)


def fmt_entry(sig: Signal, entry_source: str, trigger_reason: str) -> str:
    ts = sig.timestamp_utc.strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        f"Xauusd: {sig.direction} (ENTRY TRIGGERED)",
        f"ENTRY: {sig.entry_price:.2f} ({entry_source})",
        f"TP1: {sig.tp1:.2f}",
        f"TP2: {sig.tp2:.2f} (TP2 may not be reached — take profit when convenient)",
        f"SL: {sig.sl:.2f}",
        f"RR (to TP2): {sig.rr:.2f}",
        f"Confidence: {sig.confidence}% {sig.confidence_emoji}",
        "",
        f"{ts} UTC | Session: *{sig.session_label}*",
        f"Signal: *{sig.direction}* | Risk: *{sig.risk_tag}*",
        f"Mode: `{sig.timeframe_mode}`",
        "",
        "*Trigger*",
        f"- {trigger_reason}",
        "",
        "*Reason for Trade*",
    ]
    for b in sig.reason_bullets[:8]:
        lines.append(f"- {b}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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
    last_entry_adx: float = 0.0
    last_zone_mid: float | None = None
    last_zone_time: dt.datetime | None = None
    setup_state: SetupState | None = None

    broker_offset = float(cfg.broker_price_offset)
    effective_setup_min = max(2, int(cfg.min_confirmations))
    effective_entry_min = max(3, int(cfg.min_confirmations))

    log.info(
        "Bot started. Sessions=%s | Symbol=%s | setup_min=%s | entry_min=%s | ADX_MIN=%s",
        cfg.trading_sessions,
        cfg.symbol,
        effective_setup_min,
        effective_entry_min,
        cfg.adx_min,
    )

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)

            if cfg.block_weekends and _is_weekend_utc(now):
                _log_dedupe(
                    log, "weekend_block",
                    f"Weekend UTC. Bot paused ({cfg.poll_seconds}s).",
                    dedupe_state, 900,
                )
                time.sleep(cfg.poll_seconds)
                continue

            if not now_in_sessions_utc(sessions, now):
                _log_dedupe(
                    log, "outside_sessions",
                    f"Outside sessions. Sleeping {cfg.poll_seconds}s.",
                    dedupe_state, 600,
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
                _log_dedupe(log, "blocked_news", f"Blocked: {news.message}", dedupe_state, 300)
                time.sleep(cfg.poll_seconds)
                continue

            # --- Fetch data ---

            try:
                trend_tf = "1h"
                trend_df = td.fetch_time_series_cached(
                    cfg.symbol, "1h", outputsize=200, ttl_seconds=3600, now_utc=now
                ).df
                if len(trend_df) < 100:
                    raise RuntimeError("Insufficient H1 candles")
            except Exception as e:
                log.warning("H1 unavailable (%s). Falling back to M15.", e)
                trend_tf = "15min"
                trend_df = td.fetch_time_series_cached(
                    cfg.symbol, "15min", outputsize=200, ttl_seconds=900, now_utc=now
                ).df

            df_m15 = td.fetch_time_series_cached(
                cfg.symbol, "15min", outputsize=200, ttl_seconds=900, now_utc=now
            ).df
            df_m5 = td.fetch_time_series_cached(
                cfg.symbol, "5min", outputsize=200, ttl_seconds=300, now_utc=now
            ).df

            if len(df_m5) < 100 or len(df_m15) < 100 or len(trend_df) < 100:
                _log_dedupe(log, "insufficient_data", "Insufficient candles. Sleeping.", dedupe_state, 300)
                time.sleep(cfg.poll_seconds)
                continue

            # --- Trend ---

            trend = detect_trend(trend_df, trend_tf, cfg.ema_fast, cfg.ema_slow, cfg.ema_slope_bars)
            tradeable, direction = trend_is_tradeable(trend)

            if not tradeable:
                _log_dedupe(log, "trend_neutral", "HTF trend NEUTRAL. No setups.", dedupe_state, 300)
                setup_state = None
                time.sleep(cfg.poll_seconds)
                continue

            # --- M15 confirmation ---

            ok_confirm, confirm_reason = m15_confirms(
                df_m15, trend.direction, cfg.ema_fast, cfg.rsi_period
            )
            if not ok_confirm:
                _log_dedupe(log, "m15_not_confirm", f"M15 not confirming. {confirm_reason}", dedupe_state, 180)
                setup_state = None
                time.sleep(cfg.poll_seconds)
                continue

            # --- Compression check ---

            compressed, compression_reason = is_market_too_compressed(
                df_m5=df_m5,
                df_m15=df_m15,
                ema_fast=cfg.ema_fast,
                ema_slow=cfg.ema_slow,
                atr_period=cfg.atr_period,
                compression_ema_atr_mult=cfg.compression_ema_atr_mult,
                max_overlap_ratio=cfg.max_overlap_ratio,
            )
            if compressed:
                _log_dedupe(log, "compressed_market", compression_reason, dedupe_state, 180)
                time.sleep(cfg.poll_seconds)
                continue

            # --- Direction-specific cooldown ---

            if last_entry_time is not None and last_entry_direction == direction:
                cooldown = _effective_cooldown(last_entry_adx, float(cfg.adx_min), cfg.cooldown_minutes)
                mins_since = (now - last_entry_time).total_seconds() / 60
                if mins_since < cooldown:
                    _log_dedupe(
                        log, "cooldown",
                        f"Cooldown active: {mins_since:.1f}/{cooldown} min ({direction}).",
                        dedupe_state, 180,
                    )
                    time.sleep(cfg.poll_seconds)
                    continue

            # --- Existing setup: check for trigger ---

            if setup_state is not None:
                if now >= setup_state.expires_utc:
                    _log_dedupe(log, "setup_expired", "Setup expired.", dedupe_state, 120)
                    setup_state = None
                elif setup_state.direction != direction:
                    _log_dedupe(log, "setup_flip", "Direction flipped. Resetting setup.", dedupe_state, 120)
                    setup_state = None
                else:
                    entry_ref, entry_src = _safe_entry_from_quote_or_close(
                        td, cfg.symbol, df_m5, now, broker_offset
                    )
                    df_m1 = td.fetch_time_series_cached(
                        cfg.symbol, "1min", outputsize=200, ttl_seconds=60, now_utc=now
                    ).df

                    _atr_m15_live = setup_state.atr_m15

                    trig_ok_m1, trig_reason_m1 = trigger_entry_m1_confirmed(
                        df_m1=df_m1,
                        direction=setup_state.direction,
                        ema_period=cfg.trigger_ema_period,
                        rsi_period=cfg.rsi_period,
                        rsi_min_buy=cfg.trigger_rsi_min_buy,
                        rsi_max_sell=cfg.trigger_rsi_max_sell,
                        zone_low=setup_state.zone_low,
                        zone_high=setup_state.zone_high,
                        live_price=entry_ref,
                        atr_val=_atr_m15_live,
                    )
                    trig_ok_m5, trig_reason_m5 = trigger_entry_m5_confirmed(
                        df_m5=df_m5,
                        direction=setup_state.direction,
                        ema_period=cfg.trigger_ema_period,
                        rsi_period=cfg.rsi_period,
                        rsi_min_buy=cfg.trigger_rsi_min_buy,
                        rsi_max_sell=cfg.trigger_rsi_max_sell,
                        zone_low=setup_state.zone_low,
                        zone_high=setup_state.zone_high,
                        live_price=entry_ref,
                        atr_val=_atr_m15_live,
                    )
                    # M15 trigger: only in strong (non-pullback) H1 trends
                    is_strong_h1 = trend_tf == "1h" and "PULLBACK" not in trend.direction
                    trig_ok_m15, trig_reason_m15 = (False, "M15 trigger skipped (not strong H1)")
                    if is_strong_h1:
                        trig_ok_m15, trig_reason_m15 = trigger_entry_m15_confirmed(
                            df_m15=df_m15,
                            direction=setup_state.direction,
                            ema_period=cfg.trigger_ema_period,
                            rsi_period=cfg.rsi_period,
                            rsi_min_buy=cfg.trigger_rsi_min_buy,
                            rsi_max_sell=cfg.trigger_rsi_max_sell,
                            zone_low=setup_state.zone_low,
                            zone_high=setup_state.zone_high,
                            live_price=entry_ref,
                            atr_val=_atr_m15_live,
                        )

                    if trig_ok_m1:
                        trig_ok, trig_reason = True, trig_reason_m1
                    elif trig_ok_m5:
                        trig_ok, trig_reason = True, trig_reason_m5
                    elif trig_ok_m15:
                        trig_ok, trig_reason = True, trig_reason_m15
                    else:
                        trig_ok, trig_reason = False, trig_reason_m5

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
                            confirmations_required=effective_entry_min,
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
                            confirmations_required=effective_entry_min,
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
                        last_entry_adx = setup_state.adx_val
                        last_zone_mid = setup_state.zone_mid
                        last_zone_time = now
                        setup_state = None
                        log.info(
                            "ENTRY triggered: %s %s @ %.2f (trigger=%s)",
                            sig.direction, cfg.symbol, entry_ref, trig_reason,
                        )
                    else:
                        _log_dedupe(
                            log, "waiting_trigger",
                            f"Waiting trigger: M1={trig_reason_m1} | M5={trig_reason_m5}",
                            dedupe_state, 180,
                        )

                    time.sleep(cfg.poll_seconds)
                    continue

            # --- Score M5 for new setup ---

            confirmations, adx_val, _adx_pass, reason_bullets = score_entry_m5(
                df_m5, direction, cfg
            )

            if len(confirmations) < effective_setup_min:
                _log_dedupe(
                    log, "not_enough_setup_confirm",
                    f"Not enough confirmations: {len(confirmations)}/{effective_setup_min}",
                    dedupe_state, 180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            # Hard ADX filter — no relaxed bypass
            if adx_val < float(cfg.adx_min):
                _log_dedupe(
                    log, "adx_failed_setup",
                    f"ADX {adx_val:.1f} < min {cfg.adx_min:.1f}. Skipping.",
                    dedupe_state, 180,
                )
                time.sleep(cfg.poll_seconds)
                continue

            atr_m15 = atr_value(df_m15, cfg.atr_period)
            entry_ref, entry_src = _safe_entry_from_quote_or_close(
                td, cfg.symbol, df_m5, now, broker_offset
            )

            zone_low, zone_high, zone_mid = _build_entry_zone(
                entry_ref=entry_ref,
                direction=direction,
                atr_m15=atr_m15,
                entry_zone_atr_mult=float(cfg.entry_zone_atr_mult),
                entry_zone_min_width=float(cfg.entry_zone_min_width),
            )

            # Same-zone cooldown
            if last_zone_mid is not None and last_zone_time is not None:
                mins_since_zone = (now - last_zone_time).total_seconds() / 60
                if mins_since_zone < cfg.same_zone_cooldown_minutes and _zone_too_close(
                    zone_mid, last_zone_mid, atr_m15, cfg
                ):
                    _log_dedupe(
                        log, "same_zone_block",
                        "Skipping repeated setup in same zone.",
                        dedupe_state, 180,
                    )
                    time.sleep(cfg.poll_seconds)
                    continue

            tf_mode = "H1→M15→M5" if trend_tf == "1h" else "M15→M5 (fallback)"
            risk_tag = risk_tag_from_context(trend_tf, trend.direction)
            news_status = ("⚠️ " + news.message) if news.is_high_impact else "Normal"

            reasons: list[str] = [
                f"HTF {trend.timeframe} trend {trend.direction} "
                f"(EMA{cfg.ema_fast} vs EMA{cfg.ema_slow}, slope {trend.slope})",
                confirm_reason,
                *reason_bullets,
            ]

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
                zone_mid=zone_mid,
            )

            conf, emoji = confidence_score(
                confirmations_passed=len(confirmations),
                confirmations_required=effective_setup_min,
                adx_value=adx_val,
                adx_min=float(cfg.adx_min),
                news_is_normal=not news.is_high_impact,
            )

            tg.send_message(
                fmt_setup(setup_state, confidence=conf, emoji=emoji, effective_setup_min=effective_setup_min)
            )
            log.info(
                "SETUP sent: %s %s zone=[%.2f..%.2f] ref=%.2f(%s) ADX=%.1f",
                direction, cfg.symbol, zone_low, zone_high, entry_ref, entry_src, adx_val,
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
