from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .indicators import adx, bollinger_bands, ema, rsi, stoch_oscillator


@dataclass
class TrendState:
    timeframe: str
    direction: str  # 'BULL', 'BEAR', 'NEUTRAL'
    ema_fast: float
    ema_slow: float
    close: float
    slope: str  # 'UP', 'DOWN', 'FLAT'


@dataclass
class Signal:
    symbol: str
    timestamp_utc: dt.datetime
    direction: str  # 'BUY'/'SELL'
    risk_tag: str
    timeframe_mode: str
    session_label: str
    trend_state: TrendState
    confirmations: list[str]
    confirmations_passed: int
    confirmations_required: int
    adx_value: float
    adx_min: float
    news_status: str
    reason_bullets: list[str]


def detect_trend(df: pd.DataFrame, timeframe: str, ema_fast_p: int, ema_slow_p: int, slope_bars: int) -> TrendState:
    close = df['close']
    ef = ema(close, ema_fast_p)
    es = ema(close, ema_slow_p)

    c = float(close.iloc[-1])
    ef_now = float(ef.iloc[-1])
    es_now = float(es.iloc[-1])

    if len(ef) > slope_bars:
        ef_prev = float(ef.iloc[-1 - slope_bars])
        slope = 'UP' if ef_now > ef_prev else 'DOWN' if ef_now < ef_prev else 'FLAT'
    else:
        slope = 'FLAT'

    direction = 'NEUTRAL'
    if c > es_now and ef_now > es_now and slope == 'UP':
        direction = 'BULL'
    elif c < es_now and ef_now < es_now and slope == 'DOWN':
        direction = 'BEAR'

    return TrendState(
        timeframe=timeframe,
        direction=direction,
        ema_fast=ef_now,
        ema_slow=es_now,
        close=c,
        slope=slope,
    )


def m15_confirms(df_m15: pd.DataFrame, trend_dir: str, ema_fast_p: int, rsi_period: int) -> tuple[bool, str]:
    close = df_m15['close']
    ef = ema(close, ema_fast_p)
    r = rsi(close, rsi_period)
    c = float(close.iloc[-1])
    ef_now = float(ef.iloc[-1])
    r_now = float(r.iloc[-1])

    if trend_dir == 'BULL':
        ok = (c > ef_now) or (r_now > 50)
        reason = f"M15 confirm: close>{ema_fast_p}EMA" if c > ef_now else "M15 confirm: RSI>50" if r_now > 50 else "M15 confirm failed"
        return ok, reason
    if trend_dir == 'BEAR':
        ok = (c < ef_now) or (r_now < 50)
        reason = f"M15 confirm: close<{ema_fast_p}EMA" if c < ef_now else "M15 confirm: RSI<50" if r_now < 50 else "M15 confirm failed"
        return ok, reason
    return False, "M15 confirm: HTF neutral"


def score_entry_m5(
    df_m5: pd.DataFrame,
    direction: str,
    cfg,
) -> tuple[list[str], float, bool, list[str]]:
    """Return (confirmations_passed_labels, adx_value, adx_pass, reason_bullets)."""
    close = df_m5['close']
    high = df_m5['high']
    low = df_m5['low']
    open_ = df_m5['open']

    mid, upper, lower = bollinger_bands(close, cfg.bb_period, cfg.bb_std)
    r = rsi(close, cfg.rsi_period)
    k, d = stoch_oscillator(high, low, close, cfg.stoch_k, cfg.stoch_d, cfg.stoch_smooth)
    a = adx(high, low, close, cfg.adx_period)

    adx_val = float(a.iloc[-1])
    adx_pass = adx_val >= cfg.adx_min

    conf: list[str] = []
    reasons: list[str] = []

    # Group A: Bollinger
    c_now = float(close.iloc[-1])
    c_prev = float(close.iloc[-2])
    lower_now = float(lower.iloc[-1])
    upper_now = float(upper.iloc[-1])
    mid_now = float(mid.iloc[-1])

    if direction == 'BUY':
        if (c_prev < lower_now and c_now > lower_now) or (c_now < mid_now and c_now > lower_now and c_now > c_prev):
            conf.append('BB')
            reasons.append('Bollinger: lower-band rejection / pullback end')
    else:
        if (c_prev > upper_now and c_now < upper_now) or (c_now > mid_now and c_now < upper_now and c_now < c_prev):
            conf.append('BB')
            reasons.append('Bollinger: upper-band rejection / pullback end')

    # Group B: RSI
    r_now = float(r.iloc[-1])
    r_prev = float(r.iloc[-2])
    if direction == 'BUY':
        if (r_prev <= 50 and r_now > 50) or (r.tail(10).min() < cfg.rsi_buy_min and r_now > r_prev):
            conf.append('RSI')
            reasons.append('RSI: momentum rebound / crossed above 50')
    else:
        if (r_prev >= 50 and r_now < 50) or (r.tail(10).max() > cfg.rsi_sell_max and r_now < r_prev):
            conf.append('RSI')
            reasons.append('RSI: momentum fade / crossed below 50')

    # Group C: Stochastic
    k_now = float(k.iloc[-1])
    d_now = float(d.iloc[-1])
    k_prev = float(k.iloc[-2])
    d_prev = float(d.iloc[-2])

    if direction == 'BUY':
        crossed_up = (k_prev <= d_prev) and (k_now > d_now)
        if crossed_up and (k_now < cfg.stoch_oversold or k_prev < 30):
            conf.append('STOCH')
            reasons.append('Stoch: cross up from oversold / rising')
    else:
        crossed_down = (k_prev >= d_prev) and (k_now < d_now)
        if crossed_down and (k_now > cfg.stoch_overbought or k_prev > 70):
            conf.append('STOCH')
            reasons.append('Stoch: cross down from overbought / falling')

    # Group D: Candlestick (basic)
    o_now = float(open_.iloc[-1])
    o_prev = float(open_.iloc[-2])
    h_now = float(high.iloc[-1])
    l_now = float(low.iloc[-1])

    body_now = abs(c_now - o_now)
    body_prev = abs(c_prev - o_prev)
    upper_wick = h_now - max(c_now, o_now)
    lower_wick = min(c_now, o_now) - l_now

    bullish_engulf = (c_now > o_now) and (c_prev < o_prev) and (c_now >= o_prev) and (o_now <= c_prev)
    bearish_engulf = (c_now < o_now) and (c_prev > o_prev) and (o_now >= c_prev) and (c_now <= o_prev)

    # Pinbar-ish: long wick relative to body
    pin_bull = (lower_wick > body_now * 2) and (c_now > o_now)
    pin_bear = (upper_wick > body_now * 2) and (c_now < o_now)

    if direction == 'BUY':
        if bullish_engulf or pin_bull:
            conf.append('CANDLE')
            reasons.append('Candle: bullish engulf / pin bar')
    else:
        if bearish_engulf or pin_bear:
            conf.append('CANDLE')
            reasons.append('Candle: bearish engulf / pin bar')

    # Build reason bullets (only true conditions)
    reason_bullets = []
    for rtxt in reasons:
        reason_bullets.append(rtxt)

    # Always include ADX bullet
    reason_bullets.append(f"ADX {adx_val:.1f} {'>=' if adx_pass else '<'} {cfg.adx_min} (trend strength)")

    return conf, adx_val, adx_pass, reason_bullets


def risk_tag_from_context(trend_timeframe: str) -> str:
    # Simple default: if H1 trend is present, treat as SWING-ish; if fallback M15-only trend, treat as SCALP
    return "SWING" if trend_timeframe == "1h" else "SCALP"
