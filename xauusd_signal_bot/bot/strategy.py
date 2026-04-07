from __future__ import annotations

"""
strategy.py — Signal generation logic for LumiCh XAU/USD bot.

Key improvements vs prior version:
- All indicators imported from indicators.py (single source of truth)
- Trend detection supports BULL_PULLBACK / BEAR_PULLBACK states
- M15 confirmation thresholds symmetric and tightened
- Zone bias corrected (BUY zones pull lower, SELL zones push higher)
- Momentum confirmation requires 3-bar consistency, not single candle
- BB Expansion used as a quality filter, not a positive confirmation
- Confidence score rebalanced (confirmations drive 80%)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .indicators import (
    ema,
    rsi,
    atr,
    atr_series,
    adx,
    bollinger_bands,
    is_bullish_engulfing,
    is_bearish_engulfing,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendState:
    timeframe: str
    direction: str          # BULL | BULL_PULLBACK | BEAR | BEAR_PULLBACK | NEUTRAL
    close: float
    ema_fast: float
    ema_slow: float
    slope: str              # UP | DOWN | FLAT


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp_utc: object
    direction: str
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
    entry_price: float | None = None
    sl: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    rr: float | None = None
    confidence: int | None = None
    confidence_emoji: str | None = None


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

def detect_trend(
    df: pd.DataFrame,
    timeframe: str,
    ema_fast: int,
    ema_slow: int,
    ema_slope_bars: int,
) -> TrendState:
    """
    Detects HTF trend with pullback sub-states.

    States:
      BULL            — EMA fast > slow, slope UP
      BULL_PULLBACK   — EMA fast > slow, slope briefly DOWN (still tradeable long)
      BEAR            — EMA fast < slow, slope DOWN
      BEAR_PULLBACK   — EMA fast < slow, slope briefly UP (still tradeable short)
      NEUTRAL         — EMAs crossed / flat, no clear bias
    """
    close = df["close"].astype(float)
    ef = ema(close, ema_fast)
    es = ema(close, ema_slow)

    last_close = float(close.iloc[-1])
    last_ef = float(ef.iloc[-1])
    last_es = float(es.iloc[-1])

    if len(ef) > ema_slope_bars + 1:
        prev = float(ef.iloc[-(ema_slope_bars + 1)])
        diff = last_ef - prev
    else:
        diff = 0.0

    if abs(diff) < 0.05:
        slope = "FLAT"
    elif diff > 0:
        slope = "UP"
    else:
        slope = "DOWN"

    if last_ef > last_es:
        direction = "BULL" if slope == "UP" else "BULL_PULLBACK" if slope == "DOWN" else "NEUTRAL"
    elif last_ef < last_es:
        direction = "BEAR" if slope == "DOWN" else "BEAR_PULLBACK" if slope == "UP" else "NEUTRAL"
    else:
        direction = "NEUTRAL"

    return TrendState(
        timeframe=timeframe,
        direction=direction,
        close=last_close,
        ema_fast=last_ef,
        ema_slow=last_es,
        slope=slope,
    )


def trend_is_tradeable(trend: TrendState) -> tuple[bool, str]:
    """
    Returns (tradeable, trade_direction).
    BULL and BULL_PULLBACK → BUY
    BEAR and BEAR_PULLBACK → SELL
    NEUTRAL → not tradeable
    """
    if trend.direction in ("BULL", "BULL_PULLBACK"):
        return True, "BUY"
    if trend.direction in ("BEAR", "BEAR_PULLBACK"):
        return True, "SELL"
    return False, "NEUTRAL"


# ---------------------------------------------------------------------------
# M15 confirmation  (symmetric thresholds)
# ---------------------------------------------------------------------------

def m15_confirms(
    df_m15: pd.DataFrame,
    trend_dir: str,
    ema_fast: int,
    rsi_period: int,
) -> Tuple[bool, str]:
    """
    M15 must broadly agree with the HTF trend.
    Thresholds are symmetric: BUY needs RSI ≥ 45, SELL needs RSI ≤ 55.
    Pullback states get slightly relaxed thresholds (42 / 58) to allow entries.
    """
    close = df_m15["close"].astype(float)
    ema_line = ema(close, ema_fast)
    rsi_line = rsi(close, rsi_period)

    last_close = float(close.iloc[-1])
    last_ema = float(ema_line.iloc[-1])
    last_rsi = float(rsi_line.iloc[-1])

    is_pullback = "PULLBACK" in trend_dir

    if "BULL" in trend_dir:
        rsi_threshold = 42 if is_pullback else 45
        if last_close >= last_ema and last_rsi >= rsi_threshold:
            return True, f"M15 confirms bullish (RSI {last_rsi:.1f} ≥ {rsi_threshold}, price above EMA)"
        return False, (
            f"M15 weak for BUY (close {last_close:.2f} vs EMA{ema_fast} {last_ema:.2f}, "
            f"RSI {last_rsi:.1f} needs ≥ {rsi_threshold})"
        )

    if "BEAR" in trend_dir:
        rsi_threshold = 58 if is_pullback else 55
        if last_close <= last_ema and last_rsi <= rsi_threshold:
            return True, f"M15 confirms bearish (RSI {last_rsi:.1f} ≤ {rsi_threshold}, price below EMA)"
        return False, (
            f"M15 weak for SELL (close {last_close:.2f} vs EMA{ema_fast} {last_ema:.2f}, "
            f"RSI {last_rsi:.1f} needs ≤ {rsi_threshold})"
        )

    return False, "Trend is NEUTRAL"


# ---------------------------------------------------------------------------
# Compression filter
# ---------------------------------------------------------------------------

def is_market_too_compressed(
    df_m5: pd.DataFrame,
    df_m15: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    atr_period: int,
    compression_ema_atr_mult: float,
    max_overlap_ratio: float,
) -> tuple[bool, str]:
    close_m5 = df_m5["close"].astype(float)
    ef = ema(close_m5, ema_fast)
    es = ema(close_m5, ema_slow)
    ema_gap = abs(float(ef.iloc[-1]) - float(es.iloc[-1]))

    atr_m15 = atr(df_m15, atr_period)

    if ema_gap < compression_ema_atr_mult * atr_m15:
        return True, (
            f"Compression: EMA gap {ema_gap:.2f} < "
            f"{compression_ema_atr_mult}×ATR15 {atr_m15:.2f}"
        )

    highs = df_m5["high"].astype(float).tail(10).tolist()
    lows = df_m5["low"].astype(float).tail(10).tolist()
    overlaps = sum(
        1
        for i in range(1, len(highs))
        if max(lows[i - 1], lows[i]) <= min(highs[i - 1], highs[i])
    )
    overlap_ratio = overlaps / max(len(highs) - 1, 1)

    if overlap_ratio >= max_overlap_ratio:
        return True, f"Compression: candle overlap {overlap_ratio:.2f} ≥ {max_overlap_ratio}"

    return False, "OK"


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def is_m5_bull_structure(df_m5: pd.DataFrame, lookback: int = 8) -> bool:
    if len(df_m5) < lookback + 2:
        return False
    highs = df_m5["high"].astype(float).tail(lookback)
    lows = df_m5["low"].astype(float).tail(lookback)
    return float(highs.iloc[-1]) >= float(highs.iloc[:-2].max()) or float(
        lows.iloc[-1]
    ) > float(lows.iloc[:-2].min())


def is_m5_bear_structure(df_m5: pd.DataFrame, lookback: int = 8) -> bool:
    if len(df_m5) < lookback + 2:
        return False
    highs = df_m5["high"].astype(float).tail(lookback)
    lows = df_m5["low"].astype(float).tail(lookback)
    return float(lows.iloc[-1]) <= float(lows.iloc[:-2].min()) or float(
        highs.iloc[-1]
    ) < float(highs.iloc[:-2].max())


def _three_bar_momentum_bull(close: pd.Series) -> bool:
    """3 consecutive closes each higher than the prior (not just last vs second-to-last)."""
    if len(close) < 4:
        return False
    tail = close.tail(4).tolist()
    return tail[1] > tail[0] and tail[2] > tail[1] and tail[3] > tail[2]


def _three_bar_momentum_bear(close: pd.Series) -> bool:
    if len(close) < 4:
        return False
    tail = close.tail(4).tolist()
    return tail[1] < tail[0] and tail[2] < tail[1] and tail[3] < tail[2]


# ---------------------------------------------------------------------------
# Entry scoring
# ---------------------------------------------------------------------------

def score_entry_m5(
    df_m5: pd.DataFrame,
    direction: str,
    cfg,
) -> tuple[list[str], float, bool, list[str]]:
    """
    Returns (confirmations, adx_val, adx_pass, reason_bullets).

    Changes vs prior version:
    - Momentum now requires 3-bar consistency
    - BB Expansion removed as a positive confirmation (it is a quality blocker)
    - BB too-wide filter added (blocks entry in explosive/whipsaw conditions)
    - Zone bias not applied here — zone calculation moved to main loop
    """
    close = df_m5["close"].astype(float)
    open_ = df_m5["open"].astype(float)
    high = df_m5["high"].astype(float)
    low = df_m5["low"].astype(float)

    ema_fast = ema(close, cfg.ema_fast)
    ema_slow = ema(close, cfg.ema_slow)
    rsi_line = rsi(close, cfg.rsi_period)
    adx_series = adx(df_m5, cfg.adx_period)
    lower, _mid, upper = bollinger_bands(close, 20, 2.0)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_rsi = float(rsi_line.iloc[-1])
    adx_val = float(adx_series.iloc[-1])

    confirmations: list[str] = []
    reasons: list[str] = []

    atr_m5_val = atr(df_m5, getattr(cfg, "atr_period", 14))

    # --- Hard blocks ---

    ema_gap = abs(last_close - last_ema_fast)
    if ema_gap > float(cfg.ext_atr_mult) * atr_m5_val:
        reasons.append(
            f"Blocked: overextended vs EMA (gap {ema_gap:.2f} > "
            f"{cfg.ext_atr_mult:.2f}×ATR {atr_m5_val:.2f})"
        )
        return [], adx_val, False, reasons

    if direction == "BUY" and last_rsi > float(cfg.rsi_buy_max):
        reasons.append(f"Blocked: RSI overbought for BUY ({last_rsi:.1f} > {cfg.rsi_buy_max:.1f})")
        return [], adx_val, False, reasons
    if direction == "SELL" and last_rsi < float(cfg.rsi_sell_min):
        reasons.append(f"Blocked: RSI oversold for SELL ({last_rsi:.1f} < {cfg.rsi_sell_min:.1f})")
        return [], adx_val, False, reasons

    last_upper = float(upper.iloc[-1]) if np.isfinite(upper.iloc[-1]) else last_close
    last_lower = float(lower.iloc[-1]) if np.isfinite(lower.iloc[-1]) else last_close
    band_buffer = float(cfg.bb_band_buffer_atr) * atr_m5_val

    if direction == "BUY" and last_close >= (last_upper - band_buffer):
        reasons.append("Blocked: BUY price at/above upper Bollinger band")
        return [], adx_val, False, reasons
    if direction == "SELL" and last_close <= (last_lower + band_buffer):
        reasons.append("Blocked: SELL price at/below lower Bollinger band")
        return [], adx_val, False, reasons

    # BB too-wide block: explosive expansion signals whipsaw risk
    bb_width = max(0.0, last_upper - last_lower)
    bb_wide_threshold = atr_m5_val * 4.0
    if bb_width > bb_wide_threshold:
        reasons.append(
            f"Blocked: BB width {bb_width:.2f} too wide vs ATR {atr_m5_val:.2f} — whipsaw risk"
        )
        return [], adx_val, False, reasons

    # --- Positive confirmations ---

    if direction == "BUY" and is_m5_bull_structure(df_m5, cfg.structure_lookback):
        confirmations.append("Structure")
        reasons.append("M5 structure aligned bullish (HH or HL)")
    elif direction == "SELL" and is_m5_bear_structure(df_m5, cfg.structure_lookback):
        confirmations.append("Structure")
        reasons.append("M5 structure aligned bearish (LL or LH)")

    if direction == "BUY" and last_close >= last_ema_fast and last_rsi >= 48:
        confirmations.append("RSI")
        reasons.append(f"RSI supportive for BUY ({last_rsi:.1f} ≥ 48)")
    elif direction == "SELL" and last_close <= last_ema_fast and last_rsi <= 52:
        confirmations.append("RSI")
        reasons.append(f"RSI supportive for SELL ({last_rsi:.1f} ≤ 52)")

    ema_slow_last = float(ema_slow.iloc[-1])
    if direction == "BUY" and last_close >= last_ema_fast and last_close >= ema_slow_last:
        confirmations.append("EMA Pullback")
        reasons.append("Price above both EMAs — continuation area")
    elif direction == "SELL" and last_close <= last_ema_fast and last_close <= ema_slow_last:
        confirmations.append("EMA Pullback")
        reasons.append("Price below both EMAs — continuation area")

    lookback = int(cfg.pullback_lookback)
    touched = any(
        float(low.iloc[-i]) <= float(ema_fast.iloc[-i]) <= float(high.iloc[-i])
        for i in range(1, min(lookback + 1, len(df_m5)))
    )
    if direction == "BUY":
        rejection = (last_close > last_ema_fast) and (last_close > last_open)
        if touched and rejection:
            confirmations.append("Pullback+Rejection")
            reasons.append("M5 pulled back to EMA then rejected upward")
    else:
        rejection = (last_close < last_ema_fast) and (last_close < last_open)
        if touched and rejection:
            confirmations.append("Pullback+Rejection")
            reasons.append("M5 pulled back to EMA then rejected downward")

    if direction == "BUY" and is_bullish_engulfing(df_m5):
        confirmations.append("Engulfing")
        reasons.append("Bullish engulfing on M5")
    elif direction == "SELL" and is_bearish_engulfing(df_m5):
        confirmations.append("Engulfing")
        reasons.append("Bearish engulfing on M5")

    # 3-bar momentum (replaces single-candle check)
    if direction == "BUY" and _three_bar_momentum_bull(close):
        confirmations.append("Momentum")
        reasons.append("3-bar bullish momentum confirmed")
    elif direction == "SELL" and _three_bar_momentum_bear(close):
        confirmations.append("Momentum")
        reasons.append("3-bar bearish momentum confirmed")

    adx_pass = adx_val >= float(cfg.adx_min)
    return confirmations, adx_val, adx_pass, reasons


# ---------------------------------------------------------------------------
# Entry triggers
# ---------------------------------------------------------------------------

def trigger_entry_m1_confirmed(
    df_m1: pd.DataFrame,
    direction: str,
    ema_period: int,
    rsi_period: int,
    rsi_min_buy: float,
    rsi_max_sell: float,
    zone_low: float,
    zone_high: float,
    live_price: float,
) -> tuple[bool, str]:
    if len(df_m1) < 3:
        return False, "Not enough M1 candles"
    if not (zone_low <= live_price <= zone_high):
        return False, f"Price {live_price:.2f} not inside entry zone [{zone_low:.2f}–{zone_high:.2f}]"

    close = df_m1["close"].astype(float)
    open_ = df_m1["open"].astype(float)
    high = df_m1["high"].astype(float)
    low = df_m1["low"].astype(float)
    ema_line = ema(close, ema_period)
    rsi_line = rsi(close, rsi_period)

    c1_open = float(open_.iloc[-2])
    c1_close = float(close.iloc[-2])
    c1_high = float(high.iloc[-2])
    c1_low = float(low.iloc[-2])

    c2_close = float(close.iloc[-1])
    ema2 = float(ema_line.iloc[-1])
    rsi2 = float(rsi_line.iloc[-1])

    if direction == "BUY":
        if c1_close <= c1_open:
            return False, "M1 setup candle not bullish"
        if rsi2 < rsi_min_buy:
            return False, f"M1 RSI too weak for BUY ({rsi2:.1f} < {rsi_min_buy:.1f})"
        if c2_close > c1_high or c2_close > ema2:
            return True, "M1 bullish confirmation inside zone"
        return False, "M1 second candle did not break above setup candle high or EMA"

    # SELL
    if c1_close >= c1_open:
        return False, "M1 setup candle not bearish"
    if rsi2 > rsi_max_sell:
        return False, f"M1 RSI too strong for SELL ({rsi2:.1f} > {rsi_max_sell:.1f})"
    if c2_close < c1_low or c2_close < ema2:
        return True, "M1 bearish confirmation inside zone"
    return False, "M1 second candle did not break below setup candle low or EMA"


def trigger_entry_m5_confirmed(
    df_m5: pd.DataFrame,
    direction: str,
    ema_period: int,
    rsi_period: int,
    rsi_min_buy: float,
    rsi_max_sell: float,
    zone_low: float,
    zone_high: float,
    live_price: float,
) -> tuple[bool, str]:
    if len(df_m5) < 2:
        return False, "Not enough M5 candles"
    if not (zone_low <= live_price <= zone_high):
        return False, f"Price {live_price:.2f} not inside entry zone [{zone_low:.2f}–{zone_high:.2f}]"

    close = df_m5["close"].astype(float)
    open_ = df_m5["open"].astype(float)
    ema_line = ema(close, ema_period)
    rsi_line = rsi(close, rsi_period)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_ema = float(ema_line.iloc[-1])
    last_rsi = float(rsi_line.iloc[-1])

    if direction == "BUY":
        if last_close > last_open and last_close >= last_ema and last_rsi >= rsi_min_buy:
            return True, f"M5 bullish trigger inside zone (RSI {last_rsi:.1f})"
        return False, f"M5 BUY not confirmed (close {last_close:.2f}, EMA {last_ema:.2f}, RSI {last_rsi:.1f})"

    if last_close < last_open and last_close <= last_ema and last_rsi <= rsi_max_sell:
        return True, f"M5 bearish trigger inside zone (RSI {last_rsi:.1f})"
    return False, f"M5 SELL not confirmed (close {last_close:.2f}, EMA {last_ema:.2f}, RSI {last_rsi:.1f})"


def trigger_entry_m15_confirmed(
    df_m15: pd.DataFrame,
    direction: str,
    ema_period: int,
    rsi_period: int,
    rsi_min_buy: float,
    rsi_max_sell: float,
    zone_low: float,
    zone_high: float,
    live_price: float,
) -> tuple[bool, str]:
    """
    New: M15 trigger for strong HTF trend setups.
    A clean M15 close with EMA + RSI alignment is sufficient when H1 is strongly trending.
    """
    if len(df_m15) < 2:
        return False, "Not enough M15 candles"
    if not (zone_low <= live_price <= zone_high):
        return False, f"Price {live_price:.2f} not inside entry zone [{zone_low:.2f}–{zone_high:.2f}]"

    close = df_m15["close"].astype(float)
    open_ = df_m15["open"].astype(float)
    ema_line = ema(close, ema_period)
    rsi_line = rsi(close, rsi_period)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_ema = float(ema_line.iloc[-1])
    last_rsi = float(rsi_line.iloc[-1])

    if direction == "BUY":
        if last_close > last_open and last_close >= last_ema and last_rsi >= rsi_min_buy:
            return True, f"M15 bullish trigger inside zone (RSI {last_rsi:.1f})"
        return False, f"M15 BUY not confirmed (RSI {last_rsi:.1f})"

    if last_close < last_open and last_close <= last_ema and last_rsi <= rsi_max_sell:
        return True, f"M15 bearish trigger inside zone (RSI {last_rsi:.1f})"
    return False, f"M15 SELL not confirmed (RSI {last_rsi:.1f})"


# ---------------------------------------------------------------------------
# Risk / TP / SL
# ---------------------------------------------------------------------------

def compute_tp_sl_from_atr(
    entry: float,
    direction: str,
    atr_value: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float, float, float]:
    """
    Returns (tp2, sl, risk_reward).
    atr_value is floored so bad data never produces zero-width risk.
    """
    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = max(entry * 0.001, 1.0)

    sl_dist = atr_value * sl_mult
    tp_dist = atr_value * tp_mult

    if direction == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
        rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0.0
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
        rr = (entry - tp) / (sl - entry) if (sl - entry) > 0 else 0.0

    return float(tp), float(sl), float(rr)


def risk_tag_from_context(trend_tf: str, trend_direction: str = "") -> str:
    """
    MEDIUM for H1 or M15 primary HTF.
    LOW  for strong (non-pullback) H1 trends.
    HIGH otherwise.
    """
    if trend_tf == "1h" and "PULLBACK" not in trend_direction:
        return "LOW"
    if trend_tf in ("1h", "15min"):
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Confidence scoring  (rebalanced)
# ---------------------------------------------------------------------------

def confidence_score(
    confirmations_passed: int,
    confirmations_required: int,
    adx_value: float,
    adx_min: float,
    news_is_normal: bool,
) -> tuple[int, str]:
    """
    Confirmations drive 80 % of the score (was 70 %).
    ADX contributes up to 10 %.
    News contributes ±10 %.
    """
    base = 0.0
    if confirmations_required > 0:
        base = (confirmations_passed / confirmations_required) * 80.0

    adx_bonus = 0.0
    if adx_value >= adx_min:
        adx_bonus = 7.0
    if adx_value >= adx_min + 6:
        adx_bonus = 10.0

    news_adj = 10.0 if news_is_normal else -10.0
    score = int(max(0, min(100, base + adx_bonus + news_adj)))

    if score >= 80:
        emoji = "🔥"
    elif score >= 60:
        emoji = "🟡"
    else:
        emoji = "⚠️"

    return score, emoji
