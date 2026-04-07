from __future__ import annotations

"""
strategy.py — Signal generation logic for LumiCh XAU/USD bot.

v3 changes (more quality signals):
- FLAT slope now tradeable when EMA separation is meaningful (was blocking valid setups)
- M15 RSI gate widened: BUY ≥ 40 / SELL ≤ 60 (was 45/55)
- M15 EMA check softened: price within 0.3×ATR of EMA qualifies for pullback entries
- Zone check in triggers: ±0.5×ATR buffer outside zone edges (was hard cutoff)
- Momentum: 2-of-3 bars in direction (was strict 3-of-3 consecutive)
- New: EMA Slope confirmation (fast EMA rising/falling 3 bars)
- New: Stochastic oversold/overbought confirmation
- New: MACD crossover confirmation
- BB too-wide threshold raised to 5×ATR (was 4×ATR)
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
    stochastic_oscillator,
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
# Internal helpers
# ---------------------------------------------------------------------------

def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """Returns (macd_line, signal_line)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _ema_slope_positive(ema_series: pd.Series, bars: int = 3) -> bool:
    if len(ema_series) < bars + 1:
        return False
    return float(ema_series.iloc[-1]) > float(ema_series.iloc[-(bars + 1)])


def _ema_slope_negative(ema_series: pd.Series, bars: int = 3) -> bool:
    if len(ema_series) < bars + 1:
        return False
    return float(ema_series.iloc[-1]) < float(ema_series.iloc[-(bars + 1)])


def _two_of_three_momentum_bull(close: pd.Series) -> bool:
    """2 of the last 3 candles closed higher than the prior candle."""
    if len(close) < 4:
        return False
    tail = close.tail(4).tolist()
    ups = sum(1 for i in range(1, 4) if tail[i] > tail[i - 1])
    return ups >= 2


def _two_of_three_momentum_bear(close: pd.Series) -> bool:
    if len(close) < 4:
        return False
    tail = close.tail(4).tolist()
    downs = sum(1 for i in range(1, 4) if tail[i] < tail[i - 1])
    return downs >= 2


def _price_near_ema(price: float, ema_val: float, atr_val: float, mult: float = 0.3) -> bool:
    """Price within mult×ATR of EMA — counts as 'at EMA' for pullback detection."""
    return abs(price - ema_val) <= mult * atr_val


def _in_zone_or_near(
    price: float, zone_low: float, zone_high: float, atr_val: float, buffer_mult: float = 0.5
) -> bool:
    """
    Price qualifies if inside the zone OR within buffer_mult×ATR of zone edges.
    Prevents valid setups from being rejected because price is a few pips outside.
    """
    buf = atr_val * buffer_mult
    return (zone_low - buf) <= price <= (zone_high + buf)


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

    FLAT slope is now treated as trend continuation when EMA fast/slow are
    meaningfully separated — a brief pause in slope does not negate the trend.

    States:
      BULL            — EMA fast > slow (meaningful gap), slope UP or FLAT
      BULL_PULLBACK   — EMA fast > slow (meaningful gap), slope DOWN
      BEAR            — EMA fast < slow (meaningful gap), slope DOWN or FLAT
      BEAR_PULLBACK   — EMA fast < slow (meaningful gap), slope UP
      NEUTRAL         — EMAs too close / crossed
    """
    close = df["close"].astype(float)
    ef = ema(close, ema_fast)
    es = ema(close, ema_slow)

    last_close = float(close.iloc[-1])
    last_ef = float(ef.iloc[-1])
    last_es = float(es.iloc[-1])

    atr_val = atr(df, 14)
    # Slope threshold scales with price (0.05 was too small for $4600 gold)
    slope_threshold = max(atr_val * 0.01, 0.10)

    if len(ef) > ema_slope_bars + 1:
        prev = float(ef.iloc[-(ema_slope_bars + 1)])
        diff = last_ef - prev
    else:
        diff = 0.0

    if abs(diff) < slope_threshold:
        slope = "FLAT"
    elif diff > 0:
        slope = "UP"
    else:
        slope = "DOWN"

    # Only call it a trend if EMAs are meaningfully apart
    ema_separation = abs(last_ef - last_es)
    is_meaningful = ema_separation > atr_val * 0.05

    if last_ef > last_es and is_meaningful:
        direction = "BULL" if slope in ("UP", "FLAT") else "BULL_PULLBACK"
    elif last_ef < last_es and is_meaningful:
        direction = "BEAR" if slope in ("DOWN", "FLAT") else "BEAR_PULLBACK"
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
    if trend.direction in ("BULL", "BULL_PULLBACK"):
        return True, "BUY"
    if trend.direction in ("BEAR", "BEAR_PULLBACK"):
        return True, "SELL"
    return False, "NEUTRAL"


# ---------------------------------------------------------------------------
# M15 confirmation — widened thresholds
# ---------------------------------------------------------------------------

def m15_confirms(
    df_m15: pd.DataFrame,
    trend_dir: str,
    ema_fast: int,
    rsi_period: int,
) -> Tuple[bool, str]:
    """
    Widened RSI thresholds to catch more valid entries:
      BUY:  RSI ≥ 40  (was 45) — 40-45 is neutral-to-bullish recovery
      SELL: RSI ≤ 60  (was 55) — 55-60 is neutral-to-bearish territory
      Pullbacks: RSI ≥ 37 / ≤ 63

    EMA check softened: price within 0.3×ATR of EMA also passes (tight pullbacks
    briefly dip below EMA before recovering — don't want to miss those).
    """
    close = df_m15["close"].astype(float)
    ema_line = ema(close, ema_fast)
    rsi_line = rsi(close, rsi_period)

    last_close = float(close.iloc[-1])
    last_ema = float(ema_line.iloc[-1])
    last_rsi = float(rsi_line.iloc[-1])

    atr_m15 = atr(df_m15, 14)
    near_ema = _price_near_ema(last_close, last_ema, atr_m15, mult=0.3)
    is_pullback = "PULLBACK" in trend_dir

    if "BULL" in trend_dir:
        rsi_threshold = 37 if is_pullback else 40
        price_ok = last_close >= last_ema or near_ema
        if price_ok and last_rsi >= rsi_threshold:
            return True, (
                f"M15 confirms bullish (RSI {last_rsi:.1f} ≥ {rsi_threshold}, "
                f"price {'near' if near_ema else 'above'} EMA{ema_fast})"
            )
        return False, (
            f"M15 weak for BUY (close {last_close:.2f} vs EMA{ema_fast} {last_ema:.2f}, "
            f"RSI {last_rsi:.1f} needs ≥ {rsi_threshold})"
        )

    if "BEAR" in trend_dir:
        rsi_threshold = 63 if is_pullback else 60
        price_ok = last_close <= last_ema or near_ema
        if price_ok and last_rsi <= rsi_threshold:
            return True, (
                f"M15 confirms bearish (RSI {last_rsi:.1f} ≤ {rsi_threshold}, "
                f"price {'near' if near_ema else 'below'} EMA{ema_fast})"
            )
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


# ---------------------------------------------------------------------------
# Entry scoring — 9 possible confirmations
# ---------------------------------------------------------------------------

def score_entry_m5(
    df_m5: pd.DataFrame,
    direction: str,
    cfg,
) -> tuple[list[str], float, bool, list[str]]:
    """
    Returns (confirmations, adx_val, adx_pass, reason_bullets).

    Possible confirmations (9 total):
      1. Structure      — M5 HH/HL or LL/LH pattern
      2. RSI            — RSI above/below 48/52
      3. EMA Stack      — price above/below both EMAs
      4. EMA Slope      — EMA20 directionally sloping (new)
      5. Pullback+Rej   — touched EMA then rejected
      6. Engulfing      — bullish/bearish engulfing candle
      7. Momentum       — 2-of-3 bars in direction (was 3-of-3)
      8. Stochastic     — K < 30 for BUY, K > 70 for SELL (new)
      9. MACD Cross     — MACD line crossing signal line (new)
    """
    close = df_m5["close"].astype(float)
    open_ = df_m5["open"].astype(float)
    high = df_m5["high"].astype(float)
    low = df_m5["low"].astype(float)

    ema_fast_s = ema(close, cfg.ema_fast)
    ema_slow_s = ema(close, cfg.ema_slow)
    rsi_line = rsi(close, cfg.rsi_period)
    adx_series = adx(df_m5, cfg.adx_period)
    lower, _mid, upper = bollinger_bands(close, 20, 2.0)
    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3, smooth=3)
    macd_line, macd_signal = _macd(close)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_ema_fast = float(ema_fast_s.iloc[-1])
    last_ema_slow = float(ema_slow_s.iloc[-1])
    last_rsi = float(rsi_line.iloc[-1])
    adx_val = float(adx_series.iloc[-1])
    last_stoch_k = float(stoch_k.iloc[-1])
    last_macd = float(macd_line.iloc[-1])
    last_macd_sig = float(macd_signal.iloc[-1])
    prev_macd = float(macd_line.iloc[-2]) if len(macd_line) >= 2 else last_macd
    prev_macd_sig = float(macd_signal.iloc[-2]) if len(macd_signal) >= 2 else last_macd_sig

    confirmations: list[str] = []
    reasons: list[str] = []

    atr_m5_val = atr(df_m5, getattr(cfg, "atr_period", 14))

    # ---- Hard blocks ----

    ema_gap = abs(last_close - last_ema_fast)
    if ema_gap > float(cfg.ext_atr_mult) * atr_m5_val:
        reasons.append(
            f"Blocked: overextended (gap {ema_gap:.2f} > {cfg.ext_atr_mult:.2f}×ATR {atr_m5_val:.2f})"
        )
        return [], adx_val, False, reasons

    if direction == "BUY" and last_rsi > float(cfg.rsi_buy_max):
        reasons.append(f"Blocked: RSI overbought ({last_rsi:.1f} > {cfg.rsi_buy_max:.1f})")
        return [], adx_val, False, reasons
    if direction == "SELL" and last_rsi < float(cfg.rsi_sell_min):
        reasons.append(f"Blocked: RSI oversold ({last_rsi:.1f} < {cfg.rsi_sell_min:.1f})")
        return [], adx_val, False, reasons

    last_upper = float(upper.iloc[-1]) if np.isfinite(upper.iloc[-1]) else last_close
    last_lower = float(lower.iloc[-1]) if np.isfinite(lower.iloc[-1]) else last_close
    band_buffer = float(cfg.bb_band_buffer_atr) * atr_m5_val

    if direction == "BUY" and last_close >= (last_upper - band_buffer):
        reasons.append("Blocked: BUY at/above upper Bollinger band")
        return [], adx_val, False, reasons
    if direction == "SELL" and last_close <= (last_lower + band_buffer):
        reasons.append("Blocked: SELL at/below lower Bollinger band")
        return [], adx_val, False, reasons

    bb_width = max(0.0, last_upper - last_lower)
    if bb_width > atr_m5_val * 5.0:
        reasons.append(f"Blocked: BB width {bb_width:.2f} too wide — whipsaw risk")
        return [], adx_val, False, reasons

    # ---- Positive confirmations ----

    # 1. Structure
    if direction == "BUY" and is_m5_bull_structure(df_m5, cfg.structure_lookback):
        confirmations.append("Structure")
        reasons.append("M5 bullish structure (HH or HL)")
    elif direction == "SELL" and is_m5_bear_structure(df_m5, cfg.structure_lookback):
        confirmations.append("Structure")
        reasons.append("M5 bearish structure (LL or LH)")

    # 2. RSI
    if direction == "BUY" and last_rsi >= 48:
        confirmations.append("RSI")
        reasons.append(f"RSI bullish ({last_rsi:.1f} ≥ 48)")
    elif direction == "SELL" and last_rsi <= 52:
        confirmations.append("RSI")
        reasons.append(f"RSI bearish ({last_rsi:.1f} ≤ 52)")

    # 3. EMA Stack
    if direction == "BUY" and last_close >= last_ema_fast and last_close >= last_ema_slow:
        confirmations.append("EMA Stack")
        reasons.append("Price above both EMAs")
    elif direction == "SELL" and last_close <= last_ema_fast and last_close <= last_ema_slow:
        confirmations.append("EMA Stack")
        reasons.append("Price below both EMAs")

    # 4. EMA Slope (new)
    if direction == "BUY" and _ema_slope_positive(ema_fast_s, bars=3):
        confirmations.append("EMA Slope")
        reasons.append("EMA20 sloping up over last 3 bars")
    elif direction == "SELL" and _ema_slope_negative(ema_fast_s, bars=3):
        confirmations.append("EMA Slope")
        reasons.append("EMA20 sloping down over last 3 bars")

    # 5. Pullback + Rejection
    lookback = int(cfg.pullback_lookback)
    touched = any(
        float(low.iloc[-i]) <= float(ema_fast_s.iloc[-i]) <= float(high.iloc[-i])
        for i in range(1, min(lookback + 1, len(df_m5)))
    )
    if direction == "BUY":
        if touched and (last_close > last_ema_fast) and (last_close > last_open):
            confirmations.append("Pullback+Rejection")
            reasons.append("M5 pulled back to EMA then rejected upward")
    else:
        if touched and (last_close < last_ema_fast) and (last_close < last_open):
            confirmations.append("Pullback+Rejection")
            reasons.append("M5 pulled back to EMA then rejected downward")

    # 6. Engulfing
    if direction == "BUY" and is_bullish_engulfing(df_m5):
        confirmations.append("Engulfing")
        reasons.append("Bullish engulfing on M5")
    elif direction == "SELL" and is_bearish_engulfing(df_m5):
        confirmations.append("Engulfing")
        reasons.append("Bearish engulfing on M5")

    # 7. Momentum — 2-of-3 (was 3-of-3)
    if direction == "BUY" and _two_of_three_momentum_bull(close):
        confirmations.append("Momentum")
        reasons.append("2-of-3 bar bullish momentum")
    elif direction == "SELL" and _two_of_three_momentum_bear(close):
        confirmations.append("Momentum")
        reasons.append("2-of-3 bar bearish momentum")

    # 8. Stochastic (new)
    if direction == "BUY" and last_stoch_k < 30:
        confirmations.append("Stoch Oversold")
        reasons.append(f"Stochastic oversold ({last_stoch_k:.1f}) — reversal zone")
    elif direction == "SELL" and last_stoch_k > 70:
        confirmations.append("Stoch Overbought")
        reasons.append(f"Stochastic overbought ({last_stoch_k:.1f}) — reversal zone")

    # 9. MACD crossover (new)
    macd_crossed_up = (prev_macd <= prev_macd_sig) and (last_macd > last_macd_sig)
    macd_crossed_down = (prev_macd >= prev_macd_sig) and (last_macd < last_macd_sig)
    if direction == "BUY" and macd_crossed_up:
        confirmations.append("MACD Cross")
        reasons.append("MACD crossed above signal on M5")
    elif direction == "SELL" and macd_crossed_down:
        confirmations.append("MACD Cross")
        reasons.append("MACD crossed below signal on M5")

    adx_pass = adx_val >= float(cfg.adx_min)
    return confirmations, adx_val, adx_pass, reasons


# ---------------------------------------------------------------------------
# Entry triggers — zone tolerance buffer
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
    atr_val: float = 5.0,
) -> tuple[bool, str]:
    if len(df_m1) < 3:
        return False, "Not enough M1 candles"
    if not _in_zone_or_near(live_price, zone_low, zone_high, atr_val):
        return False, f"Price {live_price:.2f} not near zone [{zone_low:.2f}–{zone_high:.2f}]"

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
            return False, f"M1 RSI too weak ({rsi2:.1f} < {rsi_min_buy:.1f})"
        if c2_close > c1_high or c2_close > ema2:
            return True, "M1 bullish confirmation near/inside zone"
        return False, "M1 second candle did not confirm BUY"

    if c1_close >= c1_open:
        return False, "M1 setup candle not bearish"
    if rsi2 > rsi_max_sell:
        return False, f"M1 RSI too strong ({rsi2:.1f} > {rsi_max_sell:.1f})"
    if c2_close < c1_low or c2_close < ema2:
        return True, "M1 bearish confirmation near/inside zone"
    return False, "M1 second candle did not confirm SELL"


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
    atr_val: float = 5.0,
) -> tuple[bool, str]:
    if len(df_m5) < 2:
        return False, "Not enough M5 candles"
    if not _in_zone_or_near(live_price, zone_low, zone_high, atr_val):
        return False, f"Price {live_price:.2f} not near zone [{zone_low:.2f}–{zone_high:.2f}]"

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
            return True, f"M5 bullish trigger near/inside zone (RSI {last_rsi:.1f})"
        return False, f"M5 BUY not confirmed (close {last_close:.2f}, EMA {last_ema:.2f}, RSI {last_rsi:.1f})"

    if last_close < last_open and last_close <= last_ema and last_rsi <= rsi_max_sell:
        return True, f"M5 bearish trigger near/inside zone (RSI {last_rsi:.1f})"
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
    atr_val: float = 5.0,
) -> tuple[bool, str]:
    """M15 trigger — for strong H1 trends only."""
    if len(df_m15) < 2:
        return False, "Not enough M15 candles"
    if not _in_zone_or_near(live_price, zone_low, zone_high, atr_val):
        return False, f"Price {live_price:.2f} not near zone [{zone_low:.2f}–{zone_high:.2f}]"

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
            return True, f"M15 bullish trigger near/inside zone (RSI {last_rsi:.1f})"
        return False, f"M15 BUY not confirmed (RSI {last_rsi:.1f})"

    if last_close < last_open and last_close <= last_ema and last_rsi <= rsi_max_sell:
        return True, f"M15 bearish trigger near/inside zone (RSI {last_rsi:.1f})"
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
    if trend_tf == "1h" and "PULLBACK" not in trend_direction:
        return "LOW"
    if trend_tf in ("1h", "15min"):
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def confidence_score(
    confirmations_passed: int,
    confirmations_required: int,
    adx_value: float,
    adx_min: float,
    news_is_normal: bool,
) -> tuple[int, str]:
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
