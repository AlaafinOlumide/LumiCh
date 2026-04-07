from __future__ import annotations

"""
indicators.py — Single source of truth for all technical indicators.

All functions use pandas Series / DataFrames and return pandas objects
so callers can index them freely.  No duplicate implementations exist here.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (Wilder-style: adjust=False)."""
    return series.ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range — maximum of the three classical TR components."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range as a full Series (Simple/RMA rolling)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Latest ATR value as a scalar float. Returns a safe floor on bad data."""
    s = atr_series(df, period)
    v = float(s.iloc[-1])
    if not np.isfinite(v) or v <= 0:
        last_close = float(df["close"].astype(float).iloc[-1])
        return max(last_close * 0.001, 1.0)
    return v


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EWM, alpha=1/period).
    NaNs filled with 50 so downstream checks don't fail on cold-start.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Slow Stochastic.  Returns (%K smoothed, %D).
    Division-by-zero rows become NaN and are forward-filled.
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean() if smooth > 1 else raw_k
    d = k.rolling(d_period).mean()
    return k.ffill(), d.ffill()


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Welles Wilder ADX as a full Series.
    Uses EWM smoothing (alpha = 1/period) for DM and TR.
    Returns 0.0 for any leading NaN rows.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s = atr_series(df, period).replace(0, np.nan)

    plus_di = (
        100
        * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr_s
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean()
        / atr_s
    )

    dx = (
        100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    ).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def bollinger_bands(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (lower, mid, upper)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return lower, mid, upper


# ---------------------------------------------------------------------------
# Candle patterns
# ---------------------------------------------------------------------------

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """True when the last completed candle fully engulfs the prior bearish candle."""
    if len(df) < 3:
        return False
    o1 = float(df["open"].iloc[-2])
    c1 = float(df["close"].iloc[-2])
    o2 = float(df["open"].iloc[-1])
    c2 = float(df["close"].iloc[-1])
    return (c1 < o1) and (c2 > o2) and (o2 <= c1) and (c2 >= o1)


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    """True when the last completed candle fully engulfs the prior bullish candle."""
    if len(df) < 3:
        return False
    o1 = float(df["open"].iloc[-2])
    c1 = float(df["close"].iloc[-2])
    o2 = float(df["open"].iloc[-1])
    c2 = float(df["close"].iloc[-1])
    return (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1)


def candle_patterns(df: pd.DataFrame) -> dict[str, bool]:
    """
    Returns a dict of pattern booleans for the *last* candle.
    Keys: bullish_engulfing, bearish_engulfing, hammer, shooting_star, inside_bar.
    """
    if len(df) < 2:
        return {
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "hammer": False,
            "shooting_star": False,
            "inside_bar": False,
        }

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)
    upper_wick = h - c.where(c >= o, o)
    lower_wick = c.where(c <= o, o) - l

    # Patterns
    bullish_eng = (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    bearish_eng = (c < o) & (c.shift(1) > o.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1))
    hammer = (lower_wick >= 2 * body) & (upper_wick <= 0.5 * body) & (body / candle_range <= 0.35)
    shooting_star = (upper_wick >= 2 * body) & (lower_wick <= 0.5 * body) & (body / candle_range <= 0.35)
    inside_bar = (h < h.shift(1)) & (l > l.shift(1))

    idx = df.index[-1]
    return {
        "bullish_engulfing": bool(bullish_eng.loc[idx]),
        "bearish_engulfing": bool(bearish_eng.loc[idx]),
        "hammer": bool(hammer.loc[idx]),
        "shooting_star": bool(shooting_star.loc[idx]),
        "inside_bar": bool(inside_bar.loc[idx]),
    }
