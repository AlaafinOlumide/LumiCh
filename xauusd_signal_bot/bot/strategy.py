from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TrendState:
    timeframe: str
    direction: str  # "BULL", "BEAR", "NEUTRAL"
    close: float
    ema_fast: float
    ema_slow: float
    slope: str  # "UP", "DOWN", "FLAT"


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp_utc: object
    direction: str  # "BUY"/"SELL"
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

    # Execution fields
    entry_price: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    sl: float | None = None
    rr: float | None = None
    confidence: int | None = None
    confidence_emoji: str | None = None


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(df, period).replace(0, np.nan)

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0.0)


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return lower, mid, upper


def _is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    o1, c1 = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    o2, c2 = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    return (c1 < o1) and (c2 > o2) and (o2 <= c1) and (c2 >= o1)


def _is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    o1, c1 = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    o2, c2 = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    return (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1)


def detect_trend(
    df: pd.DataFrame,
    timeframe: str,
    ema_fast: int,
    ema_slow: int,
    ema_slope_bars: int,
) -> TrendState:
    close = df["close"].astype(float)
    ef = _ema(close, ema_fast)
    es = _ema(close, ema_slow)

    last_close = float(close.iloc[-1])
    last_ef = float(ef.iloc[-1])
    last_es = float(es.iloc[-1])

    if len(ef) > ema_slope_bars + 1:
        prev = float(ef.iloc[-(ema_slope_bars + 1)])
        diff = last_ef - prev
    else:
        diff = 0.0

    if diff > 0:
        slope = "UP"
    elif diff < 0:
        slope = "DOWN"
    else:
        slope = "FLAT"

    if last_ef > last_es and slope == "UP":
        direction = "BULL"
    elif last_ef < last_es and slope == "DOWN":
        direction = "BEAR"
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


def m15_confirms(df_m15: pd.DataFrame, trend_dir: str, ema_fast: int, rsi_period: int) -> Tuple[bool, str]:
    close = df_m15["close"].astype(float)
    ema = _ema(close, ema_fast)
    rsi = _rsi(close, rsi_period)

    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    last_rsi = float(rsi.iloc[-1])

    if trend_dir == "BULL":
        if last_close >= last_ema and last_rsi >= 45:
            return True, "M15 confirms bullish continuation (price above EMA, RSI supportive)"
        return False, f"M15 weak for BULL (close {last_close:.2f} vs EMA{ema_fast} {last_ema:.2f}, RSI {last_rsi:.1f})"

    if trend_dir == "BEAR":
        if last_close <= last_ema and last_rsi <= 55:
            return True, "M15 confirms bearish continuation (price below EMA, RSI supportive)"
        return False, f"M15 weak for BEAR (close {last_close:.2f} vs EMA{ema_fast} {last_ema:.2f}, RSI {last_rsi:.1f})"

    return False, "Trend is NEUTRAL"


def risk_tag_from_context(trend_tf: str) -> str:
    if trend_tf == "1h":
        return "MEDIUM"
    if trend_tf == "15min":
        return "MEDIUM"
    return "HIGH"


def score_entry_m5(df_m5: pd.DataFrame, direction: str, cfg) -> tuple[list[str], float, bool, list[str]]:
    close = df_m5["close"].astype(float)
    ema_fast = _ema(close, cfg.ema_fast)
    rsi = _rsi(close, cfg.rsi_period)
    adx_series = _adx(df_m5, cfg.adx_period)
    lower, mid, upper = _bollinger(close, 20, 2.0)

    last_close = float(close.iloc[-1])
    last_ema = float(ema_fast.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    adx_val = float(adx_series.iloc[-1])

    confirmations: list[str] = []
    reasons: list[str] = []

    if direction == "BUY" and last_rsi >= 50:
        confirmations.append("RSI")
        reasons.append(f"RSI supportive ({last_rsi:.1f} ≥ 50)")
    elif direction == "SELL" and last_rsi <= 50:
        confirmations.append("RSI")
        reasons.append(f"RSI supportive ({last_rsi:.1f} ≤ 50)")

    if direction == "BUY" and last_close >= last_ema:
        confirmations.append("EMA Pullback")
        reasons.append(f"Price above EMA{cfg.ema_fast} (continuation bias)")
    elif direction == "SELL" and last_close <= last_ema:
        confirmations.append("EMA Pullback")
        reasons.append(f"Price below EMA{cfg.ema_fast} (continuation bias)")

    if direction == "BUY" and _is_bullish_engulfing(df_m5):
        confirmations.append("Engulfing")
        reasons.append("Bullish engulfing present on M5")
    elif direction == "SELL" and _is_bearish_engulfing(df_m5):
        confirmations.append("Engulfing")
        reasons.append("Bearish engulfing present on M5")

    last_upper = float(upper.iloc[-1]) if not np.isnan(upper.iloc[-1]) else last_close
    last_lower = float(lower.iloc[-1]) if not np.isnan(lower.iloc[-1]) else last_close
    bb_width = max(0.0, last_upper - last_lower)

    if bb_width > 0:
        if len(upper) > 6:
            prev_upper = float(upper.iloc[-6]) if not np.isnan(upper.iloc[-6]) else last_upper
            prev_lower = float(lower.iloc[-6]) if not np.isnan(lower.iloc[-6]) else last_lower
            prev_width = max(0.0, prev_upper - prev_lower)
        else:
            prev_width = bb_width

        if bb_width >= prev_width:
            confirmations.append("BB Expansion")
            reasons.append("Bollinger width expanding (volatility pickup)")

    if len(close) >= 2:
        prev_close = float(close.iloc[-2])
        if direction == "BUY" and last_close > prev_close:
            confirmations.append("Momentum")
            reasons.append("Positive momentum (last candle closing higher)")
        elif direction == "SELL" and last_close < prev_close:
            confirmations.append("Momentum")
            reasons.append("Negative momentum (last candle closing lower)")

    adx_pass = adx_val >= float(cfg.adx_min)
    return confirmations, adx_val, adx_pass, reasons


def atr(df: pd.DataFrame, period: int = 14) -> float:
    s = _atr(df, period)
    v = float(s.iloc[-1])
    return v if v == v else 0.0


def compute_tp_sl_from_atr(
    entry: float,
    direction: str,
    atr_value: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float, float, float]:
    if atr_value <= 0:
        atr_value = max(entry * 0.001, 1.0)

    sl_dist = atr_value * sl_mult
    tp_dist = atr_value * tp_mult

    if direction == "BUY":
        sl = entry - sl_dist
        tp = entry + tp_dist
        rr = (tp - entry) / (entry - sl) if (entry - sl) != 0 else 0.0
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
        rr = (entry - tp) / (sl - entry) if (sl - entry) != 0 else 0.0

    return float(tp), float(sl), float(rr)


def confidence_score(
    confirmations_passed: int,
    confirmations_required: int,
    adx_value: float,
    adx_min: float,
    news_is_normal: bool,
) -> tuple[int, str]:
    base = 0.0
    if confirmations_required > 0:
        base = (confirmations_passed / confirmations_required) * 70.0

    adx_bonus = 0.0
    if adx_value >= adx_min:
        adx_bonus = 10.0
    if adx_value >= adx_min + 10:
        adx_bonus = 15.0

    news_adj = 10.0 if news_is_normal else -10.0
    score = int(max(0, min(100, base + adx_bonus + news_adj)))

    if score >= 75:
        emoji = "🔥"
    elif score >= 55:
        emoji = "🟡"
    else:
        emoji = "⚠️"

    return score, emoji