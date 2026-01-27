from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Signal:
    symbol: str
    side: str  # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    reason: str
    score: float
    risk_tag: str


# -----------------------------
# Helpers (no pandas)
# -----------------------------
def _arr(candles: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(c[key]) for c in candles], dtype=float)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nan)
    alpha = 2 / (period + 1)
    out = np.empty_like(values)
    out[:] = np.nan
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Wilder smoothing
    avg_gain = np.empty_like(close); avg_gain[:] = np.nan
    avg_loss = np.empty_like(close); avg_loss[:] = np.nan
    avg_gain[period] = gain[1:period+1].mean()
    avg_loss[period] = loss[1:period+1].mean()

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return out


def stoch_kd(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3):
    n = len(close)
    k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = np.max(high[i - k_period + 1:i + 1])
        ll = np.min(low[i - k_period + 1:i + 1])
        rng = (hh - ll) if (hh - ll) != 0 else 1e-12
        k[i] = 100 * (close[i] - ll) / rng
    # D = SMA of K
    d = np.full(n, np.nan)
    for i in range(d_period - 1, n):
        window = k[i - d_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        d[i] = window.mean()
    return k, d


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0.0
    down_move[0] = 0.0

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    # Wilder smoothing
    def wilder_smooth(x: np.ndarray, p: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        out[p] = np.sum(x[1:p+1])
        for i in range(p + 1, len(x)):
            out[i] = out[i - 1] - (out[i - 1] / p) + x[i]
        return out

    atr = wilder_smooth(tr, period)
    p_dm = wilder_smooth(plus_dm, period)
    m_dm = wilder_smooth(minus_dm, period)

    plus_di = 100 * (p_dm / (atr + 1e-12))
    minus_di = 100 * (m_dm / (atr + 1e-12))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))

    adx_out = np.full(n, np.nan)
    # first ADX value is average of DX over period
    start = period * 2
    if start < n:
        adx_out[start] = np.nanmean(dx[period+1:start+1])
        for i in range(start + 1, n):
            adx_out[i] = ((adx_out[i - 1] * (period - 1)) + dx[i]) / period
    return adx_out


# -----------------------------
# Strategy logic (no pandas)
# -----------------------------
def detect_trend(h1_candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    close = _arr(h1_candles, "close")
    high = _arr(h1_candles, "high")
    low = _arr(h1_candles, "low")

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    adx14 = adx(high, low, close, 14)

    i = len(close) - 1
    bullish = ema50[i] > ema200[i]
    strong = (not np.isnan(adx14[i])) and (adx14[i] >= 18)  # mild strength

    direction = "BUY" if bullish else "SELL"
    return {"direction": direction, "strong": strong, "adx": float(adx14[i]) if not np.isnan(adx14[i]) else None}


def m15_confirms(m15_candles: List[Dict[str, Any]], direction: str) -> bool:
    # Simple structure confirmation: last 3 closes aligned
    close = _arr(m15_candles, "close")
    if len(close) < 5:
        return False
    last = close[-1]
    prev1 = close[-2]
    prev2 = close[-3]

    if direction == "BUY":
        return (last > prev1) and (prev1 > prev2)
    else:
        return (last < prev1) and (prev1 < prev2)


def risk_tag_from_context(trend_info: Dict[str, Any]) -> str:
    # Simple tag
    if trend_info.get("strong"):
        return "A+"
    return "B"


def score_entry_m5(m5_candles: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
    close = _arr(m5_candles, "close")
    high = _arr(m5_candles, "high")
    low = _arr(m5_candles, "low")

    r = rsi(close, 14)
    k, d = stoch_kd(high, low, close, 14, 3)

    i = len(close) - 1
    if np.isnan(r[i]) or np.isnan(k[i]) or np.isnan(d[i]):
        return {"ok": False, "score": 0.0, "why": "Not enough data"}

    score = 0.0
    reasons = []

    # Trend-friendly oscillator logic
    if direction == "BUY":
        if r[i] >= 45:
            score += 1.0; reasons.append("RSI ok")
        if k[i] > d[i] and k[i] < 80:
            score += 1.0; reasons.append("Stoch cross up")
    else:
        if r[i] <= 55:
            score += 1.0; reasons.append("RSI ok")
        if k[i] < d[i] and k[i] > 20:
            score += 1.0; reasons.append("Stoch cross down")

    ok = score >= 1.5
    return {"ok": ok, "score": float(score), "why": ", ".join(reasons) if reasons else "No trigger"}


def build_signal(symbol: str,
                 direction: str,
                 h1: List[Dict[str, Any]],
                 m15: List[Dict[str, Any]],
                 m5: List[Dict[str, Any]]) -> Optional[Signal]:

    trend = detect_trend(h1)
    if trend["direction"] != direction:
        direction = trend["direction"]

    if not trend["strong"]:
        return None
    if not m15_confirms(m15, direction):
        return None

    trig = score_entry_m5(m5, direction)
    if not trig["ok"]:
        return None

    # Basic SL/TP using last swing range on M5
    close = _arr(m5, "close")
    high = _arr(m5, "high")
    low = _arr(m5, "low")
    entry = float(close[-1])

    recent_range = float(np.nanmax(high[-20:]) - np.nanmin(low[-20:])) if len(close) >= 20 else float(high[-1] - low[-1])
    recent_range = max(recent_range, 0.5)  # avoid zero

    if direction == "BUY":
        sl = entry - 0.8 * recent_range
        tp1 = entry + 1.0 * recent_range
        tp2 = entry + 1.8 * recent_range
    else:
        sl = entry + 0.8 * recent_range
        tp1 = entry - 1.0 * recent_range
        tp2 = entry - 1.8 * recent_range

    reason = f"H1 {trend['direction']} strong (ADX={trend.get('adx')}), M15 confirms, M5 trigger: {trig['why']}"
    return Signal(
        symbol=symbol,
        side=direction,
        entry=entry,
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        reason=reason,
        score=float(trig["score"]),
        risk_tag=risk_tag_from_context(trend),
    )
