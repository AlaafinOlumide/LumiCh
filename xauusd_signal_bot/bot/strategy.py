from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


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

    # NEW: execution fields
    entry_price: float | None = None
    tp: float | None = None
    sl: float | None = None
    rr: float | None = None
    confidence: int | None = None
    confidence_emoji: str | None = None


# -------------------------
# ATR helper (NEW)
# -------------------------
def atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Requires columns: high, low, close
    Returns latest ATR value (float).
    """
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

    out = tr.rolling(period).mean()
    v = float(out.iloc[-1])
    return v if v == v else 0.0  # handle NaN


def compute_tp_sl_from_atr(
    entry: float,
    direction: str,
    atr_value: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[float, float, float]:
    """
    Returns (tp, sl, rr)
    """
    if atr_value <= 0:
        # fallback to simple fixed distance (very small) if ATR not available
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
    """
    Returns (confidence_percent, emoji)
    """
    # Base from confirmations (0..70)
    base = 0.0
    if confirmations_required > 0:
        base = (confirmations_passed / confirmations_required) * 70.0

    # ADX bonus
    adx_bonus = 0.0
    if adx_value >= adx_min:
        adx_bonus = 10.0
    if adx_value >= adx_min + 10:
        adx_bonus = 15.0

    # News adjustment
    news_adj = 10.0 if news_is_normal else -10.0

    score = int(max(0, min(100, base + adx_bonus + news_adj)))

    # Emoji bands
    if score >= 75:
        emoji = "🔥"
    elif score >= 55:
        emoji = "🟡"
    else:
        emoji = "⚠️"

    return score, emoji


# -------------------------
# Keep your existing functions below
# detect_trend, m15_confirms, score_entry_m5, risk_tag_from_context
# -------------------------