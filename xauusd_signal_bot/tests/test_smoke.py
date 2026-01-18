import pandas as pd

from bot.strategy import detect_trend


def test_detect_trend_runs():
    idx = pd.date_range("2026-01-01", periods=250, freq="h", tz="UTC")
    close = pd.Series(range(250), index=idx, dtype=float)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=idx)
    t = detect_trend(df, "1h", ema_fast_p=50, ema_slow_p=200, slope_bars=10)
    assert t.direction in {"BULL", "BEAR", "NEUTRAL"}
