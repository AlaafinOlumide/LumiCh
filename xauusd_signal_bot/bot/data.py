# bot/data.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests
import pandas as pd


@dataclass
class TimeSeriesResponse:
    df: pd.DataFrame
    raw: Dict[str, Any]


class TwelveDataClient:
    """
    Minimal TwelveData REST client.
    Provides fetch_time_series(...) used by bot/main.py:
        td.fetch_time_series(symbol, "1h", outputsize=200).df
    """

    def __init__(self, api_key: str, base_url: str = "https://api.twelvedata.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def fetch_time_series(
        self,
        symbol: str,
        interval: str,
        outputsize: int = 200,
        timezone: str = "UTC",
        dp: int = 5,
    ) -> TimeSeriesResponse:
        """
        Fetch OHLCV time series from TwelveData and return a TimeSeriesResponse
        with a pandas DataFrame at .df.

        DataFrame columns:
            datetime, open, high, low, close, volume
        Sorted ascending by datetime.
        """

        if not self.api_key:
            raise RuntimeError("TWELVEDATA_API_KEY is missing/empty")

        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "timezone": timezone,
            "apikey": self.api_key,
            "dp": dp,
            "format": "JSON",
        }

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        # TwelveData uses "status":"error" with a message
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"TwelveData error: {data.get('message', data)}")

        values = data.get("values") if isinstance(data, dict) else None
        if not values:
            raise RuntimeError(f"TwelveData returned no values for {symbol} {interval}: {data}")

        df = pd.DataFrame(values)

        # Normalize expected columns
        # TwelveData typically returns: datetime, open, high, low, close, volume (all as strings)
        expected = ["datetime", "open", "high", "low", "close", "volume"]
        for col in expected:
            if col not in df.columns:
                # volume can be missing for some assets
                if col == "volume":
                    df["volume"] = 0
                else:
                    raise RuntimeError(f"Missing column '{col}' in TwelveData response: columns={list(df.columns)}")

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean + order
        df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)

        return TimeSeriesResponse(df=df, raw=data)
