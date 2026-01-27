from __future__ import annotations

import os
import requests
from typing import List, Dict, Any


class TwelveDataClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise RuntimeError("TWELVEDATA_API_KEY is required")

    def fetch_ohlcv(self, symbol: str, interval: str, outputsize: int = 300) -> List[Dict[str, Any]]:
        """
        Returns OHLCV as a list of dicts sorted oldest -> newest.
        Each item: {"datetime","open","high","low","close","volume"}
        """
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
            "timezone": "UTC",
        }

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("status") == "error":
            raise RuntimeError(f"TwelveData error: {data.get('message')}")

        values = data.get("values", [])
        if not values:
            raise RuntimeError(f"No data returned for {symbol} {interval}")

        # TwelveData returns newest -> oldest, so reverse
        values = list(reversed(values))

        out: List[Dict[str, Any]] = []
        for v in values:
            out.append({
                "datetime": v["datetime"],
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": float(v.get("volume") or 0),
            })
        return out