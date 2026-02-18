from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import requests


log = logging.getLogger(__name__)


class TwelveDataError(RuntimeError):
    pass


class TwelveDataQuotaError(TwelveDataError):
    """Raised when daily API credits are exhausted."""


@dataclass
class TimeSeriesResult:
    df: pd.DataFrame
    raw: dict


@dataclass
class QuoteResult:
    symbol: str
    price: float
    timestamp_utc: dt.datetime
    raw: dict
    source_field: str


class TwelveDataClient:
    def __init__(self, api_key: str, timeout: int = 20) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://api.twelvedata.com/time_series"
        self.quote_url = "https://api.twelvedata.com/quote"

        # cache:
        # - time series: key -> (fetched_at_utc, result)
        # - quote: symbol -> (fetched_at_utc, quote)
        self._cache: Dict[Tuple[str, str, int], Tuple[dt.datetime, TimeSeriesResult]] = {}
        self._quote_cache: Dict[str, Tuple[dt.datetime, QuoteResult]] = {}

    # =========================
    # Time series
    # =========================
    def fetch_time_series(self, symbol: str, interval: str, outputsize: int = 200) -> TimeSeriesResult:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
        }

        r = requests.get(self.base_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data: Dict[str, Any] = r.json()

        if str(data.get("status", "")).lower() == "error":
            msg = str(data.get("message", data))
            lower = msg.lower()
            if "run out of api credits" in lower or "out of api credits" in lower:
                raise TwelveDataQuotaError(f"TwelveData error: {msg}")
            raise TwelveDataError(f"TwelveData error: {msg}")

        values = data.get("values", [])
        if not values:
            raise TwelveDataError(f"TwelveData error: empty values for {symbol} {interval}. Raw={data}")

        df = pd.DataFrame(values)

        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.sort_values("datetime").reset_index(drop=True)
            df = df.rename(columns={"datetime": "time"})

        if "time" not in df.columns:
            df["time"] = pd.to_datetime(df.index, utc=True)

        return TimeSeriesResult(df=df, raw=data)

    def fetch_time_series_cached(
        self,
        symbol: str,
        interval: str,
        outputsize: int = 200,
        ttl_seconds: int = 300,
        now_utc: dt.datetime | None = None,
    ) -> TimeSeriesResult:
        now = now_utc or dt.datetime.now(dt.timezone.utc)
        key = (symbol, interval, int(outputsize))

        cached = self._cache.get(key)
        if cached:
            fetched_at, result = cached
            age = (now - fetched_at).total_seconds()
            if age < ttl_seconds:
                return result

        result = self.fetch_time_series(symbol, interval, outputsize=outputsize)
        self._cache[key] = (now, result)
        return result

    # =========================
    # Quote (live-ish)
    # =========================
    def _extract_quote_price(self, data: Dict[str, Any]) -> Tuple[Optional[float], str]:
        """
        TwelveData /quote is inconsistent by instrument.
        For XAU/USD we often see `close` but no `price`.
        We'll accept first valid numeric field from a priority list.
        """
        candidates = ["price", "last", "bid", "ask", "close", "previous_close", "open"]
        for k in candidates:
            v = data.get(k)
            if v is None:
                continue
            try:
                f = float(v)
                if f == f:
                    return f, k
            except Exception:
                continue
        return None, ""

    def fetch_quote(self, symbol: str) -> float:
        q = self.fetch_quote_cached(symbol=symbol, ttl_seconds=2)
        return float(q.price)

    def fetch_quote_cached(
        self,
        symbol: str,
        ttl_seconds: int = 2,
        now_utc: dt.datetime | None = None,
    ) -> QuoteResult:
        now = now_utc or dt.datetime.now(dt.timezone.utc)
        sym = (symbol or "").strip()

        cached = self._quote_cache.get(sym)
        if cached:
            fetched_at, q = cached
            age = (now - fetched_at).total_seconds()
            if age < ttl_seconds:
                return q

        params = {
            "symbol": sym,
            "apikey": self.api_key,
            "format": "JSON",
        }
        r = requests.get(self.quote_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data: Dict[str, Any] = r.json()

        if str(data.get("status", "")).lower() == "error":
            msg = str(data.get("message", data))
            lower = msg.lower()
            if "run out of api credits" in lower or "out of api credits" in lower:
                raise TwelveDataQuotaError(f"TwelveData error: {msg}")
            raise TwelveDataError(f"TwelveData error: {msg}")

        price, used_field = self._extract_quote_price(data)
        if price is None:
            raise TwelveDataError(f"TwelveData quote missing usable price field. Raw={data}")

        q = QuoteResult(
            symbol=sym,
            price=float(price),
            timestamp_utc=now,
            raw=data,
            source_field=used_field,
        )

        self._quote_cache[sym] = (now, q)
        return q