from __future__ import annotations

import datetime as dt
import logging
import time
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


class TwelveDataClient:
    def __init__(self, api_key: str, timeout: int = 20, max_retries: int = 2, backoff_seconds: float = 1.0) -> None:
        self.api_key = api_key
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)

        self.base_url = "https://api.twelvedata.com/time_series"
        self.quote_url = "https://api.twelvedata.com/quote"

        # cache:
        self._cache: Dict[Tuple[str, str, int], Tuple[dt.datetime, TimeSeriesResult]] = {}
        self._quote_cache: Dict[str, Tuple[dt.datetime, QuoteResult]] = {}

        self._session = requests.Session()

    def _request_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ReadTimeout as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                raise
            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                raise
        raise TwelveDataError(f"HTTP request failed: {last_err}")

    def _raise_if_error(self, data: Dict[str, Any]) -> None:
        if str(data.get("status", "")).lower() == "error":
            msg = str(data.get("message", data))
            lower = msg.lower()
            if "run out of api credits" in lower or "out of api credits" in lower:
                raise TwelveDataQuotaError(f"TwelveData error: {msg}")
            raise TwelveDataError(f"TwelveData error: {msg}")

    def fetch_time_series(self, symbol: str, interval: str, outputsize: int = 200) -> TimeSeriesResult:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": int(outputsize),
            "apikey": self.api_key,
            "format": "JSON",
        }

        data: Dict[str, Any] = self._request_json(self.base_url, params=params)
        self._raise_if_error(data)

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
            if (now - fetched_at).total_seconds() < ttl_seconds:
                return result

        result = self.fetch_time_series(symbol, interval, outputsize=outputsize)
        self._cache[key] = (now, result)
        return result

    # =========================
    # Quote
    # =========================
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
            if (now - fetched_at).total_seconds() < ttl_seconds:
                return q

        params = {
            "symbol": sym,
            "apikey": self.api_key,
            "format": "JSON",
        }

        data: Dict[str, Any] = self._request_json(self.quote_url, params=params)
        self._raise_if_error(data)

        # TwelveData quote sometimes returns 'price' OR only OHLC with 'close'
        price_raw = data.get("price", None)
        if price_raw is None:
            price_raw = data.get("close", None)

        if price_raw is None:
            raise TwelveDataError(f"TwelveData quote missing price/close. Raw={data}")

        try:
            price = float(price_raw)
        except Exception:
            raise TwelveDataError(f"TwelveData quote invalid price/close. Raw={data}")

        # Prefer seconds timestamp if present
        ts = None
        for k in ("last_quote_at", "timestamp"):
            v = data.get(k)
            if v is not None:
                try:
                    ts = dt.datetime.fromtimestamp(int(v), tz=dt.timezone.utc)
                    break
                except Exception:
                    pass
        timestamp_utc = ts or now

        q = QuoteResult(
            symbol=sym,
            price=float(price),
            timestamp_utc=timestamp_utc,
            raw=data,
        )
        self._quote_cache[sym] = (now, q)
        return q