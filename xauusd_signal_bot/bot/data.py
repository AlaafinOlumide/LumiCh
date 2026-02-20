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


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


class TwelveDataClient:
    def __init__(
        self,
        api_key: str,
        timeout: int = 20,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

        self.base_url = "https://api.twelvedata.com/time_series"
        self.quote_url = "https://api.twelvedata.com/quote"

        # cache:
        # - time series: key -> (fetched_at_utc, result)
        # - quote: symbol -> (fetched_at_utc, quote)
        self._cache: Dict[Tuple[str, str, int], Tuple[dt.datetime, TimeSeriesResult]] = {}
        self._quote_cache: Dict[str, Tuple[dt.datetime, QuoteResult]] = {}

        self._session = requests.Session()

    def _get_json(self, url: str, params: dict) -> Dict[str, Any]:
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    sleep_s = self.backoff_seconds * (attempt + 1)
                    log.warning("TwelveData request failed (%s). Retry in %.1fs...", e, sleep_s)
                    time.sleep(sleep_s)
                    continue
                break

        raise TwelveDataError(f"TwelveData request failed after retries: {last_err}")

    def _raise_if_error_payload(self, data: Dict[str, Any]) -> None:
        if str(data.get("status", "")).lower() == "error":
            msg = str(data.get("message", data))
            lower = msg.lower()
            if "run out of api credits" in lower or "out of api credits" in lower:
                raise TwelveDataQuotaError(f"TwelveData error: {msg}")
            raise TwelveDataError(f"TwelveData error: {msg}")

    # =========================
    # Time Series
    # =========================
    def fetch_time_series(self, symbol: str, interval: str, outputsize: int = 200) -> TimeSeriesResult:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
        }

        data: Dict[str, Any] = self._get_json(self.base_url, params=params)
        self._raise_if_error_payload(data)

        values = data.get("values", [])
        if not values:
            raise TwelveDataError(f"TwelveData error: empty values for {symbol} {interval}. Raw={data}")

        df = pd.DataFrame(values)

        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # TwelveData uses 'datetime'
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
    # Quote (live-ish price)
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
            age = (now - fetched_at).total_seconds()
            if age < ttl_seconds:
                return q

        params = {
            "symbol": sym,
            "apikey": self.api_key,
            "format": "JSON",
        }

        data: Dict[str, Any] = self._get_json(self.quote_url, params=params)
        self._raise_if_error_payload(data)

        # TwelveData quote responses vary by asset type.
        # Prefer mid(bid/ask) if present, else price/last, else close.
        bid = _to_float(data.get("bid"))
        ask = _to_float(data.get("ask"))

        price = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            price = (bid + ask) / 2.0

        if price is None:
            price = _to_float(data.get("price"))

        if price is None:
            price = _to_float(data.get("last"))

        if price is None:
            # Many times TwelveData returns close instead of price
            price = _to_float(data.get("close"))

        if price is None:
            raise TwelveDataError(f"TwelveData quote missing usable price fields. Raw={data}")

        q = QuoteResult(
            symbol=sym,
            price=float(price),
            timestamp_utc=now,
            raw=data,
        )

        self._quote_cache[sym] = (now, q)
        return q

    def fetch_quote(self, symbol: str) -> float:
        return float(self.fetch_quote_cached(symbol=symbol, ttl_seconds=2).price)