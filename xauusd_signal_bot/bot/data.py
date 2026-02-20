from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import requests
from requests import Response


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
    """
    TwelveData client with:
    - retry + backoff on timeouts / transient network errors / 5xx
    - cached time series per timeframe
    - cached quote with short TTL
    """

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

        self._session = requests.Session()

        # cache:
        # - time series: key -> (fetched_at_utc, result)
        # - quote: symbol -> (fetched_at_utc, quote)
        self._cache: Dict[Tuple[str, str, int], Tuple[dt.datetime, TimeSeriesResult]] = {}
        self._quote_cache: Dict[str, Tuple[dt.datetime, QuoteResult]] = {}

    # -------------------------
    # Low-level HTTP with retry
    # -------------------------
    def _request_with_retry(self, method: str, url: str, *, params: dict | None = None, data: dict | None = None) -> Response:
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.request(method, url, params=params, data=data, timeout=self.timeout)

                # Retry on 5xx (transient)
                if 500 <= r.status_code <= 599:
                    raise TwelveDataError(f"TwelveData 5xx error {r.status_code}: {r.text[:200]}")

                r.raise_for_status()
                return r

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                last_exc = e
            except requests.exceptions.ConnectionError as e:
                last_exc = e
            except TwelveDataError as e:
                # includes our synthetic 5xx raise
                last_exc = e
            except Exception as e:
                last_exc = e

            # backoff before next attempt
            if attempt < self.max_retries:
                sleep_s = self.backoff_seconds * (attempt + 1)
                time.sleep(sleep_s)

        raise TwelveDataError(f"TwelveData request failed after retries: {last_exc}")

    def _raise_if_error_payload(self, data: Dict[str, Any]) -> None:
        if str(data.get("status", "")).lower() == "error":
            msg = str(data.get("message", data))
            lower = msg.lower()
            if "run out of api credits" in lower or "out of api credits" in lower:
                raise TwelveDataQuotaError(f"TwelveData error: {msg}")
            raise TwelveDataError(f"TwelveData error: {msg}")

    # -------------------------
    # Time series
    # -------------------------
    def fetch_time_series(self, symbol: str, interval: str, outputsize: int = 200) -> TimeSeriesResult:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON",
        }

        r = self._request_with_retry("GET", self.base_url, params=params)
        data: Dict[str, Any] = r.json()

        self._raise_if_error_payload(data)

        values = data.get("values", [])
        if not values:
            raise TwelveDataError(f"TwelveData error: empty values for {symbol} {interval}. Raw={data}")

        df = pd.DataFrame(values)

        # normalize OHLC columns
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # normalize time
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.sort_values("datetime").reset_index(drop=True)
            df = df.rename(columns={"datetime": "time"})

        if "time" not in df.columns:
            df["time"] = pd.to_datetime(df.index, utc=True)

        # drop rows with NaN close to avoid indicator explosions
        if "close" in df.columns:
            df = df[df["close"].notna()].reset_index(drop=True)

        return TimeSeriesResult(df=df, raw=data)

    def fetch_time_series_cached(
        self,
        symbol: str,
        interval: str,
        outputsize: int = 200,
        ttl_seconds: int = 300,
        now_utc: dt.datetime | None = None,
    ) -> TimeSeriesResult:
        """
        Cache per timeframe to reduce API calls drastically.
        ttl_seconds should match timeframe:
          - 1min  => 60
          - 5min  => 300
          - 15min => 900
          - 1h    => 3600
        """
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

    # -------------------------
    # Quote (live price)
    # -------------------------
    def fetch_quote(self, symbol: str) -> float:
        q = self.fetch_quote_cached(symbol=symbol, ttl_seconds=2)
        return float(q.price)

    def _extract_quote_price(self, data: Dict[str, Any]) -> Optional[float]:
        """
        TwelveData /quote sometimes returns:
          - price
        or (for some instruments/accounts):
          - close (as string)
        We'll accept: price -> close -> previous_close.
        """
        for k in ("price", "close", "previous_close"):
            v = data.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return None

    def _extract_quote_timestamp(self, data: Dict[str, Any], fallback_now: dt.datetime) -> dt.datetime:
        """
        Prefer last_quote_at (epoch seconds) -> timestamp (epoch seconds) -> fallback_now.
        """
        for k in ("last_quote_at", "timestamp"):
            v = data.get(k)
            try:
                if v is not None:
                    return dt.datetime.fromtimestamp(int(v), tz=dt.timezone.utc)
            except Exception:
                pass
        return fallback_now

    def fetch_quote_cached(
        self,
        symbol: str,
        ttl_seconds: int = 2,
        now_utc: dt.datetime | None = None,
    ) -> QuoteResult:
        """
        Very short TTL quote cache to avoid burning credits.
        ttl_seconds=1-3 is enough.
        """
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

        r = self._request_with_retry("GET", self.quote_url, params=params)
        data: Dict[str, Any] = r.json()

        self._raise_if_error_payload(data)

        price = self._extract_quote_price(data)
        if price is None:
            raise TwelveDataError(f"TwelveData quote missing price/close. Raw={data}")

        ts = self._extract_quote_timestamp(data, now)

        q = QuoteResult(
            symbol=sym,
            price=float(price),
            timestamp_utc=ts,
            raw=data,
        )

        self._quote_cache[sym] = (now, q)
        return q