from __future__ import annotations

"""
high_impact_news.py — High-impact economic news filter.

Improvements vs prior version:
- File-based cache so news state survives process restarts
  (module-level variables reset to None on every deploy/restart,
   causing a burst of API calls on startup)
- Graceful fallback to in-memory cache if file system is unavailable
- Cleaner separation of cache read/write logic
"""

import datetime as dt
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache config
# ---------------------------------------------------------------------------

_CACHE_FILE = Path(os.getenv("NEWS_CACHE_PATH", "/tmp/lumich_news_cache.json"))

# In-memory fallback (used if file I/O fails)
_mem_check_at: dt.datetime | None = None
_mem_result: "NewsStatus | None" = None


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class NewsStatus:
    is_high_impact: bool
    message: str


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _read_cache() -> tuple[dt.datetime | None, NewsStatus | None]:
    """Read cached result from disk. Returns (checked_at, status) or (None, None)."""
    try:
        if not _CACHE_FILE.exists():
            return None, None
        data = json.loads(_CACHE_FILE.read_text())
        checked_at = dt.datetime.fromisoformat(data["checked_at"])
        status = NewsStatus(
            is_high_impact=bool(data["is_high_impact"]),
            message=str(data["message"]),
        )
        return checked_at, status
    except Exception as exc:
        log.debug("News cache read failed: %s", exc)
        return None, None


def _write_cache(checked_at: dt.datetime, status: NewsStatus) -> None:
    """Atomically write cache to disk."""
    try:
        payload = {
            "checked_at": checked_at.isoformat(),
            "is_high_impact": status.is_high_impact,
            "message": status.message,
        }
        # Write to temp file then rename for atomicity
        tmp = Path(tempfile.mktemp(dir=_CACHE_FILE.parent, suffix=".tmp"))
        tmp.write_text(json.dumps(payload))
        tmp.rename(_CACHE_FILE)
    except Exception as exc:
        log.debug("News cache write failed: %s", exc)


def _is_cache_fresh(checked_at: dt.datetime | None, now: dt.datetime, ttl_seconds: int) -> bool:
    if checked_at is None:
        return False
    return (now - checked_at).total_seconds() < ttl_seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_high_impact_news(
    provider: str,
    api_key: str,
    base_url: str | None,
    lookahead_min: int,
    cooldown_after_min: int,
    now_utc: dt.datetime | None = None,
    ttl_seconds: int = 300,
) -> NewsStatus:
    global _mem_check_at, _mem_result

    now = now_utc or dt.datetime.now(dt.timezone.utc)

    if not api_key:
        return NewsStatus(is_high_impact=False, message="News disabled (no API key)")

    # Try file cache first (survives restarts)
    cached_at, cached_status = _read_cache()
    if _is_cache_fresh(cached_at, now, ttl_seconds) and cached_status is not None:
        return cached_status

    # Fallback: in-memory cache (same process, faster)
    if _is_cache_fresh(_mem_check_at, now, ttl_seconds) and _mem_result is not None:
        return _mem_result

    # Fetch fresh
    try:
        if (provider or "").lower() == "fmp":
            result = _check_fmp(api_key, base_url, now, lookahead_min)
        else:
            result = NewsStatus(is_high_impact=False, message="News provider not supported; continuing")
    except requests.HTTPError as e:
        status_code = getattr(e.response, "status_code", None)
        if status_code == 401:
            log.warning("News API unauthorized (401). Disabling until key is corrected.")
            result = NewsStatus(is_high_impact=False, message="News unavailable (401 unauthorized)")
        elif status_code == 429:
            log.warning("News API rate limited (429). Will retry after TTL.")
            result = NewsStatus(is_high_impact=False, message="News unavailable (rate limited)")
        else:
            log.warning("News HTTP error %s: %s", status_code, e)
            result = NewsStatus(is_high_impact=False, message=f"News unavailable (HTTP {status_code})")
    except requests.Timeout:
        log.warning("News API timed out.")
        result = NewsStatus(is_high_impact=False, message="News unavailable (timeout)")
    except Exception as e:
        log.warning("News check failed: %s", e)
        result = NewsStatus(is_high_impact=False, message="News unavailable")

    # Persist to both caches
    _write_cache(now, result)
    _mem_check_at = now
    _mem_result = result

    return result


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _check_fmp(
    api_key: str,
    base_url: str | None,
    now: dt.datetime,
    lookahead_min: int,
) -> NewsStatus:
    """
    Fetches FMP economic calendar and checks for high-impact events
    within the lookahead window.
    """
    base = (base_url or "https://financialmodelingprep.com/api/v3").rstrip("/")
    url = f"{base}/economic_calendar"
    start = (now - dt.timedelta(days=1)).date().isoformat()
    end = (now + dt.timedelta(days=1)).date().isoformat()
    params = {"from": start, "to": end, "apikey": api_key}

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    items = r.json()
    if not isinstance(items, list):
        return NewsStatus(is_high_impact=False, message="Normal")

    lookahead_cutoff = now + dt.timedelta(minutes=lookahead_min)

    for ev in items:
        raw_date = ev.get("date")
        impact = str(ev.get("impact", "")).lower()
        if not raw_date or "high" not in impact:
            continue
        try:
            ev_time = dt.datetime.fromisoformat(
                str(raw_date).replace("Z", "+00:00")
            ).astimezone(dt.timezone.utc)
        except Exception:
            continue
        if now <= ev_time <= lookahead_cutoff:
            event_name = ev.get("event", "Unknown event")
            return NewsStatus(
                is_high_impact=True,
                message=f"High impact: {event_name} at {ev_time:%H:%M} UTC",
            )

    return NewsStatus(is_high_impact=False, message="Normal")
