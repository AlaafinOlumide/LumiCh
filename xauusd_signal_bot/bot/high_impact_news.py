from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import requests


log = logging.getLogger(__name__)


@dataclass
class NewsStatus:
    is_high_impact: bool
    message: str


# simple cache so you don’t call news endpoint every loop
_last_check_at: dt.datetime | None = None
_last_result: NewsStatus | None = None


def check_high_impact_news(
    provider: str,
    api_key: str,
    base_url: str | None,
    lookahead_min: int,
    cooldown_after_min: int,
    now_utc: dt.datetime | None = None,
    ttl_seconds: int = 300,  # check news max every 5 mins
) -> NewsStatus:
    global _last_check_at, _last_result

    now = now_utc or dt.datetime.now(dt.timezone.utc)

    # If no key → disable news checks cleanly
    if not api_key:
        return NewsStatus(is_high_impact=False, message="News disabled (no API key)")

    # caching
    if _last_check_at and _last_result:
        if (now - _last_check_at).total_seconds() < ttl_seconds:
            return _last_result

    try:
        if (provider or "").lower() == "fmp":
            res = _check_fmp(api_key, base_url, now, lookahead_min)
        else:
            # default safe fallback
            res = NewsStatus(is_high_impact=False, message="News provider not supported; continuing")
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 401:
            # IMPORTANT: do not spam logs + do not crash loop
            log.warning("News check unauthorized (401). Disabling news checks until key is fixed.")
            res = NewsStatus(is_high_impact=False, message="News unavailable (401)")
        else:
            log.warning("News check HTTP error: %s", e)
            res = NewsStatus(is_high_impact=False, message="News unavailable (HTTP error)")
    except Exception as e:
        log.warning("News check failed: %s", e)
        res = NewsStatus(is_high_impact=False, message="News unavailable")

    _last_check_at = now
    _last_result = res
    return res


def _check_fmp(api_key: str, base_url: str | None, now: dt.datetime, lookahead_min: int) -> NewsStatus:
    # FMP endpoint example: /api/v3/economic_calendar?from=YYYY-MM-DD&to=YYYY-MM-DD&apikey=KEY
    # You can adjust filter logic later; for now: just fetch and decide.
    base = base_url or "https://financialmodelingprep.com/api/v3"
    url = f"{base}/economic_calendar"

    start = (now - dt.timedelta(days=1)).date().isoformat()
    end = (now + dt.timedelta(days=1)).date().isoformat()

    params = {"from": start, "to": end, "apikey": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    items = r.json()

    # Minimal safe logic: if any event within lookahead is “high”, flag it.
    # (Keep your existing filter logic if you already had it.)
    lookahead = now + dt.timedelta(minutes=lookahead_min)

    for ev in items if isinstance(items, list) else []:
        # Many FMP events have "date" string
        d = ev.get("date")
        impact = str(ev.get("impact", "")).lower()
        if not d:
            continue
        try:
            ev_time = dt.datetime.fromisoformat(str(d).replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        except Exception:
            continue

        if now <= ev_time <= lookahead and ("high" in impact):
            return NewsStatus(is_high_impact=True, message=f"High impact news at {ev_time:%H:%M} UTC")

    return NewsStatus(is_high_impact=False, message="Normal")