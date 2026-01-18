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
    provider: str | None = None


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _in_window(event_time: dt.datetime, now: dt.datetime, lookahead_min: int, cooldown_after_min: int) -> bool:
    start = now - dt.timedelta(minutes=cooldown_after_min)
    end = now + dt.timedelta(minutes=lookahead_min)
    return start <= event_time <= end


def check_high_impact_news(
    provider: str,
    api_key: str,
    base_url: str | None,
    lookahead_min: int,
    cooldown_after_min: int,
) -> NewsStatus:
    """Returns whether a high-impact news window is active.

    Providers:
      - fmp: Financial Modeling Prep economic calendar (high-impact filter)

    If provider is missing/misconfigured, returns is_high_impact=False.
    """
    provider = (provider or "").lower()
    if not api_key:
        return NewsStatus(False, "News API key not set", provider=provider or None)

    now = _now_utc()

    try:
        if provider == "fmp":
            return _check_fmp(api_key, base_url, now, lookahead_min, cooldown_after_min)
        return NewsStatus(False, f"Unknown news provider: {provider}", provider=provider)
    except Exception as e:
        log.exception("News check failed")
        return NewsStatus(False, f"News check error: {e}", provider=provider)


def _check_fmp(api_key: str, base_url: str | None, now: dt.datetime, lookahead_min: int, cooldown_after_min: int) -> NewsStatus:
    """Financial Modeling Prep Economic Calendar.

    Expected endpoint:
      GET https://financialmodelingprep.com/api/v3/economic_calendar?from=YYYY-MM-DD&to=YYYY-MM-DD&apikey=KEY

    Notes:
      - Response is a list of events.
      - Many FMP calendar events include fields like 'country', 'event', 'date' and may include 'impact'.
      - If 'impact' is missing, we fall back to keyword heuristics.
    """
    base = (base_url or "https://financialmodelingprep.com/api/v3").rstrip("/")
    url = f"{base}/economic_calendar"

    # Pull a 2-day window to cover lookahead around midnight
    frm = (now - dt.timedelta(days=1)).date().isoformat()
    to = (now + dt.timedelta(days=1)).date().isoformat()

    params = {"from": frm, "to": to, "apikey": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    events = r.json()
    if not isinstance(events, list):
        return NewsStatus(False, "Unexpected FMP response", provider="fmp")

    high_events: list[tuple[dt.datetime, str]] = []
    for ev in events:
        try:
            country = str(ev.get("country", "")).upper()
            if country and country not in {"USD", "UNITED STATES", "US"}:
                # Only USD events by default. (You can broaden later.)
                continue

            # Parse event datetime
            # FMP often uses 'date' like '2024-01-01 13:30:00'
            date_str = ev.get("date") or ev.get("datetime") or ev.get("time")
            if not date_str:
                continue

            # Try multiple formats
            event_dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    parsed = dt.datetime.strptime(date_str, fmt)
                    event_dt = parsed
                    break
                except Exception:
                    continue
            if event_dt is None:
                # Last resort: let fromisoformat try
                try:
                    event_dt = dt.datetime.fromisoformat(str(date_str))
                except Exception:
                    continue

            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=dt.timezone.utc)
            else:
                event_dt = event_dt.astimezone(dt.timezone.utc)

            if not _in_window(event_dt, now, lookahead_min, cooldown_after_min):
                continue

            name = str(ev.get("event") or ev.get("title") or "Economic event")
            impact = str(ev.get("impact") or ev.get("importance") or "").lower()

            # High-impact decision
            is_high = False
            if impact in {"high", "3", "high impact", "important"}:
                is_high = True
            else:
                # Keyword heuristic fallback
                key_words = ["cpi", "core cpi", "ppi", "nonfarm", "nfp", "fed", "fomc", "powell", "interest rate", "rate decision", "gdp", "unemployment", "jobless", "retail sales"]
                lowered = name.lower()
                if any(k in lowered for k in key_words):
                    is_high = True

            if is_high:
                high_events.append((event_dt, name))
        except Exception:
            continue

    if not high_events:
        return NewsStatus(False, "No high-impact USD events in window", provider="fmp")

    high_events.sort(key=lambda x: x[0])
    soonest_dt, soonest_name = high_events[0]
    msg = f"HIGH IMPACT NEWS: {soonest_name} at {soonest_dt.strftime('%Y-%m-%d %H:%M')} UTC"
    return NewsStatus(True, msg, provider="fmp")
