from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytz


@dataclass(frozen=True)
class Session:
    start: dt.time
    end: dt.time


def parse_sessions(spec: str) -> list[Session]:
    """Parse 'HH:MM-HH:MM,HH:MM-HH:MM' into Session list."""
    sessions: list[Session] = []
    if not spec:
        return sessions
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        left, right = part.split("-")
        sh, sm = map(int, left.strip().split(":"))
        eh, em = map(int, right.strip().split(":"))
        sessions.append(Session(start=dt.time(sh, sm), end=dt.time(eh, em)))
    return sessions


def now_in_sessions_utc(sessions: list[Session], now_utc: dt.datetime | None = None) -> bool:
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    t = now.timetz().replace(tzinfo=None)
    for s in sessions:
        # same-day window
        if s.start <= s.end:
            if s.start <= t <= s.end:
                return True
        else:
            # window that crosses midnight
            if t >= s.start or t <= s.end:
                return True
    return False


def session_label(sessions: list[Session], now_utc: dt.datetime | None = None) -> str:
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    t = now.timetz().replace(tzinfo=None)
    for s in sessions:
        if s.start <= s.end:
            if s.start <= t <= s.end:
                return f"{s.start.strftime('%H:%M')}-{s.end.strftime('%H:%M')} GMT"
        else:
            if t >= s.start or t <= s.end:
                return f"{s.start.strftime('%H:%M')}-{s.end.strftime('%H:%M')} GMT"
    return "OUTSIDE SESSIONS"
