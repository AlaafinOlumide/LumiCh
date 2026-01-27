# bot/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return None
    v = v.strip()
    return v if v != "" else None


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Config:
    # --- Core ---
    symbol: str
    poll_seconds: int

    # --- TwelveData ---
    twelvedata_api_key: str

    # --- Telegram ---
    telegram_bot_token: str
    telegram_chat_id: str

    # --- Trading sessions ---
    # IMPORTANT: sessions.py expects a STRING like "00:00-03:00,07:00-11:00,12:00-20:00"
    trading_sessions: str

    # --- News filter (high-impact) ---
    news_api_provider: str        # e.g. "fmp"
    news_api_key: str             # API key for provider
    news_base_url: str            # optional override (empty means provider default inside your news module)
    news_lookahead_min: int       # e.g. 60 (minutes ahead)
    news_cooldown_after_min: int  # e.g. 30 (minutes after event)

    # --- Optional toggles ---
    enable_news_filter: bool
    log_level: str

    @staticmethod
    def from_env() -> "Config":
        # Core
        symbol = (_env("SYMBOL", "XAUUSD") or "XAUUSD").strip()
        poll_seconds = _env_int("POLL_SECONDS", 60)

        # TwelveData
        td_key = _env("TWELVEDATA_API_KEY") or ""

        # Telegram
        tg_token = _env("TELEGRAM_BOT_TOKEN") or ""
        tg_chat = _env("TELEGRAM_CHAT_ID") or ""

        # Sessions: must be a STRING (comma-separated)
        sessions = _env(
            "TRADING_SESSIONS",
            "00:00-03:00,07:00-11:00,12:00-20:00",
        ) or "00:00-03:00,07:00-11:00,12:00-20:00"

        # News filter
        provider = (_env("NEWS_API_PROVIDER", "fmp") or "fmp").strip().lower()
        news_key = _env("NEWS_API_KEY") or ""
        news_base_url = _env("NEWS_BASE_URL", "") or ""  # keep empty string if not set
        lookahead = _env_int("NEWS_LOOKAHEAD_MIN", 60)
        cooldown_after = _env_int("NEWS_COOLDOWN_AFTER_MIN", 30)

        enable_news = _env_bool("ENABLE_NEWS_FILTER", True)
        log_level = (_env("LOG_LEVEL", "INFO") or "INFO").strip().upper()

        return Config(
            symbol=symbol,
            poll_seconds=poll_seconds,
            twelvedata_api_key=td_key,
            telegram_bot_token=tg_token,
            telegram_chat_id=tg_chat,
            trading_sessions=sessions,
            news_api_provider=provider,
            news_api_key=news_key,
            news_base_url=news_base_url,
            news_lookahead_min=lookahead,
            news_cooldown_after_min=cooldown_after,
            enable_news_filter=enable_news,
            log_level=log_level,
        )

    # Backwards compatible alias (in case any file still calls Config.load())
    @staticmethod
    def load() -> "Config":
        return Config.from_env()
