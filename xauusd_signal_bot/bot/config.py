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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _normalize_xau_symbol_for_twelvedata(symbol: str) -> str:
    """
    TwelveData commonly accepts forex metals like 'XAU/USD' not 'XAUUSD'.
    We'll normalize common inputs.
    """
    s = (symbol or "").strip().upper()
    if s == "XAUUSD":
        return "XAU/USD"
    if s == "XAU-USD":
        return "XAU/USD"
    return s


@dataclass(frozen=True)
class Config:
    # --- Core ---
    symbol: str                 # your internal/bot symbol name (keep as XAUUSD)
    poll_seconds: int

    # --- TwelveData ---
    twelvedata_api_key: str
    symbol_twelvedata: str      # what TwelveData should receive (e.g. XAU/USD)

    # --- Telegram ---
    telegram_bot_token: str
    telegram_chat_id: str

    # --- Trading sessions (string spec) ---
    trading_sessions: str       # "00:00-03:00,07:00-11:00,12:00-20:00"

    # --- News filter (high-impact) ---
    enable_news_filter: bool
    news_api_provider: str      # "fmp"
    news_api_key: str
    news_base_url: str
    news_lookahead_min: int
    news_cooldown_after_min: int

    # --- Logging ---
    log_level: str

    @staticmethod
    def from_env() -> "Config":
        # Core
        symbol = (_env("SYMBOL", "XAUUSD") or "XAUUSD").strip().upper()
        poll_seconds = _env_int("POLL_SECONDS", 60)

        # TwelveData
        td_key = _env("TWELVEDATA_API_KEY") or ""
        # Allow explicit override; otherwise normalize from SYMBOL
        symbol_td_raw = _env("SYMBOL_TWELVEDATA")
        symbol_td = _normalize_xau_symbol_for_twelvedata(symbol_td_raw or symbol)

        # Telegram
        tg_token = _env("TELEGRAM_BOT_TOKEN") or ""
        tg_chat = _env("TELEGRAM_CHAT_ID") or ""

        # Sessions must be STRING (sessions.py uses .split(","))
        sessions = _env(
            "TRADING_SESSIONS",
            "00:00-03:00,07:00-11:00,12:00-20:00",
        ) or "00:00-03:00,07:00-11:00,12:00-20:00"

        # News
        provider = (_env("NEWS_API_PROVIDER", "fmp") or "fmp").strip().lower()
        news_key = _env("NEWS_API_KEY") or ""
        news_base_url = _env("NEWS_BASE_URL", "") or ""
        lookahead = _env_int("NEWS_LOOKAHEAD_MIN", 60)
        cooldown_after = _env_int("NEWS_COOLDOWN_AFTER_MIN", 30)

        # If user enabled news filter but key is missing, auto-disable to prevent 401 spam/crashes
        enable_news = _env_bool("ENABLE_NEWS_FILTER", True)
        if enable_news and provider == "fmp" and not news_key:
            enable_news = False

        log_level = (_env("LOG_LEVEL", "INFO") or "INFO").strip().upper()

        return Config(
            symbol=symbol,
            poll_seconds=poll_seconds,
            twelvedata_api_key=td_key,
            symbol_twelvedata=symbol_td,
            telegram_bot_token=tg_token,
            telegram_chat_id=tg_chat,
            trading_sessions=sessions,
            enable_news_filter=enable_news,
            news_api_provider=provider,
            news_api_key=news_key,
            news_base_url=news_base_url,
            news_lookahead_min=lookahead,
            news_cooldown_after_min=cooldown_after,
            log_level=log_level,
        )

    @staticmethod
    def load() -> "Config":
        # Backwards-compatible alias
        return Config.from_env()
