# bot/config.py

from dataclasses import dataclass
import os


def get_env(name: str, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass
class Config:
    # ===== Core =====
    symbol: str = "XAUUSD"

    # ===== Trading Sessions =====
    # MUST stay a string (parse_sessions uses .split)
    trading_sessions: str = "00:00-03:00,07:00-11:00,12:00-20:00"

    # ===== Market Data =====
    twelvedata_api_key: str = ""

    # ===== Telegram =====
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ===== News Filter =====
    news_api_provider: str = "fmp"
    news_api_key: str = ""
    news_base_url: str = ""

    # Minutes BEFORE high-impact news to block trades
    news_lookahead_min: int = 60

    # Minutes AFTER high-impact news to block trades
    news_cooldown_after_min: int = 30  # ✅ FIX

    # ===== Runtime =====
    poll_seconds: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            symbol=get_env("SYMBOL", "XAUUSD"),

            trading_sessions=get_env(
                "TRADING_SESSIONS",
                "00:00-03:00,07:00-11:00,12:00-20:00"
            ),

            twelvedata_api_key=get_env("TWELVEDATA_API_KEY", ""),

            telegram_bot_token=get_env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=get_env("TELEGRAM_CHAT_ID", ""),

            news_api_provider=get_env("NEWS_API_PROVIDER", "fmp"),
            news_api_key=get_env("NEWS_API_KEY", ""),
            news_base_url=get_env("NEWS_BASE_URL", ""),

            news_lookahead_min=int(get_env("NEWS_LOOKAHEAD_MIN", 60)),
            news_cooldown_after_min=int(get_env("NEWS_COOLDOWN_AFTER_MIN", 30)),

            poll_seconds=int(get_env("POLL_SECONDS", 60)),
        )
