import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # ===== API KEYS =====
    TWELVEDATA_API_KEY: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # ===== SYMBOL & TIMEFRAMES =====
    SYMBOL: str = "XAUUSD"
    M5_INTERVAL: str = "5min"
    M15_INTERVAL: str = "15min"
    H1_INTERVAL: str = "1h"

    # ===== TRADING SESSIONS (GMT) =====
    SESSION_1_START: int = 0     # 00:00
    SESSION_1_END: int = 3       # 03:00
    SESSION_2_START: int = 7     # 07:00
    SESSION_2_END: int = 11      # 11:00
    SESSION_3_START: int = 12    # 12:00
    SESSION_3_END: int = 20      # 20:00

    # ===== RISK RULES (EQUITY EDGE SAFE) =====
    RISK_PER_TRADE_PCT: float = 0.25
    MAX_DAILY_LOSS_PCT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PCT: float = 10.0
    MAX_TRADES_PER_DAY: int = 3
    COOLDOWN_MINUTES: int = 30

    # ===== STRATEGY SETTINGS =====
    RSI_PERIOD: int = 14
    STOCH_K: int = 14
    STOCH_D: int = 3
    ADX_PERIOD: int = 14
    BB_PERIOD: int = 20
    BB_STDDEV: float = 2.0

    # ===== LOGGING =====
    LOG_LEVEL: str = "INFO"

    @classmethod
    def from_env(cls):
        """
        Load config safely from environment variables (Render compatible)
        """

        def get_env(name: str, default=None, required=False):
            value = os.getenv(name, default)
            if required and not value:
                raise RuntimeError(f"Missing required environment variable: {name}")
            return value

        return cls(
            TWELVEDATA_API_KEY=get_env("TWELVEDATA_API_KEY", required=True),
            TELEGRAM_BOT_TOKEN=get_env("TELEGRAM_BOT_TOKEN", required=True),
            TELEGRAM_CHAT_ID=get_env("TELEGRAM_CHAT_ID", required=True),

            SYMBOL=get_env("SYMBOL", "XAUUSD"),
            LOG_LEVEL=get_env("LOG_LEVEL", "INFO"),

            RISK_PER_TRADE_PCT=float(get_env("RISK_PER_TRADE_PCT", 0.25)),
            MAX_DAILY_LOSS_PCT=float(get_env("MAX_DAILY_LOSS_PCT", 3.0)),
            MAX_TOTAL_DRAWDOWN_PCT=float(get_env("MAX_TOTAL_DRAWDOWN_PCT", 10.0)),
            MAX_TRADES_PER_DAY=int(get_env("MAX_TRADES_PER_DAY", 3)),
            COOLDOWN_MINUTES=int(get_env("COOLDOWN_MINUTES", 30)),
        )
