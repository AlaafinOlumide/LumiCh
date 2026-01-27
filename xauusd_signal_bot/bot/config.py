import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # ===== Required secrets =====
    TWELVEDATA_API_KEY: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # ===== Core =====
    SYMBOL: str = "XAUUSD"
    LOG_LEVEL: str = "INFO"

    # ===== Trading sessions (what main.py expects) =====
    # Format: "HH:MM-HH:MM" in GMT/UTC
    trading_sessions: str = "00:00-03:00,07:00-11:00,12:00-20:00"  # will be set in from_env()

    # ===== Risk rules =====
    RISK_PER_TRADE_PCT: float = 0.25
    MAX_DAILY_LOSS_PCT: float = 3.0
    MAX_TOTAL_DRAWDOWN_PCT: float = 10.0
    MAX_TRADES_PER_DAY: int = 3
    COOLDOWN_MINUTES: int = 30

    # ===== Strategy params (optional / safe defaults) =====
    RSI_PERIOD: int = 14
    STOCH_K: int = 14
    STOCH_D: int = 3
    ADX_PERIOD: int = 14
    BB_PERIOD: int = 20
    BB_STDDEV: float = 2.0

    @classmethod
    def from_env(cls):
        """
        Render-friendly config loader.
        Supports either:
          - TRADING_SESSIONS="00:00-03:00,07:00-11:00,12:00-20:00"
        or defaults to the sessions above.
        """

        def get_env(name: str, default=None, required: bool = False):
            value = os.getenv(name, default)
            if required and (value is None or str(value).strip() == ""):
                raise RuntimeError(f"Missing required environment variable: {name}")
            return value

        # Sessions: env override or default
        sessions_raw = get_env("TRADING_SESSIONS", "00:00-03:00,07:00-11:00,12:00-20:00")
        sessions_list = [s.strip() for s in sessions_raw.split(",") if s.strip()]

        return cls(
            TWELVEDATA_API_KEY=get_env("TWELVEDATA_API_KEY", required=True),
            TELEGRAM_BOT_TOKEN=get_env("TELEGRAM_BOT_TOKEN", required=True),
            TELEGRAM_CHAT_ID=get_env("TELEGRAM_CHAT_ID", required=True),

            SYMBOL=get_env("SYMBOL", "XAUUSD"),
            LOG_LEVEL=get_env("LOG_LEVEL", "INFO"),

            trading_sessions=sessions_list,

            RISK_PER_TRADE_PCT=float(get_env("RISK_PER_TRADE_PCT", 0.25)),
            MAX_DAILY_LOSS_PCT=float(get_env("MAX_DAILY_LOSS_PCT", 3.0)),
            MAX_TOTAL_DRAWDOWN_PCT=float(get_env("MAX_TOTAL_DRAWDOWN_PCT", 10.0)),
            MAX_TRADES_PER_DAY=int(get_env("MAX_TRADES_PER_DAY", 3)),
            COOLDOWN_MINUTES=int(get_env("COOLDOWN_MINUTES", 30)),
        )
