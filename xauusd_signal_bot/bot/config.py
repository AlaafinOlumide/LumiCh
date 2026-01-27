import os
from dataclasses import dataclass


@dataclass
class Config:
    # ===== Required secrets (match what main.py expects) =====
    twelvedata_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str

    # ===== Core =====
    symbol: str = "XAUUSD"
    log_level: str = "INFO"

    # ===== Trading sessions (string for sessions.py) =====
    # Format: "HH:MM-HH:MM,HH:MM-HH:MM,HH:MM-HH:MM" in GMT/UTC
    trading_sessions: str = "00:00-03:00,07:00-11:00,12:00-20:00"

    # ===== Risk rules =====
    risk_per_trade_pct: float = 0.25
    max_daily_loss_pct: float = 3.0
    max_total_drawdown_pct: float = 10.0
    max_trades_per_day: int = 3
    cooldown_minutes: int = 30

    # ===== Strategy params =====
    rsi_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    adx_period: int = 14
    bb_period: int = 20
    bb_stddev: float = 2.0

    @classmethod
    def from_env(cls):
        """
        Loads config from Render environment variables.
        """

        def get_env(name: str, default=None, required: bool = False):
            value = os.getenv(name, default)
            if required and (value is None or str(value).strip() == ""):
                raise RuntimeError(f"Missing required environment variable: {name}")
            return value

        return cls(
            twelvedata_api_key=get_env("TWELVEDATA_API_KEY", required=True),
            telegram_bot_token=get_env("TELEGRAM_BOT_TOKEN", required=True),
            telegram_chat_id=get_env("TELEGRAM_CHAT_ID", required=True),

            symbol=get_env("SYMBOL", "XAUUSD"),
            log_level=get_env("LOG_LEVEL", "INFO"),
            trading_sessions=get_env("TRADING_SESSIONS", "00:00-03:00,07:00-11:00,12:00-20:00"),

            risk_per_trade_pct=float(get_env("RISK_PER_TRADE_PCT", 0.25)),
            max_daily_loss_pct=float(get_env("MAX_DAILY_LOSS_PCT", 3.0)),
            max_total_drawdown_pct=float(get_env("MAX_TOTAL_DRAWDOWN_PCT", 10.0)),
            max_trades_per_day=int(get_env("MAX_TRADES_PER_DAY", 3)),
            cooldown_minutes=int(get_env("COOLDOWN_MINUTES", 30)),
        )
