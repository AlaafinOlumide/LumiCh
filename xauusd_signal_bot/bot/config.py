import os
from dataclasses import dataclass


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def _getint(name: str, default: int) -> int:
    v = _getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _getfloat(name: str, default: float) -> float:
    v = _getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    symbol: str
    twelvedata_api_key: str

    telegram_bot_token: str
    telegram_chat_id: str

    trading_sessions: str

    min_confirmations: int
    adx_min: float
    cooldown_minutes: int
    poll_seconds: int

    ema_fast: int
    ema_slow: int
    ema_slope_bars: int

    rsi_period: int
    stoch_k: int
    stoch_d: int
    stoch_smooth: int
    stoch_oversold: float
    stoch_overbought: float
    bb_period: int
    bb_std: float
    adx_period: int

    news_mode: str
    news_api_provider: str
    news_api_key: str
    news_base_url: str | None
    news_lookahead_min: int
    news_cooldown_after_min: int


def load_config() -> Config:
    symbol = _getenv("SYMBOL", "XAU/USD")
    td_key = _getenv("TWELVEDATA_API_KEY", "")
    tg_token = _getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = _getenv("TELEGRAM_CHAT_ID", "")

    return Config(
        symbol=symbol,
        twelvedata_api_key=td_key,
        telegram_bot_token=tg_token,
        telegram_chat_id=tg_chat,
        trading_sessions=_getenv("TRADING_SESSIONS", "00:00-03:00,07:00-11:00,12:00-20:00"),
        min_confirmations=_getint("MIN_CONFIRMATIONS", 2),
        adx_min=_getfloat("ADX_MIN", 18.0),
        cooldown_minutes=_getint("COOLDOWN_MINUTES", 20),
        poll_seconds=_getint("POLL_SECONDS", 60),
        ema_fast=_getint("EMA_FAST", 50),
        ema_slow=_getint("EMA_SLOW", 200),
        ema_slope_bars=_getint("EMA_SLOPE_BARS", 10),
        rsi_period=_getint("RSI_PERIOD", 14),
        stoch_k=_getint("STOCH_K", 14),
        stoch_d=_getint("STOCH_D", 3),
        stoch_smooth=_getint("STOCH_SMOOTH", 3),
        stoch_oversold=_getfloat("STOCH_OVERSOLD", 20.0),
        stoch_overbought=_getfloat("STOCH_OVERBOUGHT", 80.0),
        bb_period=_getint("BB_PERIOD", 20),
        bb_std=_getfloat("BB_STD", 2.0),
        adx_period=_getint("ADX_PERIOD", 14),
        news_mode=(_getenv("NEWS_MODE", "WARN") or "WARN").upper(),
        news_api_provider=_getenv("NEWS_API_PROVIDER", "fmp"),
        news_api_key=_getenv("NEWS_API_KEY", ""),
        news_base_url=_getenv("NEWS_BASE_URL", None),
        news_lookahead_min=_getint("NEWS_LOOKAHEAD_MIN", 60),
        news_cooldown_after_min=_getint("NEWS_COOLDOWN_AFTER_MIN", 30),
    )
