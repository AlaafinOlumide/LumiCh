from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def _normalize_symbol_for_twelvedata(sym: str | None) -> str:
    if not sym:
        return "XAU/USD"
    s = sym.strip().upper()
    if s in ("XAUUSD", "XAU-USD", "XAU USD"):
        return "XAU/USD"
    return s


@dataclass(frozen=True)
class Config:
    # Core
    symbol: str
    poll_seconds: int

    # Trading windows
    trading_sessions: str

    # TwelveData
    twelvedata_api_key: str
    http_timeout: int
    http_max_retries: int
    http_backoff_seconds: float

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # Strategy params
    ema_fast: int
    ema_slow: int
    ema_slope_bars: int
    rsi_period: int
    adx_period: int
    adx_min: float
    min_confirmations: int
    cooldown_minutes: int

    # ATR-based risk model
    atr_period: int
    atr_sl_mult: float
    atr_tp_mult: float

    # Setup + trigger model
    setup_ttl_minutes: int
    entry_zone_atr_mult: float          # zone_half = ATR15 * this
    entry_zone_min_width: float         # minimum $ width for zone_half
    trigger_tf: str                     # "1min"
    trigger_ema_period: int             # EMA on trigger TF
    trigger_rsi_min_buy: float          # optional
    trigger_rsi_max_sell: float         # optional

    # Peak-chase guards (M5)
    ext_atr_mult: float
    rsi_buy_max: float
    rsi_sell_min: float
    bb_band_buffer_atr: float
    pullback_lookback: int

    # News filter
    news_mode: str
    news_api_provider: str
    news_api_key: str
    news_base_url: str
    news_lookahead_min: int
    news_cooldown_after_min: int

    # Price calibration (optional)
    broker_price_offset: float

    # Weekend blocking
    block_weekends: bool

    @staticmethod
    def from_env() -> "Config":
        symbol = _normalize_symbol_for_twelvedata(_env("SYMBOL", "XAU/USD"))

        return Config(
            symbol=symbol,
            poll_seconds=_env_int("POLL_SECONDS", 60),
            trading_sessions=_env("TRADING_SESSIONS", "00:00-03:00,07:00-11:00,12:00-20:00") or "",

            twelvedata_api_key=_env("TWELVEDATA_API_KEY", "") or "",
            http_timeout=_env_int("HTTP_TIMEOUT", 20),
            http_max_retries=_env_int("HTTP_MAX_RETRIES", 2),
            http_backoff_seconds=_env_float("HTTP_BACKOFF_SECONDS", 1.0),

            telegram_bot_token=_env("TELEGRAM_BOT_TOKEN", "") or "",
            telegram_chat_id=_env("TELEGRAM_CHAT_ID", "") or "",

            ema_fast=_env_int("EMA_FAST", 20),
            ema_slow=_env_int("EMA_SLOW", 50),
            ema_slope_bars=_env_int("EMA_SLOPE_BARS", 10),
            rsi_period=_env_int("RSI_PERIOD", 14),
            adx_period=_env_int("ADX_PERIOD", 14),
            adx_min=_env_float("ADX_MIN", 20.0),
            min_confirmations=_env_int("MIN_CONFIRMATIONS", 5),
            cooldown_minutes=_env_int("COOLDOWN_MINUTES", 30),

            atr_period=_env_int("ATR_PERIOD", 14),
            atr_sl_mult=_env_float("ATR_SL_MULT", 1.5),
            atr_tp_mult=_env_float("ATR_TP_MULT", 3.0),

            setup_ttl_minutes=_env_int("SETUP_TTL_MINUTES", 60),
            entry_zone_atr_mult=_env_float("ENTRY_ZONE_ATR_MULT", 0.25),
            entry_zone_min_width=_env_float("ENTRY_ZONE_MIN_WIDTH", 2.0),
            trigger_tf=_env("TRIGGER_TF", "1min") or "1min",
            trigger_ema_period=_env_int("TRIGGER_EMA_PERIOD", 20),
            trigger_rsi_min_buy=_env_float("TRIGGER_RSI_MIN_BUY", 45.0),
            trigger_rsi_max_sell=_env_float("TRIGGER_RSI_MAX_SELL", 55.0),

            ext_atr_mult=_env_float("EXT_ATR_MULT", 0.9),
            rsi_buy_max=_env_float("RSI_BUY_MAX", 68.0),
            rsi_sell_min=_env_float("RSI_SELL_MIN", 32.0),
            bb_band_buffer_atr=_env_float("BB_BAND_BUFFER_ATR", 0.15),
            pullback_lookback=_env_int("PULLBACK_LOOKBACK", 6),

            news_mode=(_env("NEWS_MODE", "WARN") or "WARN").upper(),
            news_api_provider=(_env("NEWS_API_PROVIDER", "fmp") or "fmp").lower(),
            news_api_key=_env("NEWS_API_KEY", "") or "",
            news_base_url=_env("NEWS_BASE_URL", "https://financialmodelingprep.com/api/v3") or "",
            news_lookahead_min=_env_int("NEWS_LOOKAHEAD_MIN", 60),
            news_cooldown_after_min=_env_int("NEWS_COOLDOWN_AFTER_MIN", 30),

            broker_price_offset=_env_float("BROKER_PRICE_OFFSET", 0.0),
            block_weekends=_env_bool("BLOCK_WEEKENDS", True),
        )