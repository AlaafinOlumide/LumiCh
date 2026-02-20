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
    """
    TwelveData accepts XAU/USD (and often also XAUUSD depending on endpoint).
    We normalize to XAU/USD by default to avoid 'symbol missing/invalid' errors.
    """
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
    trading_sessions: str  # e.g. "00:00-03:00,07:00-11:00,12:00-20:00"

    # TwelveData
    twelvedata_api_key: str

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

    # News filter
    news_mode: str  # "BLOCK" or "WARN"
    news_api_provider: str  # e.g. "fmp"
    news_api_key: str
    news_base_url: str
    news_lookahead_min: int
    news_cooldown_after_min: int

    # No-trade alerts
    send_no_trade_alerts: bool
    no_trade_alert_cooldown_min: int

    # ✅ NEW: price calibration (fixes discrepancy with MT5 broker feed)
    broker_price_offset: float

    # ✅ NEW: weekend blocking
    block_weekends: bool
    weekend_timezone: str  # informational (we use UTC in code)

    @staticmethod
    def from_env() -> "Config":
        symbol = _normalize_symbol_for_twelvedata(_env("SYMBOL", "XAU/USD"))

        return Config(
            symbol=symbol,
            poll_seconds=_env_int("POLL_SECONDS", 60),
            trading_sessions=_env("TRADING_SESSIONS", "00:00-03:00,07:00-11:00,12:00-20:00") or "",

            twelvedata_api_key=_env("TWELVEDATA_API_KEY", "") or "",

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

            news_mode=(_env("NEWS_MODE", "WARN") or "WARN").upper(),
            news_api_provider=(_env("NEWS_API_PROVIDER", "fmp") or "fmp").lower(),
            news_api_key=_env("NEWS_API_KEY", "") or "",
            news_base_url=_env("NEWS_BASE_URL", "https://financialmodelingprep.com/api/v3") or "",
            news_lookahead_min=_env_int("NEWS_LOOKAHEAD_MIN", 60),
            news_cooldown_after_min=_env_int("NEWS_COOLDOWN_AFTER_MIN", 30),

            send_no_trade_alerts=_env_bool("SEND_NO_TRADE_ALERTS", True),
            no_trade_alert_cooldown_min=_env_int("NO_TRADE_ALERT_COOLDOWN_MIN", 20),

            # ✅ NEW ENV VARS
            broker_price_offset=_env_float("BROKER_PRICE_OFFSET", 0.0),
            block_weekends=_env_bool("BLOCK_WEEKENDS", True),
            weekend_timezone=_env("WEEKEND_TIMEZONE", "UTC") or "UTC",
        )