# XAUUSD Telegram Signal Bot (Render-ready)

A production-style signal bot for **XAUUSD only**, using:
- M5 entry triggers
- M15 confirmation
- H1 trend detection (fallback to M15 when H1 is unavailable)
- Indicators: Bollinger Bands, RSI, Stochastic, Candlestick patterns, ADX
- Telegram signal output **including “Reason for Trade”**
- Optional high-impact news gate/warn via `high_impact_news.py`

## 1) Quick start (local)

1. Create a virtual env and install:

```bash
pip install -r requirements.txt
```

2. Copy env template and fill in keys:

```bash
cp .env.example .env
```

3. Run:

```bash
python -m bot.main
```

## 2) Deploy to Render

### Option A: Using `render.yaml`
- Push this repo to GitHub
- In Render: **New** → **Blueprint** → connect the repo
- Set environment variables in Render (same keys as in `.env.example`)

### Option B: Manual Render worker
- Create **Background Worker**
- Build command: `pip install -r requirements.txt`
- Start command: `python -m bot.main`

## 3) Required environment variables

- `TWELVEDATA_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Optional:
- News filter: `NEWS_API_KEY`, `NEWS_MODE=BLOCK|WARN`, `NEWS_API_PROVIDER=fmp`

## 4) Sessions (GMT)
Configure sessions via:

- `TRADING_SESSIONS=00:00-03:00,07:00-11:00,12:00-20:00`

## 5) Notes
- This bot **sends signals only**; it does NOT place trades.
- If TwelveData is rate-limited, reduce `POLL_SECONDS` frequency or upgrade your plan.
- Default risk tag:
  - `SWING` if H1 is available
  - `SCALP` if using M15 fallback
