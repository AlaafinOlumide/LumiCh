from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger(__name__)


@dataclass
class CandleData:
    df: pd.DataFrame  # indexed by datetime (UTC)
    interval: str


class TwelveDataClient:
    def __init__(self, api_key: str, base_url: str = 'https://api.twelvedata.com'):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def fetch_time_series(self, symbol: str, interval: str, outputsize: int = 200) -> CandleData:
        url = f'{self.base_url}/time_series'
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': int(outputsize),
            'format': 'JSON',
            'apikey': self.api_key,
            'order': 'ASC',
        }

        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and data.get('status') == 'error':
            raise RuntimeError(f"TwelveData error: {data.get('message')}")

        values = data.get('values') if isinstance(data, dict) else None
        if not values:
            raise RuntimeError(f'No candle values returned (interval={interval})')

        df = pd.DataFrame(values)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        return CandleData(df=df, interval=interval)
