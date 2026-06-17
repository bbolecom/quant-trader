"""Alpaca Markets 专业行情数据源（券商级 IEX/SIP 数据）。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

from .base import MarketDataProvider, normalize_ohlcv, DataError


class AlpacaProvider(MarketDataProvider):
    name = "alpaca"
    label = "Alpaca Markets"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        data_url: str = "https://data.alpaca.markets",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.data_url = data_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key or not self.api_secret:
            raise DataError(
                "Alpaca 需要 API Key 与 Secret，请在 secrets.toml 或环境变量中配置。"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _fetch_bars(
        self,
        symbols: list[str],
        start: date | str,
        end: date | str,
    ) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
        # Alpaca end 为开区间，+1 天以包含 end 当日。
        end_s = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"{self.data_url}/v2/stocks/bars"
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start_s,
            "end": end_s,
            "adjustment": "all",
            "limit": 10000,
        }
        resp = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
        if resp.status_code == 401:
            raise DataError("Alpaca API 凭证无效。")
        if resp.status_code == 429:
            raise DataError("Alpaca 请求频率超限，请稍后重试。")
        if resp.status_code >= 400:
            raise DataError(f"Alpaca 请求失败 ({resp.status_code})：{resp.text[:200]}")

        payload = resp.json()
        bars = payload.get("bars") or {}
        out: dict[str, pd.DataFrame] = {}
        for sym, items in bars.items():
            if not items:
                continue
            rows = [{
                "Date": pd.to_datetime(b["t"]),
                "Open": b.get("o"),
                "High": b.get("h"),
                "Low": b.get("l"),
                "Close": b.get("c"),
                "Volume": b.get("v"),
            } for b in items]
            df = pd.DataFrame(rows).set_index("Date")
            norm = normalize_ohlcv(df)
            if not norm.empty:
                out[sym.upper()] = norm
        return out

    def fetch_history(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise DataError("Alpaca 当前仅支持日线 interval=1d。")
        ticker = ticker.strip().upper()
        batch = self._fetch_bars([ticker], start, end)
        if ticker not in batch:
            raise DataError(f"Alpaca 未返回 {ticker} 的行情数据。")
        return batch[ticker]

    def fetch_batch(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        if interval != "1d":
            raise DataError("Alpaca 当前仅支持日线 interval=1d。")
        syms = [t.strip().upper() for t in tickers if t.strip()]
        out: dict[str, pd.DataFrame] = {}
        # Alpaca 单次最多约 100 只，分批请求。
        chunk = 50
        for i in range(0, len(syms), chunk):
            part = self._fetch_bars(syms[i:i + chunk], start, end)
            out.update(part)
        return out
