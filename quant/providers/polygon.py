"""Polygon.io 专业行情数据源（美股交易所聚合数据）。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from .base import MarketDataProvider, normalize_ohlcv, DataError


class PolygonProvider(MarketDataProvider):
    name = "polygon"
    label = "Polygon.io"

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout
        if not self.api_key:
            raise DataError("Polygon 需要 API Key，请在 secrets.toml 或环境变量 POLYGON_API_KEY 中配置。")

    def fetch_history(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise DataError("Polygon 当前仅支持日线 interval=1d。")

        ticker = ticker.strip().upper()
        start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start_s}/{end_s}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 401:
            raise DataError("Polygon API Key 无效或未授权。")
        if resp.status_code == 429:
            raise DataError("Polygon 请求频率超限，请稍后重试或升级套餐。")
        if resp.status_code >= 400:
            raise DataError(f"Polygon 请求失败 ({resp.status_code})：{resp.text[:200]}")

        payload = resp.json()
        results = payload.get("results") or []
        if not results:
            raise DataError(f"Polygon 未返回 {ticker} 在 {start_s}~{end_s} 的数据。")

        rows = []
        for bar in results:
            rows.append({
                "Date": pd.to_datetime(bar["t"], unit="ms"),
                "Open": bar.get("o"),
                "High": bar.get("h"),
                "Low": bar.get("l"),
                "Close": bar.get("c"),
                "Volume": bar.get("v"),
            })
        df = pd.DataFrame(rows).set_index("Date")
        out = normalize_ohlcv(df)
        # 日线对齐到当天零点，避免与纯日期索引比较时错位。
        out.index = out.index.normalize()
        return out
