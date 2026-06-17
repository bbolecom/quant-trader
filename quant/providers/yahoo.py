"""Yahoo Finance 数据源（免费备用）。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .base import MarketDataProvider, normalize_ohlcv, DataError

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


class YahooProvider(MarketDataProvider):
    name = "yahoo"
    label = "Yahoo Finance"

    def fetch_history(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if yf is None:
            raise DataError("未安装 yfinance，请运行: pip install yfinance")

        ticker = ticker.strip().upper()
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")

        raw = yf.download(
            ticker,
            start=start_str,
            end=end_str,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            raise DataError(f"Yahoo 未返回 {ticker} 的行情数据。")
        return normalize_ohlcv(raw)

    def fetch_batch(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        if yf is None or not tickers:
            return {}
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(
            tickers,
            start=start_str,
            end=end_str,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw is None or raw.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t in tickers:
            try:
                sub = raw[t].dropna(how="all") if multi else raw.dropna(how="all")
                if sub is None or sub.empty:
                    continue
                df = normalize_ohlcv(sub)
                if not df.empty:
                    out[t] = df
            except Exception:  # noqa: BLE001
                continue
        return out
