"""行情数据源抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataError(Exception):
    """数据获取或解析过程中出现的错误。"""


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """统一 OHLCV 格式。"""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if set(REQUIRED_COLUMNS).issubset(set(level0)):
            df.columns = level0
        else:
            df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns=lambda c: str(c).strip().title())
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    return df[keep].dropna(how="all").dropna(subset=["Close"])


class MarketDataProvider(ABC):
    name: str = "base"
    label: str = "Base"

    @abstractmethod
    def fetch_history(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_batch(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """默认逐只拉取；子类可覆盖为批量接口。"""
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                df = self.fetch_history(t, start, end, interval=interval)
                if df is not None and not df.empty:
                    out[t] = df
            except Exception:  # noqa: BLE001
                continue
        return out
