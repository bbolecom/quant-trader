"""行情数据获取模块。

支持多数据源（可在 secrets.toml / 环境变量中切换）：
    - polygon  — Polygon.io，交易所级专业数据（推荐）
    - alpaca   — Alpaca Markets，券商级行情
    - yahoo    — Yahoo Finance，免费备用
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .providers import get_provider, load_data_config, provider_label, reset_provider_cache
from .providers.base import DataError, REQUIRED_COLUMNS

__all__ = [
    "DataError",
    "REQUIRED_COLUMNS",
    "fetch_history",
    "fetch_history_batch",
    "get_data_source_info",
    "load_data_config",
    "provider_label",
    "reset_provider_cache",
]


def get_data_source_info() -> dict[str, str]:
    """返回当前数据源名称与说明（供 UI 展示）。"""
    cfg = load_data_config()
    return {
        "provider": cfg.provider,
        "label": provider_label(cfg),
    }


def fetch_history(
    ticker: str,
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """拉取单只美股的历史行情（自动使用当前配置的数据源）。"""
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataError("股票代码不能为空。")

    if end is None:
        end = date.today()
    if start is None:
        start = (pd.Timestamp(end) - pd.DateOffset(years=3)).date()

    provider = get_provider()
    df = provider.fetch_history(ticker, start, end, interval=interval)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"行情数据缺少必要列: {missing}")
    if df.empty:
        raise DataError(f"未获取到 {ticker} 的行情数据，请检查代码或日期范围。")
    return df


def fetch_history_batch(
    tickers: list[str],
    start: date | str,
    end: date | str,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """批量拉取多标的行情，返回 {代码: DataFrame}。"""
    syms = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not syms:
        return {}
    provider = get_provider()
    return provider.fetch_batch(syms, start, end, interval=interval)


# 兼容旧代码：yfinance 规范化函数仍可从 data 模块引用。
def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    from .providers.base import normalize_ohlcv
    return normalize_ohlcv(raw)
