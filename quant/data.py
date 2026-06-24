"""行情数据获取模块。

支持多数据源（可在 secrets.toml / 环境变量中切换）：
    - polygon  — Polygon.io，交易所级专业数据（推荐）
    - alpaca   — Alpaca Markets，券商级行情
    - yahoo    — Yahoo Finance，免费备用

本地磁盘缓存（research/.cache/market）减少 App 重复拉取与断连。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .market_cache import read_cached, write_cached
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
    from .market_cache import cache_stats
    stats = cache_stats()
    return {
        "provider": cfg.provider,
        "label": provider_label(cfg),
        "cache_entries": str(stats.get("entries", 0)),
    }


def fetch_history(
    ticker: str,
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str = "1d",
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """拉取单只美股的历史行情（自动使用当前配置的数据源）。"""
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataError("股票代码不能为空。")

    if end is None:
        end = date.today()
    if start is None:
        start = (pd.Timestamp(end) - pd.DateOffset(years=3)).date()

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    cfg = load_data_config()
    provider_name = cfg.provider

    if use_cache:
        hit = read_cached(provider_name, ticker, start_s, end_s, interval)
        if hit is not None:
            return hit

    provider = get_provider(cfg)
    df = provider.fetch_history(ticker, start_s, end_s, interval=interval)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"行情数据缺少必要列: {missing}")
    if df.empty:
        raise DataError(f"未获取到 {ticker} 的行情数据，请检查代码或日期范围。")

    if use_cache:
        write_cached(provider_name, ticker, start_s, end_s, df, interval)
    return df


def fetch_history_batch(
    tickers: list[str],
    start: date | str,
    end: date | str,
    interval: str = "1d",
    *,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """批量拉取多标的行情，返回 {代码: DataFrame}。优先读缓存，只拉缺失部分。"""
    syms = sorted(dict.fromkeys(t.strip().upper() for t in tickers if t and str(t).strip()))
    if not syms:
        return {}

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    cfg = load_data_config()
    provider_name = cfg.provider

    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    if use_cache:
        for t in syms:
            hit = read_cached(provider_name, t, start_s, end_s, interval)
            if hit is not None:
                out[t] = hit
            else:
                missing.append(t)
    else:
        missing = syms

    if missing:
        provider = get_provider(cfg)
        fetched = provider.fetch_batch(missing, start_s, end_s, interval=interval)
        for t, df in fetched.items():
            if df is None or df.empty:
                continue
            out[t.upper()] = df
            if use_cache:
                write_cached(provider_name, t.upper(), start_s, end_s, df, interval)

    return out


# 兼容旧代码：yfinance 规范化函数仍可从 data 模块引用。
def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    from .providers.base import normalize_ohlcv
    return normalize_ohlcv(raw)
