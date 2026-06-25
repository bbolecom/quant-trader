"""Yahoo Finance 数据源（免费备用）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

from .base import MarketDataProvider, normalize_ohlcv, DataError

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# 单次 batch 不宜过大，避免 Yahoo 超时/断连
_BATCH_CHUNK = 40
_DOWNLOAD_TIMEOUT = 45
_BATCH_WORKERS = 8


def _download(tickers, start_str: str, end_str: str, interval: str = "1d"):
    if yf is None:
        raise DataError("未安装 yfinance，请运行: pip install yfinance")
    multi = isinstance(tickers, (list, tuple)) and len(tickers) > 1
    kwargs = dict(
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if multi:
        kwargs["group_by"] = "ticker"
    try:
        return yf.download(tickers, timeout=_DOWNLOAD_TIMEOUT, **kwargs)
    except TypeError:
        return yf.download(tickers, **{k: v for k, v in kwargs.items() if k != "timeout"})


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
        ticker = ticker.strip().upper()
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")

        raw = _download(ticker, start_str, end_str, interval=interval)
        if raw is None or raw.empty:
            raise DataError(f"Yahoo 未返回 {ticker} 的行情数据。")
        return normalize_ohlcv(raw)

    def fetch_batch(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
        interval: str = "1d",
        *,
        max_workers: int = _BATCH_WORKERS,
    ) -> dict[str, pd.DataFrame]:
        if yf is None or not tickers:
            return {}
        start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
        end_str = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
        syms = [t.strip().upper() for t in tickers if t and str(t).strip()]
        if not syms:
            return {}
        chunks = [syms[i : i + _BATCH_CHUNK] for i in range(0, len(syms), _BATCH_CHUNK)]
        out: dict[str, pd.DataFrame] = {}

        def _fetch_chunk(chunk: list[str]) -> dict[str, pd.DataFrame]:
            partial: dict[str, pd.DataFrame] = {}
            try:
                raw = _download(
                    chunk if len(chunk) > 1 else chunk[0],
                    start_str,
                    end_str,
                    interval=interval,
                )
            except Exception:  # noqa: BLE001
                return partial
            if raw is None or raw.empty:
                return partial
            multi = isinstance(raw.columns, pd.MultiIndex)
            for t in chunk:
                try:
                    sub = raw[t].dropna(how="all") if multi else raw.dropna(how="all")
                    if sub is None or sub.empty:
                        continue
                    df = normalize_ohlcv(sub)
                    if not df.empty:
                        partial[t] = df
                except Exception:  # noqa: BLE001
                    continue
            return partial

        workers = min(max(1, int(max_workers)), len(chunks))
        if workers <= 1:
            for chunk in chunks:
                out.update(_fetch_chunk(chunk))
            return out
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_chunk, chunk) for chunk in chunks]
            for fut in as_completed(futures):
                try:
                    out.update(fut.result())
                except Exception:  # noqa: BLE001
                    continue
        return out
