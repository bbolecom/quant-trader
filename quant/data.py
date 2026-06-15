"""行情数据获取模块。

使用 yfinance 拉取美股历史日线数据，并对结果做规范化处理。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - 友好提示
    yf = None


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataError(Exception):
    """数据获取或解析过程中出现的错误。"""


def _ensure_yfinance() -> None:
    if yf is None:
        raise DataError(
            "未安装 yfinance，请先运行: pip install yfinance"
        )


def fetch_history(
    ticker: str,
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """拉取单只美股的历史行情。

    参数
    ----
    ticker: 股票代码，如 "AAPL"。
    start / end: 起止日期，默认最近 3 年。
    interval: K 线周期，默认日线 "1d"。

    返回
    ----
    含 Open/High/Low/Close/Volume 列、以日期为索引的 DataFrame。
    """
    _ensure_yfinance()

    ticker = ticker.strip().upper()
    if not ticker:
        raise DataError("股票代码不能为空。")

    if end is None:
        end = date.today()
    if start is None:
        start = (pd.Timestamp(end) - pd.DateOffset(years=3)).date()

    # yfinance 对 end 参数要求纯日期字符串；带 "00:00:00" 会解析失败。
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
        raise DataError(f"未获取到 {ticker} 的行情数据，请检查代码或日期范围。")

    df = _normalize(raw)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"行情数据缺少必要列: {missing}")

    return df


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """规范化 yfinance 返回的数据结构。

    yfinance 在不同版本/单/多标的下可能返回 MultiIndex 列，这里统一压平。
    """
    df = raw.copy()

    # 压平可能存在的多级列索引。
    if isinstance(df.columns, pd.MultiIndex):
        # 优先保留字段名（Open/Close...），丢弃标的层级。
        level0 = df.columns.get_level_values(0)
        if set(REQUIRED_COLUMNS).issubset(set(level0)):
            df.columns = level0
        else:
            df.columns = df.columns.get_level_values(-1)

    df = df.rename(columns=lambda c: str(c).strip().title())

    # 索引转为不带时区的日期。
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    keep = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[keep].dropna(how="all").dropna(subset=["Close"])

    return df
