"""Tests for quant.ui_helpers (从 app.py 抽离的纯展示助手)。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from quant import backtest
from quant.ui_helpers import (
    compare_chart,
    equity_chart,
    fmt_dollar_m,
    fmt_mcap,
    fmt_num,
    fmt_pct,
    parse_tickers,
    price_chart,
)


def test_fmt_pct_and_num() -> None:
    assert fmt_pct(0.1234) == "12.34%"
    assert fmt_num(1234.5) == "1,234.50"


def test_fmt_mcap() -> None:
    assert fmt_mcap(2.5e9) == "2.5B"
    assert fmt_mcap(5e8) == "500M"
    assert fmt_mcap(float("nan")) == "-"


def test_fmt_dollar_m() -> None:
    assert fmt_dollar_m(1.2e8) == "$120.0M"
    assert fmt_dollar_m(float("nan")) == "-"


def test_parse_tickers_dedupes_and_normalizes() -> None:
    assert parse_tickers("aapl, nvda\nmsft aapl") == ["AAPL", "NVDA", "MSFT"]
    assert parse_tickers("  ") == []


def test_price_chart_returns_figure(ohlcv: pd.DataFrame) -> None:
    fig = price_chart(ohlcv, "双均线交叉", {"fast": 20, "slow": 60})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # K线 + 成交量(+均线)


def test_equity_chart_returns_figure(ohlcv: pd.DataFrame) -> None:
    pos = pd.Series(1.0, index=ohlcv.index)
    res = backtest.run_backtest(ohlcv, pos)
    fig = equity_chart(res, "测试策略")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # 策略净值 + 基准 + 回撤


def test_compare_chart_returns_figure(ohlcv: pd.DataFrame) -> None:
    eq = (1 + ohlcv["Close"].pct_change().fillna(0)).cumprod()
    fig = compare_chart({"A": eq, "B": eq * 1.1})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
