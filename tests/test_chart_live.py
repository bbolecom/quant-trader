"""Tests for live chart fetch."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from quant.chart_live import chart_period_args, fetch_live_chart


def test_chart_period_args() -> None:
    assert chart_period_args("daily") == ("6mo", "1d")
    assert chart_period_args("weekly") == ("2y", "1wk")
    # 未知 period 安全回退日 K
    assert chart_period_args("garbage") == ("6mo", "1d")


def test_fetch_live_chart_rejects_blank() -> None:
    assert fetch_live_chart("") is None
    assert fetch_live_chart("—") is None


def test_fetch_live_chart_mocked(monkeypatch) -> None:
    """用假的 yfinance 模块验证解析逻辑，不联网。"""
    idx = pd.to_datetime(["2026-06-20", "2026-06-23", "2026-06-24"])
    hist = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [10.5, 11.5, 12.5],
            "Low": [9.5, 10.5, 11.5],
            "Close": [10.2, 11.2, 12.2],
            "Volume": [1000, 2000, 3000],
        },
        index=idx,
    )

    class _FakeTicker:
        def __init__(self, sym: str) -> None:
            self.sym = sym

        def history(self, **_kwargs):
            return hist

    fake_yf = types.SimpleNamespace(Ticker=_FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    doc = fetch_live_chart("aapl", period="6mo", interval="1d")
    assert doc is not None
    assert doc["ticker"] == "AAPL"
    assert doc["source"] == "yfinance-live"
    assert len(doc["bars"]) == 3
    last = doc["bars"][-1]
    assert last["close"] == 12.2
    assert last["date"] == "2026-06-24"


@pytest.mark.network
def test_fetch_live_chart_spy() -> None:
    doc = fetch_live_chart("SPY", period="1mo", interval="1d")
    assert doc is not None
    assert doc["ticker"] == "SPY"
    assert len(doc["bars"]) >= 5
    bar = doc["bars"][-1]
    assert "close" in bar and bar["close"] > 0
