"""Tests for live chart fetch."""

from __future__ import annotations

from quant.chart_live import chart_period_args, fetch_live_chart


def test_chart_period_args() -> None:
    assert chart_period_args("daily") == ("6mo", "1d")
    assert chart_period_args("weekly") == ("2y", "1wk")


def test_fetch_live_chart_spy() -> None:
    doc = fetch_live_chart("SPY", period="1mo", interval="1d")
    assert doc is not None
    assert doc["ticker"] == "SPY"
    assert len(doc["bars"]) >= 5
    bar = doc["bars"][-1]
    assert "close" in bar and bar["close"] > 0
