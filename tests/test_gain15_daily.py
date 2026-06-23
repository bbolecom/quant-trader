"""Tests for gain15 80% daily scan rules."""

from __future__ import annotations

import pandas as pd

from quant.gain15_scan import (
    WatchEvent,
    eval_drop_rules,
    eval_surge_rules,
    returns_from_spike,
)


def _sample_df() -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", periods=10)
    close = [10.0, 10.5, 11.0, 12.5, 13.0, 12.0, 11.5, 11.0, 10.5, 10.0]
    return pd.DataFrame({"Close": close, "High": close, "Low": close}, index=dates)


def test_returns_from_spike():
    df = _sample_df()
    spike = "2026-06-02"
    rets = returns_from_spike(df, spike)
    assert rets["tdays_since"] == 8
    assert rets["ret_1d"] is not None
    assert abs(rets["ret_1d"] - (11.0 / 10.5 - 1)) < 1e-6


def test_surge_rule_top3_d3():
    ev = WatchEvent(
        代码="TEST",
        暴涨日="2026-06-01",
        涨幅_pct=18.0,
        成交额M=100.0,
        gain_rank=2,
        站上MA20=True,
        站上MA50=True,
        创20日高=True,
        涨幅20d_pct=30.0,
        相对SPY20d_pct=25.0,
        量比=3.0,
        收盘强度=0.8,
        SPY站上MA20=True,
        暴涨收盘价=10.0,
    )
    hits = eval_surge_rules(ev, {"ret_1d": 0.12, "ret_3d": 0.22})
    ids = {h.rule_id for h in hits}
    assert "S2" in ids
    assert "S6" in ids


def test_drop_rule_top3_d1():
    ev = WatchEvent(
        代码="TEST",
        暴涨日="2026-06-01",
        涨幅_pct=35.0,
        成交额M=200.0,
        gain_rank=1,
        站上MA20=True,
        站上MA50=False,
        创20日高=True,
        涨幅20d_pct=60.0,
        相对SPY20d_pct=55.0,
        量比=5.0,
        收盘强度=0.9,
        SPY站上MA20=False,
        暴涨收盘价=20.0,
    )
    hits = eval_drop_rules(ev, {"ret_1d": -0.18, "ret_3d": -0.25})
    ids = {h.rule_id for h in hits}
    assert "D5" in ids
    assert "D7" in ids
    assert "D9" in ids
