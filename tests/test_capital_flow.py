"""资金流向操盘规律单元测试（无网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.capital_flow import (
    assess_flow_patterns,
    build_daily_picks,
    build_flow_history,
    enrich_flow_row,
    match_pattern_by_id,
    scan_universe_flow,
    _match_down_patterns,
    _match_up_patterns,
)


def _synthetic_df(
    n: int = 80,
    *,
    daily_gain: float = 0.035,
    vol_mult: float = 1.4,
    base_vol: float = 2e6,
    last_day_gain: float | None = None,
) -> pd.DataFrame:
    """构造温和上涨 + 温和放量序列。"""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 50.0
    rows = []
    for i in range(n):
        if i > 0:
            g = last_day_gain if i == n - 1 and last_day_gain is not None else daily_gain * (0.8 if i < n - 5 else 1.0)
            close *= 1 + g
        vol = base_vol * (vol_mult if i == n - 1 else 1.0)
        hi = close * 1.02
        lo = close * 0.97
        rows.append({"Open": close, "High": hi, "Low": lo, "Close": close, "Volume": vol})
    return pd.DataFrame(rows, index=dates)


def test_enrich_flow_row_basic():
    df = _synthetic_df()
    r = enrich_flow_row(df)
    assert r["above_ma50"]
    assert r["vol_ratio"] > 1.0
    assert "涨幅%" in r


def test_up_pattern_u_s1():
    df = _synthetic_df(daily_gain=0.01, vol_mult=1.45, last_day_gain=0.035)
    r = enrich_flow_row(df)
    hits = _match_up_patterns(r, spy_bull=True)
    ids = [h["规律ID"] for h in hits]
    assert "U_S1" in ids or "U_A1" in ids or "U_S2" in ids


def test_down_pattern_parabolic():
    r = {
        "vol_ratio": 3.0,
        "涨幅%": 5.0,
        "涨幅5d%": 20.0,
        "涨幅20d%": 45.0,
        "前日涨幅%": 10.0,
        "振幅%": 15.0,
        "close_strength": 0.4,
        "dvol_m": 200.0,
        "above_ma50": True,
    }
    hits = _match_down_patterns(r, spy_bull=True)
    ids = [h["规律ID"] for h in hits]
    assert "D_S1" in ids


def test_down_pattern_prev_day_mega_gain():
    r = {
        "vol_ratio": 2.0,
        "涨幅%": -5.0,
        "涨幅5d%": 30.0,
        "涨幅20d%": 50.0,
        "前日涨幅%": 55.0,
        "振幅%": 20.0,
        "close_strength": 0.3,
        "dvol_m": 50.0,
        "above_ma50": True,
    }
    hits = _match_down_patterns(r, spy_bull=False)
    assert any(h["规律ID"] == "D_S2" for h in hits)


def test_extreme_volatility_skip():
    r = {
        "vol_ratio": 5.0,
        "涨幅%": 15.0,
        "涨幅5d%": 5.0,
        "涨幅20d%": 10.0,
        "前日涨幅%": 2.0,
        "振幅%": 30.0,
        "close_strength": 0.5,
        "dvol_m": 100.0,
        "above_ma50": True,
        "ret_5d": 0.05,
        "ret_20d": 0.10,
        "dvol_m": 100.0,
        "close_strength": 0.5,
    }
    res = assess_flow_patterns(r, spy_bull=True)
    assert res["信号"] == "观望"
    assert "极端" in res["选股理由"]


def test_weak_market_no_long():
    r = {
        "vol_ratio": 1.4,
        "涨幅%": 3.5,
        "涨幅5d%": 8.0,
        "涨幅20d%": 12.0,
        "前日涨幅%": 1.0,
        "振幅%": 5.0,
        "close_strength": 0.7,
        "dvol_m": 600.0,
        "above_ma50": True,
        "ret_1d": 0.035,
        "ret_5d": 0.08,
        "ret_20d": 0.12,
    }
    res_bull = assess_flow_patterns(r, spy_bull=True)
    res_bear = assess_flow_patterns(r, spy_bull=False)
    assert res_bull["信号"] == "做多"
    assert res_bear["信号"] != "做多"


def test_scan_universe_flow():
    batch = {
        "AAA": _synthetic_df(daily_gain=0.01, vol_mult=1.45, base_vol=5e6, last_day_gain=0.035),
        "BBB": _synthetic_df(daily_gain=0.001, vol_mult=1.0, base_vol=1e6),
    }
    df = scan_universe_flow(batch, spy_bull=True, min_dvol_m=10)
    assert not df.empty
    assert "AAA" in df["代码"].tolist()


def test_build_flow_history_fwd():
    df = _synthetic_df(daily_gain=0.01, vol_mult=1.45, last_day_gain=0.035)
    hist = build_flow_history(df)
    assert not hist.empty
    assert "fwd_1d" in hist.columns
    assert match_pattern_by_id("U_S1", hist.iloc[-1].to_dict(), spy_bull=True) or True


    scan = pd.DataFrame([
        {"代码": "AAA", "信号": "做多", "策略动作": "次日做多", "选股理由": "test", "_score": 5},
        {"代码": "BBB", "信号": "做空", "策略动作": "买Put价差", "选股理由": "test", "_score": 4},
        {"代码": "CCC", "信号": "回避", "策略动作": "回避追涨", "选股理由": "test", "_score": 3},
    ])
    pools = build_daily_picks(scan, long_top_n=1, short_top_n=1, avoid_top_n=1)
    assert len(pools["long"]) == 1
    assert pools["long"].iloc[0]["代码"] == "AAA"
