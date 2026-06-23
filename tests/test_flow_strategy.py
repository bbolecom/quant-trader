"""flow_strategy 单元测试。"""

from __future__ import annotations

import pandas as pd

from quant.flow_strategy import (
    FlowStrategyParams,
    evaluate_actionable_signal,
    run_portfolio_backtest,
    select_picks_for_date,
)


def test_long_signal_u_s1():
    params = FlowStrategyParams()
    r = {
        "dvol_m": 100,
        "现价": 50,
        "vol_ratio": 1.45,
        "涨幅%": 3.5,
        "涨幅5d%": 8.0,
        "close_strength": 0.7,
        "above_ma50": True,
        "前日涨幅%": 1.0,
    }
    sig = evaluate_actionable_signal(r, params, spy_bull=True)
    assert sig["signal"] == "long"
    assert "U_S1" in sig["规律"]


def test_short_signal_d_s2():
    params = FlowStrategyParams()
    r = {
        "dvol_m": 80,
        "现价": 7,
        "vol_ratio": 2.0,
        "涨幅%": -5.0,
        "涨幅5d%": 30.0,
        "涨幅20d%": 50.0,
        "close_strength": 0.3,
        "above_ma50": True,
        "前日涨幅%": 55.0,
    }
    sig = evaluate_actionable_signal(r, params, spy_bull=True)
    assert sig["signal"] == "short"
    assert "D_S2" in sig["规律"] or "D_OFFERING" in sig["规律"]


def test_flat_d_b2():
    params = FlowStrategyParams()
    r = {
        "dvol_m": 50,
        "现价": 10,
        "vol_ratio": 5.0,
        "涨幅%": 15.0,
        "振幅%": 30.0,
        "涨幅5d%": 5.0,
        "close_strength": 0.5,
        "above_ma50": True,
        "前日涨幅%": 2.0,
    }
    sig = evaluate_actionable_signal(r, params, spy_bull=True)
    assert sig["signal"] == "flat"


def test_portfolio_backtest_synthetic():
    params = FlowStrategyParams(long_top_n=1, short_top_n=1)
    panel = pd.DataFrame([
        {"日期": pd.Timestamp("2024-01-02"), "代码": "AAA", "signal": "long", "fwd_1d": 0.02, "score": 5, "规律": "U_A2", "涨幅%": 3, "量比": 1.4},
        {"日期": pd.Timestamp("2024-01-02"), "代码": "BBB", "signal": "short", "fwd_1d": 0.03, "score": 4, "规律": "D_S2", "涨幅%": -2, "量比": 2},
        {"日期": pd.Timestamp("2024-01-03"), "代码": "CCC", "signal": "long", "fwd_1d": -0.01, "score": 5, "规律": "U_A2", "涨幅%": 4, "量比": 1.5},
    ])
    res = run_portfolio_backtest(panel, params, initial_capital=10000)
    assert res["final_equity"] != 10000
    assert res["总笔数"] == 3


def test_u_s2_gain_filter():
    params = FlowStrategyParams(
        long_patterns=frozenset({"U_S2"}),
        long_s2_min_gain_pct=7.0,
        long_s2_max_gain_pct=15.0,
        long_min_close_strength=0.65,
        long_min_vol_ratio=1.5,
    )
    low = {
        "dvol_m": 100, "现价": 50, "vol_ratio": 1.6, "涨幅%": 4.0,
        "涨幅5d%": 10.0, "close_strength": 0.7, "above_ma50": True, "前日涨幅%": 1.0,
    }
    high = dict(low)
    high["涨幅%"] = 9.0
    assert evaluate_actionable_signal(low, params, spy_bull=True)["signal"] == "flat"
    assert evaluate_actionable_signal(high, params, spy_bull=True)["signal"] == "long"


    params = FlowStrategyParams(long_top_n=1, short_top_n=1)
    panel = pd.DataFrame([
        {"日期": pd.Timestamp("2024-01-02"), "代码": "A", "signal": "long", "score": 5, "fwd_1d": 0.01},
        {"日期": pd.Timestamp("2024-01-02"), "代码": "B", "signal": "long", "score": 3, "fwd_1d": 0.02},
        {"日期": pd.Timestamp("2024-01-02"), "代码": "C", "signal": "short", "score": 4, "fwd_1d": -0.01},
    ])
    picks = select_picks_for_date(panel, pd.Timestamp("2024-01-02"), params)
    assert len(picks) == 2
    assert set(picks["代码"]) == {"A", "C"}
