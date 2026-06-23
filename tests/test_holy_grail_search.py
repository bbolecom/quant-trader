"""holy_grail_search 单元测试（无网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.holy_grail_search import (
    apply_leverage,
    apply_stop_overlay,
    apply_vol_target,
    classify_tier,
    gen_portfolio_mixes,
    metrics_from_returns,
    mix_returns,
    returns_to_equity,
    theoretical_bounds,
    walk_forward_pass,
)
from research.triple_target_scan import set_scan_targets


def test_theoretical_bounds_structure():
    set_scan_targets(preset="strict")
    b = theoretical_bounds(win_rate=0.80, max_dd=-0.10, target_ann=1.0)
    assert "ann_upper_heuristic" in b
    assert b["payoff_ratio_needed"] > 1.0
    assert "conclusion" in b


def test_leverage_doubles_vol():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, len(idx)), index=idx)
    lev = apply_leverage(r, 2.0)
    assert abs(lev.std() - 2 * r.std()) < 1e-9


def test_stop_halts_after_drawdown():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    r = pd.Series(0.0, index=idx)
    r.iloc[10:15] = -0.05
    out = apply_stop_overlay(r, -0.10)
    assert (out.iloc[20:] == 0).all()


def test_mix_returns_weights():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    a = pd.Series(0.01, index=idx)
    b = pd.Series(0.02, index=idx)
    m = mix_returns([(a, 0.5), (b, 0.5)])
    assert abs(m.iloc[-1] - 0.015) < 1e-9


def test_metrics_from_returns():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    r = pd.Series(0.002, index=idx)
    m = metrics_from_returns(r)
    assert m["ann_return"] > 0.5
    assert m["win_rate"] == 1.0


def test_walk_forward_fail_on_bad_series():
    idx = pd.date_range("2019-01-01", periods=1500, freq="B")
    r = pd.Series(-0.001, index=idx)
    assert walk_forward_pass(r, 0.3) is False


def test_gen_portfolio_mixes_empty():
    assert gen_portfolio_mixes([], 10) == []


def test_classify_tier_strict():
    set_scan_targets(preset="strict")
    assert classify_tier(1.1, -0.05, 0.85) == "A"
    assert classify_tier(0.5, -0.05, 0.85) == "B"
