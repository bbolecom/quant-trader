#!/usr/bin/env python3
"""longshort_combo_strategy unit tests (no network)."""

from __future__ import annotations

from quant.longshort_combo_strategy import (
    LongShortComboConfig,
    config_from_dict,
    merge_and_select,
    passes_quality_filter,
    quality_score,
)


def test_quality_score_l1_high_on_weak_close():
    cfg = LongShortComboConfig()
    sig = {"策略ID": "L1", "leg": "e20", "收盘强度": 0.15, "成交额M": 120, "量比": 2.0}
    assert quality_score(sig, cfg) > 0.6


def test_s1_filter_rejects_shallow_drop():
    cfg = LongShortComboConfig(s1_pre20_drop_pct=35.0)
    sig = {"策略ID": "S1", "leg": "e20", "前20日%": -20, "涨幅%": 25, "量比": 1.5, "成交额M": 90}
    assert not passes_quality_filter(sig, cfg, spy_bull=False)


def test_merge_dedupes_ticker():
    cfg = LongShortComboConfig(min_quality_score=0.0, min_dvol_m_boost=0)
    e20 = [{"策略ID": "L1", "代码": "AAA", "side": "long", "成交额M": 100, "收盘强度": 0.2, "量比": 2}]
    flow = [{"代码": "AAA", "方向": "做多", "side": "long", "量比": 1.5, "setup_win_rate": 0.6}]
    out = merge_and_select(e20, flow, cfg, spy_bull=True)
    assert len(out) == 1
    assert out[0]["代码"] == "AAA"


def test_config_from_dict_nested():
    cfg = config_from_dict({"min_quality_score": 0.6, "extreme20": {"min_dvol_m": 80}})
    assert cfg.min_quality_score == 0.6
    assert cfg.extreme20["min_dvol_m"] == 80
