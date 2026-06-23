"""三重目标扫描器单元测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.decline_income import trades_to_equity, equity_metrics_from_trades, _summarize_csp
from research.triple_target_scan import (
    classify_tier,
    gap_score,
    pareto_frontier,
    ScanResult,
    set_scan_targets,
    split_equity_oos,
)


def test_trades_to_equity_compounds():
    eq, r = trades_to_equity([0.10, -0.05, 0.08], alloc_pct=1.0, initial=1.0)
    assert len(eq) == 3
    assert abs(eq.iloc[-1] - 1.10 * 0.95 * 1.08) < 1e-9


def test_equity_max_dd_not_zero_on_losses():
    """真实净值回撤：有亏损序列时不应出现 0% 回撤 artifact。"""
    rors = [0.02, 0.01, -0.15, 0.03, 0.02]
    stats = equity_metrics_from_trades(rors, alloc_pct=0.10)
    assert stats["最大回撤"] < -0.001
    assert stats["胜率"] == pytest.approx(0.8)


def test_summarize_csp_uses_equity_not_synthetic_only():
    rors = [0.01] * 50 + [-0.20]
    stats = _summarize_csp(rors, alloc_pct=0.10)
    assert "最大回撤" in stats
    assert stats["最大回撤"] < 0.0
    assert "合成回撤" in stats


def test_classify_tier_a_requires_all_three():
    set_scan_targets(preset="strict")
    assert classify_tier(1.1, -0.05, 0.85, oos=True) == "A"
    assert classify_tier(1.1, -0.05, 0.85, oos=False) == "C"
    assert classify_tier(0.5, -0.05, 0.85, oos=True) == "B"
    assert classify_tier(1.1, -0.20, 0.85, oos=True) == "B"
    assert classify_tier(1.1, -0.05, 0.70, oos=True) == "B"
    set_scan_targets(preset="relaxed")
    assert classify_tier(0.55, -0.10, 0.88, oos=True) == "A"


def test_gap_score_lower_is_better():
    set_scan_targets(preset="strict")
    perfect = gap_score(1.1, -0.05, 0.90)
    bad = gap_score(0.1, -0.50, 0.50)
    assert perfect < bad
    set_scan_targets(preset="relaxed")


def test_pareto_frontier():
    rows = [
        ScanResult("a", "a", "t", 0.5, -0.20, 0.70, 10, 1.0),
        ScanResult("b", "b", "t", 0.8, -0.30, 0.80, 10, 1.0),
        ScanResult("c", "c", "t", 0.3, -0.10, 0.90, 10, 1.0),
    ]
    front = pareto_frontier(rows)
    ids = {r.strategy_id for r in front}
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids


def test_split_equity_oos_datetime():
    idx = pd.date_range("2019-01-01", periods=1000, freq="B")
    eq = pd.Series(range(1, 1001), index=idx, dtype=float)
    train, test = split_equity_oos(eq, train_end="2022-12-31")
    assert len(train) > 0
    assert len(test) > 0
    assert train.index.max() <= pd.Timestamp("2022-12-31")


def test_split_equity_oos_range_index():
    eq = pd.Series([1.0, 1.05, 1.10, 0.95, 1.0, 1.02])
    train, test = split_equity_oos(eq)
    assert len(train) == 4
    assert len(test) == 2
