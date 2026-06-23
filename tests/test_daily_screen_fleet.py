"""daily_screen_fleet 指标计算测试。"""

from __future__ import annotations

import pandas as pd

from quant.daily_screen_fleet import (
    _stats_from_historical_picks,
    meets_target_profile,
    load_target_profile,
)


def test_stats_from_historical_picks_basic():
    daily = pd.DataFrame([
        {"选股日期": "2024-01-02", "代码": "AAPL", "后20日收益": 0.05, "策略后向收益": 0.04},
        {"选股日期": "2024-01-02", "代码": "MSFT", "后20日收益": 0.03, "策略后向收益": 0.02},
        {"选股日期": "2024-01-08", "代码": "NVDA", "后20日收益": -0.02, "策略后向收益": -0.01},
    ])
    stats = _stats_from_historical_picks(daily, forward_days=20, years=1.0, initial_capital=10_000.0)
    assert stats["rebalance_count"] == 2
    assert stats["period_win_rate"] == 0.5
    assert stats["trade_win_rate"] > 0.5
    assert stats["final_equity"] > 10_000.0


def test_meets_target_profile_csp_anchor():
    stats = {
        "ann_return": 1.128,
        "max_dd": 0.0,
        "trade_win_rate": 1.0,
    }
    targets = {"ann_return": 0.80, "max_dd": -0.10, "win_rate": 0.85}
    assert meets_target_profile(stats, targets)


def test_meets_target_profile_fails_momentum():
    stats = {"ann_return": 0.80, "max_dd": -0.50, "trade_win_rate": 0.40}
    targets = load_target_profile({"target_profile": {"ann_return": 0.80, "max_dd": -0.10, "win_rate": 0.85}})
    assert not meets_target_profile(stats, targets)
