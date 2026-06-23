"""decline_income 模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import decline_income as di


def _drifting_down(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.exp(-0.0003 * np.arange(n))  # 缓慢下跌
    close *= 1 + np.random.default_rng(1).normal(0, 0.02, n)
    close = np.abs(close)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.02, "Low": close * 0.98,
         "Close": close, "Volume": 8e6},
        index=idx,
    )


def test_classify_decline_trend_slow():
    df = _drifting_down()
    label, r20, r60 = di.classify_decline_trend(df["Close"])
    assert r60 < 0
    assert "缓跌" in label or "跌" in label or "横盘" in label


def test_estimate_bear_call_spread():
    ks, kl, credit, max_l, y = di.estimate_bear_call_spread(100.0, 0.5)
    assert ks < kl
    assert credit > 0
    assert max_l >= 0
    assert y > 0


def test_backtest_bear_call_spread_on_decline():
    df = _drifting_down(400)
    stats = di.backtest_bear_call_spread(df["Close"])
    assert stats.get("周期数", 0) >= 3
    assert stats.get("胜率", 0) >= 0


def test_analyze_ticker():
    df = _drifting_down(250)
    plan = di.analyze_ticker("TEST", df, di.DeclineFilters(min_dollar_vol_m=0, min_ret_60d_pct=-99))
    assert plan is not None
    assert plan.primary_strategy == "熊市认购价差"
    assert len(plan.playbook) >= 2


def _trending_up(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.exp(0.0015 * np.arange(n))  # 强势上涨
    close *= 1 + np.random.default_rng(7).normal(0, 0.03, n)
    close = np.abs(close)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.03, "Low": close * 0.97,
         "Close": close, "Volume": 1e7},
        index=idx,
    )


def test_backtest_csp_income():
    df = _trending_up(400)
    stats = di.backtest_csp_income(df["Close"])
    assert stats.get("交易数", 0) >= 5
    assert 0.0 <= stats.get("胜率", -1) <= 1.0
    # 顺势 CSP 在上涨股上应是高胜率
    assert stats["胜率"] >= 0.6


def test_compare_income_strategies():
    df = _trending_up(400)
    tbl = di.compare_income_strategies(df["Close"])
    assert not tbl.empty
    assert "信息比" in tbl.columns
    assert len(tbl) == 5
    # 顺势 CSP 在上涨股上应是高胜率（方向性优劣由真实数据回测证明）
    csp_win = tbl[tbl["策略"].str.contains("现金担保")]["胜率"].iloc[0]
    assert csp_win >= 0.7


def test_csp_income_plan():
    df = _trending_up(300)
    plan = di.csp_income_plan("TEST", df)
    assert plan is not None
    assert plan.put_strike < plan.close          # 卖下方 put
    assert plan.premium > 0
    assert plan.take_profit_price < plan.premium  # 止盈目标小于初始权利金
    assert plan.breakeven < plan.put_strike
    assert len(plan.playbook) >= 5


def test_estimate_put_credit_spread():
    ks, kl, credit, margin, max_loss, zp = di.estimate_put_credit_spread(100.0, 0.5, width=5.0)
    assert ks > kl
    assert credit > 0
    assert margin == 500.0
    assert max_loss >= 0
    assert 0 < zp < 1


def test_weekly_put_soup_plan():
    df = _trending_up(300)
    plan = di.weekly_put_soup_plan("TEST", df, account_size=10_000)
    assert plan is not None
    assert plan.short_strike > plan.long_strike
    assert plan.credit_per_contract > 0
    assert plan.margin_per_contract > 0
    assert plan.take_profit_price < plan.credit_per_share
    assert len(plan.playbook) >= 5


def test_scan_weekly_soup_configs():
    tbl = di.scan_weekly_soup_configs(100.0, 0.5, account_size=10_000)
    assert not tbl.empty
    assert "归零概率" in tbl.columns
    assert len(tbl) == 9
