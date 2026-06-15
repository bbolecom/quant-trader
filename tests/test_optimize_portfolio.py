"""寻优、策略对比、组合回测与风险控制测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import optimize, portfolio


def test_grid_search_returns_best(ohlcv):
    grid = {"fast": [5, 10, 20], "slow": [40, 60, 90]}
    res = optimize.grid_search(ohlcv, "双均线交叉", grid, sort_by="夏普比率")
    assert set(res.best_params) == {"fast", "slow"}
    # 结果按夏普降序，首行应为最优。
    assert res.table.iloc[0]["夏普比率"] == res.table["夏普比率"].max()
    # 不合理组合（fast>=slow）应被跳过。
    assert (res.table["fast"] < res.table["slow"]).all()


def test_grid_search_combo_limit(ohlcv):
    grid = {"fast": list(range(2, 60)), "slow": list(range(60, 200))}
    with pytest.raises(ValueError):
        optimize.grid_search(ohlcv, "双均线交叉", grid, max_combos=100)


def test_compare_strategies(multi_data):
    df = list(multi_data.values())[0]
    table, curves = optimize.compare_strategies(df)
    assert len(curves) == len(table)
    assert "夏普比率" in table.columns


def test_normalize_weights_sums_to_one():
    w = portfolio.normalize_weights({"A": 2, "B": 1, "C": 1})
    assert np.isclose(sum(w.values()), 1.0)
    assert np.isclose(w["A"], 0.5)


def test_weight_cap_respected():
    w = portfolio.apply_weight_cap({"A": 10, "B": 1, "C": 1}, cap=0.4)
    assert np.isclose(sum(w.values()), 1.0)
    assert max(w.values()) <= 0.4 + 1e-9


def test_inverse_vol_favors_low_vol():
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    lowvol = 100 * np.cumprod(1 + rng.normal(0, 0.005, len(idx)))
    highvol = 100 * np.cumprod(1 + rng.normal(0, 0.03, len(idx)))
    data = {
        "LOW": pd.DataFrame({"Open": lowvol, "High": lowvol, "Low": lowvol, "Close": lowvol, "Volume": 1e6}, index=idx),
        "HIGH": pd.DataFrame({"Open": highvol, "High": highvol, "Low": highvol, "Close": highvol, "Volume": 1e6}, index=idx),
    }
    w = portfolio.compute_weights(data, mode="逆波动率")
    assert w["LOW"] > w["HIGH"]


def test_vol_target_reduces_volatility(multi_data):
    w = portfolio.compute_weights(multi_data, mode="等权")
    base = portfolio.run_portfolio(multi_data, w, "买入持有（基准）")
    targeted = portfolio.run_portfolio(multi_data, w, "买入持有（基准）", vol_target=0.10, max_leverage=1.0)
    assert targeted.leverage is not None
    # 目标波动率 <= 基准波动率时，调节后的年化波动应不高于原始。
    assert targeted.stats["年化波动率"] <= base.stats["年化波动率"] + 1e-6
