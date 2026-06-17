"""短线策略搜索器测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import strategy_search as ss


def test_build_search_space_non_empty():
    space = ss.build_search_space()
    assert len(space) > 20
    ideas = {c.idea for c in space}
    assert {"强势动量", "突破", "超跌反弹"}.issubset(ideas)
    # 关闭做空后组合数应减少。
    fewer = ss.build_search_space(include_short=False)
    assert len(fewer) < len(space)


def test_metrics_basic():
    m = ss._metrics([0.05, -0.02, 0.03, -0.01, 0.04])
    assert m["笔数"] == 5
    assert 0.0 <= m["胜率"] <= 1.0
    assert m["盈亏比"] > 0
    assert m["最差单笔%"] < 0


def test_evaluate_and_search_on_synthetic(multi_data):
    # 复用 conftest 的 multi_data（4 只、各 500 根）。
    combo = ss.build_search_space(rebalance_options=(5,), include_short=False)[0]
    ev = ss.evaluate_combo(multi_data, combo, min_trades=5)
    assert "train" in ev and "test" in ev
    assert "robust" in ev

    table, results = ss.search_short_term(
        multi_data, rebalance_options=(5,), include_short=False, min_trades=5,
    )
    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert "稳健通过" in table.columns
    assert "样本内评分" in table.columns


def test_combo_to_preset():
    combo = ss.build_search_space()[0]
    preset = ss.combo_to_preset(combo, "测试预设", "测试依据")
    assert preset.name == "测试预设"
    assert preset.horizon == "短线"
    assert preset.trading_strategy == combo.trading_strategy
    assert preset.forward_eval_days == combo.forward_days
