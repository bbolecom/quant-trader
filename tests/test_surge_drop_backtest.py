"""暴涨/暴跌池策略回测测试。"""

from __future__ import annotations

from quant.surge_drop_backtest import get_strategy_rule, list_strategy_presets


def test_strategy_presets_exist():
    presets = list_strategy_presets()
    assert "drop_rebound" in presets
    assert "surge_fade" in presets
    rule = get_strategy_rule("drop_rebound")
    assert rule.direction == "drop"
    assert rule.side == "long"
