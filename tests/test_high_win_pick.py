"""high_win_pick 单元测试。"""

from __future__ import annotations

from quant.high_win_pick import StatsStore, enrich_pick, filter_high_win_picks


def test_fleet_ticker_stats():
    store = StatsStore()
    pick = {"模块": "5×舰队·CSP", "代码": "AMPX", "状态": "可开仓", "方向": "卖Put"}
    out = enrich_pick(pick, store)
    assert out.get("历史胜率") is not None
    assert float(out["历史胜率"]) >= 0.80


def test_gain15_rule_stats():
    store = StatsStore()
    pick = {
        "模块": "暴涨80%·追多",
        "代码": "GME",
        "状态": "可开仓",
        "方向": "做多",
        "规则": "Top3+3日累计涨>15%",
        "历史命中率": "83%",
    }
    out = enrich_pick(pick, store)
    assert out.get("高胜率达标") is True


def test_filter_high_win():
    picks = [
        {"代码": "A", "状态": "可开仓", "历史胜率": 0.85, "历史年化": 0.4, "最大回撤": -0.1},
        {"代码": "B", "状态": "可开仓", "历史胜率": 0.55},
        {"代码": "C", "状态": "观望", "历史胜率": 0.90},
    ]
    out = filter_high_win_picks(picks, min_win_rate=0.80, actionable_only=True)
    assert len(out) == 1
    assert out[0]["代码"] == "A"
