"""strategy_catalog 单元测试。"""

from __future__ import annotations

from quant.strategy_catalog import (
    build_strategy_summary_doc,
    strategy_registry,
    summarize_picks_by_module,
)


def test_registry_has_gain15():
    ids = {s.id for s in strategy_registry()}
    assert "gain15" in ids
    assert "daily_pick" in ids


def test_summarize_picks_by_module():
    picks = [
        {"模块": "暴涨80%·追多", "状态": "可开仓", "代码": "GME"},
        {"模块": "暴涨80%·观察", "状态": "观望", "代码": "AMC"},
        {"模块": "资金流向", "状态": "可开仓", "代码": "NVDA"},
    ]
    out = summarize_picks_by_module(picks)
    assert out["暴涨80%·追多"]["可开仓"] == 1
    assert out["资金流向"]["可开仓"] == 1
    assert out["暴涨80%·观察"]["观望"] == 1


def test_build_strategy_summary_doc():
    doc = build_strategy_summary_doc(
        picks=[{"模块": "测试", "状态": "可开仓", "代码": "AAPL"}],
        modules_summary={"测试": {"可开仓": 1, "观望": 0, "总条目": 1, "代码": ["AAPL"]}},
        regime={"bull": True, "label": "test"},
    )
    assert doc["integrated_count"] >= 1
    assert isinstance(doc["catalog"], list)
