"""strategy_catalog 单元测试。"""

from __future__ import annotations

from quant.strategy_catalog import (
    CORE_STRATEGY_IDS,
    _count_actionable,
    build_strategy_audit,
    build_strategy_summary_doc,
    enrich_catalog_from_daily_pick,
    strategy_registry,
    summarize_picks_by_module,
)


def test_registry_has_gain15():
    ids = {s.id for s in strategy_registry()}
    assert "gain15" in ids
    assert "extreme20" in ids
    assert "longshort_combo" in ids
    assert "whipsaw_short" in ids
    assert "gainer10" in ids
    assert "daily_pick" in ids
    assert len(CORE_STRATEGY_IDS) == 12


def test_count_actionable_whipsaw_scan_stats():
    doc = {
        "date": "2026-06-25",
        "scan_stats": {"候选": 2, "可开仓": 1},
        "candidates": [
            {"代码": "ABC", "信号": "卖Call价差", "建议张数": 2},
            {"代码": "XYZ", "信号": "无可行价差"},
        ],
    }
    stats = _count_actionable(doc)
    assert stats["可开仓"] == 1
    assert stats["总条目"] == 2
    assert stats["date"] == "2026-06-25"


def test_core_twelve_only():
    reg = strategy_registry()
    core = [s for s in reg if s.id != "daily_pick"]
    assert len(core) == 12
    assert {s.id for s in core} == set(CORE_STRATEGY_IDS)


def test_build_strategy_audit_has_ranks():
    doc = build_strategy_audit()
    rows = doc["rows"]
    assert rows
    assert rows[0]["audit_rank"] == 1
    assert all("audit_tier" in r for r in rows)
    assert any(r["id"] == "longshort_combo" for r in rows)


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
        pick_date="2026-06-24",
        summary={"可开仓": 1, "观望": 0, "总条目": 1},
    )
    assert doc["integrated_count"] >= 1
    assert isinstance(doc["catalog"], list)
    hub = next(r for r in doc["catalog"] if r["id"] == "daily_pick")
    assert hub["可开仓"] == 1
    assert hub["今日有数据"] is True


def test_enrich_catalog_from_daily_pick():
    catalog = [
        {"id": "daily_pick", "可开仓": 0, "观望": 0, "总条目": 0, "今日有数据": False},
        {"id": "fleet_csp", "可开仓": 0, "模块标签": "5×舰队·CSP", "今日有数据": False},
    ]
    dp = {
        "选股日期": "2026-06-24",
        "summary": {"可开仓": 3, "观望": 1, "总条目": 4},
        "modules_summary": {
            "5×舰队·CSP": {"可开仓": 1, "观望": 0, "总条目": 1, "代码": ["QUBT"]},
        },
    }
    out = enrich_catalog_from_daily_pick(catalog, dp)
    hub = next(r for r in out if r["id"] == "daily_pick")
    fleet = next(r for r in out if r["id"] == "fleet_csp")
    assert hub["可开仓"] == 3
    assert hub["今日有数据"] is True
    assert fleet["可开仓"] == 1
    assert fleet["今日有数据"] is True
