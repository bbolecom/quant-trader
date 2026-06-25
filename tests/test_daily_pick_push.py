"""daily_pick_push 过滤逻辑。"""

from __future__ import annotations

from quant.daily_pick_push import (
    build_push_block,
    enrich_pick_data_source,
    is_push_eligible,
    push_priority,
    sanitize_pick_row,
)


def test_option_model_not_push_eligible():
    row = enrich_pick_data_source({
        "代码": "QUBT",
        "状态": "可开仓",
        "方向": "卖Put",
        "模块": "5×舰队·CSP",
        "选股理由": "收$42/张（模型估值）",
    })
    assert row["数据源"] == "模型估算"
    assert is_push_eligible(row) is False


def test_real_chain_push_eligible():
    row = enrich_pick_data_source({
        "代码": "QUBT",
        "状态": "可开仓",
        "方向": "卖Put",
        "模块": "5×舰队·CSP",
        "数据源": "真实链",
        "选股理由": "QUBT $10.78 · 真实链 卖P$9.5 @2026-06-26",
    })
    assert is_push_eligible(row) is True


def test_equity_real_market_push_eligible():
    row = enrich_pick_data_source({
        "代码": "PLTR",
        "状态": "可开仓",
        "方向": "做多",
        "模块": "规律·Ultra80",
        "选股理由": "OOS高胜率规律触发",
    })
    assert row["数据源"] == "真实行情"
    assert is_push_eligible(row) is True


def test_build_push_block_filters_model():
    doc = {
        "选股日期": "2026-06-24",
        "选股时间": "2026-06-24 12:00:00",
        "regime": {"label": "牛市", "bull": True},
        "summary": {},
        "picks": [
            {
                "代码": "AAPL",
                "状态": "可开仓",
                "方向": "卖Call价差",
                "模块": "收入·卖Call",
                "数据源": "真实链",
                "选股理由": "真实链 卖C$200/买C$220",
            },
            {
                "代码": "QUBT",
                "状态": "可开仓",
                "方向": "卖Put",
                "模块": "CSP",
                "选股理由": "模型估值",
                "数据源": "模型估算",
            },
        ],
    }
    push = build_push_block(doc, {"push": {"require_real_data": True}})
    assert push["count"] == 1
    assert push["picks"][0]["代码"] == "AAPL"
    assert push["stats"]["skipped_watch"] >= 1


def test_sanitize_clears_fake_option_fields():
    row = sanitize_pick_row({
        "代码": "SNDK",
        "状态": "可开仓",
        "方向": "卖Put",
        "模块": "5×舰队·CSP",
        "数据源": "真实链不可用",
        "卖Put行权价": 1720,
        "权利金$": 25800,
        "建议张数": 1,
        "策略动作": "卖Put 30D",
        "选股理由": "SNDK $1914 · 真实链不可用",
    })
    assert row["状态"] == "观望"
    assert row["方向"] == "观望"
    assert row["数据有效"] is False
    assert row["可交易"] is False
    assert row["卖Put行权价"] == ""
    assert row["权利金$"] == ""
    assert row["建议张数"] == ""
    assert row["策略动作"] == ""


def test_sanitize_keeps_real_chain_quotes():
    row = sanitize_pick_row({
        "代码": "SOFI",
        "状态": "可开仓",
        "方向": "卖Put",
        "模块": "5×舰队·CSP",
        "数据源": "真实链",
        "可开仓": "✅",
        "卖Put行权价": 15,
        "权利金$": 42,
        "建议张数": 1,
        "策略动作": "卖Put $15",
    })
    assert row["数据有效"] is True
    assert row["可交易"] is True
    assert row["卖Put行权价"] == 15
    assert row["权利金$"] == 42


def test_push_block_sorts_by_strategy_priority():
    doc = {
        "选股日期": "2026-06-24",
        "选股时间": "2026-06-24 12:00:00",
        "regime": {"label": "牛市", "bull": True},
        "summary": {},
        "picks": [
            {
                "代码": "ZZZ",
                "状态": "可开仓",
                "方向": "做多",
                "模块": "低优先",
                "数据源": "真实行情",
                "策略排名": 9,
                "历史胜率": 0.55,
                "选股理由": "低优先级真实行情",
            },
            {
                "代码": "AAA",
                "状态": "可开仓",
                "方向": "做多",
                "模块": "多空组合",
                "数据源": "真实行情",
                "策略排名": 3,
                "策略评级": "A",
                "历史胜率": 0.59,
                "推送优先级": 120,
                "选股理由": "高排名策略",
            },
        ],
    }
    push = build_push_block(doc, {"push": {"require_real_data": True}})
    assert push["picks"][0]["代码"] == "AAA"
    assert push_priority(push["picks"][0]) > push_priority(push["picks"][1])
    assert "#3A" in push["lines"][0]
