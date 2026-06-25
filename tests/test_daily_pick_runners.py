"""daily_pick_runners 单元测试。"""

from __future__ import annotations

from quant.daily_pick_runners import (
    RUNNER_REGISTRY,
    pick_row,
    run_registered,
)


def test_runner_registry_covers_core():
    expected = {
        "longshort_combo", "flow_strategy", "vrp", "sndk_iron",
        "extreme20", "whipsaw_short", "gainer10",
    }
    assert expected == set(RUNNER_REGISTRY.keys())


def test_pick_row_shape():
    r = pick_row(
        module="测试",
        account="A",
        ticker="AAPL",
        status="可开仓",
        direction="做多",
        reason="test",
    )
    assert r["模块"] == "测试"
    assert r["代码"] == "AAPL"


def test_quick_skips_heavy():
    cfg = {"quick": True, "modules": {"extreme20": True}}
    rows = run_registered("extreme20", cfg, bull=True)
    assert len(rows) == 1
    assert "quick" in rows[0]["选股理由"]


def test_extreme20_disabled_returns_empty():
    from quant.daily_pick_runners import run_extreme20

    rows = run_extreme20({"modules": {"extreme20": False}})
    assert rows == []


def test_whipsaw_short_disabled_returns_empty():
    from quant.daily_pick_runners import run_whipsaw_short

    rows = run_whipsaw_short({"modules": {"whipsaw_short": False}})
    assert rows == []


def test_gainer10_disabled_returns_empty():
    from quant.daily_pick_runners import run_gainer10

    rows = run_gainer10({"modules": {"gainer10": False}})
    assert rows == []


def test_longshort_combo_disabled_returns_empty():
    from quant.daily_pick_runners import run_longshort_combo

    rows = run_longshort_combo({"modules": {"longshort_combo": False}})
    assert rows == []


def test_gainer10_maps_signals(monkeypatch):
    from quant.daily_pick_runners import run_gainer10

    fake_plan = {
        "date": "2026-06-25",
        "note": "",
        "scan_stats": {"续涨A": 1, "做空S": 1, "分板块多": 0, "分板块空": 0},
        "buy_a": [{
            "代码": "NVDA", "信号": "续涨A", "动作": "hold20", "涨幅_pct": 12.0, "成交额M": 500,
            "板块": "科技", "跳空_pct": 6.0, "乖离20_pct": 45.0, "RSI": 78,
            "规则说明": "科技强动量", "历史胜率": "~59%", "历史均收益": "~+19%",
            "限价入场": 120.0, "止盈_pct": 25.0, "止损_pct": 12.0, "现价": 125.0,
        }],
        "buy_b": [],
        "buy_sector": [],
        "short_s": [{
            "代码": "BIOT", "信号": "做空S", "动作": "收盘做空", "涨幅_pct": 11.0, "成交额M": 150,
            "板块": "医疗", "跳空_pct": -1.0, "乖离20_pct": -3.0, "RSI": 42,
            "规则说明": "弱板块衰竭", "历史胜率": "~69%", "历史均收益": "~+3.6%",
            "现价": 8.5,
        }],
        "short_sector": [],
        "picks": [],
    }
    monkeypatch.setattr("gainer10_daily.run_gainer10_scan", lambda cfg: fake_plan)
    monkeypatch.setattr("gainer10_daily.save_outputs", lambda plan, raw: None)
    monkeypatch.setattr("gainer10_daily.load_config", lambda p: {})
    monkeypatch.setattr("gainer10_daily.cfg_from_dict", lambda raw: raw)
    rows = run_gainer10({"modules": {"gainer10": True}})
    assert len(rows) == 2
    assert rows[0]["代码"] == "NVDA"
    assert rows[0]["状态"] == "可开仓"
    assert rows[1]["方向"] == "做空"


def test_extreme20_maps_signals(monkeypatch):
    from quant.daily_pick_runners import run_extreme20
    import pandas as pd

    fake = pd.DataFrame([{
        "策略ID": "L1",
        "代码": "TEST",
        "方向": "做多",
        "策略": "弱收盘续涨·顺风",
        "依据": "test signal",
        "持有": "5日",
        "入场": "次日开盘",
        "止损价≈": 9.0,
        "止盈价≈": 11.0,
        "止损%": 10.0,
        "止盈%": 15.0,
        "side": "long",
    }])

    monkeypatch.setattr("quant.extreme20_strategy.scan_live", lambda *a, **k: fake)
    rows = run_extreme20({"modules": {"extreme20": True}})
    assert len(rows) == 1
    assert rows[0]["状态"] == "可开仓"
    assert rows[0]["代码"] == "TEST"
    assert rows[0]["策略ID"] == "L1"
