"""daily_pick.py 单元测试（无网络）。"""

from __future__ import annotations

import json
from pathlib import Path

import daily_pick as dp


def _bear_cfg() -> dict:
    base = json.loads((Path(__file__).resolve().parents[1] / "daily_pick_config.json").read_text())
    base["regime"] = {"mock": {"bull": False, "spy": 400.0, "ma50": 420.0}, "trajectory_bull_only": True}
    base["frequency_profile"] = "standard"
    base["modules"] = {
        "bear_call": False,
        "bear_iron_etf": False,
        "fleet_csp": False,
        "trajectory_highwin": True,
        "gain15": False,
        "extreme20": False,
        "capital_flow": False,
        "meme_long": False,
        "pattern_daily": False,
        "flow_strategy": False,
        "vrp": False,
        "calendar": False,
        "universal_playbook": False,
        "sndk_iron": False,
        "strategy_rank": False,
        "screen_daily": False,
        "scan_daily": False,
        "ticker_pattern": False,
        "legacy_screen": False,
    }
    return base


def _bull_cfg() -> dict:
    base = _bear_cfg()
    base["regime"]["mock"] = {"bull": True, "spy": 520.0, "ma50": 500.0}
    return base


def test_get_market_regime_mock_bear():
    reg = dp.get_market_regime(_bear_cfg())
    assert reg["bull"] is False
    assert reg["mode"] == "bear"
    assert "弱市" in reg["label"] or "SPY" in reg["label"]


def test_trajectory_skipped_in_bear():
    rows = dp._run_trajectory_highwin(bull=False, bull_only=True)
    assert len(rows) == 1
    assert rows[0]["状态"] == "观望"
    assert "弱市" in rows[0]["选股理由"]


def test_run_daily_pick_bear_includes_regime():
    doc = dp.run_daily_pick(_bear_cfg())
    assert doc["regime"]["mode"] == "bear"
    assert doc["summary"]["模式"] == "弱市偏空收租"
    traj = [p for p in doc["picks"] if p["模块"] == "轨迹·高置信"]
    assert len(traj) == 1
    assert traj[0]["状态"] == "观望"


def test_run_daily_pick_bull_mode_label():
    doc = dp.run_daily_pick(_bull_cfg())
    assert doc["regime"]["mode"] == "bull"
    assert doc["summary"]["模式"] == "牛市三引擎"


def test_meme_long_skipped_when_disabled():
    cfg = _bull_cfg()
    cfg["modules"]["meme_long"] = False
    doc = dp.run_daily_pick(cfg)
    assert not any(
        p.get("模块") in ("规律·纯多头", "规律·Ultra80", "规律·Ultra80准入")
        for p in doc.get("picks") or []
    )


def test_meme_long_skipped_in_bear():
    cfg = _bear_cfg()
    cfg["modules"]["meme_long"] = True
    cfg["meme_long"] = {"enabled": True, "bull_only": True, "tickers": ["MSTR"]}
    rows = dp._run_meme_long(cfg, bull=False)
    assert len(rows) == 1
    assert rows[0]["状态"] == "观望"
    assert "弱市" in rows[0]["选股理由"]


def test_run_daily_pick_includes_strategy_summary():
    doc = dp.run_daily_pick(_bull_cfg())
    assert "strategy_summary" in doc
    assert "modules_summary" in doc
    assert "catalog" in doc["strategy_summary"]


def test_notify_bear_empty_day():
    doc = {
        "regime": {"bull": False, "label": "🔴 弱市（SPY<MA50）"},
        "summary": {"可开仓": 0},
        "picks": [],
        "push": {"picks": [], "stats": {"skipped_model": 2}},
    }
    cfg = {"notify": {"desktop": True, "only_when_action": False}, "push": {"require_real_data": True}}
    # 不应抛错（无 scan_daily 时静默跳过）
    dp.notify(doc, cfg)
