"""strategy_ranker 单元测试（无网络）。"""

from __future__ import annotations

from research.strategy_ranker import CATALOG, StrategyMeta, StrategyPick, _static_score, format_playbook


def test_static_score_avoids_short():
    short = next(m for m in CATALOG if m.id == "short_overheat")
    assert _static_score(short) < 0


def test_static_score_income_positive():
    cs = next(m for m in CATALOG if m.id == "call_spread")
    assert _static_score(cs) > 0.5


def test_format_playbook_structure():
    class Reg:
        label = "🟢 牛市"
        spy = 500.0
        ma50 = 480.0

    pick = StrategyPick(
        meta=next(m for m in CATALOG if m.id == "call_spread"),
        score=1.2, signal_ok=True, regime_ok=True,
        detail="NVDA 卖C$250/买C$280",
        trades=[{"代码": "NVDA", "卖Call": 250, "买Call": 280, "建议张数": 1,
                 "预计收租$": 120, "最大亏损$": 800, "占用$": 800, "占比%": 8.0}],
    )
    result = {
        "regime": Reg(),
        "account": 10_000,
        "profile": "balanced",
        "top3": [pick],
        "portfolio": [{
            "引擎": "卖看涨价差", "代码": "NVDA", "结构": "卖Call $250 / 买Call $280",
            "张数": 1, "预估收租$": 120, "最大亏损$": 800, "占用$": 800, "占比%": 8.0,
        }],
    }
    lines = format_playbook(result)
    assert any("Top3" in ln for ln in lines)
    assert any("NVDA" in ln for ln in lines)
