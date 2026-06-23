"""daily_pick_runners 单元测试。"""

from __future__ import annotations

from quant.daily_pick_runners import (
    RUNNER_REGISTRY,
    pick_row,
    run_registered,
)


def test_runner_registry_covers_all_standalone():
    expected = {
        "pattern_daily",
        "flow_strategy",
        "vrp",
        "calendar",
        "universal_playbook",
        "sndk_iron",
        "strategy_rank",
        "screen_daily",
        "scan_daily",
        "ticker_pattern",
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
    cfg = {"quick": True}
    rows = run_registered("pattern_daily", cfg, bull=True)
    assert len(rows) == 1
    assert "quick" in rows[0]["选股理由"]
