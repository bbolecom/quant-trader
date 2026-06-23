"""Tests for universal_playbook."""

from __future__ import annotations

from research.universal_playbook import (
    TICKER_RATIONALE,
    build_universal_playbook,
    format_playbook_text,
    playbook_to_dataframe,
)


def test_ticker_rationale_has_core_etfs():
    for tk in ("SPY", "QQQ", "IWM"):
        assert tk in TICKER_RATIONALE
        assert len(TICKER_RATIONALE[tk]["选股理由"]) >= 2


def test_build_playbook_structure():
    pb = build_universal_playbook(account=10_000, mode="stable")
    assert pb.total_capital == 50_000
    assert len(pb.slots) == 5
    assert pb.slots[0].ticker in {"SPY", "QQQ", "IWM", "XLF", "DIA"}
    assert len(pb.slots[0].selection_reason) >= 1
    assert len(pb.slots[0].execution_steps) >= 3


def test_format_and_csv():
    pb = build_universal_playbook(account=10_000, mode="stable")
    lines = format_playbook_text(pb)
    assert any("选股理由" in ln or "为什么选" in ln for ln in lines)
    df = playbook_to_dataframe(pb)
    assert "执行腿" in df.columns
    assert len(df) >= 5
