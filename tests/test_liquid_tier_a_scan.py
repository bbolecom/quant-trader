"""Tests for liquid_tier_a_scan."""

from __future__ import annotations

import pandas as pd
import pytest

from research.liquid_tier_a_scan import (
    _avg_dollar_vol,
    build_candidate_pool,
    pick_fleet_tickers,
)


def test_build_candidate_pool_dedupes():
    pool = build_candidate_pool(use_broad=False)
    assert len(pool) == len(set(pool))
    assert "NVDA" in pool
    assert "AAPL" in pool
    short = build_candidate_pool(use_broad=False, max_names=30)
    assert len(short) == 30


def test_avg_dollar_vol():
    close = pd.Series([100.0] * 25)
    vol = pd.Series([1_000_000.0] * 25)
    assert _avg_dollar_vol(close, vol) == pytest.approx(100_000_000.0)


def test_pick_fleet_tickers_from_csv(tmp_path, monkeypatch):
    csv = tmp_path / "results.csv"
    csv.write_text(
        "代码,策略,年化,最大回撤,胜率,成交额M,tier,gap_score,fits_10k\n"
        "NVDA,CSP,0.60,-0.05,0.90,500,A,0.1,False\n"
        "AMD,偏斜铁鹰,0.55,-0.08,0.88,200,A,0.2,True\n"
        "INTC,偏斜铁鹰,0.50,-0.10,0.86,150,A,0.3,True\n"
        "MU,偏斜铁鹰,0.48,-0.12,0.85,120,B,0.4,True\n"
        "WDC,偏斜铁鹰,0.45,-0.11,0.84,80,B,0.5,True\n"
    )
    import research.liquid_tier_a_scan as mod

    monkeypatch.setattr(mod, "RESULTS_CSV", csv)
    # use_patterns=False：强制走 CSV 路径（默认 True 会先读规律模型，绕过本测试目标）。
    picks = pick_fleet_tickers(
        3, account_size=10_000, prefer_weekly_for_small=True, use_patterns=False
    )
    assert len(picks) == 3
    assert all(p in {"AMD", "INTC", "MU", "WDC", "NVDA"} for p in picks)
