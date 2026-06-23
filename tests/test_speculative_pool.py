"""SPCE 类投机票池测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.speculative_pool import (
    SpeculativePoolConfig,
    aggregate_ticker_stats,
    build_pool_from_events,
    load_pool_tickers,
    similarity_score,
)


@pytest.fixture
def mini_events(tmp_path: Path) -> Path:
    rows = []
    # SPCE-like
    for i in range(8):
        rows.append({
            "日期": f"2024-0{(i % 9) + 1}-15",
            "代码": "SPCE",
            "涨幅%": 18.0 + i,
            "dvol_m": 200 + i * 20,
            "收盘价": 3.0,
        })
    for i in range(6):
        rows.append({
            "日期": f"2024-0{(i % 9) + 1}-20",
            "代码": "RKLB",
            "涨幅%": 16.0 + i * 0.5,
            "dvol_m": 250,
            "收盘价": 20.0,
        })
    # 不符合：暴涨次数少
    rows.append({"日期": "2024-03-01", "代码": "ZZZZ", "涨幅%": 5.0, "dvol_m": 100, "收盘价": 10.0})
    # ETF 应排除
    for i in range(10):
        rows.append({
            "日期": f"2024-04-{i+1:02d}",
            "代码": "SOXL",
            "涨幅%": 20.0,
            "dvol_m": 500,
            "收盘价": 30.0,
        })
    p = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_aggregate_ticker_stats(mini_events: Path):
    df = pd.read_csv(mini_events)
    stats = aggregate_ticker_stats(df)
    spce = stats[stats["代码"] == "SPCE"].iloc[0]
    assert int(spce["spikes15"]) == 8


def test_build_pool_includes_spce_like(mini_events: Path, tmp_path: Path):
    cfg = SpeculativePoolConfig(
        min_spikes15=5,
        min_max_gain_pct=15,
        pool_size=10,
        core_size=5,
        min_med_dvol_m=10,
        max_med_dvol_m=1000,
    )
    members, meta = build_pool_from_events(cfg, events_path=mini_events, shares_path=tmp_path / "x.json")
    tickers = {m.代码 for m in members}
    assert "SPCE" in tickers
    assert "RKLB" in tickers
    assert "SOXL" not in tickers
    assert "ZZZZ" not in tickers
    assert meta["archetype"] == "SPCE"


def test_similarity_score_range():
    arch = {"spikes15": 10, "spikes30": 2, "max_gain": 40, "med_dvol_m": 300, "n": 100}
    row = pd.Series({"spikes15": 10, "spikes30": 2, "max_gain": 40, "med_dvol_m": 300, "n": 100})
    s = similarity_score(row, arch)
    assert 0.9 <= s <= 1.0


def test_load_pool_tickers_missing(tmp_path: Path):
    assert load_pool_tickers(tmp_path / "missing.json") == []
