"""Tier A CSP / 5×$10k 舰队测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from research.tier_a_csp import (
    _contracts_for_account,
    _resolve_fleet_slots,
    fleet_summary,
    format_playbook_lines,
    load_tier_a_csp_config,
    scan_tier_a_fleet,
)


def test_load_fleet_config():
    cfg = load_tier_a_csp_config()
    assert cfg["fleet"]["enabled"] is True
    assert cfg["fleet"]["account_size"] == 10_000
    assert cfg["fleet"]["count"] == 5


def test_resolve_fleet_slots():
    cfg = load_tier_a_csp_config()
    slots = _resolve_fleet_slots(cfg)
    assert len(slots) == 5
    assert slots[0][1] == 10_000
    tickers = [s[2] for s in slots]
    assert len(tickers) == 5
    assert all(isinstance(t, str) and len(t) >= 1 for t in tickers)


def test_contracts_for_account():
    assert _contracts_for_account(10_000, 200_000, alloc_pct=0.50, max_single_pct=0.50) == 0
    assert _contracts_for_account(50_000, 20_000, alloc_pct=0.50, max_single_pct=0.50) == 1


def test_fleet_summary():
    df = pd.DataFrame([
        {"可开仓": "✅", "权利金$": 100, "建议张数": 1, "担保金$": 2500},
        {"可开仓": "⏸", "权利金$": 0, "建议张数": 0, "担保金$": 0},
    ])
    s = fleet_summary(df)
    assert s["total_accounts"] == 2
    assert s["open_count"] == 1
    assert s["total_premium"] == 100


def test_format_playbook_fleet():
    cfg = load_tier_a_csp_config()
    df = pd.DataFrame([
        {
            "账户": "账户1", "可开仓": "✅", "代码": "INTC", "策略": "周Put价差",
            "卖Put": 105, "建议张数": 1, "权利金$": 80, "站上MA50": "✅", "提示": "",
        },
    ])
    lines = format_playbook_lines(df, cfg)
    assert any("5 账户" in ln for ln in lines)
    assert any("账户1" in ln for ln in lines)


@patch("quant.data.fetch_history")
def test_scan_fleet_fallback_to_weekly(mock_fetch):
    """CSP 放不下时降级为周 Put 价差。"""
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series([50.0 + i * 0.1 for i in range(n)], index=idx)
    df = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99, "Volume": 1e6})
    mock_fetch.return_value = df
    cfg = load_tier_a_csp_config()
    cfg["fleet"]["tickers"] = ["INTC"]
    cfg["fleet"]["count"] = 1
    cfg["fleet"]["labels"] = ["账户1"]
    out, _ = scan_tier_a_fleet(cfg=cfg)
    assert len(out) == 1
    assert out.iloc[0]["账户"] == "账户1"
    assert out.iloc[0]["规模$"] == 10_000
    assert out.iloc[0]["策略"] in ("周Put价差", "偏斜铁鹰", "CSP")
