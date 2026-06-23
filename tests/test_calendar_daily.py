"""calendar_daily.py 单元测试（无网络）。"""

from __future__ import annotations

import pandas as pd

from quant.calendar_spread import CalendarSpreadPlan, calendar_spread_plan, efficiency_ratio, iv_rank_from_close
from calendar_daily import build_history_row, build_playbook, format_notification, resolve_tickers


def _fake_df(n: int = 300, drift: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 * (1 + pd.Series(range(n), index=idx) * 0.0001 + drift)
    noise = pd.Series([(-1) ** i * 0.5 for i in range(n)], index=idx)
    return pd.DataFrame({"Close": close + noise})


def test_efficiency_ratio_choppy():
    df = _fake_df()
    er = efficiency_ratio(df["Close"], 30)
    assert 0 <= er <= 1


def test_iv_rank_low_in_calm_series():
    df = _fake_df()
    iv, rank = iv_rank_from_close(df["Close"])
    assert iv > 0
    assert 0 <= rank <= 1


def test_calendar_plan_blocks_high_iv_rank():
    df = _fake_df()
    # 末尾注入高波动，使 IV Rank 升高
    c = df["Close"].copy()
    c.iloc[-10:] = c.iloc[-11] * (1 + pd.Series([0.08, -0.07, 0.09, -0.08, 0.07, -0.06, 0.08, -0.07, 0.06, -0.05]))
    df = pd.DataFrame({"Close": c})
    plan = calendar_spread_plan("TEST", df, iv_pct_max=0.20, check_earnings=False)
    assert plan is not None
    assert plan.can_open is False
    assert plan.flags


def test_format_notification_open():
    p = CalendarSpreadPlan(
        ticker="NVDA", close=120.0, rv_pct=40.0, iv_pct=52.0, iv_rank=0.25, er=0.3,
        can_open=True, call_strike=125.0, put_strike=115.0,
        debit_per_share=1.2, debit_per_contract=120.0, debit_pct_account=1.2,
        profit_zone_pct=5.0, theta_est_contract=8.0,
        short_d=14, long_d=21, hold_trading_days=5, max_contracts=1,
    )
    title, body = format_notification({"plans": [p], "errors": []})
    assert "可开" in title
    assert "NVDA" in title
    assert "25%" in body


def test_format_notification_blocked():
    p = CalendarSpreadPlan(
        ticker="SNDK", close=2000.0, rv_pct=90.0, iv_pct=117.0, iv_rank=0.85, er=0.6,
        can_open=False, call_strike=2300.0, put_strike=1700.0,
        debit_per_share=12.0, debit_per_contract=1200.0, debit_pct_account=12.0,
        profit_zone_pct=14.0, theta_est_contract=30.0,
        short_d=14, long_d=21, hold_trading_days=5, max_contracts=0,
        flags=["IV Rank 85% > 40%"],
    )
    title, body = format_notification({"plans": [p], "errors": []})
    assert "无信号" in title


def test_resolve_tickers():
    assert resolve_tickers({"tickers": "NVDA, AMD"}) == ["NVDA", "AMD"]


def test_build_history_row():
    p = CalendarSpreadPlan(
        ticker="PLTR", close=25.0, rv_pct=50.0, iv_pct=65.0, iv_rank=0.3, er=0.25,
        can_open=True, call_strike=27.0, put_strike=23.0,
        debit_per_share=0.5, debit_per_contract=50.0, debit_pct_account=0.5,
        profit_zone_pct=6.0, theta_est_contract=3.0,
        short_d=14, long_d=21, hold_trading_days=5, max_contracts=2,
    )
    row = build_history_row({"plans": [p], "errors": [], "config": {"iv_pct_max": 0.4}})
    assert row["首选"] == "PLTR"
    assert row["可开数"] == 1


def test_build_playbook_lists_open():
    p = CalendarSpreadPlan(
        ticker="QQQ", close=500.0, rv_pct=20.0, iv_pct=26.0, iv_rank=0.2, er=0.2,
        can_open=True, call_strike=510.0, put_strike=490.0,
        debit_per_share=2.0, debit_per_contract=200.0, debit_pct_account=2.0,
        profit_zone_pct=3.0, theta_est_contract=10.0,
        short_d=14, long_d=21, hold_trading_days=5, max_contracts=1,
        playbook=["步骤1"],
    )
    lines = build_playbook({"plans": [p], "errors": [], "config": {"iv_pct_max": 0.4}})
    assert any("可开" in ln for ln in lines)
