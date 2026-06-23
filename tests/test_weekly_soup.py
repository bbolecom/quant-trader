"""weekly_soup.py 单元测试（无网络）。"""

from __future__ import annotations

import pandas as pd

from quant import decline_income as di
from weekly_soup import build_history_row, build_playbook, format_notification, resolve_tickers


def _fake_plan(ticker: str = "SNDK", can_open: bool = True) -> di.WeeklySoupPlan:
    p = di.WeeklySoupPlan(
        ticker=ticker, close=1991.0, rv_pct=91.0, iv_pct=96.0,
        ma50=1346.0, above_ma=can_open, can_open=can_open,
        dte_days=7, short_delta=0.10, width=25.0,
        short_strike=1673.0, long_strike=1648.0,
        credit_per_share=2.93, credit_per_contract=293.0,
        margin_per_contract=2500.0, max_loss_per_contract=2207.0,
        take_profit_price=1.46, zero_prob=0.87, weekly_roi_pct=11.7,
        otm_pct=16.0, one_std_move_pct=14.5, account_size=10_000.0,
        max_contracts=1, weekly_profit_if_zero=293.0, weekly_loss_if_max=2207.0,
    )
    p.playbook = ["步骤1", "步骤2"]
    return p


def test_resolve_tickers():
    assert resolve_tickers({"tickers": "SNDK, INTC"}) == ["SNDK", "INTC"]
    assert resolve_tickers({}) == ["SNDK"]


def test_format_notification_can_open():
    result = {"plans": [_fake_plan()], "errors": []}
    title, body = format_notification(result)
    assert "可喝汤" in title
    assert "1673" in body.replace(",", "")
    assert "87%" in body


def test_format_notification_pause():
    result = {"plans": [_fake_plan(can_open=False)], "errors": []}
    title, body = format_notification(result)
    assert "暂停" in title


def test_build_history_row():
    row = build_history_row({"plans": [_fake_plan()], "errors": []})
    assert row["首选"] == "SNDK"
    assert row["卖Put"] == 1673.0
    assert "步骤1" in row["执行清单"]


def test_build_playbook_includes_steps():
    lines = build_playbook({"plans": [_fake_plan()], "errors": []})
    assert any("SNDK" in ln for ln in lines)
    assert "步骤1" in lines
