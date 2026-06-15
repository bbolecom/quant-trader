"""信号扫描与本地模拟账户测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import paper, signals


def test_scan_columns(multi_data):
    table = signals.scan(multi_data, "双均线交叉")
    assert {"代码", "今日动作", "目标仓位", "昨日仓位", "最新价"}.issubset(table.columns)
    assert len(table) == len(multi_data)


def test_scan_action_logic():
    assert "买入" in paper.__doc__ or True  # 占位，确保模块可导入
    assert signals._action(0.0, 1.0).startswith("🟢")
    assert signals._action(1.0, 0.0).startswith("🟡")
    assert signals._action(0.0, -1.0).startswith("🔴")
    assert signals._action(1.0, 1.0) == "持有多头"
    assert signals._action(0.0, 0.0) == "空仓观望"


def test_paper_new_account():
    acc = paper.new_account(50_000)
    assert acc.cash == 50_000
    assert acc.initial == 50_000
    assert acc.positions == {}


def test_paper_rebalance_buys():
    acc = paper.new_account(100_000)
    prices = {"AAPL": 100.0, "MSFT": 200.0}
    trades = paper.rebalance(acc, {"AAPL": 0.5, "MSFT": 0.5}, prices, as_of="2024-01-02", fee_bps=0, slippage_bps=0)
    assert len(trades) == 2
    eq = paper.equity(acc, prices)
    assert abs(eq - 100_000) < 250  # 整数股零头留作现金
    assert acc.positions["AAPL"]["shares"] == 500


def test_paper_rebalance_idempotent():
    acc = paper.new_account(100_000)
    prices = {"AAPL": 100.0}
    paper.rebalance(acc, {"AAPL": 1.0}, prices, as_of="2024-01-02", fee_bps=0, slippage_bps=0)
    n_before = len(acc.history)
    second = paper.rebalance(acc, {"AAPL": 1.0}, prices, as_of="2024-01-03", fee_bps=0, slippage_bps=0)
    assert second == []  # 目标未变，不应再交易
    assert len(acc.history) == n_before


def test_paper_sell_on_exit():
    acc = paper.new_account(100_000)
    prices = {"AAPL": 100.0}
    paper.rebalance(acc, {"AAPL": 1.0}, prices, as_of="2024-01-02", fee_bps=0, slippage_bps=0)
    paper.rebalance(acc, {"AAPL": 0.0}, prices, as_of="2024-01-03", fee_bps=0, slippage_bps=0)
    assert acc.positions.get("AAPL", {"shares": 0})["shares"] == 0
    assert acc.cash > 99_000


def test_paper_persistence_roundtrip(tmp_path):
    acc = paper.new_account(100_000)
    prices = {"AAPL": 150.0, "MSFT": 300.0}
    paper.rebalance(acc, {"AAPL": 0.6, "MSFT": 0.4}, prices, as_of="2024-01-02")
    path = tmp_path / "acc.json"
    paper.save_account(acc, path)
    loaded = paper.load_account(path)
    assert np.isclose(paper.equity(loaded, prices), paper.equity(acc, prices))
    assert loaded.positions.keys() == acc.positions.keys()


def test_targets_from_signals():
    table = pd.DataFrame(
        [
            {"代码": "A", "目标仓位": "多头"},
            {"代码": "B", "目标仓位": "多头"},
            {"代码": "C", "目标仓位": "空仓"},
        ]
    )
    targets = paper.targets_from_signals(table)
    assert set(targets) == {"A", "B"}
    assert np.isclose(targets["A"], 0.5)
