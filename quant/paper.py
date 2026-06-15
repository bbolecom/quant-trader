"""本地模拟交易账户（Paper Trading）引擎。

完全本地、无需券商账户、零资金风险。按策略信号把一个虚拟账户调仓到
目标权重（在当前所有"做多"信号的标的间等权分配），记录每一笔成交与盈亏，
账户状态以 JSON 持久化，可反复执行、长期跟踪。

设计说明：
- 仅支持做多 + 现金（不加杠杆、不做空），贴近个人模拟盘的真实约束。
- 每次 rebalance 传入"目标权重"与"最新价格"，引擎计算需要买卖的股数并撮合。
- 成交按最新收盘价 + 手续费/滑点；不足整数股时向下取整（保留零头为现金）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class Position:
    shares: float = 0.0
    avg_cost: float = 0.0


@dataclass
class Account:
    initial: float = 100_000.0
    cash: float = 100_000.0
    positions: dict[str, dict] = field(default_factory=dict)   # ticker -> {shares, avg_cost}
    history: list[dict] = field(default_factory=list)          # 成交流水
    equity_curve: list[dict] = field(default_factory=list)     # {date, equity}
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_account(initial: float = 100_000.0) -> Account:
    return Account(initial=float(initial), cash=float(initial))


def load_account(path: Path) -> Account | None:
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Account(**data)


def save_account(account: Account, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(account), f, ensure_ascii=False, indent=2)


def market_value(account: Account, prices: dict[str, float]) -> float:
    """持仓市值（按给定价格）。"""
    total = 0.0
    for t, p in account.positions.items():
        px = prices.get(t)
        if px is not None:
            total += p["shares"] * px
    return total


def equity(account: Account, prices: dict[str, float]) -> float:
    """账户总权益 = 现金 + 持仓市值。"""
    return account.cash + market_value(account, prices)


def holdings_table(account: Account, prices: dict[str, float]) -> pd.DataFrame:
    """当前持仓明细（含浮动盈亏）。"""
    rows = []
    for t, p in account.positions.items():
        if p["shares"] <= 0:
            continue
        px = prices.get(t, p["avg_cost"])
        mv = p["shares"] * px
        cost = p["shares"] * p["avg_cost"]
        rows.append(
            {
                "标的": t,
                "股数": round(p["shares"], 4),
                "成本价": round(p["avg_cost"], 2),
                "现价": round(px, 2),
                "市值": round(mv, 2),
                "浮动盈亏": round(mv - cost, 2),
                "盈亏%": (mv / cost - 1.0) if cost > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def rebalance(
    account: Account,
    targets: dict[str, float],
    prices: dict[str, float],
    as_of: str | None = None,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> list[dict]:
    """把账户调仓到目标权重。

    targets: {ticker: 目标权重}（权重之和应 <= 1；剩余留作现金）。
    prices:  {ticker: 最新价格}。
    返回本次产生的成交流水列表。
    """
    as_of = as_of or datetime.now().strftime("%Y-%m-%d")
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    total_eq = equity(account, prices)

    trades: list[dict] = []

    # 计算每个相关标的的目标股数。
    universe = set(targets) | set(account.positions)
    target_shares: dict[str, float] = {}
    for t in universe:
        px = prices.get(t)
        if px is None or px <= 0:
            # 无价格则维持原持仓。
            target_shares[t] = account.positions.get(t, {}).get("shares", 0.0)
            continue
        w = max(0.0, targets.get(t, 0.0))
        target_dollar = total_eq * w
        target_shares[t] = float(int(target_dollar / px))  # 向下取整到整数股

    # 先卖后买，保证现金充足。
    for t in sorted(universe):
        px = prices.get(t)
        if px is None or px <= 0:
            continue
        cur = account.positions.get(t, {}).get("shares", 0.0)
        delta = target_shares[t] - cur
        if delta < 0:  # 卖出
            sell_shares = -delta
            proceeds = sell_shares * px
            fee = proceeds * cost_rate
            account.cash += proceeds - fee
            _apply_fill(account, t, -sell_shares, px)
            trades.append(_record(as_of, t, "卖出", sell_shares, px, -(proceeds - fee)))

    for t in sorted(universe):
        px = prices.get(t)
        if px is None or px <= 0:
            continue
        cur = account.positions.get(t, {}).get("shares", 0.0)
        delta = target_shares[t] - cur
        if delta > 0:  # 买入
            cost = delta * px
            fee = cost * cost_rate
            need = cost + fee
            if need > account.cash + 1e-6:
                # 现金不足则按可用现金缩减买入量。
                affordable = int(account.cash / (px * (1 + cost_rate)))
                delta = max(0.0, float(affordable))
                if delta <= 0:
                    continue
                cost = delta * px
                fee = cost * cost_rate
            account.cash -= cost + fee
            _apply_fill(account, t, delta, px)
            trades.append(_record(as_of, t, "买入", delta, px, cost + fee))

    account.history.extend(trades)
    account.equity_curve.append({"date": as_of, "equity": round(equity(account, prices), 2)})
    return trades


def _apply_fill(account: Account, ticker: str, shares_delta: float, price: float) -> None:
    """更新持仓与成本价。shares_delta 正为买入、负为卖出。"""
    pos = account.positions.get(ticker, {"shares": 0.0, "avg_cost": 0.0})
    cur = pos["shares"]
    new_shares = cur + shares_delta
    if shares_delta > 0:
        # 买入：加权平均成本。
        total_cost = cur * pos["avg_cost"] + shares_delta * price
        pos["avg_cost"] = total_cost / new_shares if new_shares > 0 else 0.0
    pos["shares"] = new_shares
    if pos["shares"] <= 1e-9:
        account.positions.pop(ticker, None)
    else:
        account.positions[ticker] = pos


def _record(date: str, ticker: str, action: str, shares: float, price: float, cash_flow: float) -> dict:
    return {
        "日期": date,
        "标的": ticker,
        "动作": action,
        "股数": round(shares, 4),
        "成交价": round(price, 2),
        "现金变动": round(-cash_flow, 2),
    }


def targets_from_signals(signal_table: pd.DataFrame, max_positions: int = 0) -> dict[str, float]:
    """根据信号表生成目标权重：在所有"多头"标的间等权分配。

    signal_table 需含 "代码" 与 "目标仓位" 两列（来自 quant.signals.scan）。
    max_positions > 0 时只取前若干只（按出现顺序）。
    """
    longs = [r["代码"] for _, r in signal_table.iterrows() if r.get("目标仓位") == "多头"]
    if max_positions and max_positions > 0:
        longs = longs[:max_positions]
    if not longs:
        return {}
    w = 1.0 / len(longs)
    return {t: w for t in longs}


def summary(account: Account, prices: dict[str, float]) -> dict[str, float]:
    eq = equity(account, prices)
    return {
        "总权益": eq,
        "现金": account.cash,
        "持仓市值": market_value(account, prices),
        "累计收益率": eq / account.initial - 1.0 if account.initial > 0 else 0.0,
        "盈亏金额": eq - account.initial,
        "持仓数量": len([p for p in account.positions.values() if p["shares"] > 0]),
        "成交笔数": len(account.history),
    }
