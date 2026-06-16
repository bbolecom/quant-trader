"""期权策略损益（payoff）计算模块。

只做到期日(expiry)的损益结构计算与关键指标（最大盈利/亏损、盈亏平衡点），
不涉及定价模型（Black-Scholes）；权利金由用户按券商实际报价输入。

约定：
    - 每张期权合约对应 100 股（CONTRACT_MULTIPLIER）。
    - 权利金、行权价均按"每股"金额输入。
    - 多头 direction=+1，空头 direction=-1。
    - 现金流：买入支付权利金（负），卖出收取权利金（正）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100  # 每张合约股数


@dataclass
class Leg:
    """单腿头寸：期权或股票。"""

    kind: str          # "call" | "put" | "stock"
    direction: int     # +1 多 / -1 空
    quantity: float = 1.0     # 期权=合约张数；股票=股数
    strike: float = 0.0       # 期权行权价；股票为建仓价
    premium: float = 0.0      # 期权权利金（每股）；股票为 0

    def multiplier(self) -> float:
        return CONTRACT_MULTIPLIER if self.kind in ("call", "put") else 1.0

    def intrinsic(self, price: np.ndarray | float):
        p = np.asarray(price, dtype=float)
        if self.kind == "call":
            return np.maximum(p - self.strike, 0.0)
        if self.kind == "put":
            return np.maximum(self.strike - p, 0.0)
        if self.kind == "stock":
            return p - self.strike
        raise ValueError(f"未知头寸类型：{self.kind}")

    def payoff(self, price: np.ndarray | float):
        """该腿在到期价 price 下的盈亏（含权利金成本）。"""
        units = self.quantity * self.multiplier()
        if self.kind == "stock":
            return self.direction * units * (np.asarray(price, dtype=float) - self.strike)
        # 期权：到期价值 - 已付/已收权利金
        value = self.intrinsic(price) - self.premium
        return self.direction * units * value

    def cost(self) -> float:
        """建仓现金流（正=收入，负=支出）。股票按建仓市值计为支出。"""
        units = self.quantity * self.multiplier()
        if self.kind == "stock":
            return -self.direction * units * self.strike
        return -self.direction * units * self.premium


@dataclass
class StrategyResult:
    legs: list[Leg]
    prices: np.ndarray
    payoff: np.ndarray
    max_profit: float
    max_loss: float
    breakevens: list[float]
    net_cost: float          # 建仓净现金流（正=净收入/收权利金，负=净支出）


def strategy_payoff(legs: list[Leg], prices: np.ndarray) -> np.ndarray:
    """组合各腿到期损益。"""
    total = np.zeros_like(np.asarray(prices, dtype=float))
    for leg in legs:
        total = total + leg.payoff(prices)
    return total


def find_breakevens(prices: np.ndarray, payoff: np.ndarray) -> list[float]:
    """线性插值找出盈亏=0 的价格点。"""
    bes: list[float] = []
    for i in range(1, len(payoff)):
        y0, y1 = payoff[i - 1], payoff[i]
        if y0 == 0.0:
            bes.append(float(prices[i - 1]))
        elif y0 * y1 < 0:
            x0, x1 = prices[i - 1], prices[i]
            be = x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0)
            bes.append(float(be))
    # 去重（合并相近点）
    out: list[float] = []
    for b in bes:
        if not any(abs(b - o) < 1e-6 for o in out):
            out.append(round(b, 4))
    return out


def analyze(legs: list[Leg], spot: float, width: float = 0.6, n: int = 401) -> StrategyResult:
    """在 spot 上下 width 比例区间内计算损益曲线与关键指标。"""
    if not legs:
        raise ValueError("至少需要一条头寸腿。")
    lo = max(0.01, spot * (1.0 - width))
    hi = spot * (1.0 + width)
    prices = np.linspace(lo, hi, n)
    payoff = strategy_payoff(legs, prices)
    net_cost = sum(leg.cost() for leg in legs)
    return StrategyResult(
        legs=legs,
        prices=prices,
        payoff=payoff,
        max_profit=float(np.max(payoff)),
        max_loss=float(np.min(payoff)),
        breakevens=find_breakevens(prices, payoff),
        net_cost=float(net_cost),
    )


# ---------------------------------------------------------------------------
# 常见策略构造器：给定参数返回腿列表
# ---------------------------------------------------------------------------
def long_call(strike: float, premium: float, qty: float = 1.0) -> list[Leg]:
    return [Leg("call", +1, qty, strike, premium)]


def long_put(strike: float, premium: float, qty: float = 1.0) -> list[Leg]:
    return [Leg("put", +1, qty, strike, premium)]


def covered_call(spot: float, call_strike: float, call_premium: float, qty: float = 1.0) -> list[Leg]:
    """备兑开仓：持股 + 卖出认购。"""
    return [
        Leg("stock", +1, qty * CONTRACT_MULTIPLIER, spot, 0.0),
        Leg("call", -1, qty, call_strike, call_premium),
    ]


def cash_secured_put(put_strike: float, put_premium: float, qty: float = 1.0) -> list[Leg]:
    """现金担保卖出认沽。"""
    return [Leg("put", -1, qty, put_strike, put_premium)]


def bull_call_spread(low_strike: float, low_premium: float,
                     high_strike: float, high_premium: float, qty: float = 1.0) -> list[Leg]:
    """牛市认购价差：买低行权 call + 卖高行权 call。"""
    return [
        Leg("call", +1, qty, low_strike, low_premium),
        Leg("call", -1, qty, high_strike, high_premium),
    ]


def bear_put_spread(high_strike: float, high_premium: float,
                    low_strike: float, low_premium: float, qty: float = 1.0) -> list[Leg]:
    """熊市认沽价差：买高行权 put + 卖低行权 put。"""
    return [
        Leg("put", +1, qty, high_strike, high_premium),
        Leg("put", -1, qty, low_strike, low_premium),
    ]


def collar(spot: float, put_strike: float, put_premium: float,
           call_strike: float, call_premium: float, qty: float = 1.0) -> list[Leg]:
    """领口：持股 + 买保护性认沽 + 卖认购。"""
    return [
        Leg("stock", +1, qty * CONTRACT_MULTIPLIER, spot, 0.0),
        Leg("put", +1, qty, put_strike, put_premium),
        Leg("call", -1, qty, call_strike, call_premium),
    ]


def long_straddle(strike: float, call_premium: float, put_premium: float, qty: float = 1.0) -> list[Leg]:
    """买入跨式：同行权价买 call + 买 put（赌大波动）。"""
    return [
        Leg("call", +1, qty, strike, call_premium),
        Leg("put", +1, qty, strike, put_premium),
    ]


def iron_condor(put_long: float, put_long_prem: float, put_short: float, put_short_prem: float,
                call_short: float, call_short_prem: float, call_long: float, call_long_prem: float,
                qty: float = 1.0) -> list[Leg]:
    """铁鹰：卖出认沽价差 + 卖出认购价差（赌区间震荡）。"""
    return [
        Leg("put", +1, qty, put_long, put_long_prem),
        Leg("put", -1, qty, put_short, put_short_prem),
        Leg("call", -1, qty, call_short, call_short_prem),
        Leg("call", +1, qty, call_long, call_long_prem),
    ]


# 策略元信息（用于 UI 展示与说明）
STRATEGY_INFO: dict[str, dict] = {
    "买入认购 (Long Call)": {
        "view": "看涨", "risk": "有限（权利金）", "reward": "理论无限",
        "desc": "看涨且想用小成本博上涨。最大亏损=权利金，涨越多赚越多。",
    },
    "买入认沽 (Long Put)": {
        "view": "看跌", "risk": "有限（权利金）", "reward": "大（股价归零封顶）",
        "desc": "看跌且想控制风险的首选，替代做空。最大亏损=权利金，不会无限亏。",
    },
    "备兑开仓 (Covered Call)": {
        "view": "温和看涨/持股收租", "risk": "大（股票下跌）", "reward": "有限（封顶）",
        "desc": "已持股，卖认购收权利金。代价是放弃大涨；下跌只比裸持股略好。",
    },
    "现金担保认沽 (Cash-Secured Put)": {
        "view": "温和看涨/愿低位接货", "risk": "大（股价大跌）", "reward": "有限（权利金）",
        "desc": "在愿意买入的价位卖认沽收租；跌破则按行权价接货。高波动股慎用。",
    },
    "牛市认购价差 (Bull Call Spread)": {
        "view": "看涨", "risk": "有限（净权利金）", "reward": "有限（价差-成本）",
        "desc": "买低行权+卖高行权，降低成本、亏损封顶，适合高 IV 下看涨。",
    },
    "熊市认沽价差 (Bear Put Spread)": {
        "view": "看跌", "risk": "有限（净权利金）", "reward": "有限（价差-成本）",
        "desc": "买高行权+卖低行权 put，控制看跌成本，亏损封顶。",
    },
    "领口 (Collar)": {
        "view": "持股锁定区间", "risk": "有限", "reward": "有限",
        "desc": "持股+买 put 保护+卖 call 抵成本，把盈亏锁在区间内。适合锁利润。",
    },
    "买入跨式 (Long Straddle)": {
        "view": "赌大波动(方向不限)", "risk": "有限（双权利金）", "reward": "大",
        "desc": "同行权价买 call+put，大涨大跌都赚，横盘最亏。高 IV 时很贵。",
    },
    "铁鹰 (Iron Condor)": {
        "view": "赌区间震荡", "risk": "有限", "reward": "有限（净收权利金）",
        "desc": "卖出上下两个价差收租，赌不大动。单边暴走会触及最大亏损，抛物线股慎用。",
    },
}


def list_strategies() -> list[str]:
    return list(STRATEGY_INFO.keys())


def compare_results(
    strategies_legs: dict[str, list[Leg]],
    spot: float,
    width: float = 0.6,
) -> tuple[pd.DataFrame, dict[str, StrategyResult]]:
    """多策略并排对比，返回汇总表与各策略结果。"""
    results: dict[str, StrategyResult] = {}
    rows: list[dict] = []
    for name, legs in strategies_legs.items():
        res = analyze(legs, spot, width=width)
        results[name] = res
        rr = (res.max_profit / abs(res.max_loss)) if res.max_loss < 0 and res.max_profit < 5e7 else None
        rows.append({
            "策略": name,
            "最大盈利": res.max_profit,
            "最大亏损": res.max_loss,
            "盈亏平衡": " / ".join(f"${b:,.0f}" for b in res.breakevens) if res.breakevens else "-",
            "建仓现金流": res.net_cost,
            "盈亏比": rr,
        })
    table = pd.DataFrame(rows)
    return table, results


def recommend_for_regime(
    *,
    trend_label: str,
    direction: str = "中性",
    vol_pct: float,
    owns_shares: bool = False,
    bearish_view: bool = False,
) -> list[tuple[str, str]]:
    """根据判市结果给出期权策略方向建议（文字级，非定价）。"""
    recs: list[tuple[str, str]] = []
    high_iv = vol_pct >= 60
    uptrend = trend_label == "趋势市" and direction == "上行"
    if owns_shares and uptrend:
        recs.append(("领口 (Collar)", "已持股且趋势仍向上，用领口锁利润、限制回撤。"))
        recs.append(("备兑开仓 (Covered Call)", "温和看涨时卖 call 收租，但会封顶上涨。"))
    if bearish_view:
        recs.append(("熊市认沽价差 (Bear Put Spread)", "看跌但想控风险，比直接做空更安全。"))
        recs.append(("买入认沽 (Long Put)", "强看跌且接受权利金成本，最大亏损=权利金。"))
    elif uptrend and not owns_shares:
        recs.append(("牛市认购价差 (Bull Call Spread)", "看涨且 IV 高时，价差比裸买 call 更省。"))
    if high_iv and not bearish_view and trend_label == "震荡市":
        recs.append(("铁鹰 (Iron Condor)", "仅当预期横盘；高波动抛物线股慎用。"))
    if not recs:
        recs.append(("买入跨式 (Long Straddle)", "方向不明但预期大波动时可考虑（IV 高时很贵）。"))
    return recs
