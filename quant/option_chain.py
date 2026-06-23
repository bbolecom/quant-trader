"""真实期权链：只用券商盘上真实存在的行权价 / 到期日 / bid-ask 报价定价。

与 income_engine / vol_decay 的 Black-Scholes 估值不同，这里拿真实期权链：
  · 行权价、到期日来自交易所实际挂牌
  · 权利金按真实盘口保守成交价（卖腿用 bid，买腿用 ask）
  · 用持仓量 / 买卖价差过滤掉无法成交的合约
  · 没有可行结构 → 返回 None + 原因（宁可观望，不虚构）

数据源 yfinance（延迟约 15 分钟），结构真实可对照券商。
回测无法使用（免费数据没有历史期权链），仅供 live 选股。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

DEFAULT_MIN_OI = 25
DEFAULT_MAX_SPREAD_PCT = 0.60
DEFAULT_MIN_DTE = 2
DEFAULT_MAX_DTE = 45

CHAIN_TTL_SEC = 600  # 真实期权链缓存 10 分钟（yfinance 延迟约 15 分钟，无需更频繁）
_CHAIN_CACHE: dict[tuple, tuple[float, tuple]] = {}


def clear_chain_cache() -> None:
    _CHAIN_CACHE.clear()


@dataclass
class OptionLeg:
    action: str          # sell / buy
    right: str           # C / P
    strike: float
    bid: float
    ask: float
    mid: float
    oi: int
    volume: int
    iv: float

    def label(self) -> str:
        verb = "卖" if self.action == "sell" else "买"
        return f"{verb}{self.right}${self.strike:g}"


@dataclass
class SpreadPlan:
    ticker: str
    structure: str       # bear_call / put_credit / bear_put_debit / csp
    expiry: str
    dte: int
    spot: float
    legs: list[OptionLeg] = field(default_factory=list)
    net_per_share: float = 0.0   # +=收权利金 -=付权利金
    width: float = 0.0
    max_loss: float = 0.0        # $/张
    max_profit: float = 0.0      # $/张
    collateral: float = 0.0      # $/张（CSP 占用现金）
    contracts: int = 0
    note: str = ""

    @property
    def net_per_contract(self) -> float:
        return self.net_per_share * 100

    def legs_label(self) -> str:
        return " / ".join(leg.label() for leg in self.legs)


# ---------- 纯逻辑：从链 DataFrame 里选合约（可单测，无网络） ----------

def _leg_ok(row: pd.Series, *, min_oi: int, max_spread_pct: float) -> bool:
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return False
    mid = (bid + ask) / 2
    if mid <= 0 or (ask - bid) / mid > max_spread_pct:
        return False
    oi = float(row.get("openInterest", 0) or 0)
    if oi < min_oi:
        return False
    return True


def _mk_leg(row: pd.Series, action: str, right: str) -> OptionLeg:
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    return OptionLeg(
        action=action, right=right, strike=float(row["strike"]),
        bid=bid, ask=ask, mid=(bid + ask) / 2,
        oi=int(float(row.get("openInterest", 0) or 0)),
        volume=int(float(row.get("volume", 0) or 0)),
        iv=float(row.get("impliedVolatility", 0) or 0),
    )


def _nearest_at_or_above(df: pd.DataFrame, strike_min: float) -> pd.Series | None:
    sub = df[df["strike"] >= strike_min].sort_values("strike")
    return sub.iloc[0] if not sub.empty else None


def _nearest_at_or_below(df: pd.DataFrame, strike_max: float) -> pd.Series | None:
    sub = df[df["strike"] <= strike_max].sort_values("strike")
    return sub.iloc[-1] if not sub.empty else None


def _liquid_at_or_above(
    df: pd.DataFrame, strike_min: float, *, min_oi: int, max_spread_pct: float,
    max_hops: int = 6,
) -> pd.Series | None:
    """从 strike_min 往上找第一个流动腿（跳过稀薄中间档）。"""
    sub = df[df["strike"] >= strike_min].sort_values("strike")
    for _, row in sub.head(max_hops).iterrows():
        if _leg_ok(row, min_oi=min_oi, max_spread_pct=max_spread_pct):
            return row
    return None


def _liquid_at_or_below(
    df: pd.DataFrame, strike_max: float, *, min_oi: int, max_spread_pct: float,
    max_hops: int = 6,
) -> pd.Series | None:
    sub = df[df["strike"] <= strike_max].sort_values("strike", ascending=False)
    for _, row in sub.head(max_hops).iterrows():
        if _leg_ok(row, min_oi=min_oi, max_spread_pct=max_spread_pct):
            return row
    return None


def pick_bear_call(
    calls: pd.DataFrame, spot: float, *,
    otm: float = 0.10, width_pct: float = 0.10,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[OptionLeg | None, OptionLeg | None, str]:
    """卖看涨价差：卖近端OTM call，买更高 call。保守净权利金=短腿bid-长腿ask。"""
    if calls is None or calls.empty or spot <= 0:
        return None, None, "无 call 链"
    short_row = _liquid_at_or_above(
        calls, spot * (1 + otm), min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if short_row is None:
        return None, None, "无足够 OTM 流动卖腿"
    long_row = _liquid_at_or_above(
        calls, float(short_row["strike"]) * (1 + width_pct) + 1e-9,
        min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if long_row is None or float(long_row["strike"]) <= float(short_row["strike"]):
        return None, None, "无更高流动行权价做保护腿"
    return _mk_leg(short_row, "sell", "C"), _mk_leg(long_row, "buy", "C"), ""


def pick_put_credit(
    puts: pd.DataFrame, spot: float, *,
    otm: float = 0.10, width_pct: float = 0.10,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[OptionLeg | None, OptionLeg | None, str]:
    """卖看跌价差：卖近端OTM put，买更低 put 保护。"""
    if puts is None or puts.empty or spot <= 0:
        return None, None, "无 put 链"
    short_row = _liquid_at_or_below(
        puts, spot * (1 - otm), min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if short_row is None:
        return None, None, "无足够 OTM 流动卖腿"
    long_row = _liquid_at_or_below(
        puts, float(short_row["strike"]) * (1 - width_pct) - 1e-9,
        min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if long_row is None or float(long_row["strike"]) >= float(short_row["strike"]):
        return None, None, "无更低流动行权价做保护腿"
    return _mk_leg(short_row, "sell", "P"), _mk_leg(long_row, "buy", "P"), ""


def pick_bear_put_debit(
    puts: pd.DataFrame, spot: float, *,
    otm: float = 0.0, width_pct: float = 0.10,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[OptionLeg | None, OptionLeg | None, str]:
    """买看跌价差（做空替代）：买近 ATM put，卖更低 put 降成本。"""
    if puts is None or puts.empty or spot <= 0:
        return None, None, "无 put 链"
    long_row = _nearest_at_or_below(puts, spot * (1 - otm))
    if long_row is None:
        return None, None, "无合适行权价"
    if not _leg_ok(long_row, min_oi=min_oi, max_spread_pct=max_spread_pct):
        return None, None, f"买腿 ${long_row['strike']:g} 流动性不足"
    short_row = _nearest_at_or_below(puts, float(long_row["strike"]) * (1 - width_pct) - 1e-9)
    if short_row is None or float(short_row["strike"]) >= float(long_row["strike"]):
        return None, None, "无更低行权价"
    if not _leg_ok(short_row, min_oi=min_oi, max_spread_pct=max_spread_pct):
        return None, None, f"卖腿 ${short_row['strike']:g} 流动性不足"
    return _mk_leg(long_row, "buy", "P"), _mk_leg(short_row, "sell", "P"), ""


def pick_iron_condor(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float, *,
    call_otm: float = 0.10, put_otm: float = 0.10, width_pct: float = 0.02,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[OptionLeg | None, OptionLeg | None, OptionLeg | None, OptionLeg | None, str]:
    """铁鹰：卖看涨价差 + 卖看跌价差（偏斜可用不同 otm）。"""
    cs, cl, cwhy = pick_bear_call(
        calls, spot, otm=call_otm, width_pct=width_pct,
        min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if cs is None or cl is None:
        return None, None, None, None, f"看涨腿: {cwhy}"
    ps, pl, pwhy = pick_put_credit(
        puts, spot, otm=put_otm, width_pct=width_pct,
        min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if ps is None or pl is None:
        return None, None, None, None, f"看跌腿: {pwhy}"
    return cs, cl, ps, pl, ""


def pick_csp(
    puts: pd.DataFrame, spot: float, *,
    otm: float = 0.10,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[OptionLeg | None, str]:
    """现金担保卖 put：卖 OTM put，收 bid。"""
    if puts is None or puts.empty or spot <= 0:
        return None, "无 put 链"
    short_row = _nearest_at_or_below(puts, spot * (1 - otm))
    if short_row is None:
        return None, "无足够 OTM 行权价"
    if not _leg_ok(short_row, min_oi=min_oi, max_spread_pct=max_spread_pct):
        return None, f"行权价 ${short_row['strike']:g} 流动性不足"
    return _mk_leg(short_row, "sell", "P"), ""


def _credit_spread_plan(
    ticker: str, structure: str, expiry: str, dte: int, spot: float,
    short_leg: OptionLeg, long_leg: OptionLeg, account: float, risk_per_trade: float,
) -> SpreadPlan:
    net = short_leg.bid - long_leg.ask          # 保守：收短腿 bid，付长腿 ask
    width = abs(long_leg.strike - short_leg.strike)
    max_loss = max(0.0, (width - net)) * 100
    max_profit = net * 100
    contracts = int(max(0, (account * risk_per_trade) // max(max_loss, 1.0))) if max_loss > 0 else 0
    return SpreadPlan(
        ticker=ticker, structure=structure, expiry=expiry, dte=dte, spot=spot,
        legs=[short_leg, long_leg], net_per_share=net, width=width,
        max_loss=round(max_loss, 0), max_profit=round(max_profit, 0), contracts=contracts,
    )


def _debit_spread_plan(
    ticker: str, structure: str, expiry: str, dte: int, spot: float,
    long_leg: OptionLeg, short_leg: OptionLeg, account: float, risk_per_trade: float,
) -> SpreadPlan:
    debit = long_leg.ask - short_leg.bid        # 保守：付长腿 ask，收短腿 bid
    width = abs(long_leg.strike - short_leg.strike)
    max_loss = max(0.0, debit) * 100            # 买价差最大亏=净付出
    max_profit = max(0.0, width - debit) * 100
    contracts = int(max(0, (account * risk_per_trade) // max(max_loss, 1.0))) if max_loss > 0 else 0
    return SpreadPlan(
        ticker=ticker, structure=structure, expiry=expiry, dte=dte, spot=spot,
        legs=[long_leg, short_leg], net_per_share=-debit, width=width,
        max_loss=round(max_loss, 0), max_profit=round(max_profit, 0), contracts=contracts,
    )


# ---------- 取链（网络） ----------

def _yf_ticker(sym: str):
    import yfinance as yf
    return yf.Ticker(sym)


def fetch_chain(
    sym: str, *, min_dte: int = DEFAULT_MIN_DTE, max_dte: int = DEFAULT_MAX_DTE,
    use_cache: bool = True,
) -> tuple[str | None, int | None, pd.DataFrame, pd.DataFrame]:
    """返回 (到期日, dte, calls, puts)；无可用到期返回 (None,None,空,空)。

    带 TTL 缓存（CHAIN_TTL_SEC），避免一次扫描里对同一标的重复拉链。
    """
    key = (str(sym).upper(), int(min_dte), int(max_dte))
    now = time.time()
    if use_cache:
        hit = _CHAIN_CACHE.get(key)
        if hit is not None and now - hit[0] < CHAIN_TTL_SEC:
            return hit[1]

    def _store(result: tuple) -> tuple:
        _CHAIN_CACHE[key] = (now, result)
        return result

    try:
        t = _yf_ticker(sym)
        exps = list(t.options or [])
    except Exception:  # noqa: BLE001
        return _store((None, None, pd.DataFrame(), pd.DataFrame()))
    today = date.today()
    chosen, chosen_dte = None, None
    in_window: list[tuple[str, int]] = []
    for e in exps:
        try:
            d = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        if min_dte <= d <= max_dte:
            in_window.append((e, d))
    if in_window:
        chosen, chosen_dte = min(in_window, key=lambda x: x[1])
    else:
        rest = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days) for e in exps]
        rest = [c for c in rest if c[1] >= min_dte]
        if rest:
            chosen, chosen_dte = min(rest, key=lambda x: x[1])
    if chosen is None:
        return _store((None, None, pd.DataFrame(), pd.DataFrame()))
    try:
        ch = t.option_chain(chosen)
        return _store((chosen, chosen_dte, ch.calls.copy(), ch.puts.copy()))
    except Exception:  # noqa: BLE001
        return _store((None, None, pd.DataFrame(), pd.DataFrame()))


def build_bear_call_spread(
    sym: str, spot: float, account: float, *,
    otm: float = 0.10, width_pct: float = 0.10, risk_per_trade: float = 0.02,
    min_dte: int = DEFAULT_MIN_DTE, max_dte: int = DEFAULT_MAX_DTE,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[SpreadPlan | None, str]:
    expiry, dte, calls, _ = fetch_chain(sym, min_dte=min_dte, max_dte=max_dte)
    if expiry is None:
        return None, "无可交易到期日（无周/近月期权）"
    short_leg, long_leg, why = pick_bear_call(
        calls, spot, otm=otm, width_pct=width_pct, min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if short_leg is None or long_leg is None:
        return None, why or "无可行卖Call价差"
    plan = _credit_spread_plan(sym, "bear_call", expiry, dte, spot, short_leg, long_leg, account, risk_per_trade)
    if plan.net_per_share <= 0:
        return None, "真实盘口净权利金≤0（价差太窄/价差过宽）"
    return plan, ""


def build_bear_put_debit_spread(
    sym: str, spot: float, account: float, *,
    otm: float = 0.0, width_pct: float = 0.10, risk_per_trade: float = 0.02,
    min_dte: int = DEFAULT_MIN_DTE, max_dte: int = DEFAULT_MAX_DTE,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[SpreadPlan | None, str]:
    expiry, dte, _, puts = fetch_chain(sym, min_dte=min_dte, max_dte=max_dte)
    if expiry is None:
        return None, "无可交易到期日"
    long_leg, short_leg, why = pick_bear_put_debit(
        puts, spot, otm=otm, width_pct=width_pct, min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if long_leg is None or short_leg is None:
        return None, why or "无可行Put价差"
    plan = _debit_spread_plan(sym, "bear_put_debit", expiry, dte, spot, long_leg, short_leg, account, risk_per_trade)
    if plan.max_loss <= 0:
        return None, "真实盘口净成本≤0"
    return plan, ""


def build_put_credit_spread(
    sym: str, spot: float, account: float, *,
    put_otm: float = 0.12, width_pct: float = 0.02,
    risk_per_trade: float = 0.25, max_margin_pct: float = 0.25,
    min_dte: int = DEFAULT_MIN_DTE, max_dte: int = DEFAULT_MAX_DTE,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
    expiry_override: str | None = None,
) -> tuple[SpreadPlan | None, str]:
    """真实链 Put 信用价差：卖 OTM put + 买更低 put（顺势收租，无 Call 腿）。"""
    if expiry_override:
        from datetime import datetime as _dt
        try:
            dte = (_dt.strptime(expiry_override, "%Y-%m-%d").date() - date.today()).days
        except ValueError:
            return None, "到期日格式错误"
        try:
            t = _yf_ticker(sym)
            ch = t.option_chain(expiry_override)
            puts, expiry = ch.puts.copy(), expiry_override
        except Exception:  # noqa: BLE001
            return None, "无法取指定到期链"
    else:
        expiry, dte, _, puts = fetch_chain(sym, min_dte=min_dte, max_dte=max_dte)
    if expiry is None:
        return None, "无可交易到期日"
    ps, pl, why = pick_put_credit(
        puts, spot, otm=put_otm, width_pct=width_pct,
        min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if ps is None or pl is None:
        return None, why or "无可行 Put 价差"
    net = ps.bid - pl.ask
    if net <= 0:
        return None, "真实盘口净权利金≤0"
    width = abs(ps.strike - pl.strike)
    max_loss = max(0.0, width - net) * 100
    margin = width * 100
    budget = account * min(risk_per_trade, max_margin_pct)
    contracts = int(budget // margin) if margin > 0 else 0
    return SpreadPlan(
        ticker=sym, structure="put_credit", expiry=expiry, dte=dte, spot=spot,
        legs=[ps, pl], net_per_share=net, width=width,
        max_loss=round(max_loss, 0), max_profit=round(net * 100, 0),
        collateral=round(margin, 0), contracts=contracts,
        note=f"股价>${ps.strike:g} 盈利",
    ), ""


def build_iron_condor(
    sym: str, spot: float, account: float, *,
    call_otm: float = 0.12, put_otm: float = 0.12, width_pct: float = 0.02,
    risk_per_trade: float = 0.25, max_margin_pct: float = 0.25,
    min_dte: int = DEFAULT_MIN_DTE, max_dte: int = DEFAULT_MAX_DTE,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
    expiry_override: str | None = None,
) -> tuple[SpreadPlan | None, str]:
    """真实链铁鹰：双卖价差，保证金取较宽一侧，张数按账户保证金上限。"""
    if expiry_override:
        from datetime import datetime as _dt
        try:
            dte = (_dt.strptime(expiry_override, "%Y-%m-%d").date() - date.today()).days
        except ValueError:
            return None, "到期日格式错误"
        try:
            t = _yf_ticker(sym)
            ch = t.option_chain(expiry_override)
            calls, puts, expiry = ch.calls.copy(), ch.puts.copy(), expiry_override
        except Exception:  # noqa: BLE001
            return None, "无法取指定到期链"
    else:
        expiry, dte, calls, puts = fetch_chain(sym, min_dte=min_dte, max_dte=max_dte)
    if expiry is None:
        return None, "无可交易到期日"
    cs, cl, ps, pl, why = pick_iron_condor(
        calls, puts, spot, call_otm=call_otm, put_otm=put_otm, width_pct=width_pct,
        min_oi=min_oi, max_spread_pct=max_spread_pct,
    )
    if cs is None:
        return None, why or "无可行铁鹰"
    net = (cs.bid - cl.ask) + (ps.bid - pl.ask)
    if net <= 0:
        return None, "真实盘口净权利金≤0"
    call_w = abs(cl.strike - cs.strike)
    put_w = abs(ps.strike - pl.strike)
    width = max(call_w, put_w)
    max_loss = max(0.0, width - net) * 100
    margin = width * 100
    budget = account * min(risk_per_trade, max_margin_pct)
    contracts = int(budget // margin) if margin > 0 else 0
    return SpreadPlan(
        ticker=sym, structure="iron_condor", expiry=expiry, dte=dte, spot=spot,
        legs=[cs, cl, ps, pl], net_per_share=net, width=width,
        max_loss=round(max_loss, 0), max_profit=round(net * 100, 0),
        collateral=round(margin, 0), contracts=contracts,
        note=f"盈利区间 ${ps.strike:g}~${cs.strike:g}",
    ), ""


def build_csp(
    sym: str, spot: float, account: float, *,
    otm: float = 0.10, max_collateral_pct: float = 1.0,
    min_dte: int = DEFAULT_MIN_DTE, max_dte: int = DEFAULT_MAX_DTE,
    min_oi: int = DEFAULT_MIN_OI, max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> tuple[SpreadPlan | None, str]:
    expiry, dte, _, puts = fetch_chain(sym, min_dte=min_dte, max_dte=max_dte)
    if expiry is None:
        return None, "无可交易到期日"
    short_leg, why = pick_csp(puts, spot, otm=otm, min_oi=min_oi, max_spread_pct=max_spread_pct)
    if short_leg is None:
        return None, why or "无可行 CSP"
    collateral = short_leg.strike * 100
    budget = account * max_collateral_pct
    contracts = int(budget // collateral) if collateral > 0 else 0
    return SpreadPlan(
        ticker=sym, structure="csp", expiry=expiry, dte=dte, spot=spot,
        legs=[short_leg], net_per_share=short_leg.bid, width=0.0,
        max_loss=round((short_leg.strike - short_leg.bid) * 100, 0),
        max_profit=round(short_leg.bid * 100, 0),
        collateral=round(collateral, 0), contracts=contracts,
    ), ""
