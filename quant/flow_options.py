"""资金流向选股 · 真实期权链 enrich（Put/Call 价差）。"""

from __future__ import annotations

from typing import Any

from quant.option_chain import SpreadPlan, build_bear_call_spread, build_bear_put_debit_spread


def wants_put_spread(pick: dict) -> bool:
    act = str(pick.get("策略动作", ""))
    down = str(pick.get("下跌规律", ""))
    if pick.get("信号") == "做空":
        return True
    if "Put" in act:
        return True
    for pid in ("D_OFFERING", "D_S2", "D_A3", "D_B3"):
        if pid in down:
            return True
    return False


def wants_bear_call(pick: dict) -> bool:
    return "卖Call" in str(pick.get("策略动作", ""))


def _lc(lc: dict, key: str, default: Any) -> Any:
    return lc.get(key, default)


def plan_summary(plan: SpreadPlan) -> str:
    pay = plan.net_per_contract
    verb = "收" if pay > 0 else "付"
    return (
        f"{plan.legs_label()} @{plan.expiry}({plan.dte}d) · "
        f"{verb}${abs(pay):.0f}/张 · 最大亏${plan.max_loss:.0f}"
        + (f" × {plan.contracts}张" if plan.contracts >= 1 else " · 账户不够1张")
    )


def enrich_pick_with_chain(
    pick: dict,
    account: float,
    lc: dict | None = None,
) -> dict:
    """为单条选股附加真实链报价；失败则保持观望并写明原因。"""
    lc = lc or {}
    if not lc.get("enabled", True):
        return pick
    sym = str(pick.get("代码", "")).upper()
    spot = pick.get("现价")
    if not sym or sym == "—" or spot is None:
        return pick
    try:
        spot_f = float(spot)
    except (TypeError, ValueError):
        return pick
    if spot_f <= 0:
        return pick

    row = dict(pick)
    risk = float(lc.get("risk_per_trade", 0.02))

    if wants_put_spread(row):
        plan, why = build_bear_put_debit_spread(
            sym, spot_f, account,
            otm=float(_lc(lc, "put_debit_otm", 0.0)),
            width_pct=float(_lc(lc, "put_debit_width_pct", 0.10)),
            risk_per_trade=risk,
            min_dte=int(_lc(lc, "min_dte", 2)),
            max_dte=int(_lc(lc, "max_dte", 45)),
            min_oi=int(_lc(lc, "min_open_interest", 25)),
            max_spread_pct=float(_lc(lc, "max_spread_pct", 0.60)),
        )
        if plan is None:
            row["状态"] = "观望"
            row["期权备注"] = f"真实链Put价差：{why}"
            return row
        can = plan.contracts >= 1
        row["状态"] = "可开仓" if can else "观望"
        row["方向"] = "做空"
        row["策略动作"] = "买Put价差"
        row["期权结构"] = plan.legs_label()
        row["到期"] = plan.expiry
        row["DTE"] = plan.dte
        row["建议张数"] = plan.contracts if can else 0
        row["最大亏损$"] = plan.max_loss
        row["净成本$"] = round(-plan.net_per_contract, 0)
        row["期权备注"] = plan_summary(plan)
        prev = str(row.get("选股理由", ""))
        row["选股理由"] = prev + " · " + row["期权备注"] if prev else row["期权备注"]
        return row

    if wants_bear_call(row):
        plan, why = build_bear_call_spread(
            sym, spot_f, account,
            otm=float(_lc(lc, "bear_call_otm", 0.08)),
            width_pct=float(_lc(lc, "bear_call_width_pct", 0.10)),
            risk_per_trade=risk,
            min_dte=int(_lc(lc, "min_dte", 2)),
            max_dte=int(_lc(lc, "max_dte", 45)),
            min_oi=int(_lc(lc, "min_open_interest", 25)),
            max_spread_pct=float(_lc(lc, "max_spread_pct", 0.60)),
        )
        if plan is None:
            row["状态"] = "观望"
            row["期权备注"] = f"真实链Call价差：{why}"
            return row
        can = plan.contracts >= 1
        row["状态"] = "可开仓" if can else "观望"
        row["策略动作"] = "卖Call价差"
        row["期权结构"] = plan.legs_label()
        row["到期"] = plan.expiry
        row["DTE"] = plan.dte
        row["建议张数"] = plan.contracts if can else 0
        row["最大亏损$"] = plan.max_loss
        row["净权利金$"] = round(plan.net_per_contract, 0)
        row["期权备注"] = plan_summary(plan)
        prev = str(row.get("选股理由", ""))
        row["选股理由"] = prev + " · " + row["期权备注"] if prev else row["期权备注"]
        return row

    return row


def enrich_picks_with_live_chain(
    picks: list[dict],
    account: float,
    lc: dict | None = None,
) -> list[dict]:
    if not picks or not (lc or {}).get("enabled", True):
        return picks
    return [enrich_pick_with_chain(p, account, lc) for p in picks]
