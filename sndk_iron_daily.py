#!/usr/bin/env python3
"""5×$10k 舰队每日推送 · mixed_balanced（真实期权链）。

策略组合（回测 mixed_balanced / etf3_csp2）：
  · SNDK → Put 信用价差（顺势卖下方，无 Call 腿）
  · INTC → 廉价 CSP（δ≈OTM10%，月到期）
  · QQQ/SPY/IWM → 月铁鹰（低波 ETF）

全部用 yfinance 真实 bid/ask；卖腿收 bid、买腿付 ask。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.option_chain import (
    build_csp,
    build_iron_condor,
    build_put_credit_spread,
    clear_chain_cache,
)
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "sndk_iron_config.json"
HISTORY_FILE = ROOT / "sndk_iron_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {
            "tickers": ["SNDK"],
            "account_size": 10000,
            "call_otm": 0.15,
            "put_otm": 0.12,
            "width_pct": 0.02,
            "max_margin_pct": 0.25,
            "min_dte": 2,
            "max_dte": 14,
            "min_oi": 50,
            "expiry": None,
            "notify": {"desktop": True, "email": {"enabled": False}},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _spot(sym: str) -> float | None:
    try:
        import yfinance as yf
        return float(yf.Ticker(sym).fast_info["last_price"])
    except Exception:  # noqa: BLE001
        return None


def run_scan(cfg: dict) -> dict:
    clear_chain_cache()
    account = float(cfg.get("account_size", 10_000))
    tickers = cfg.get("tickers") or ["SNDK"]
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.replace(",", " ").split()]
    plans, errors = [], []
    for tk in tickers:
        spot = _spot(tk)
        if spot is None:
            errors.append(f"{tk}: 无法取现价")
            continue
        plan, why = build_iron_condor(
            tk, spot, account,
            call_otm=float(cfg.get("call_otm", 0.15)),
            put_otm=float(cfg.get("put_otm", 0.12)),
            width_pct=float(cfg.get("width_pct", 0.02)),
            max_margin_pct=float(cfg.get("max_margin_pct", 0.25)),
            risk_per_trade=float(cfg.get("max_margin_pct", 0.25)),
            min_dte=int(cfg.get("min_dte", 2)),
            max_dte=int(cfg.get("max_dte", 14)),
            min_oi=int(cfg.get("min_oi", 50)),
            expiry_override=cfg.get("expiry") or None,
        )
        if plan:
            plans.append(plan)
        else:
            errors.append(f"{tk}: {why}")
    return {"plans": plans, "errors": errors, "config": cfg}


def _build_account_plan(acc: dict, tk: str, spot: float, size: float, cfg: dict):
    """按账户 strategy 字段构建真实链方案。"""
    strat = str(acc.get("strategy", "iron_condor")).lower()
    min_dte = int(acc.get("min_dte", cfg.get("min_dte", 2)))
    max_dte = int(acc.get("max_dte", cfg.get("max_dte", 14)))
    min_oi = int(acc.get("min_oi", cfg.get("min_oi", 50)))
    expiry = acc.get("expiry") or cfg.get("expiry") or None
    margin_pct = float(acc.get("max_margin_pct", cfg.get("max_margin_pct", 0.25)))

    if strat in ("put_credit", "put_spread", "pcs"):
        return build_put_credit_spread(
            tk, spot, size,
            put_otm=float(acc.get("put_otm", cfg.get("put_otm", 0.12))),
            width_pct=float(acc.get("width_pct", cfg.get("width_pct", 0.02))),
            max_margin_pct=margin_pct,
            risk_per_trade=margin_pct,
            min_dte=min_dte, max_dte=max_dte, min_oi=min_oi,
            expiry_override=expiry,
        )
    if strat == "csp":
        return build_csp(
            tk, spot, size,
            otm=float(acc.get("otm", 0.10)),
            max_collateral_pct=margin_pct,
            min_dte=min_dte, max_dte=max_dte, min_oi=min_oi,
        )
    return build_iron_condor(
        tk, spot, size,
        call_otm=float(acc.get("call_otm", cfg.get("call_otm", 0.10))),
        put_otm=float(acc.get("put_otm", cfg.get("put_otm", 0.10))),
        width_pct=float(acc.get("width_pct", cfg.get("width_pct", 0.02))),
        max_margin_pct=margin_pct,
        risk_per_trade=margin_pct,
        min_dte=min_dte, max_dte=max_dte, min_oi=min_oi,
        expiry_override=expiry,
    )


def _strategy_label(p) -> str:
    if p is None:
        return ""
    return {"iron_condor": "铁鹰", "put_credit": "Put价差", "csp": "CSP"}.get(p.structure, p.structure)


def _plan_structure_line(p) -> str:
    if p.structure == "iron_condor":
        cs, cl, ps, pl = p.legs
        return f"卖C${cs.strike:g}/买C${cl.strike:g} + 卖P${ps.strike:g}/买P${pl.strike:g}"
    if p.structure == "put_credit":
        ps, pl = p.legs
        return f"卖P${ps.strike:g}/买P${pl.strike:g}"
    if p.structure == "csp":
        leg = p.legs[0]
        return f"卖P${leg.strike:g} CSP"
    return p.note or p.structure


def format_pick_action(plan) -> str:
    """每日选股「策略动作」：策略名 + 四腿/两腿结构 + 盈利区间。"""
    if plan is None:
        return ""
    strat = _strategy_label(plan)
    struct = _plan_structure_line(plan)
    zone, _, _ = _profit_zone(plan)
    return f"{strat} · {struct} · 盈利区{zone}"


def format_pick_reason(plan) -> str:
    """每日选股「选股理由」：现价、到期、张数、收租与保证金。"""
    if plan is None:
        return ""
    n_ct = plan.contracts
    credit = plan.net_per_contract * n_ct
    margin = plan.collateral * n_ct
    return (
        f"现价${plan.spot:,.2f} · {plan.expiry} DTE{plan.dte} · "
        f"{n_ct}张 · 收${credit:,.0f} · 保证金${margin:,.0f}"
    )


def _profit_zone(p) -> tuple[str, float | None, float | None]:
    if p.structure == "iron_condor":
        cs, _, ps, _ = p.legs
        up = (cs.strike / p.spot - 1) * 100
        dn = (ps.strike / p.spot - 1) * 100
        return f"${ps.strike:g}~${cs.strike:g} ({dn:+.0f}%/{up:+.0f}%)", dn, up
    if p.structure == "put_credit":
        ps, _ = p.legs
        dn = (1 - ps.strike / p.spot) * 100
        return f"股价>${ps.strike:g} (-{dn:.0f}%保护)", dn, None
    if p.structure == "csp":
        leg = p.legs[0]
        dn = (1 - leg.strike / p.spot) * 100
        return f"股价>${leg.strike:g} (-{dn:.0f}%)", dn, None
    return p.note or "", None, None


def run_fleet(cfg: dict) -> dict:
    """舰队模式：每户按 strategy 跑真实链（Put价差 / CSP / 铁鹰）。"""
    clear_chain_cache()
    fleet = cfg.get("fleet") or {}
    size = float(fleet.get("account_size", cfg.get("account_size", 10_000)))
    accounts = fleet.get("accounts") or []
    profile = fleet.get("description") or cfg.get("profile") or "mixed_balanced"
    rows, errors = [], []
    spot_cache: dict[str, float | None] = {}
    for acc in accounts:
        tk = str(acc.get("ticker", "")).strip().upper()
        label = acc.get("label", tk)
        strat = str(acc.get("strategy", "iron_condor")).lower()
        margin_pct = float(acc.get("max_margin_pct", cfg.get("max_margin_pct", 0.25)))
        if not tk:
            continue
        if tk not in spot_cache:
            spot_cache[tk] = _spot(tk)
        spot = spot_cache[tk]
        if spot is None:
            errors.append(f"{label}/{tk}: 无法取现价")
            rows.append({"label": label, "ticker": tk, "strategy": strat, "plan": None})
            continue
        plan, why = _build_account_plan(acc, tk, spot, size, cfg)
        if plan is not None and plan.contracts < 1:
            why = why or f"账户 ${size:,.0f} 按 {margin_pct:.0%} 上限不够 1 张"
            errors.append(f"{label}/{tk}: {why}")
            plan = None
        elif plan is None:
            errors.append(f"{label}/{tk}: {why}")
        rows.append({
            "label": label, "ticker": tk, "strategy": strat,
            "plan": plan, "size": size, "note": acc.get("note", ""),
        })
    return {
        "fleet_rows": rows, "errors": errors, "config": cfg,
        "account_size": size, "profile": profile,
    }


def fleet_lines(result: dict) -> list[str]:
    rows = result.get("fleet_rows") or []
    size = result.get("account_size", 10_000)
    n = len(rows)
    prof = result.get("profile") or "mixed_balanced"
    lines = [f"5×${size:,.0f} 舰队 · {prof} · 总本金 ${n * size:,.0f}", ""]
    tot_credit = tot_margin = 0.0
    open_n = 0
    for r in rows:
        p = r["plan"]
        strat = _strategy_label(p) if p else r.get("strategy", "")
        if p is None:
            lines.append(f"  [{r['label']}] {r['ticker']}({strat}): ⏸ 无可行")
            continue
        open_n += 1
        n_ct = p.contracts
        credit = p.net_per_contract * n_ct
        margin = p.collateral * n_ct
        tot_credit += credit
        tot_margin += margin
        zone, _, _ = _profit_zone(p)
        lines.append(
            f"  [{r['label']}] {r['ticker']}·{strat} ${p.spot:,.0f} · {p.expiry}({p.dte}天) · {n_ct}张"
        )
        lines.append(f"      {_plan_structure_line(p)}")
        lines.append(
            f"      收${credit:,.0f} 保证金${margin:,.0f}({margin/size:.0%}) ROI{_roi(p):.0%} · {zone}"
        )
    cash_pct = max(0.0, 1.0 - tot_margin / (n * size)) if n * size else 0
    lines.append("")
    lines.append(
        f"  舰队合计：{open_n}/{n} 户可开 · 收租≈${tot_credit:,.0f} · "
        f"占用≈${tot_margin:,.0f}({tot_margin/(n*size):.0%}) · 现金纪律≈{cash_pct:.0%}"
    )
    for e in result.get("errors") or []:
        lines.append(f"  ⚠ {e}")
    return lines


def _roi(plan) -> float:
    return plan.max_profit / plan.collateral if plan.collateral else 0.0


def build_lines(result: dict) -> list[str]:
    lines: list[str] = []
    for p in result.get("plans") or []:
        cs, cl, ps, pl = p.legs
        up = (cs.strike / p.spot - 1) * 100
        dn = (ps.strike / p.spot - 1) * 100
        lines.append(
            f"{p.ticker} 现价 ${p.spot:,.0f} · 到期 {p.expiry}({p.dte}天) 真实链铁鹰："
        )
        lines.append(
            f"  卖C${cs.strike:g}/买C${cl.strike:g} + 卖P${ps.strike:g}/买P${pl.strike:g}"
        )
        lines.append(
            f"  真实收 ${p.net_per_contract:,.0f}/张 · 保证金 ${p.collateral:,.0f} · "
            f"ROI {_roi(p):.1%} · 建议 {p.contracts} 张"
        )
        lines.append(
            f"  盈利区间 ${ps.strike:g}~${cs.strike:g} ({dn:+.0f}%/{up:+.0f}%) · "
            f"最大亏 ${p.max_loss:,.0f}/张 · 50%止盈≈${p.max_profit/2:,.0f}"
        )
        lines.append(
            f"  流动性 OI 卖C{cs.oi}/卖P{ps.oi} · 卖C IV{cs.iv:.0%}"
        )
    for e in result.get("errors") or []:
        lines.append(f"⚠ {e}")
    if not result.get("plans"):
        lines.append("今日无可行铁鹰（流动性不足 / 净权利金≤0 / 无近月到期）")
    return lines


def format_notification(result: dict) -> tuple[str, str]:
    plans = result.get("plans") or []
    if not plans:
        return "⏸ SNDK 铁鹰 · 今日无信号", "；".join(result.get("errors") or ["无可行结构"])[:200]
    p = plans[0]
    cs, cl, ps, pl = p.legs
    title = f"🦅 {p.ticker} 真实铁鹰可开"
    body = (
        f"卖C${cs.strike:g}/买C${cl.strike:g}+卖P${ps.strike:g}/买P${pl.strike:g} "
        f"收${p.net_per_contract:,.0f} ROI{_roi(p):.0%} 区间${ps.strike:g}~${cs.strike:g}"
    )
    return title, body[:200]


def append_history(result: dict) -> None:
    if result.get("fleet_rows") is not None:
        rows = result["fleet_rows"]
        open_rows = [r for r in rows if r["plan"] is not None]
        tot_credit = sum(r["plan"].net_per_contract * r["plan"].contracts for r in open_rows)
        tot_margin = sum(r["plan"].collateral * r["plan"].contracts for r in open_rows)
        row = {
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "模式": result.get("profile") or "mixed_balanced",
            "可开户数": f"{len(open_rows)}/{len(rows)}",
            "合计收租": round(tot_credit, 0),
            "合计保证金": round(tot_margin, 0),
            "清单": " | ".join(fleet_lines(result)),
            "错误": "；".join(result.get("errors") or []),
        }
        df = pd.DataFrame([row])
        header = not HISTORY_FILE.exists()
        df.to_csv(HISTORY_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")
        return
    plans = result.get("plans") or []
    p = plans[0] if plans else None
    row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "标的": ",".join(x.ticker for x in plans),
        "首选": p.ticker if p else "",
        "现价": round(p.spot, 0) if p else None,
        "到期": p.expiry if p else "",
        "结构": (f"C{p.legs[0].strike:g}/{p.legs[1].strike:g} "
                 f"P{p.legs[2].strike:g}/{p.legs[3].strike:g}") if p else "",
        "真实收/张": p.net_per_contract if p else None,
        "保证金": p.collateral if p else None,
        "建议张数": p.contracts if p else None,
        "清单": " | ".join(build_lines(result)),
        "错误": "；".join(result.get("errors") or []),
    }
    df = pd.DataFrame([row])
    header = not HISTORY_FILE.exists()
    df.to_csv(HISTORY_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")


def fleet_to_dataframe(result: dict) -> pd.DataFrame:
    """舰队结果 → 表格（供 app.py / 报表展示）。"""
    rows_out: list[dict] = []
    size = float(result.get("account_size", 10_000))
    for r in result.get("fleet_rows") or []:
        p = r.get("plan")
        strat = _strategy_label(p) if p else r.get("strategy", "")
        if p is None:
            rows_out.append({
                "账户": r.get("label", ""),
                "策略": strat,
                "代码": r.get("ticker", ""),
                "可开": "⏸",
                "现价": None,
                "到期": "",
                "DTE": None,
                "结构": "",
                "真实收$": None,
                "保证金$": None,
                "张数": 0,
                "ROI": None,
                "盈利区间": "",
                "50%止盈$": None,
            })
            continue
        n_ct = p.contracts
        credit = p.net_per_contract * n_ct
        margin = p.collateral * n_ct
        zone, _, _ = _profit_zone(p)
        struct = _plan_structure_line(p).replace("卖", "").replace("买", "")
        row = {
            "账户": r.get("label", ""),
            "策略": strat,
            "代码": p.ticker,
            "可开": "✅",
            "现价": round(p.spot, 2),
            "到期": p.expiry,
            "DTE": p.dte,
            "结构": struct,
            "真实收$": round(credit, 0),
            "保证金$": round(margin, 0),
            "占用%": round(margin / size * 100, 0) if size else None,
            "张数": n_ct,
            "ROI": round(_roi(p) * 100, 1),
            "盈利区间": zone,
            "50%止盈$": round(credit / 2, 0),
        }
        if p.structure == "iron_condor":
            cs, _, ps, _ = p.legs
            row["卖C_OI"] = cs.oi
            row["卖P_OI"] = ps.oi
        elif p.structure == "put_credit":
            ps, _ = p.legs
            row["卖P_OI"] = ps.oi
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def fleet_summary_metrics(result: dict) -> dict:
    rows = result.get("fleet_rows") or []
    size = float(result.get("account_size", 10_000))
    open_rows = [r for r in rows if r.get("plan") is not None]
    tot_credit = sum(r["plan"].net_per_contract * r["plan"].contracts for r in open_rows)
    tot_margin = sum(r["plan"].collateral * r["plan"].contracts for r in open_rows)
    n = len(rows) or 1
    return {
        "open_count": len(open_rows),
        "total_accounts": n,
        "total_credit": tot_credit,
        "total_margin": tot_margin,
        "fleet_capital": n * size,
        "margin_pct": tot_margin / (n * size) if n * size else 0,
        "cash_pct": max(0.0, 1.0 - tot_margin / (n * size)) if n * size else 1.0,
        "roi": tot_credit / tot_margin if tot_margin else 0,
    }


def fleet_notification(result: dict) -> tuple[str, str]:
    rows = result.get("fleet_rows") or []
    open_rows = [r for r in rows if r.get("plan") is not None]
    if not open_rows:
        return "⏸ 舰队 · 今日无信号", "；".join(result.get("errors") or ["无可行"])[:200]
    tot = sum(r["plan"].net_per_contract * r["plan"].contracts for r in open_rows)
    tickers = "/".join(dict.fromkeys(r["ticker"] for r in open_rows))
    return (
        f"🦅 舰队 {len(open_rows)}/{len(rows)} 户可开",
        f"{tickers} mixed_balanced · 合计收租≈${tot:,.0f}"[:200],
    )


def write_today_json(result: dict, cfg: dict) -> Path:
    """写入云端 JSON feed（App + GitHub raw）。"""
    out = cfg.get("outputs") or {}
    tj = ROOT / out.get("today_json", "research/sndk_iron_today.json")
    tj.parent.mkdir(parents=True, exist_ok=True)
    if result.get("fleet_rows") is not None:
        df = fleet_to_dataframe(result)
        rows = df.to_dict(orient="records")
        metrics = fleet_summary_metrics(result)
        actionable = int(metrics.get("open_count", 0))
        summary_block = metrics
    else:
        rows = []
        for p in result.get("plans") or []:
            rows.append({
                "代码": p.ticker,
                "可开": "✅",
                "现价": round(p.spot, 2),
                "到期": p.expiry,
                "DTE": p.dte,
                "结构": _plan_structure_line(p),
                "真实收$": round(p.net_per_contract * p.contracts, 0),
                "保证金$": round(p.collateral * p.contracts, 0),
                "张数": p.contracts,
                "ROI": round(_roi(p) * 100, 1),
                "盈利区间": _profit_zone(p)[0],
            })
        actionable = len(rows)
        summary_block = None
    doc = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy_id": "sndk_iron",
        "title": "SNDK铁鹰收租",
        "profile": result.get("profile") or cfg.get("profile"),
        "scan_stats": {
            "可开仓": actionable,
            "合计": len(rows),
            "观望": max(0, len(rows) - actionable),
        },
        "summary": summary_block,
        "rows": rows,
        "picks": [r for r in rows if r.get("可开") == "✅"],
        "errors": result.get("errors") or [],
    }
    payload = json.dumps(doc, ensure_ascii=False, indent=2)
    tj.write_text(payload, encoding="utf-8")
    ios = out.get("ios_bundle")
    if ios:
        ip = ROOT / ios
        ip.parent.mkdir(parents=True, exist_ok=True)
        ip.write_text(payload, encoding="utf-8")
    return tj


def print_report(result: dict) -> None:
    print("=" * 78)
    mode = "舰队" if result.get("fleet_rows") is not None else "单标的"
    prof = result.get("profile") or "mixed_balanced"
    print(f"期权舰队[{mode}]·{prof} · {datetime.now():%Y-%m-%d %H:%M} · yfinance 实时(延迟~15min)")
    print("=" * 78)
    lines = fleet_lines(result) if result.get("fleet_rows") is not None else build_lines(result)
    for line in lines:
        print(f"  {line}")
    print("\n纪律：50%止盈 · SPY>MA50 · 财报前不开 · SNDK只卖Put不卖Call · 每户留≥55%现金")


def main() -> None:
    ap = argparse.ArgumentParser(description="SNDK 真实链铁鹰每日推送")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--single", action="store_true", help="强制单标的模式（忽略 fleet）")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    fleet_on = bool((cfg.get("fleet") or {}).get("enabled")) and not args.single
    result = run_fleet(cfg) if fleet_on else run_scan(cfg)
    write_today_json(result, cfg)
    print_report(result)
    append_history(result)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify = cfg.get("notify", {})
    title, body = fleet_notification(result) if fleet_on else format_notification(result)
    body_lines = fleet_lines(result) if fleet_on else build_lines(result)
    if notify.get("desktop"):
        desktop_notify(title, body)
    if notify.get("email", {}).get("enabled"):
        email_notify(notify["email"], title, "\n".join(body_lines) or body)


if __name__ == "__main__":
    main()
