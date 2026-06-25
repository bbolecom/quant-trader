#!/usr/bin/env python3
"""多空组合策略 · 每日扫描。

Extreme20 L1/S1 + Flow U_S2/D_S2 · 质量分排序 · 高胜率过滤

用法:
    python3 longshort_combo_daily.py
    python3 longshort_combo_daily.py --dry-run
    python3 longshort_combo_daily.py -c longshort_combo_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.extreme20_strategy import market_regime
from quant.longshort_combo_strategy import config_from_dict, load_rules, scan_live
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "longshort_combo_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def run_scan(cfg_dict: dict) -> dict:
    cfg = config_from_dict(cfg_dict)
    reg = market_regime()
    picks = scan_live(cfg)
    signals = picks.to_dict(orient="records") if not picks.empty else []
    actionable = [s for s in signals if s.get("信号") == "可开仓"]
    rules = load_rules()

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "title": "多空组合策略",
        "strategy": cfg.name,
        "market": reg,
        "backtest": rules.get("best_metrics") or {},
        "scan_stats": {
            "扫描命中": len(signals),
            "可开仓": len(actionable),
            "多头": sum(1 for s in actionable if s.get("side") == "long"),
            "空头": sum(1 for s in actionable if s.get("side") == "short"),
            "e20腿": sum(1 for s in actionable if s.get("leg") == "e20"),
            "flow腿": sum(1 for s in actionable if s.get("leg") == "flow"),
            "均质量分": round(
                sum(float(s.get("质量分", 0)) for s in actionable) / max(len(actionable), 1), 3
            ),
        },
        "signals": signals,
        "picks": actionable,
    }


def format_lines(plan: dict) -> list[str]:
    reg = plan.get("market") or {}
    st = plan.get("scan_stats") or {}
    bt = plan.get("backtest") or {}
    lines = [
        f"多空组合 · {plan['date']}",
        f"  {reg.get('regime', '')}  SPY {reg.get('SPY')} / MA20 {reg.get('MA20')}",
        f"  可开仓 {st.get('可开仓', 0)} · 多{st.get('多头', 0)}/空{st.get('空头', 0)} · "
        f"E20={st.get('e20腿', 0)} Flow={st.get('flow腿', 0)} · 均分{st.get('均质量分', 0)}",
        "",
    ]
    picks = plan.get("picks") or []
    if not picks:
        lines.append("  今日无高质量多空信号（正常空仓）")
    for s in picks:
        arrow = "📈" if s.get("side") == "long" else "📉"
        leg = s.get("leg", "?")
        lines.append(
            f"  {arrow} [{s.get('策略ID', leg)}] {s.get('代码')} "
            f"分{s.get('质量分', 0)} · {s.get('方向', s.get('side', ''))}"
        )
        reason = str(s.get("依据", s.get("选股理由", "")))[:70]
        if reason:
            lines.append(f"     {reason}")
    if bt.get("胜率"):
        lines.append("")
        lines.append(
            f"  回测: {bt.get('交易次数')}笔 胜{bt.get('胜率', 0):.0%} "
            f"年化{bt.get('年化', 0):+.0%} OOS胜{bt.get('OOS胜率', 0):.0%}"
        )
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    today_json = ROOT / out.get("today_json", "research/longshort_combo_today.json")
    today_csv = ROOT / out.get("today_csv", "research/longshort_combo_today.csv")
    hist = ROOT / out.get("history_csv", "longshort_combo_history.csv")
    today_json.parent.mkdir(parents=True, exist_ok=True)
    today_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if plan.get("signals"):
        pd.DataFrame(plan["signals"]).to_csv(today_csv, index=False, encoding="utf-8-sig")

    st = plan.get("scan_stats") or {}
    row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "日期": plan["date"],
        "大盘": (plan.get("market") or {}).get("regime"),
        "可开仓": st.get("可开仓", 0),
        "多头": st.get("多头", 0),
        "空头": st.get("空头", 0),
        "均质量分": st.get("均质量分", 0),
    }
    if hist.exists():
        pd.concat([pd.read_csv(hist, encoding="utf-8-sig"), pd.DataFrame([row])]).to_csv(
            hist, index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame([row]).to_csv(hist, index=False, encoding="utf-8-sig")

    ios = ROOT / "ios" / "Resources" / "longshort_combo_today.json"
    ios.parent.mkdir(parents=True, exist_ok=True)
    ios.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="多空组合策略每日扫描")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plan = run_scan(cfg)
    save_outputs(plan, cfg)
    print("\n".join(format_lines(plan)))

    notify = cfg.get("notify") or {}
    if not args.dry_run and notify.get("desktop", True):
        n = plan["scan_stats"]["可开仓"]
        if n > 0 or not notify.get("only_on_signal", True):
            desktop_notify(
                f"多空组合 {plan['date']}",
                f"可开仓 {n} · 多{plan['scan_stats']['多头']}/空{plan['scan_stats']['空头']}",
            )


if __name__ == "__main__":
    main()
