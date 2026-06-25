#!/usr/bin/env python3
"""暴涨/暴跌 ≥20% 事件策略 · 每日扫描（L1/S1/L2/S2）。

回测组合 L1+S1：胜率59% · OOS 70% · 年化+60%

用法:
    python extreme20_daily.py
    python extreme20_daily.py --dry-run
    python extreme20_daily.py -c extreme20_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.extreme20_strategy import Extreme20Config, config_from_dict, market_regime, scan_live
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "extreme20_config.json"
PLAYBOOK = ROOT / "research" / "surge20_refined_playbook.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def run_scan(cfg_dict: dict) -> dict:
    cfg = config_from_dict(cfg_dict)
    reg = market_regime()
    picks = scan_live(cfg, screen_count=int(cfg_dict.get("screen_count", 300)))
    signals = picks.to_dict(orient="records") if not picks.empty else []

    by_id: dict[str, list] = {k: [] for k in ("L1", "S1", "L2", "S2")}
    for s in signals:
        sid = str(s.get("策略ID", ""))
        if sid in by_id:
            by_id[sid].append(s)

    actionable = [s for s in signals if s.get("信号") == "可开仓"]
    playbook = {}
    if PLAYBOOK.exists():
        try:
            playbook = json.loads(PLAYBOOK.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            playbook = {}

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "title": "暴涨暴跌20%事件策略",
        "market": reg,
        "config": {
            "threshold_pct": cfg.threshold_pct,
            "min_price": cfg.min_price,
            "min_dvol_m": cfg.min_dvol_m,
            "enabled": list(cfg.enabled),
        },
        "backtest_combo": (playbook.get("combo_L1_S1") or {}).get("backtest", {}),
        "scan_stats": {
            "扫描命中": len(signals),
            "可开仓": len(actionable),
            "多头": sum(1 for s in actionable if s.get("side") == "long"),
            "空头": sum(1 for s in actionable if s.get("side") == "short"),
            "L1": len(by_id["L1"]),
            "S1": len(by_id["S1"]),
            "L2": len(by_id["L2"]),
            "S2": len(by_id["S2"]),
            "combo_mode": bool(cfg_dict.get("combo_mode", True)),
        },
        "signals": signals,
        "picks": actionable,
    }


def format_lines(plan: dict) -> list[str]:
    reg = plan.get("market") or {}
    st = plan.get("scan_stats") or {}
    lines = [
        f"暴涨暴跌20%策略 · {plan['date']}",
        f"  {reg.get('regime', '')}  SPY {reg.get('SPY')} / MA20 {reg.get('MA20')}",
        f"  可开仓 {st.get('可开仓', 0)} · 多{st.get('多头', 0)}/空{st.get('空头', 0)} · "
        f"L1={st.get('L1')} S1={st.get('S1')} L2={st.get('L2')} S2={st.get('S2')}",
        "",
    ]
    picks = plan.get("picks") or []
    if not picks:
        lines.append("  今日无命中信号（正常空仓，约85%交易日无操作）")
    for s in picks:
        arrow = "📈" if s.get("side") == "long" else "📉"
        pct = s.get("涨幅%") or s.get("跌幅%")
        lines.append(
            f"  {arrow} [{s['策略ID']}] {s['代码']} ${s['最新价']} "
            f"{s.get('方向')} · {pct}% · {s.get('持有', '')}"
        )
        lines.append(
            f"     止损${s.get('止损价≈')} 止盈${s.get('止盈价≈')} · {s.get('依据', '')[:60]}"
        )
    combo = plan.get("backtest_combo") or {}
    if combo.get("胜率"):
        lines.append("")
        lines.append(
            f"  组合回测: 胜{combo.get('胜率', 0):.0%} 年化{combo.get('年化', 0):+.0%} "
            f"OOS胜{combo.get('OOS胜率', 0):.0%}"
        )
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    today_json = ROOT / out.get("today_json", "research/extreme20_today.json")
    today_csv = ROOT / out.get("today_csv", "research/extreme20_today.csv")
    hist = ROOT / out.get("history_csv", "extreme20_history.csv")
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
        "L1": st.get("L1", 0),
        "S1": st.get("S1", 0),
        "L2": st.get("L2", 0),
        "S2": st.get("S2", 0),
    }
    if hist.exists():
        pd.concat([pd.read_csv(hist, encoding="utf-8-sig"), pd.DataFrame([row])]).to_csv(
            hist, index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame([row]).to_csv(hist, index=False, encoding="utf-8-sig")

    ios = ROOT / "ios" / "Resources" / "extreme20_today.json"
    ios.parent.mkdir(parents=True, exist_ok=True)
    ios.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="暴涨暴跌20%事件策略每日扫描")
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
                f"Extreme20 {plan['date']}",
                f"可开仓 {n} · L1={plan['scan_stats']['L1']} S1={plan['scan_stats']['S1']}",
            )


if __name__ == "__main__":
    main()
