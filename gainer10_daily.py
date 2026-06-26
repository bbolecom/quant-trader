#!/usr/bin/env python3
"""Gainer10+ 分板块高胜率 · 每日扫描。

策略模式（gainer10_config.json · mode）：
  high_win  — 仅分板块规则 L≥60%+avg≥3 S≥80%（默认，组合胜率~75%）
  balanced  — 分板块 + 续涨A/B/S 回退（胜率~70%）
  legacy    — v2 统一 A/B/S 规则

用法：
    python gainer10_daily.py
    python gainer10_daily.py --dry-run
    python gainer10_daily.py -c gainer10_config.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.gainer10_strategy import Gainer10Config, config_from_dict, run_gainer10_scan
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "gainer10_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def cfg_from_dict(raw: dict) -> Gainer10Config:
    return config_from_dict(raw)


def format_lines(plan: dict) -> list[str]:
    reg = plan.get("market") or {}
    st = plan.get("scan_stats") or {}
    strat = plan.get("strategy") or {}
    regime = "🟢 大盘MA20上" if reg.get("站上MA20") else "🔴 大盘MA20下"
    n_long = st.get("分板块多", 0) + st.get("续涨A", 0) + st.get("续涨B", 0)
    n_short = st.get("分板块空", 0) + st.get("做空S", 0)
    lines = [
        f"Gainer10+ 分板块多空 · {plan['date']}",
        f"  {regime}  SPY {reg.get('SPY')}/MA20 {reg.get('MA20')}",
        f"  多 {n_long}（分板块{st.get('分板块多', 0)} · A{st.get('续涨A', 0)} · B{st.get('续涨B', 0)}）"
        f" · 空 {n_short}（分板块{st.get('分板块空', 0)} · S{st.get('做空S', 0)}）",
    ]
    if strat.get("active_long_sectors"):
        lines.append(f"  多头板块: {', '.join(strat['active_long_sectors'])}")
    if strat.get("active_short_sectors"):
        lines.append(f"  空头板块: {', '.join(strat['active_short_sectors'])}")
    if plan.get("note"):
        lines.append(f"  ℹ️ {plan['note']}")
    lines.append("")
    for key, emoji, title in [
        ("buy_sector", "🎯", "分板块·续涨"),
        ("buy_a", "🚀", "续涨A·科技强动量(hold20)"),
        ("buy_b", "✅", "续涨B·均衡回踩限价"),
        ("short_sector", "🔻", "分板块·做空"),
        ("short_s", "📉", "做空S·弱板块衰竭(hold10)"),
    ]:
        rows = plan.get(key) or []
        lines.append(f"【{title}】({len(rows)})")
        if not rows:
            lines.append("  无")
        for r in rows:
            lines.append(
                f"  {emoji} {r['代码']} ${r['现价']} 涨{r['涨幅_pct']}% "
                f"{r['板块']} 跳空{r['跳空_pct']}% 乖离{r['乖离20_pct']}% RSI{r['RSI']}"
            )
            lines.append(f"     → {r['动作']} | {r['规则说明']} | 历史{r['历史胜率']} {r['历史均收益']}")
            if r.get("限价入场"):
                lines.append(f"     限价 ${r['限价入场']} TP{r.get('止盈_pct')}% SL{r.get('止损_pct')}%")
        lines.append("")
    lines.append("法则: 分板块规则优先·空头高胜率板块优先·大盘MA20下暂停多头。")
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    tj = ROOT / out.get("today_json", "research/gainer10_today.json")
    tc = ROOT / out.get("today_csv", "research/gainer10_today.csv")
    hc = ROOT / out.get("history_csv", "research/gainer10_history.csv")
    tj.parent.mkdir(parents=True, exist_ok=True)
    doc = json.dumps(plan, ensure_ascii=False, indent=2)
    tj.write_text(doc, encoding="utf-8")
    ios = out.get("ios_bundle")
    if ios:
        p = ROOT / ios
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
    if plan.get("picks"):
        pd.DataFrame(plan["picks"]).to_csv(tc, index=False, encoding="utf-8-sig")
    row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "日期": plan["date"],
        "续涨A": plan["scan_stats"].get("续涨A", 0),
        "续涨B": plan["scan_stats"].get("续涨B", 0),
        "分板块多": plan["scan_stats"].get("分板块多", 0),
        "做空S": plan["scan_stats"].get("做空S", 0),
        "分板块空": plan["scan_stats"].get("分板块空", 0),
    }
    if hc.exists():
        pd.concat([pd.read_csv(hc, encoding="utf-8-sig"), pd.DataFrame([row])]).to_csv(
            hc, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([row]).to_csv(hc, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gainer10+ 动量续涨每日扫描")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = load_config(args.config)
    plan = run_gainer10_scan(cfg_from_dict(raw))
    save_outputs(plan, raw)
    print("\n".join(format_lines(plan)))

    st = plan["scan_stats"]
    n_long = st.get("分板块多", 0) + st.get("续涨A", 0) + st.get("续涨B", 0)
    n_short = st.get("分板块空", 0) + st.get("做空S", 0)
    if not args.dry_run and (raw.get("notify") or {}).get("desktop", True):
        desktop_notify(
            f"Gainer10+ {plan['date']}",
            f"多 {n_long} · 空 {n_short}",
        )

    # 手机推送：仅当有可开仓信号时才推（push_when=actionable）；always=每天都推
    if not args.dry_run:
        push_when = str(raw.get("push_when", "actionable"))
        has_action = (n_long + n_short) > 0
        if push_when == "always" or has_action:
            try:
                from quant.mobile_push import push_mobile
                longs = plan.get("buy_sector", []) + plan.get("buy_a", []) + plan.get("buy_b", [])
                shorts = plan.get("short_sector", []) + plan.get("short_s", [])
                reg = plan.get("market") or {}
                mkt = "🟢强市" if reg.get("站上MA20") else "🔴弱市(暂停追多)"
                if has_action:
                    title = f"📈涨幅榜信号 多{n_long}空{n_short} · {plan['date']}"
                    parts = []
                    if longs:
                        parts.append("多:" + "、".join(
                            f"{r['代码']}(涨{r['涨幅_pct']}%)" for r in longs[:4]))
                    if shorts:
                        parts.append("空:" + "、".join(
                            f"{r['代码']}(涨{r['涨幅_pct']}%)" for r in shorts[:4]))
                    body = f"{mkt}\n" + "\n".join(parts)
                else:
                    title = f"涨幅榜扫描 · {plan['date']}"
                    body = f"{mkt} · 命中{st.get('扫描',0)}只但无高胜率信号 → 观望"
                push_mobile(raw, title, body)
            except Exception as e:  # noqa: BLE001
                print(f"[手机推送] 异常: {e}", file=__import__('sys').stderr)


if __name__ == "__main__":
    main()
