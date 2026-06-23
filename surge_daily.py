#!/usr/bin/env python3
"""暴涨扫描 · 突破型 / 延续型 / 前兆蓄势。

A 类突破（7%~20% + BOLL 收口 + 创 20 日高）：类似 SMCI 5/20
B 类延续（20 日涨≥30% + 沿 BOLL 上轨 + WR 超买）：类似 SMCI 6/17
C 类前兆（BOLL 收口 + 贴近 20 日高 + 尚未放量）：提前 1~3 日盯盘

用法：
    python surge_daily.py
    python surge_daily.py --quick
    python surge_daily.py --ticker SMCI --history 180
    python surge_daily.py -c surge_scan_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.surge_scan import run_surge_scan
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "surge_scan_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def format_lines(plan: dict) -> list[str]:
    lines: list[str] = []
    stats = plan.get("scan_stats") or {}
    mkt = plan.get("market") or {}
    regime = "🟢 大盘MA20上" if mkt.get("站上MA20") else "🔴 大盘MA20下"
    lines.append(
        f"暴涨扫描 · {plan['date']} · 扫描 {stats.get('universe', 0)} 只 · "
        f"命中 {stats.get('total', 0)}"
    )
    if mkt.get("SPY") and mkt.get("MA20"):
        lines.append(f"  {regime}  SPY ${mkt['SPY']:.2f} / MA20 ${mkt['MA20']:.2f}")
    lines.append("")

    def _section(title: str, key: str, emoji: str) -> None:
        items = plan.get(key) or []
        lines.append(f"【{title}】({len(items)} 只)")
        if not items:
            lines.append("  无")
        else:
            for r in items[:12]:
                lines.append(
                    f"  {emoji} {r['代码']} {r['类型名']} 涨{r['涨幅_pct']:+.1f}% "
                    f"20d{r['涨幅20d_pct']:+.0f}% 量比{r['量比']:.1f} "
                    f"${r['成交额M']:.0f}M · {r['说明']}"
                )
            if len(items) > 12:
                lines.append(f"  … 另有 {len(items) - 12} 只")
        lines.append("")

    _section("A 突破型 · 7%~20% 放量突破", "breakout", "🚀")
    _section("B 延续/高潮型 · 趋势加速", "continuation", "⚡")
    _section("C 前兆蓄势 · 提前盯盘", "precursor", "👁")

    hist = plan.get("history") or []
    if hist:
        lines.append(f"【历史回溯 · {hist[0]['代码']}】({len(hist)} 个)")
        for r in hist[-15:]:
            lines.append(
                f"  📅 {r['日期']} {r['类型名']} 涨{r['涨幅_pct']:+.1f}% · {r['说明']}"
            )
        lines.append("")

    lines.append("说明：A 类适合突破跟进；B 类多为高潮区，谨慎追高；C 类等放量确认。")
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    today_json = ROOT / out.get("today_json", "research/surge_scan_today.json")
    today_csv = ROOT / out.get("today_csv", "research/surge_scan_today.csv")
    history = ROOT / out.get("history_csv", "surge_scan_history.csv")

    today_json.parent.mkdir(parents=True, exist_ok=True)
    today_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for pool_name, key in [
        ("突破型", "breakout"),
        ("延续型", "continuation"),
        ("前兆", "precursor"),
        ("历史", "history"),
    ]:
        for r in plan.get(key) or []:
            rows.append({"扫描日期": plan["date"], "池": pool_name, **r})
    df = pd.DataFrame(rows)
    df.to_csv(today_csv, index=False, encoding="utf-8-sig")

    hist_row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "日期": plan["date"],
        "突破型": len(plan.get("breakout") or []),
        "延续型": len(plan.get("continuation") or []),
        "前兆": len(plan.get("precursor") or []),
    }
    if history.exists():
        pd.concat([pd.read_csv(history, encoding="utf-8-sig"), pd.DataFrame([hist_row])]).to_csv(
            history, index=False, encoding="utf-8-sig",
        )
    else:
        pd.DataFrame([hist_row]).to_csv(history, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="暴涨扫描 · 突破/延续/前兆")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--dry-run", action="store_true", help="不发送桌面通知")
    parser.add_argument("--quick", action="store_true", help="缩小股票池加速")
    parser.add_argument("--ticker", type=str, default="", help="单票历史回溯")
    parser.add_argument("--history", type=int, default=0, help="回溯天数（需配合 --ticker）")
    parser.add_argument("--date", type=str, default="", help="指定扫描日期 YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg["quick"] = True
    if args.ticker:
        cfg["history_ticker"] = args.ticker.upper()
        cfg["history_days"] = args.history or 180

    plan = run_surge_scan(cfg, as_of=args.date or None)
    save_outputs(plan, cfg)

    lines = format_lines(plan)
    text = "\n".join(lines)
    print(text)

    notify_cfg = cfg.get("notify") or {}
    if not args.dry_run and notify_cfg.get("desktop", True):
        title = f"暴涨扫描 {plan['date']}"
        stats = plan.get("scan_stats") or {}
        subtitle = f"突破{stats.get('breakout', 0)} · 延续{stats.get('continuation', 0)} · 前兆{stats.get('precursor', 0)}"
        desktop_notify(title, subtitle)


if __name__ == "__main__":
    main()
