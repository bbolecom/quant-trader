#!/usr/bin/env python3
"""暴涨80%规则 · 每日扫描。

流程：
  1. 扫描当日涨幅>15%、成交额≥5000万美元的标的 → 观察池
  2. 对观察池内 T+1~T+5 的标的检验 80% 确认规则
  3. 输出：新暴涨 / 追多确认 / 回避确认 / 待观察

用法：
    python gain15_daily.py
    python gain15_daily.py --dry-run
    python gain15_daily.py --quick
    python gain15_daily.py -c gain15_daily_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.gain15_scan import run_gain15_scan
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "gain15_daily_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def format_lines(plan: dict) -> list[str]:
    lines: list[str] = []
    mkt = plan.get("market") or {}
    stats = plan.get("scan_stats") or {}
    spy = mkt.get("SPY")
    ma20 = mkt.get("MA20")
    regime = "🟢 大盘MA20上" if mkt.get("站上MA20") else "🔴 大盘MA20下"
    lines.append(f"暴涨80%规则 · {plan['date']} · 扫描 {stats.get('universe', 0)} 只")
    if spy and ma20:
        lines.append(f"  {regime}  SPY ${spy:.2f} / MA20 ${ma20:.2f}")
    lines.append("")

    lines.append(f"【① 今日新暴涨 · 观察池】({stats.get('new_spikes', 0)} 只)")
    spikes = plan.get("new_spikes") or []
    if not spikes:
        lines.append("  今日无新暴涨入选（>15% 且 成交额≥门槛）")
    else:
        for s in spikes[:15]:
            ma = "MA20上" if s.get("站上MA20") else "MA20下"
            lines.append(
                f"  👁 {s['代码']} 涨{s['涨幅_pct']:.1f}% Top{s['gain_rank']} "
                f"${s['成交额M']:.0f}M {ma} → 等次日/3日确认"
            )
        if len(spikes) > 15:
            lines.append(f"  … 另有 {len(spikes) - 15} 只")

    lines.append("")
    lines.append("【② 追多确认 · 80%+继续暴涨】")
    buys = plan.get("buy_confirmed") or []
    if not buys:
        lines.append("  今日无追多确认")
    else:
        for b in buys:
            lines.append(
                f"  ✅ {b['代码']} [{b['规则ID']}] {b['规则']} "
                f"命中{b['历史命中率']} 历史5日均{b['历史5日均']}"
            )
            lines.append(
                f"     暴涨日 {b['暴涨日']} 涨{b['暴涨日涨幅%']:.1f}% Top{b['涨幅榜排名']} "
                f"| 次日{b.get('次日涨跌%')}% 3日{b.get('3日累计%')}%"
            )

    lines.append("")
    lines.append("【③ 回避/做空确认 · 80%+大幅回调】")
    avoids = plan.get("avoid_confirmed") or []
    if not avoids:
        lines.append("  今日无回避确认")
    else:
        for a in avoids:
            lines.append(
                f"  ⛔ {a['代码']} [{a['规则ID']}] {a['规则']} "
                f"命中{a['历史命中率']} 历史5日均{a['历史5日均']}"
            )
            lines.append(
                f"     暴涨日 {a['暴涨日']} 涨{a['暴涨日涨幅%']:.1f}% Top{a['涨幅榜排名']} "
                f"| 次日{a.get('次日涨跌%')}% 3日{a.get('3日累计%')}%"
            )

    lines.append("")
    lines.append(f"【④ 观察中 · 待确认】({len(plan.get('watching') or [])} 只)")
    watching = plan.get("watching") or []
    if not watching:
        lines.append("  观察池无待确认标的")
    else:
        for w in watching[:12]:
            hint = w.get("早期提示") or []
            hint_s = f" · {hint[0]}" if hint else ""
            lines.append(
                f"  ⏳ {w['代码']} 暴涨日{w['暴涨日']} Top{w['涨幅榜排名']} "
                f"T+{w.get('已过交易日', 0)} "
                f"次日{w.get('次日涨跌%')}% 3日{w.get('3日累计%')}{hint_s}"
            )
        if len(watching) > 12:
            lines.append(f"  … 另有 {len(watching) - 12} 只")

    lines.append("")
    lines.append("规则说明：追多需3日累计涨≥15~20%或次日涨>10%+3日涨>20%；"
                 "回避需3日累计跌≥10~20%或次日跌≥15%。")
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    today_json = ROOT / out.get("today_json", "research/gain15_daily_today.json")
    today_csv = ROOT / out.get("today_csv", "research/gain15_daily_today.csv")
    history = ROOT / out.get("history_csv", "gain15_daily_history.csv")

    today_json.parent.mkdir(parents=True, exist_ok=True)
    today_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for pool_name, key in [
        ("新暴涨", "new_spikes"),
        ("追多确认", "buy_confirmed"),
        ("回避确认", "avoid_confirmed"),
        ("观察中", "watching"),
    ]:
        for r in plan.get(key) or []:
            row = {"日期": plan["date"], "池": pool_name, **r}
            if "早期提示" in row:
                row["早期提示"] = "；".join(row["早期提示"]) if row["早期提示"] else ""
            if "全部命中" in row:
                row["全部命中"] = "、".join(row["全部命中"])
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(today_csv, index=False, encoding="utf-8-sig")

    hist_row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "日期": plan["date"],
        "新暴涨": len(plan.get("new_spikes") or []),
        "追多确认": len(plan.get("buy_confirmed") or []),
        "回避确认": len(plan.get("avoid_confirmed") or []),
        "观察中": len(plan.get("watching") or []),
    }
    if history.exists():
        pd.concat([pd.read_csv(history, encoding="utf-8-sig"), pd.DataFrame([hist_row])]).to_csv(
            history, index=False, encoding="utf-8-sig",
        )
    else:
        pd.DataFrame([hist_row]).to_csv(history, index=False, encoding="utf-8-sig")


def maybe_notify(plan: dict, cfg: dict, *, dry_run: bool) -> None:
    ncfg = cfg.get("notify") or {}
    if dry_run or not ncfg.get("desktop", True):
        return
    buys = plan.get("buy_confirmed") or []
    avoids = plan.get("avoid_confirmed") or []
    only_action = ncfg.get("only_when_action", True)
    if only_action and not buys and not avoids:
        return
    parts: list[str] = []
    if buys:
        parts.append("追多: " + ", ".join(b["代码"] for b in buys))
    if avoids:
        parts.append("回避: " + ", ".join(a["代码"] for a in avoids))
    title = "暴涨80%规则"
    body = " · ".join(parts) if parts else f"新暴涨 {len(plan.get('new_spikes') or [])} 只"
    desktop_notify(title, body)


def main() -> None:
    ap = argparse.ArgumentParser(description="暴涨80%规则每日扫描")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CFG))
    ap.add_argument("--dry-run", action="store_true", help="不写入文件/不通知")
    ap.add_argument("--quick", action="store_true", help="缩小股票池加速")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.quick:
        cfg["quick"] = True

    print("正在扫描…")
    plan = run_gain15_scan(cfg)
    for line in format_lines(plan):
        print(line)

    if not args.dry_run:
        save_outputs(plan, cfg)
        maybe_notify(plan, cfg, dry_run=False)
        out = cfg.get("outputs") or {}
        print(f"\n→ {ROOT / out.get('today_json', 'research/gain15_daily_today.json')}")
        print(f"→ {ROOT / out.get('today_csv', 'research/gain15_daily_today.csv')}")
        print(f"→ 观察池 {ROOT / cfg.get('watch_pool', 'research/gain15_watch_pool.json')}")


if __name__ == "__main__":
    main()
