#!/usr/bin/env python3
"""双日历价差每日择时提醒（IV Rank 低位才开 + 财报回避）。

功能：
    1. 读取 calendar_config.json：标的池、IV 分位上限、财报缓冲等。
    2. 扫描双日历方案，仅 IV Rank ≤ 阈值且无财报风险时标记「可开」。
    3. 弹 macOS 桌面通知（可选）并发邮件（可选）。
    4. 追加写入 calendar_history.csv。

用法：
    python calendar_daily.py
    python calendar_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant.calendar_spread import CalendarSpreadPlan, scan_calendar_plans
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "calendar_config.json"
HISTORY_FILE = ROOT / "calendar_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tickers(cfg: dict) -> list[str]:
    raw = cfg.get("tickers") or []
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    return [str(t).strip().upper() for t in raw if str(t).strip()]


def run_calendar_scan(cfg: dict) -> dict:
    end = date.today().isoformat()
    lookback = int(cfg.get("lookback_days", 400))
    start = (date.today() - timedelta(days=lookback)).isoformat()
    tickers = resolve_tickers(cfg)
    plans, errors = scan_calendar_plans(
        tickers, start, end,
        account_size=float(cfg.get("account_size", 10_000)),
        short_d=int(cfg.get("short_d", 14)),
        long_d=int(cfg.get("long_d", 21)),
        hold_trading_days=int(cfg.get("hold_trading_days", 5)),
        k_sigma=float(cfg.get("k_sigma", 1.0)),
        iv_mult=float(cfg.get("iv_mult", 1.1)),
        iv_pct_max=float(cfg.get("iv_pct_max", 0.40)),
        iv_window=int(cfg.get("iv_window", 252)),
        max_er=float(cfg.get("max_er", 0.45)),
        earnings_buffer_days=int(cfg.get("earnings_buffer_days", 7)),
        max_debit_pct=float(cfg.get("max_debit_pct", 0.50)),
    )
    return {"plans": plans, "errors": errors, "config": cfg}


def format_notification(result: dict) -> tuple[str, str]:
    plans: list[CalendarSpreadPlan] = result.get("plans") or []
    open_plans = [p for p in plans if p.can_open]
    if open_plans:
        p = open_plans[0]
        title = f"📅 {p.ticker} 双日历可开"
        body = (
            f"IV Rank {p.iv_rank:.0%}｜付 ${p.debit_per_contract:,.0f}/张 "
            f"7日θ≈+${p.theta_est_contract:,.0f} "
            f"C${p.call_strike:,.0f}/P${p.put_strike:,.0f}"
        )
        if len(open_plans) > 1:
            body += f" · 另{len(open_plans)-1}只"
        return title, body[:220]

    if plans:
        p = plans[0]
        title = "⏸ 双日历 · 今日无信号"
        reason = p.flags[0] if p.flags else "条件未满足"
        body = f"{p.ticker} {reason[:80]}"
        blocked = len([x for x in plans if not x.can_open])
        if blocked > 1:
            body += f" · {blocked}只均暂停"
        return title, body[:220]

    title = "⏸ 双日历 · 无方案"
    body = "；".join(result.get("errors") or ["扫描失败"])[:220]
    return title, body


def build_playbook(result: dict) -> list[str]:
    lines: list[str] = []
    plans: list[CalendarSpreadPlan] = result.get("plans") or []
    open_plans = [p for p in plans if p.can_open]
    if open_plans:
        lines.append(f"✅ 今日 {len(open_plans)} 只可开（IV Rank ≤ {result['config'].get('iv_pct_max', 0.4):.0%} 且无财报风险）：")
        for p in open_plans[:5]:
            lines.append(
                f"  · {p.ticker} IV Rank {p.iv_rank:.0%}｜付 ${p.debit_per_contract:,.0f} "
                f"θ≈+${p.theta_est_contract:,.0f}｜C${p.call_strike:,.0f}/P${p.put_strike:,.0f}"
            )
            for step in p.playbook[:3]:
                lines.append(f"    {step}")
    else:
        lines.append("❌ 今日无可开仓（IV 偏高 / 财报临近 / 趋势过强 / 成本过高）：")
        for p in plans[:5]:
            flag = p.flags[0] if p.flags else "未知"
            lines.append(f"  · {p.ticker} IV Rank {p.iv_rank:.0%} — {flag}")
    if result.get("errors"):
        lines.append("错误：" + "；".join(result["errors"][:3]))
    return lines


def build_history_row(result: dict) -> dict:
    plans: list[CalendarSpreadPlan] = result.get("plans") or []
    open_plans = [p for p in plans if p.can_open]
    top = open_plans[0] if open_plans else (plans[0] if plans else None)
    return {
        "日期": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "可开数": len(open_plans),
        "扫描数": len(plans),
        "首选": top.ticker if top else "",
        "IVRank": round(top.iv_rank, 2) if top else None,
        "净付$": round(top.debit_per_contract, 0) if top else None,
        "可开": top.can_open if top else False,
        "执行清单": " | ".join(build_playbook(result)[:6]),
    }


def append_history(result: dict) -> None:
    row = build_history_row(result)
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def print_report(result: dict) -> None:
    cfg = result.get("config") or {}
    print("=" * 88)
    print(f"双日历择时扫描｜IV Rank ≤ {float(cfg.get('iv_pct_max', 0.4)):.0%}｜"
          f"财报缓冲 {int(cfg.get('earnings_buffer_days', 7))} 天｜账户 ${float(cfg.get('account_size', 10000)):,.0f}")
    print("=" * 88)
    plans: list[CalendarSpreadPlan] = result.get("plans") or []
    if not plans:
        print("\n无可用方案。")
        for e in result.get("errors") or []:
            print(f"  ⚠ {e}")
        return

    open_n = sum(1 for p in plans if p.can_open)
    print(f"\n扫描 {len(plans)} 只｜可开 {open_n} 只\n")
    for p in plans:
        tag = "✅ 可开" if p.can_open else "❌ 暂停"
        print(f"[{p.ticker}] {tag}  现价 ${p.close:,.2f}  IV Rank {p.iv_rank:.0%}  ER {p.er:.2f}")
        print(f"  双日历 卖{p.short_d}/买{p.long_d}天  C${p.call_strike:,.0f} P${p.put_strike:,.0f}")
        print(f"  净付 ${p.debit_per_contract:,.0f}/张({p.debit_pct_account:.0f}%账户)  "
              f"7日θ≈+${p.theta_est_contract:,.0f}  盈利区±{p.profit_zone_pct:.0f}%")
        if p.earnings_days is not None:
            print(f"  下次财报约 {p.earnings_days} 天后")
        if p.flags:
            print(f"  ⚠ {'；'.join(p.flags)}")
        print()

    print("[今日执行清单]")
    for line in build_playbook(result):
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="双日历价差每日择时提醒")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    result = run_calendar_scan(cfg)
    print_report(result)
    append_history(result)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify_cfg = cfg.get("notify", {})
    title, body = format_notification(result)
    if notify_cfg.get("desktop"):
        desktop_notify(title, body)
    if notify_cfg.get("email", {}).get("enabled"):
        email_body = "\n".join(build_playbook(result))
        if result.get("errors"):
            email_body += "\n\n错误：\n" + "\n".join(result["errors"])
        email_notify(notify_cfg["email"], title, email_body or body)


if __name__ == "__main__":
    main()
