#!/usr/bin/env python3
"""每日策略排名：汇总回测分 + 实时信号 → Top3 + $10k 仓位表。

用法：
    python strategy_daily.py
    python strategy_daily.py --profile income   # 偏稳定收租
    python strategy_daily.py --profile growth   # 偏高收益
    python strategy_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from research.strategy_ranker import evaluate_strategies, format_playbook
from research.holy_grail_search import format_summary_lines, load_holy_grail_summary
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "strategy_config.json"
HISTORY_FILE = ROOT / "strategy_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {"account_size": 10000, "profile": "balanced", "notify": {"desktop": True}}
    return json.loads(path.read_text(encoding="utf-8"))


def format_notification(result: dict) -> tuple[str, str]:
    top3 = result.get("top3") or []
    reg = result["regime"]
    active = [p for p in top3 if p.signal_ok]
    if active:
        p = active[0]
        title = f"🏆 策略Top1·{p.meta.name[:8]}"
        body = f"{reg.label[:6]} | {p.detail[:120]}"
    else:
        title = "⏸ 策略排名·今日观望"
        body = f"{reg.label} | 无满足条件的开仓信号"
    return title, body[:220]


def append_history(result: dict) -> None:
    top3 = result.get("top3") or []
    names = " > ".join(p.meta.name for p in top3)
    pf = result.get("portfolio") or []
    exec_n = sum(1 for r in pf if r.get("引擎") not in ("现金储备",))
    row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "账户": result["account"],
        "风格": result["profile"],
        "大盘": result["regime"].label,
        "Top3": names,
        "可执行数": exec_n,
    }
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        df = pd.concat([pd.read_csv(HISTORY_FILE), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def print_report(result: dict) -> None:
    print("=" * 88)
    print(f"每日策略排名 · {datetime.now().strftime('%Y-%m-%d')} · 账户 ${result['account']:,.0f}")
    print("=" * 88)
    for line in format_playbook(result):
        print(line)
    print("\n【全策略得分榜】")
    print(f"{'策略':<22} {'分类':<6} {'信号':<6} {'得分':>5}  说明")
    print("-" * 80)
    for p in result["ranked"]:
        sig = "有" if p.signal_ok else "无"
        print(f"{p.meta.name:<22} {p.meta.category:<6} {sig:<6} {p.score:>5.2f}  {p.detail[:40]}")




def main() -> None:
    p = argparse.ArgumentParser(description="每日策略排名 Top3")
    p.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--account", type=float, default=None)
    p.add_argument("--profile", choices=["balanced", "income", "growth"], default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    account = float(args.account or cfg.get("account_size", 10_000))
    profile = args.profile or cfg.get("profile", "balanced")

    result = evaluate_strategies(account=account, profile=profile)
    print_report(result)
    print("\n【圣杯距离】")
    for line in format_summary_lines():
        print(f"  {line}")
    append_history(result)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify = cfg.get("notify", {})
    title, body = format_notification(result)
    if notify.get("desktop"):
        desktop_notify(title, body)
    if notify.get("email", {}).get("enabled"):
        email_notify(notify["email"], title, "\n".join(format_playbook(result)))


if __name__ == "__main__":
    main()
