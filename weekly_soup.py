#!/usr/bin/env python3
"""每周 PUT 价差「喝汤」提醒脚本（命令行，可用 launchd / cron 定时执行）。

功能：
    1. 读取 weekly_soup_config.json：标的、账户资金、Delta、价差宽度等。
    2. 生成本周 Put 信用价差方案（行权价、收租、归零概率、可否开仓）。
    3. 弹 macOS 桌面通知（可选）并发邮件（可选）。
    4. 追加写入 weekly_soup_history.csv。

用法：
    python weekly_soup.py                 # 默认 weekly_soup_config.json
    python weekly_soup.py -c my.json
    python weekly_soup.py --dry-run       # 只打印，不通知
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant import decline_income as di
from quant.data import DataError, fetch_history
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "weekly_soup_config.json"
HISTORY_FILE = ROOT / "weekly_soup_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tickers(cfg: dict) -> list[str]:
    raw = cfg.get("tickers") or ["SNDK"]
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    return [str(t).strip().upper() for t in raw if str(t).strip()]


def run_weekly_soup(cfg: dict) -> dict:
    """扫描配置中的标的，返回结构化结果。"""
    end = date.today().isoformat()
    lookback = int(cfg.get("lookback_days", 550))
    start = (date.today() - timedelta(days=lookback)).isoformat()
    account = float(cfg.get("account_size", 10_000))
    short_delta = float(cfg.get("short_delta", di.WEEKLY_SOUP_DELTA))
    width = float(cfg.get("spread_width", di.WEEKLY_SOUP_WIDTH))
    dte = int(cfg.get("dte_days", di.WEEKLY_DTE))
    max_margin_pct = float(cfg.get("max_margin_pct", 0.25))
    ic_cfg = cfg.get("iron_condor", {})
    add_call = bool(ic_cfg.get("enabled", False))
    call_delta = float(ic_cfg.get("call_delta", 0.05))
    call_width = ic_cfg.get("call_width")
    call_width = float(call_width) if call_width else None

    plans: list[di.WeeklySoupPlan] = []
    errors: list[str] = []
    for tk in resolve_tickers(cfg):
        try:
            df = fetch_history(tk, start=start, end=end)
            plan = di.weekly_put_soup_plan(
                tk, df,
                account_size=account,
                short_delta=short_delta,
                width=width,
                dte_days=dte,
                max_margin_pct=max_margin_pct,
                add_call=add_call,
                call_delta=call_delta,
                call_width=call_width,
            )
            if plan:
                plans.append(plan)
            else:
                errors.append(f"{tk}：数据不足，无法生成方案")
        except DataError as e:
            errors.append(f"{tk}：{e}")

    return {"plans": plans, "errors": errors, "config": cfg}


def format_notification(result: dict) -> tuple[str, str]:
    plans: list[di.WeeklySoupPlan] = result.get("plans") or []
    if not plans:
        title = "周PUT喝汤 · 无方案"
        body = "；".join(result.get("errors") or ["无可用标的"])
        return title, body[:220]

    open_plans = [p for p in plans if p.can_open]
    if open_plans:
        p = open_plans[0]
        title = f"🍲 {p.ticker} 本周可喝汤"
        body = (
            f"卖${p.short_strike:,.0f}/买${p.long_strike:,.0f} "
            f"收${p.credit_per_contract:,.0f} 归零{p.zero_prob:.0%} "
            f"止盈≤${p.take_profit_price:.2f}"
        )
    else:
        p = plans[0]
        title = f"⏸ {p.ticker} 本周暂停"
        body = f"跌破MA50 ${p.ma50:,.0f}，暂不开仓"

    if len(plans) > 1:
        others = "、".join(pl.ticker for pl in plans[1:3])
        body += f" · 另: {others}"
    if result.get("errors"):
        body += " ⚠"
    return title, body[:220]


def build_playbook(result: dict) -> list[str]:
    lines: list[str] = []
    for p in result.get("plans") or []:
        status = "✅ 可开" if p.can_open else "❌ 暂停"
        line = (
            f"{p.ticker} {status}：SELL Put ${p.short_strike:,.0f} / "
            f"BUY Put ${p.long_strike:,.0f}，收 ${p.credit_per_contract:,.0f}，"
            f"归零 {p.zero_prob:.0%}，建议 {p.max_contracts or 1} 张"
        )
        if p.iron_condor and p.call_short_strike > 0:
            line += (f"｜双开 SELL Call ${p.call_short_strike:,.0f}/BUY ${p.call_long_strike:,.0f}"
                     f" 多收 ${p.call_credit_per_contract:,.0f}，合计 ${p.total_credit_per_contract:,.0f}"
                     f"，留区间 {p.range_prob:.0%}")
        lines.append(line)
    if result.get("plans"):
        for step in result["plans"][0].playbook:
            lines.append(step)
    return lines


def build_history_row(result: dict) -> dict:
    plans: list[di.WeeklySoupPlan] = result.get("plans") or []
    primary = plans[0] if plans else None
    return {
        "扫描时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "标的": ",".join(p.ticker for p in plans),
        "可开仓": ",".join(p.ticker for p in plans if p.can_open),
        "首选": primary.ticker if primary else "",
        "卖Put": primary.short_strike if primary else None,
        "买Put": primary.long_strike if primary else None,
        "收租/张": primary.credit_per_contract if primary else None,
        "归零概率": primary.zero_prob if primary else None,
        "建议张数": primary.max_contracts if primary else None,
        "执行清单": " | ".join(build_playbook(result)),
        "错误": "；".join(result.get("errors") or []),
    }


def append_history(result: dict) -> None:
    row = build_history_row(result)
    df = pd.DataFrame([row])
    header = not HISTORY_FILE.exists()
    df.to_csv(HISTORY_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")


def print_report(result: dict) -> None:
    print(f"=== 周 PUT 喝汤提醒 {datetime.now():%Y-%m-%d %H:%M} ===")
    if result.get("errors"):
        for e in result["errors"]:
            print(f"[警告] {e}", file=sys.stderr)

    plans: list[di.WeeklySoupPlan] = result.get("plans") or []
    if not plans:
        print("\n无可用方案。")
        return

    for p in plans:
        print(f"\n[{p.ticker}] 现价 ${p.close:,.2f}  RV {p.rv_pct:.0f}%  IV {p.iv_pct:.0f}%")
        print(f"  趋势: {'✅ 站上 MA50' if p.above_ma else '❌ 跌破 MA50'} (${p.ma50:,.2f})")
        print(f"  组合: SELL Put ${p.short_strike:,.0f}  +  BUY Put ${p.long_strike:,.0f}  (宽 ${p.width:.0f})")
        print(f"  收租 ${p.credit_per_contract:,.0f}/张  保证金 ${p.margin_per_contract:,.0f}  "
              f"归零 {p.zero_prob:.0%}  周ROI {p.weekly_roi_pct:.1f}%")
        if p.iron_condor and p.call_short_strike > 0:
            print(f"  双开: + SELL Call ${p.call_short_strike:,.0f} / BUY Call ${p.call_long_strike:,.0f}"
                  f"  (Delta {p.call_delta:.02f}，约 +{p.call_otm_pct:.0f}%)  多收 ${p.call_credit_per_contract:,.0f}")
            print(f"  合计收 ${p.total_credit_per_contract:,.0f}/张  周ROI {p.combined_roi_pct:.1f}%  "
                  f"留区间[${p.short_strike:,.0f},${p.call_short_strike:,.0f}] {p.range_prob:.0%}")
        print(f"  止盈：总权利金赚 50% 整体平  建议 {p.max_contracts or 1} 张")
        print(f"  顺 +${p.weekly_profit_if_zero:,.0f}  /  逆 -${p.weekly_loss_if_max:,.0f}")
        if p.flags:
            print(f"  ⚠ {'；'.join(p.flags)}")

    print("\n[本周执行清单]")
    for step in build_playbook(result):
        print(f"  {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="每周 PUT 价差喝汤提醒")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不发送通知")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    result = run_weekly_soup(cfg)
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
