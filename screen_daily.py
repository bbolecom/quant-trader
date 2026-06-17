#!/usr/bin/env python3
"""每日自动选股推送脚本（命令行，可用 launchd / cron 定时执行）。

功能：
    1. 读取 screen_config.json 中的股票池、筛选条件、回测策略。
    2. 建池 → 按涨幅/成交额/换手率/市值/行业筛选 → 对入选标的批量回测。
    3. 把当日选股结果弹 macOS 桌面通知（可选）并发邮件（可选）。
    4. 把每次选股结果追加写入 screen_history.csv。

用法：
    python screen_daily.py                 # 使用默认 screen_config.json
    python screen_daily.py -c my.json      # 指定配置文件
    python screen_daily.py --dry-run       # 只打印，不发送通知
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant import screener
from quant.data import DataError

# 复用 scan_daily 的桌面/邮件通知实现，避免重复。
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "screen_config.json"
HISTORY_FILE = ROOT / "screen_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_filters(cfg: dict) -> screener.ScreenFilters:
    f = cfg.get("filters", {})
    return screener.ScreenFilters(
        min_gain_pct=float(f.get("min_gain_pct", -100.0)),
        max_gain_pct=float(f.get("max_gain_pct", 1000.0)),
        min_dollar_vol_m=float(f.get("min_dollar_vol_m", 0.0)),
        min_turnover_pct=float(f.get("min_turnover_pct", 0.0)),
        max_turnover_pct=float(f.get("max_turnover_pct", 100.0)),
        min_mcap_b=float(f.get("min_mcap_b", 0.0)),
        max_mcap_b=float(f.get("max_mcap_b", 10_000.0)),
        lookback_days=int(f.get("lookback_days", 20)),
        sectors=f.get("sectors") or None,
    )


def run(cfg: dict) -> dict:
    filters = build_filters(cfg)
    sel = screener.normalize_selection_date(cfg.get("selection_date") or date.today())
    start = (pd.Timestamp(sel) - timedelta(days=int(cfg.get("history_days", 500)))).strftime("%Y-%m-%d")
    cost = cfg.get("cost", {})
    return screener.run_screen(
        filters,
        start,
        sel,
        selection_date=sel,
        pool=cfg.get("pool", "day_gainers"),
        pool_size=int(cfg.get("pool_size", 50)),
        custom_tickers=cfg.get("custom_tickers"),
        strategy_name=cfg.get("strategy"),
        params=cfg.get("params", {}),
        max_backtest=int(cfg.get("max_backtest", 20)),
        allow_short=bool(cfg.get("allow_short", False)),
        initial_capital=float(cost.get("capital", 100_000.0)),
        fee_bps=float(cost.get("fee_bps", 5.0)),
        slippage_bps=float(cost.get("slippage_bps", 2.0)),
    )


def append_history(merged: pd.DataFrame, cfg: dict) -> None:
    if merged.empty:
        return
    record = merged.drop(columns=["_行业EN"], errors="ignore").copy()
    now = datetime.now()
    if "选股日期" not in record.columns:
        raise ValueError("选股结果缺少「选股日期」，无法写入历史记录。")
    record.insert(0, "选股时间", now.strftime("%Y-%m-%d %H:%M:%S"))
    if "股票池" not in record.columns:
        record.insert(1, "股票池", cfg.get("pool", ""))
    if "策略" not in record.columns:
        record.insert(2, "策略", cfg.get("strategy", ""))
    header = not HISTORY_FILE.exists()
    record.to_csv(HISTORY_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")


def format_lines(merged: pd.DataFrame, top: int = 10) -> list[str]:
    lines: list[str] = []
    for _, r in merged.head(top).iterrows():
        gain = r.get("涨幅%")
        ret = r.get("策略累计收益")
        sig = r.get("当前信号", "")
        reason = r.get("选股理由", "")
        gain_s = f"{gain:+.1f}%" if pd.notna(gain) else "-"
        ret_s = f"{ret:+.1%}" if pd.notna(ret) else "-"
        line = f"{r['代码']}：涨幅 {gain_s} ｜ 策略 {ret_s} ｜ {sig}"
        if reason:
            line += f" ｜ {reason}"
        lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="每日自动选股推送")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不发送通知")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print(f"=== 每日选股 {datetime.now():%Y-%m-%d %H:%M} | 池：{cfg.get('pool')} | 策略：{cfg.get('strategy')} ===")

    try:
        result = run(cfg)
    except DataError as e:
        raise SystemExit(f"选股失败：{e}")

    snapshot = result["snapshot"]
    filtered = result["filtered"]
    merged = result["merged"]
    summary = result["summary"]

    print(f"初选 {len(snapshot)} 只 → 筛选后 {len(filtered)} 只符合条件")

    if merged.empty:
        msg = "今日没有符合条件的标的。" if filtered.empty else "已筛出标的但回测无有效结果。"
        print(msg)
        if not args.dry_run and cfg.get("notify", {}).get("desktop"):
            desktop_notify("每日选股", msg)
        return

    pd.set_option("display.unicode.east_asian_width", True)
    show_cols = [c for c in ["代码", "名称", "涨幅%", "换手率%", "行业", "策略累计收益", "夏普比率", "当前信号"]
                 if c in merged.columns]
    print(merged[show_cols].head(int(cfg.get("max_backtest", 20))).to_string(index=False))

    append_history(merged, cfg)

    lines = format_lines(merged, top=int(cfg.get("notify_top", 10)))
    print("\n📬 入选标的：")
    for ln in lines:
        print("  -", ln)

    if summary:
        print(
            f"\n汇总：回测 {int(summary.get('入选数量', 0))} 只 ｜ 平均收益 {summary.get('平均累计收益', 0):+.1%} "
            f"｜ 盈利占比 {summary.get('盈利标的占比', 0):.0%} ｜ 平均夏普 {summary.get('平均夏普', 0):.2f}"
        )

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify_cfg = cfg.get("notify", {})
    pool_cn = screener.UNIVERSE_PRESETS.get(cfg.get("pool", ""), cfg.get("pool", ""))
    title = f"每日选股 · {pool_cn} · {len(merged)} 只入选"
    body_head = "；".join(lines[:3])
    if notify_cfg.get("desktop"):
        desktop_notify(title, body_head)
    if notify_cfg.get("email", {}).get("enabled"):
        body = (
            f"股票池：{pool_cn}\n策略：{cfg.get('strategy')}\n"
            f"时间：{datetime.now():%Y-%m-%d %H:%M}\n\n" + "\n".join(lines)
        )
        if summary:
            body += (
                f"\n\n汇总：平均收益 {summary.get('平均累计收益', 0):+.1%} ｜ "
                f"盈利占比 {summary.get('盈利标的占比', 0):.0%}"
            )
        email_notify(notify_cfg["email"], title, body)


if __name__ == "__main__":
    main()
