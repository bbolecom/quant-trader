#!/usr/bin/env python3
"""定时自动信号扫描脚本（命令行，可用 launchd / cron 定时执行）。

功能：
    1. 读取 scan_config.json 中的自选股、策略与参数。
    2. 拉取最近行情，计算每只股票"今天"的交易动作。
    3. 对触发买/卖/平仓信号的标的：弹出 macOS 桌面通知（可选），并发送邮件（可选）。
    4. 把每次扫描结果追加写入 scan_history.csv。

用法：
    python scan_daily.py                # 使用默认 scan_config.json
    python scan_daily.py -c my.json     # 指定配置文件
    python scan_daily.py --dry-run      # 不发送任何通知，仅打印结果
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import smtplib
import subprocess
import sys
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from quant import paper, signals
from quant.data import DataError, fetch_history

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "scan_config.json"
HISTORY_FILE = ROOT / "scan_history.csv"

CHANGE_MARK = "🟢|🔴|🟡"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 数据 + 扫描
# ---------------------------------------------------------------------------
def gather_data(watchlist: list[str], lookback_days: int) -> tuple[dict[str, pd.DataFrame], list[str]]:
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = date.today().isoformat()
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for t in watchlist:
        try:
            data[t] = fetch_history(t, start=start, end=end)
        except (DataError, Exception):  # noqa: BLE001
            failed.append(t)
    return data, failed


def run_scan(cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data, failed = gather_data(cfg["watchlist"], int(cfg.get("lookback_days", 400)))
    if failed:
        print(f"[警告] 以下标的获取失败，已忽略：{', '.join(failed)}", file=sys.stderr)
    if not data:
        raise SystemExit("没有可用的行情数据，扫描终止。")

    table = signals.scan(
        data,
        cfg["strategy"],
        params=cfg.get("params", {}),
        allow_short=bool(cfg.get("allow_short", False)),
    )
    return table, data


def update_paper(cfg: dict, table: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    """如配置启用，则按今日信号对本地模拟账户调仓。"""
    pcfg = cfg.get("paper", {})
    if not pcfg.get("enabled"):
        return
    account_file = ROOT / pcfg.get("account_file", "paper_account.json")
    account = paper.load_account(account_file)
    if account is None:
        account = paper.new_account(float(pcfg.get("initial", 100_000.0)))
        print(f"[模拟盘] 新建账户，初始资金 ${account.initial:,.0f}")

    targets = paper.targets_from_signals(table, max_positions=int(pcfg.get("max_positions", 0)))
    prices = {t: float(df["Close"].iloc[-1]) for t, df in data.items()}
    as_of = max(pd.Timestamp(df.index[-1]) for df in data.values()).strftime("%Y-%m-%d")
    trades = paper.rebalance(
        account, targets, prices, as_of=as_of,
        fee_bps=float(pcfg.get("fee_bps", 5.0)), slippage_bps=float(pcfg.get("slippage_bps", 2.0)),
    )
    paper.save_account(account, account_file)
    s = paper.summary(account, prices)
    print(f"[模拟盘] {as_of} 调仓 {len(trades)} 笔 ｜ 权益 ${s['总权益']:,.0f}（{s['累计收益率']:+.2%}）｜ 持仓 {int(s['持仓数量'])} 只")


# ---------------------------------------------------------------------------
# 通知
# ---------------------------------------------------------------------------
def desktop_notify(title: str, message: str) -> None:
    """macOS 桌面通知（其它系统则打印到终端）。"""
    if platform.system() == "Darwin":
        safe_msg = message.replace('"', "'")
        safe_title = title.replace('"', "'")
        script = f'display notification "{safe_msg}" with title "{safe_title}" sound name "Glass"'
        try:
            subprocess.run(["osascript", "-e", script], check=False)
            return
        except FileNotFoundError:
            pass
    print(f"[通知] {title} — {message}")


def email_notify(email_cfg: dict, subject: str, body: str) -> None:
    if not email_cfg.get("enabled"):
        return
    password = os.environ.get(email_cfg.get("password_env", ""), "")
    if not password:
        print(f"[警告] 未设置环境变量 {email_cfg.get('password_env')}，跳过邮件发送。", file=sys.stderr)
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = email_cfg["username"]
    msg["To"] = email_cfg["to"]
    try:
        with smtplib.SMTP(email_cfg["smtp_host"], int(email_cfg["smtp_port"]), timeout=20) as s:
            s.starttls()
            s.login(email_cfg["username"], password)
            s.sendmail(email_cfg["username"], [email_cfg["to"]], msg.as_string())
        print("[邮件] 已发送。")
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 邮件发送失败：{e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 记录
# ---------------------------------------------------------------------------
def append_history(table: pd.DataFrame, strategy: str) -> None:
    record = table.copy()
    record.insert(0, "扫描时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    record.insert(1, "策略", strategy)
    header = not HISTORY_FILE.exists()
    record.to_csv(HISTORY_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="定时自动信号扫描")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不发送通知")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print(f"=== 信号扫描 {datetime.now():%Y-%m-%d %H:%M} | 策略：{cfg['strategy']} ===")

    table, data = run_scan(cfg)
    pd.set_option("display.unicode.east_asian_width", True)
    print(table.to_string(index=False))

    if not args.dry_run:
        update_paper(cfg, table, data)

    changed = table[table["今日动作"].str.contains(CHANGE_MARK, regex=True)]
    append_history(table, cfg["strategy"])

    if changed.empty:
        print("今日没有标的触发新的买卖信号。")
        if not args.dry_run and cfg.get("notify", {}).get("desktop"):
            desktop_notify("美股信号扫描", "今日无新信号")
        return

    lines = [f"{r['代码']}: {r['今日动作']} @ ${r['最新价']}" for _, r in changed.iterrows()]
    summary = "；".join(lines)
    print("\n📬 触发信号：")
    for ln in lines:
        print("  -", ln)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify_cfg = cfg.get("notify", {})
    if notify_cfg.get("desktop"):
        desktop_notify(f"美股信号 · {len(changed)} 只触发", summary)
    if notify_cfg.get("email", {}).get("enabled"):
        body = f"策略：{cfg['strategy']}\n时间：{datetime.now():%Y-%m-%d %H:%M}\n\n" + "\n".join(lines)
        email_notify(notify_cfg["email"], f"美股信号扫描 · {len(changed)} 只触发", body)


if __name__ == "__main__":
    main()
