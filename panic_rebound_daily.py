#!/usr/bin/env python3
"""恐慌反弹做多 · 每日实盘扫描。

规律（全市场 400 只 / 5 年回测）：已深跌≥30% 的票当日再暴跌≥10% 放恐慌盘，
次日开盘做多、持有数日反弹。样本外年化 +54%~+77%、胜率 54%~63%、回撤 -18%。

用法：
    python3 panic_rebound_daily.py
    python3 panic_rebound_daily.py --dry-run
    python3 panic_rebound_daily.py -c panic_rebound_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.panic_rebound import PanicReboundConfig, scan_live
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "panic_rebound_config.json"
HISTORY_FILE = ROOT / "panic_rebound_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {
            "drop_pct": 10.0, "pre20_drop_pct": 30.0, "min_price": 5.0,
            "min_dvol_m": 100.0, "hold_days": 3, "stop_loss_pct": 0.08,
            "take_profit_pct": 0.15, "max_positions": 3,
            "notify": {"desktop": True},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def build_config(cfg: dict) -> PanicReboundConfig:
    return PanicReboundConfig(
        drop_pct=float(cfg.get("drop_pct", 10.0)),
        pre20_drop_pct=float(cfg.get("pre20_drop_pct", 30.0)),
        min_price=float(cfg.get("min_price", 5.0)),
        min_dvol_m=float(cfg.get("min_dvol_m", 100.0)),
        hold_days=int(cfg.get("hold_days", 3)),
        stop_loss_pct=float(cfg.get("stop_loss_pct", 0.08)),
        take_profit_pct=float(cfg.get("take_profit_pct", 0.15)),
        max_positions=int(cfg.get("max_positions", 3)),
    )


def format_lines(picks: pd.DataFrame, pcfg: PanicReboundConfig) -> list[str]:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"恐慌反弹做多 · {today}",
        f"规则: 前20日跌≥{pcfg.pre20_drop_pct:.0f}% + 当日跌≥{pcfg.drop_pct:.0f}% + "
        f"成交额≥${pcfg.min_dvol_m:.0f}M + 价≥${pcfg.min_price:.0f}",
        f"操作: 次日开盘做多 · 持{pcfg.hold_days}日 · 止损{pcfg.stop_loss_pct:.0%}/止盈{pcfg.take_profit_pct:.0%} · "
        f"每日最多{pcfg.max_positions}只等权",
        "",
    ]
    if picks is None or picks.empty:
        lines.append("  今日无恐慌反弹候选（无票同时满足深跌+暴跌+流动性）。")
        return lines
    top = picks.head(pcfg.max_positions)
    lines.append(f"【今日候选 {len(top)} 只（共命中 {len(picks)}）】")
    for _, r in top.iterrows():
        lines.append(
            f"  ✅ {r['代码']:<6} 现价${r['最新价']} 当日{r['当日跌%']:+.1f}% "
            f"前20日{r['前20日跌%']:+.0f}% 量比{r.get('量比', '—')} 成交额${r['成交额M']}M"
        )
        lines.append(
            f"     次日开盘做多 → 止损≈${r['止损价≈']}({-pcfg.stop_loss_pct:.0%}) "
            f"止盈≈${r['止盈价≈']}(+{pcfg.take_profit_pct:.0%}) 持{pcfg.hold_days}日"
        )
    lines.append("")
    lines.append("纪律: 等权小仓 · 触止盈/止损或到期即平 · 财报日临近降仓 · 仅做流动性好的票")
    return lines


def format_notification(picks: pd.DataFrame) -> tuple[str, str]:
    n = 0 if picks is None or picks.empty else len(picks)
    if n == 0:
        return "🩹 恐慌反弹·今日无候选", "无票同时满足深跌+暴跌+流动性"
    top = picks.iloc[0]
    title = f"🩹 恐慌反弹·{n}只候选"
    body = f"{top['代码']} 当日{top['当日跌%']:+.1f}% 前20日{top['前20日跌%']:+.0f}% → 次日开盘做多"
    return title, body[:200]


def append_history(picks: pd.DataFrame) -> None:
    codes = "" if picks is None or picks.empty else ",".join(picks["代码"].astype(str).tolist())
    row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "候选数": 0 if picks is None or picks.empty else len(picks),
        "候选清单": codes,
    }
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        df = pd.concat([pd.read_csv(HISTORY_FILE), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="恐慌反弹做多 · 每日实盘扫描")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--pool", choices=["market", "rgti", "both"], default=None,
                    help="扫描范围：market=全市场榜单 / rgti=RGTI类似高波池 / both=并集（默认读配置）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    pcfg = build_config(cfg)
    pool = args.pool or cfg.get("pool", "market")

    pool_label = {"market": "全市场跌幅榜/活跃榜", "rgti": "RGTI类似高波池", "both": "全市场+RGTI池"}.get(pool, pool)
    print(f"扫描范围：{pool_label}，匹配恐慌反弹规律…")
    picks = scan_live(pcfg, pool=pool)

    print("=" * 78)
    for line in format_lines(picks, pcfg):
        print(line)
    print("=" * 78)
    append_history(picks)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify = cfg.get("notify", {})
    title, body = format_notification(picks)
    if notify.get("desktop"):
        desktop_notify(title, body)
    if notify.get("email", {}).get("enabled"):
        email_notify(notify["email"], title, "\n".join(format_lines(picks, pcfg)))


if __name__ == "__main__":
    main()
