#!/usr/bin/env python3
"""全市场快速扫描 · 5 分钟内探测机会信号。

用法：
    python market_scan_fast.py
    python market_scan_fast.py --dry-run
    python market_scan_fast.py -c market_scan_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "market_scan_config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="全市场 5 分钟快扫")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--dry-run", action="store_true", help="只打印摘要，不写文件")
    args = parser.parse_args()

    cfg_raw = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    from quant.market_scan_fast import config_from_dict, run_market_scan, save_scan

    scan_cfg = config_from_dict(cfg_raw)
    doc = run_market_scan(scan_cfg)
    stats = doc.get("scan_stats") or {}
    summary = doc.get("summary") or {}
    print("=" * 50)
    print(f"  全市场快扫 · {doc.get('扫描时间')}")
    print("=" * 50)
    print(f"  候选池 {stats.get('universe', 0)} 只 → 信号 {stats.get('signals', 0)} 条")
    print(f"  耗时 {doc.get('elapsed_sec')}s / 预算 {doc.get('budget_sec')}s"
          f" {'✓' if doc.get('within_budget') else '⚠超时'}")
    print(f"  Phase1 {stats.get('phase1_sec')}s · Phase2 RV {stats.get('phase2_sec')}s")
    if summary.get("Gainer10+"):
        print(f"  Gainer10+ {summary.get('Gainer10+')} 条")
    for s in (doc.get("signals") or [])[:8]:
        tags = "/".join(s.get("标签") or [])
        print(f"  · {s.get('代码')} {s.get('涨幅%'):+.1f}% [{tags}] {s.get('选股理由', '')[:60]}")
    if len(doc.get("signals") or []) > 8:
        print(f"  … 另有 {len(doc['signals']) - 8} 条")

    if args.dry_run:
        return 0

    outs = cfg_raw.get("outputs") or {}
    jpath = ROOT / outs.get("today_json", "research/market_scan_today.json")
    save_scan(doc, jpath)
    ios = outs.get("ios_bundle")
    if ios:
        save_scan(doc, ROOT / ios)
    print(f"\n已写入 {jpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
