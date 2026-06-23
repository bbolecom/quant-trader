#!/usr/bin/env python3
"""SPCE 类投机票池 · 每日更新。

从历史暴涨事件筛出与 SPCE 相似的标的，可选标注当日 A/B/C 阶段。

用法：
    python speculative_pool_daily.py
    python speculative_pool_daily.py --offline
    python speculative_pool_daily.py -c speculative_pool_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.speculative_pool import run_speculative_pool
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "speculative_pool_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def format_lines(doc: dict) -> list[str]:
    lines: list[str] = []
    meta = doc.get("meta") or {}
    arch = meta.get("archetype_stats") or {}
    lines.append(
        f"SPCE类投机池 · {doc.get('updated')} · "
        f"共 {meta.get('pool_size', 0)} 只（核心 {meta.get('core_size', 0)}）"
    )
    lines.append(
        f"  原型 {meta.get('archetype', 'SPCE')} · "
        f"15%+×{arch.get('spikes15', '?')} · 最大涨{arch.get('max_gain', '?'):.0f}%"
    )
    lines.append("")

    pre = doc.get("today_precursors") or []
    brk = doc.get("today_breakouts") or []
    lines.append(f"【今日前兆 C 类 · 提前盯盘】({len(pre)} 只)")
    if not pre:
        lines.append("  无")
    else:
        for r in pre[:10]:
            lines.append(
                f"  👁 {r['代码']} 相似{r['相似分']:.2f} ${r.get('现价') or '?'} · {r.get('说明', '')}"
            )

    lines.append("")
    lines.append(f"【今日突破 A 类】({len(brk)} 只)")
    if not brk:
        lines.append("  无")
    else:
        for r in brk[:10]:
            lines.append(
                f"  🚀 {r['代码']} 相似{r['相似分']:.2f} · {r.get('说明', '')}"
            )

    lines.append("")
    lines.append("【核心池 Top15】")
    for r in (doc.get("core") or [])[:15]:
        stage = r.get("阶段") or "—"
        mcap = f"${r['市值B']:.2f}B" if r.get("市值B") else "市值?"
        lines.append(
            f"  {r['代码']} 分{r['相似分']:.2f} {stage} {mcap} · {r.get('说明', '')[:40]}"
        )

    lines.append("")
    lines.append("说明：核心池=与 SPCE 历史暴涨画像最相近；C 类前兆优先盯盘。")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="SPCE 类投机票池")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--offline", action="store_true", help="仅历史事件，不拉行情")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.offline:
        cfg["enrich_live"] = False

    doc = run_speculative_pool(cfg)
    paths = cfg.get("outputs") or {}
    out_json = ROOT / paths.get("json", "research/speculative_pool.json")
    out_csv = ROOT / paths.get("csv", "research/speculative_pool.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = doc.get("members") or []
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = format_lines(doc)
    text = "\n".join(lines)
    print(text)
    print(f"\n→ {out_json}")
    print(f"→ {out_csv}")

    notify = cfg.get("notify") or {}
    if notify.get("desktop"):
        desktop_notify("SPCE类投机池", "\n".join(lines[:12]))


if __name__ == "__main__":
    main()
