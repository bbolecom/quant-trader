#!/usr/bin/env python3
"""Meme 规律纯多头 · 每日扫描（MSTR / SMCI / COIN）。

用法：
    python ticker_pattern_daily.py
    python ticker_pattern_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.ticker_pattern_strategy import parse_meme_long, scan_meme_long

DEFAULT_CFG = ROOT / "daily_pick_config.json"
OUT_JSON = ROOT / "research" / "ticker_pattern_today.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_spy_bull(cfg: dict) -> tuple[bool, str]:
    rcfg = cfg.get("regime") or {}
    mock = rcfg.get("mock")
    if mock is not None:
        bull = bool(mock.get("bull", True))
        label = mock.get("label") or ("牛市" if bull else "弱市")
        return bull, label
    from research.income_engine import get_regime

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    reg = get_regime(yahoo)
    return reg.bull, reg.label


def run_scan(cfg: dict) -> dict:
    mlc = parse_meme_long(cfg)
    bull, label = get_spy_bull(cfg)
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=400)).isoformat()
    batch = yahoo.fetch_batch(mlc.tickers, start, end)
    picks = scan_meme_long(batch, spy_bull=bull, mlc=mlc)
    actionable = sum(1 for p in picks if p.get("状态") == "可开仓")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "选股日期": date.today().isoformat(),
        "选股时间": now,
        "regime": {"bull": bull, "label": label},
        "summary": {
            "可开仓": actionable,
            "总条目": len(picks),
            "是否空仓日": actionable == 0,
        },
        "picks": picks,
    }


def print_report(doc: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Meme规律纯多头  {doc['选股日期']}  {doc['选股时间']}")
    reg = doc.get("regime") or {}
    print(f"大盘：{reg.get('label', '')}")
    s = doc.get("summary") or {}
    print(f"可开仓 {s.get('可开仓', 0)} / {s.get('总条目', 0)}")
    for p in doc.get("picks") or []:
        mark = "✓" if p.get("状态") == "可开仓" else "·"
        print(f"  {mark} {p.get('代码','')} {p.get('状态','')} {p.get('方向','')}")
        print(f"      {p.get('选股理由','')}")
    print(f"{'=' * 60}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Meme规律纯多头每日扫描")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    doc = run_scan(cfg)
    print_report(doc)
    if not args.dry_run:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {OUT_JSON}")


if __name__ == "__main__":
    main()
