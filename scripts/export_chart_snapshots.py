#!/usr/bin/env python3
"""从 yfinance 导出 K 线快照 → research/charts/{TICKER}.json"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CHARTS = ROOT / "research" / "charts"
IOS_CHARTS = ROOT / "ios" / "Resources" / "charts"
DAILY = ROOT / "research" / "daily_pick_today.json"
SPECULATIVE_POOL = ROOT / "research" / "speculative_pool.json"


def _collect_tickers(extra: list[str] | None = None) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()

    def add(raw: object) -> None:
        tk = str(raw or "").strip().upper()
        if not tk or tk in {"—", "-"}:
            return
        if not tk.replace(".", "").replace("-", "").isalnum():
            return
        if tk not in seen:
            seen.add(tk)
            tickers.append(tk)

    for t in extra or []:
        add(t)
    if DAILY.exists():
        doc = json.loads(DAILY.read_text(encoding="utf-8"))
        for key in ("picks",):
            for row in doc.get(key) or []:
                add(row.get("代码") or row.get("ticker"))
        hw = doc.get("high_win") or {}
        for row in (hw.get("picks") or []) + (hw.get("watch") or []):
            add(row.get("代码"))

    if SPECULATIVE_POOL.exists():
        doc = json.loads(SPECULATIVE_POOL.read_text(encoding="utf-8"))
        for key in ("today_breakouts", "today_precursors", "core", "extended", "members"):
            for row in doc.get(key) or []:
                add(row.get("代码") or row.get("ticker"))

    for path in ROOT.glob("research/*_today.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for row in doc.get("picks") or []:
            add(row.get("代码") or row.get("ticker"))
    return tickers


def export_ticker(ticker: str, *, period: str = "6mo") -> dict | None:
    from quant.chart_live import fetch_live_chart

    return fetch_live_chart(ticker, period=period, interval="1d")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export OHLCV chart snapshots")
    ap.add_argument("--tickers", nargs="*", default=[], help="Extra tickers")
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    tickers = _collect_tickers(args.tickers)[: args.limit]
    CHARTS.mkdir(parents=True, exist_ok=True)
    IOS_CHARTS.mkdir(parents=True, exist_ok=True)

    ok = 0
    for tk in tickers:
        doc = export_ticker(tk)
        if not doc:
            print(f"  skip {tk}")
            continue
        out = CHARTS / f"{tk}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (IOS_CHARTS / f"{tk}.json").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        ok += 1
        print(f"  ✓ {tk} ({len(doc['bars'])} bars)")

    index = {"updated": date.today().isoformat(), "tickers": sorted(p.stem for p in CHARTS.glob("*.json"))}
    (CHARTS / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (IOS_CHARTS / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone: {ok}/{len(tickers)} charts → research/charts + ios/Resources/charts")


if __name__ == "__main__":
    main()
