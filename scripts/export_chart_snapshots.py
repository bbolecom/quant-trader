#!/usr/bin/env python3
"""从 yfinance 导出 K 线快照 → research/charts/{TICKER}.json"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "research" / "charts"
IOS_CHARTS = ROOT / "ios" / "Resources" / "charts"
DAILY = ROOT / "research" / "daily_pick_today.json"


def _collect_tickers(extra: list[str] | None = None) -> list[str]:
    tickers: set[str] = set()
    for t in extra or []:
        if t and t not in {"—", "-"}:
            tickers.add(t.upper())
    if DAILY.exists():
        doc = json.loads(DAILY.read_text(encoding="utf-8"))
        for key in ("picks",):
            for row in doc.get(key) or []:
                tk = str(row.get("代码") or row.get("ticker") or "").strip().upper()
                if tk and tk not in {"—", "-"} and tk.replace(".", "").isalnum():
                    tickers.add(tk)
        hw = doc.get("high_win") or {}
        for row in (hw.get("picks") or []) + (hw.get("watch") or []):
            tk = str(row.get("代码") or "").strip().upper()
            if tk and tk not in {"—", "-"}:
                tickers.add(tk)
    for path in ROOT.glob("research/*_today.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for row in doc.get("picks") or []:
            tk = str(row.get("代码") or "").strip().upper()
            if len(tk) >= 1 and tk not in {"—", "-"}:
                tickers.add(tk)
    return sorted(tickers)


def export_ticker(ticker: str, *, period: str = "6mo") -> dict | None:
    import yfinance as yf

    sym = ticker.upper()
    try:
        hist = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=False)
    except Exception:  # noqa: BLE001
        return None
    if hist is None or hist.empty:
        return None
    bars = []
    for idx, row in hist.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        bars.append({
            "date": d.isoformat(),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": round(float(row.get("Volume") or 0)),
        })
    return {
        "ticker": sym,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": period,
        "source": "yfinance",
        "bars": bars,
    }


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
