"""资金流向操盘规律 · 历史胜率回测。

对 FLOW_CATALOG 每条规律统计次日收益胜率（方向调整）：
  · 做多规律：fwd_1d > 0 为胜
  · 做空/回避规律：fwd_1d < 0 为胜

用法：
    python research/flow_pattern_backtest.py
    python research/flow_pattern_backtest.py --years 3 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.capital_flow import FLOW_CATALOG, build_flow_history, match_pattern_by_id
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

OUT_JSON = ROOT / "research" / "flow_pattern_stats.json"


def _direction_win(pattern_id: str, fwd: float) -> bool:
    pat = next((p for p in FLOW_CATALOG if p.id == pattern_id), None)
    if pat is None:
        return False
    if pat.side == "up":
        return fwd > 0
    return fwd < 0


def backtest_patterns(
    data: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    *,
    min_dvol_m: float = 30.0,
) -> dict[str, dict]:
    """全历史面板统计每条规律。"""
    panels: list[pd.DataFrame] = []
    for tk, df in data.items():
        hist = build_flow_history(df, spy_close)
        if hist.empty:
            continue
        hist = hist[hist["dvol_m"] >= min_dvol_m]
        hist["代码"] = tk
        panels.append(hist)
    if not panels:
        return {}
    panel = pd.concat(panels, ignore_index=True)
    results: dict[str, dict] = {}
    for pat in FLOW_CATALOG:
        if pat.id == "D_OFFERING":
            # 无 SEC 历史：用暴涨代理（前日>30%）
            mask = panel["前日涨幅%"] > 30
            sub = panel[mask]
        else:
            mask = []
            for _, row in panel.iterrows():
                r = row.to_dict()
                spy_bull = bool(row.get("spy_bull", True))
                if match_pattern_by_id(pat.id, r, spy_bull=spy_bull):
                    mask.append(True)
                else:
                    mask.append(False)
            sub = panel[pd.Series(mask, index=panel.index)]
        if sub.empty or len(sub) < 15:
            continue
        fwd = pd.to_numeric(sub["fwd_1d"], errors="coerce").dropna()
        if fwd.empty:
            continue
        wins = [_direction_win(pat.id, float(x)) for x in fwd]
        results[pat.id] = {
            "name": pat.name,
            "side": pat.side,
            "tier": pat.tier,
            "action": pat.action,
            "sample_n": int(len(fwd)),
            "win_rate_1d": float(np.mean(wins)),
            "mean_ret_1d_pct": float(fwd.mean() * 100),
            "median_ret_1d_pct": float(fwd.median() * 100),
            "std_ret_1d_pct": float(fwd.std() * 100),
            "worst_1d_pct": float(fwd.min() * 100),
            "best_1d_pct": float(fwd.max() * 100),
        }
    return results


def run_backtest(*, years: float = 3.0, quick: bool = False, min_dvol_m: float = 30.0) -> dict:
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(years * 365) + 120)).isoformat()
    pool = LIQUID100 if quick else GAINER_MOMENTUM
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    batch = yahoo.fetch_batch(pool, start, end)
    patterns = backtest_patterns(batch, spy, min_dvol_m=min_dvol_m)
    doc = {
        "generated": date.today().isoformat(),
        "years": years,
        "universe_size": len(pool),
        "tickers_with_data": len(batch),
        "patterns": patterns,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def print_summary(doc: dict) -> None:
    print(f"\n资金流向规律回测 · {doc.get('generated')} · {doc.get('years')}年")
    print(f"股票池 {doc.get('universe_size')} · 有数据 {doc.get('tickers_with_data')}")
    print("-" * 72)
    patterns = doc.get("patterns") or {}
    for pat in FLOW_CATALOG:
        row = patterns.get(pat.id)
        if not row:
            print(f"  {pat.id:12} {pat.name:16} — 样本不足")
            continue
        print(
            f"  {pat.id:12} {pat.name:16} "
            f"胜率{row['win_rate_1d']:.0%} n={row['sample_n']:4d} "
            f"均收益{row['mean_ret_1d_pct']:+.2f}% "
            f"[{row['tier']}] {pat.action}"
        )
    print(f"\n→ {OUT_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="资金流向规律回测")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--quick", action="store_true", help="仅 LIQUID100")
    parser.add_argument("--min-dvol", type=float, default=30.0)
    args = parser.parse_args()
    doc = run_backtest(years=args.years, quick=args.quick, min_dvol_m=args.min_dvol)
    print_summary(doc)


if __name__ == "__main__":
    main()
