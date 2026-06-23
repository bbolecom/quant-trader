#!/usr/bin/env python3
"""全市场搜寻：高流动性标的 × CSP / 周铁鹰 参数网格，筛选三标达标组合。

默认目标：年化>80%、回撤<10%、胜率>85%

用法：
    python research/market_triple_scan.py
    python research/market_triple_scan.py --ann 0.8 --max-dd -0.10 --win 0.85 --min-dvol-m 30
    python research/market_triple_scan.py --quick          # 缩小池子加速
    python research/market_triple_scan.py --pick-fleet 5   # 输出 5 账户推荐
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.decline_income import (
    CSP_DTE_CAL,
    CSP_HOLD_TD,
    CSP_STEP_TD,
    DEFAULT_VRP,
    backtest_weekly_put_spread,
    equity_metrics_from_trades,
    realized_vol,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS, bs_put_price, strike_for_put_delta
from research.liquid_tier_a_scan import _avg_dollar_vol, build_candidate_pool
from research.triple_target_scan import (
    ScanResult,
    classify_tier,
    gap_score,
    oos_meets_target,
    set_scan_targets,
    targets_label,
)

RESULTS_CSV = ROOT / "research" / "market_triple_scan_results.csv"
PARETO_CSV = ROOT / "research" / "market_triple_scan_pareto.csv"
FLEET_JSON = ROOT / "research" / "market_triple_fleet.json"


def _csp_returns(
    close: pd.Series,
    *,
    delta: float,
    ma_window: int,
) -> list[float]:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma_s = close.rolling(ma_window).mean() if ma_window else None
    T = CSP_DTE_CAL / TRADING_DAYS
    rors: list[float] = []
    i = max(25, ma_window)
    while i + CSP_HOLD_TD < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CSP_STEP_TD
            continue
        if ma_s is not None and not (S > float(ma_s.iloc[i])):
            i += CSP_STEP_TD
            continue
        iv = sigma * (1 + DEFAULT_VRP)
        K = strike_for_put_delta(S, T, iv, target_delta=delta)
        credit = bs_put_price(S, K, T, iv)
        if K <= 0:
            i += CSP_STEP_TD
            continue
        ST = float(close.iloc[i + CSP_HOLD_TD])
        rors.append((credit - max(0.0, K - ST)) / K)
        i += CSP_STEP_TD
    return rors


def scan_ticker_grid(
    ticker: str,
    df: pd.DataFrame,
    dvol_m: float,
    *,
    min_trades: int,
    account_size: float,
) -> list[dict]:
    close = df["Close"].astype(float)
    px = float(close.iloc[-1])
    rv_pct = float(realized_vol(close).iloc[-1]) * 100
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else px
    above_ma = px > ma50
    cyc_yr = TRADING_DAYS / CSP_HOLD_TD
    rows: list[dict] = []

    for delta, ma, alloc in product(
        [0.15, 0.20, 0.25],
        [0, 50],
        [0.35, 0.50, 0.75],
    ):
        rors = _csp_returns(close, delta=delta, ma_window=ma)
        if len(rors) < min_trades:
            continue
        stats = equity_metrics_from_trades(rors, alloc_pct=alloc, cycles_per_year=cyc_yr)
        eq = stats.get("净值曲线")
        if eq is None or len(eq) < 2:
            continue
        ann = float(stats["年化收益率"])
        dd = float(stats["最大回撤"])
        win = float(stats["胜率"])
        oos = oos_meets_target(eq, win)
        tier = classify_tier(ann, dd, win, oos=oos)
        rows.append({
            "代码": ticker,
            "现价": round(px, 2),
            "成交额M": round(dvol_m, 1),
            "RV%": round(rv_pct, 1),
            "站上MA50": above_ma,
            "策略": "CSP",
            "delta": delta,
            "ma_window": ma,
            "alloc": alloc,
            "take_profit": 0.5,
            "年化": ann,
            "最大回撤": dd,
            "胜率": win,
            "交易数": len(rors),
            "tier": tier,
            "gap_score": gap_score(ann, dd, win),
            "oos_pass": oos,
        })

    wk = backtest_weekly_put_spread(close)
    if wk and wk.get("净值曲线") is not None:
        eq = wk["净值曲线"]
        ann = float(wk.get("年化", 0.0))
        dd = float(wk.get("最大回撤", 0.0))
        win = float(wk.get("胜率", 0.0))
        oos = oos_meets_target(eq, win)
        tier = classify_tier(ann, dd, win, oos=oos)
        rows.append({
            "代码": ticker,
            "现价": round(px, 2),
            "成交额M": round(dvol_m, 1),
            "RV%": round(rv_pct, 1),
            "站上MA50": above_ma,
            "策略": "偏斜铁鹰",
            "delta": 0.10,
            "ma_window": 50,
            "alloc": 0.25,
            "take_profit": 0.5,
            "年化": ann,
            "最大回撤": dd,
            "胜率": win,
            "交易数": int(wk.get("交易数", 0)),
            "tier": tier,
            "gap_score": gap_score(ann, dd, win),
            "oos_pass": oos,
        })
    return rows


def run_market_scan(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 30.0,
    min_trades: int = 30,
    quick: bool = False,
    account_size: float = 10_000.0,
) -> pd.DataFrame:
    end = end or date.today().isoformat()
    pool = build_candidate_pool(use_broad=not quick, max_names=80 if quick else 0)
    print(f"候选池 {len(pool)} 只 · 流动性 ≥ ${min_dvol_m:.0f}M/日 · 目标 {targets_label()}")

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)
    print(f"有效行情 {len(batch)} 只")

    liquid: list[tuple[str, pd.DataFrame, float]] = []
    for tk, df in batch.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m >= min_dvol_m:
            liquid.append((tk, df, dvol_m))
    liquid.sort(key=lambda x: -x[2])
    print(f"通过流动性 {len(liquid)} 只 · 开始参数网格 …")

    all_rows: list[dict] = []
    for i, (tk, df, dvol_m) in enumerate(liquid):
        if i % 10 == 0:
            print(f"  [{i + 1}/{len(liquid)}] {tk} …")
        all_rows.extend(
            scan_ticker_grid(tk, df, dvol_m, min_trades=min_trades, account_size=account_size)
        )

    df_out = pd.DataFrame(all_rows)
    if df_out.empty:
        print("无结果")
        return df_out

    tier_order = {"A": 0, "B": 1, "C": 2}
    df_out["_ord"] = df_out["tier"].map(tier_order).fillna(9)
    df_out = df_out.sort_values(["_ord", "gap_score", "成交额M"], ascending=[True, True, False])
    df_out = df_out.drop(columns=["_ord"])
    df_out.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    tier_a = df_out[df_out["tier"] == "A"].drop_duplicates(subset=["代码", "策略", "delta", "ma_window", "alloc"])
    tier_a.to_csv(PARETO_CSV, index=False, encoding="utf-8-sig")
    return df_out


def pick_fleet_from_results(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """从 Tier A 中按 ticker 去重，选 gap 最小的 n 套策略。"""
    tier_a = df[df["tier"] == "A"].copy()
    if tier_a.empty:
        tier_a = df.nsmallest(n * 3, "gap_score")
    picks: list[dict] = []
    seen_tickers: set[str] = set()
    for _, r in tier_a.sort_values("gap_score").iterrows():
        tk = str(r["代码"])
        if tk in seen_tickers and len(picks) >= n:
            continue
        seen_tickers.add(tk)
        picks.append({
            "ticker": tk,
            "strategy": r["策略"],
            "delta": float(r["delta"]),
            "ma_window": int(r["ma_window"]),
            "alloc": float(r["alloc"]),
            "ann_return": float(r["年化"]),
            "max_dd": float(r["最大回撤"]),
            "win_rate": float(r["胜率"]),
            "dvol_m": float(r["成交额M"]),
        })
        if len(picks) >= n:
            break
    return picks


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 64}\n全市场三标扫描 · {targets_label()}\n{'=' * 64}")
    if df.empty:
        return
    for tier in ["A", "B", "C"]:
        print(f"  Tier {tier}: {int((df['tier'] == tier).sum())} 条")
    tier_a = df[df["tier"] == "A"].drop_duplicates(subset=["代码", "delta", "ma_window", "alloc"])
    print(f"\n✅ Tier A 达标组合（去重后 {len(tier_a)} 条）：")
    for _, r in tier_a.head(20).iterrows():
        ma = f"MA{int(r['ma_window'])}" if r["ma_window"] else "无MA"
        print(
            f"  {r['代码']:6s} {r['策略']:8s} δ={r['delta']} {ma} alloc={r['alloc']:.0%} "
            f"年化={r['年化']:.1%} 回撤={r['最大回撤']:.1%} 胜率={r['胜率']:.1%} "
            f"成交额=${r['成交额M']:.0f}M"
        )
    tickers_a = sorted(tier_a["代码"].unique())
    print(f"\n达标标的（{len(tickers_a)} 只）：{', '.join(tickers_a) if tickers_a else '无'}")
    near = df[df["tier"] == "B"].nsmallest(8, "gap_score")
    if not near.empty:
        print("\n⚠ Tier B（差一项，Top 8）：")
        for _, r in near.iterrows():
            print(
                f"  {r['代码']:6s} {r['策略']} 年化={r['年化']:.1%} "
                f"回撤={r['最大回撤']:.1%} 胜率={r['胜率']:.1%}"
            )
    print(f"\n→ {RESULTS_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="全市场三标策略扫描")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--ann", type=float, default=0.80)
    p.add_argument("--max-dd", type=float, default=-0.10)
    p.add_argument("--win", type=float, default=0.85)
    p.add_argument("--min-dvol-m", type=float, default=30.0)
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--pick-fleet", type=int, default=5)
    args = p.parse_args()

    set_scan_targets(ann=args.ann, max_dd=args.max_dd, win=args.win, preset=None)
    df = run_market_scan(
        start=args.start, end=args.end,
        min_dvol_m=args.min_dvol_m, min_trades=args.min_trades,
        quick=args.quick, account_size=args.account,
    )
    _print_summary(df)
    if args.pick_fleet > 0 and not df.empty:
        picks = pick_fleet_from_results(df, n=args.pick_fleet)
        doc = {
            "updated": date.today().isoformat(),
            "targets": {"ann_return": args.ann, "max_dd": args.max_dd, "win_rate": args.win},
            "accounts": [
                {
                    "id": f"acct{i + 1}",
                    "label": f"账户{i + 1}",
                    "ticker": p["ticker"],
                    "strategy_type": "csp" if p["strategy"] == "CSP" else "weekly_ic",
                    "csp_params": {
                        "delta": p["delta"],
                        "ma_window": p["ma_window"],
                        "alloc_pct": p["alloc"],
                        "take_profit": 0.5,
                        "dte_days": 35,
                    },
                    "anchor_stats": {
                        "ann_return": p["ann_return"],
                        "max_dd": p["max_dd"],
                        "win_rate": p["win_rate"],
                        "source": "market_triple_scan",
                    },
                }
                for i, p in enumerate(picks)
            ],
        }
        FLEET_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n舰队推荐 {len(picks)} 账户 → {FLEET_JSON}")
        for a in doc["accounts"]:
            s = a["anchor_stats"]
            print(f"  {a['label']}: {a['ticker']} 年化={s['ann_return']:.1%} 回撤={s['max_dd']:.1%} 胜率={s['win_rate']:.1%}")


if __name__ == "__main__":
    main()
