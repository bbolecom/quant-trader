#!/usr/bin/env python3
"""S8U Ultra80 · 高流通票全市场回测。

在 LIQUID100 / 广谱流动性池中，用路径止盈+2%/止损-5% 回测 S8U。
MSTR/SMCI/COIN 用分标的规则；其余票用通用精英门槛。

用法：
    python research/s8u_liquid_universe_backtest.py
    python research/s8u_liquid_universe_backtest.py --min-dvol-m 100 --quick
    python research/s8u_liquid_universe_backtest.py --ticker NVDA AMD PLTR
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.move_pattern import extract_trajectory_features_5d
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import LIQUID100
from research.liquid_tier_a_scan import MEGA_LIQUID, _avg_dollar_vol, build_candidate_pool
from research.medallion_short import fetch_shares
from research.ticker_pattern_backtest import (
    FEE,
    _attach_spy,
    _backtest_long_only,
    summarize_trades,
)
from research.ticker_pattern_mine import TRAIN_END

OUT_JSON = ROOT / "research" / "s8u_liquid_universe_backtest.json"
OUT_CSV = ROOT / "research" / "s8u_liquid_universe_backtest.csv"
APPROVED_JSON = ROOT / "research" / "s8u_approved_tickers.json"

# 杠杆/宽基 ETF 不参与单票路径策略
EXCLUDE = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "TQQQ", "SQQQ", "SOXL", "SOXS", "LABU", "FNGU",
    "NVDL", "TSLL", "MSTX", "CONL", "XLE", "XLF", "XLV", "XLK", "XLI", "XLP",
    "XLY", "XLB", "XLU", "XLRE", "ARKK", "UVXY", "VXX",
})


def build_liquid_universe(
    *,
    min_dvol_m: float = 50.0,
    quick: bool = False,
    extra: list[str] | None = None,
) -> list[str]:
    pool = build_candidate_pool(use_broad=not quick, max_names=80 if quick else 0)
    for t in LIQUID100 + list(MEGA_LIQUID):
        u = str(t).upper()
        if u not in pool:
            pool.append(u)
    if extra:
        for t in extra:
            u = str(t).upper()
            if u not in pool:
                pool.append(u)
    seen: set[str] = set()
    out: list[str] = []
    for tk in pool:
        u = tk.upper()
        if u in EXCLUDE or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def build_panel_from_df(ticker: str, df: pd.DataFrame, shares_out: float | None) -> pd.DataFrame:
    feat = extract_trajectory_features_5d(
        df, shares_out=shares_out, horizon=5, up_threshold=0.02, down_threshold=0.02,
    )
    if feat.empty:
        return feat
    feat["代码"] = ticker.upper()
    feat["avg_dvol_m"] = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
    feat["日期"] = pd.to_datetime(feat["日期"]).dt.strftime("%Y-%m-%d")
    return feat


def backtest_one(
    ticker: str,
    df: pd.DataFrame,
    spy_close: pd.Series,
    *,
    start: str,
    end: str,
    shares_out: float | None,
    alloc_pct: float,
) -> dict | None:
    panel = build_panel_from_df(ticker, df, shares_out)
    if panel.empty:
        return None
    panel = _attach_spy(panel, spy_close)
    close = df["Close"].astype(float)

    periods = [
        ("全样本", None, None),
        ("样本内2019-2023", None, TRAIN_END),
        ("样本外2024+", "2024-01-01", None),
    ]
    rows: list[dict] = []
    for label, p_start, p_end in periods:
        ps = start if p_start is None else max(start, p_start)
        pe = end if p_end is None else min(end, p_end)
        if pd.Timestamp(ps) > pd.Timestamp(pe):
            continue
        tr = _backtest_long_only(panel, close, df, start=ps, end=pe, ultra=True)
        sm = summarize_trades(tr, alloc_pct=alloc_pct)
        rows.append({"区间": label, **sm})

    if not rows:
        return None
    dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
    return {
        "代码": ticker.upper(),
        "成交额M": round(dvol_m, 1),
        "results": rows,
    }


def run_universe(
    tickers: list[str],
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 50.0,
    min_trades: int = 3,
    alloc_pct: float = 0.25,
) -> dict:
    end = end or date.today().isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)

    print(f"拉取 {len(tickers)} 只行情 …")
    batch = yahoo.fetch_batch(tickers, start, end)
    shares = fetch_shares(list(batch.keys()))

    flat: list[dict] = []
    skipped: list[dict] = []
    for i, tk in enumerate(tickers):
        df = batch.get(tk)
        if df is None or df.empty:
            skipped.append({"代码": tk, "原因": "无行情"})
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m < min_dvol_m:
            skipped.append({"代码": tk, "原因": f"成交额{dvol_m:.0f}M<{min_dvol_m}M"})
            continue
        if (i + 1) % 25 == 0:
            print(f"  回测 {i + 1}/{len(tickers)} …")
        doc = backtest_one(
            tk, df, spy, start=start, end=end,
            shares_out=shares.get(tk.upper()),
            alloc_pct=alloc_pct,
        )
        if doc is None:
            skipped.append({"代码": tk, "原因": "特征不足"})
            continue
        for r in doc["results"]:
            flat.append({
                "代码": doc["代码"],
                "成交额M": doc["成交额M"],
                "区间": r["区间"],
                "笔数": r["笔数"],
                "胜率": r["胜率"],
                "年化": r["年化"],
                "最大回撤": r["最大回撤"],
                "夏普": r["夏普"],
                "均单笔%": r["均单笔%"],
            })

    df_all = pd.DataFrame(flat)
    summary = _aggregate(df_all, min_trades=min_trades)
    approved = export_oos_approved(
        df_all, min_trades=5, min_win=0.80, period="样本外2024+",
    ) if not df_all.empty else {}
    doc = {
        "generated": date.today().isoformat(),
        "strategy": "S8U",
        "start": start,
        "end": end,
        "min_dvol_m": min_dvol_m,
        "min_trades": min_trades,
        "alloc_pct": alloc_pct,
        "fee_bps_roundtrip": FEE * 10000,
        "universe_size": len(tickers),
        "backtested": len(df_all["代码"].unique()) if not df_all.empty else 0,
        "skipped": skipped,
        "summary_table": flat,
        "aggregate": summary,
        "oos_approved": approved,
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if not df_all.empty:
        df_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return doc


def export_oos_approved(
    df: pd.DataFrame,
    *,
    min_trades: int = 5,
    min_win: float = 0.80,
    period: str = "样本外2024+",
) -> dict:
    """导出 OOS 准入清单 → s8u_approved_tickers.json。"""
    sub = df[(df["区间"] == period) & (df["笔数"] >= min_trades) & (df["胜率"] >= min_win)].copy()
    sub = sub.sort_values(["胜率", "笔数"], ascending=[False, False])
    details = []
    for _, r in sub.iterrows():
        details.append({
            "代码": r["代码"],
            "oos_n": int(r["笔数"]),
            "oos_win": round(float(r["胜率"]), 4),
            "笔数": int(r["笔数"]),
            "胜率": round(float(r["胜率"]), 4),
            "年化": round(float(r["年化"]), 4),
            "最大回撤": round(float(r["最大回撤"]), 4),
            "成交额M": round(float(r["成交额M"]), 1),
        })
    tickers = [d["代码"] for d in details]
    total_n = int(sub["笔数"].sum()) if not sub.empty else 0
    w_win = float((sub["胜率"] * sub["笔数"]).sum() / total_n) if total_n else 0.0
    doc = {
        "generated": date.today().isoformat(),
        "criteria": {
            "period": period,
            "min_win_rate": min_win,
            "min_trades": min_trades,
        },
        "tickers": tickers,
        "pool_stats": {
            "count": len(tickers),
            "total_oos_trades": total_n,
            "weighted_win_rate": round(w_win, 4),
        },
        "details": details,
    }
    APPROVED_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def _aggregate(df: pd.DataFrame, *, min_trades: int) -> dict:
    if df.empty:
        return {}
    out: dict = {}
    for period in ("全样本", "样本外2024+"):
        sub = df[df["区间"] == period].copy()
        if sub.empty:
            continue
        # 单票筛选：至少 min_trades 笔
        eligible = sub[sub["笔数"] >= min_trades].copy()
        eligible = eligible.sort_values("胜率", ascending=False)

        winners = eligible[eligible["胜率"] >= 0.80]
        total_n = int(eligible["笔数"].sum())
        if total_n > 0:
            # 按笔数加权胜率
            w_win = float((eligible["胜率"] * eligible["笔数"]).sum() / total_n)
        else:
            w_win = 0.0

        out[period] = {
            "eligible_tickers": int(len(eligible)),
            "win_ge_80pct_tickers": int(len(winners)),
            "total_trades": total_n,
            "weighted_win_rate": round(w_win, 4),
            "median_win_rate": round(float(eligible["胜率"].median()), 4) if len(eligible) else 0.0,
            "top10": eligible.head(10)[
                ["代码", "笔数", "胜率", "年化", "最大回撤", "成交额M"]
            ].to_dict(orient="records"),
            "bottom5": eligible.tail(5)[
                ["代码", "笔数", "胜率", "年化", "最大回撤"]
            ].to_dict(orient="records") if len(eligible) >= 5 else [],
        }
    return out


def print_report(doc: dict) -> None:
    print(f"\n{'=' * 88}")
    print(
        f"S8U 高流通票回测 · {doc.get('generated')} · "
        f"{doc.get('start')} ~ {doc.get('end')}"
    )
    print(
        f"流动性 ≥ ${doc.get('min_dvol_m', 50):.0f}M/日 · "
        f"回测 {doc.get('backtested', 0)} 只 · "
        f"跳过 {len(doc.get('skipped') or [])} 只"
    )
    print(f"{'=' * 88}")

    agg = doc.get("aggregate") or {}
    for period in ("全样本", "样本外2024+"):
        s = agg.get(period)
        if not s:
            continue
        print(f"\n【{period}】")
        print(
            f"  有效标的 {s['eligible_tickers']} 只 · "
            f"合计 {s['total_trades']} 笔 · "
            f"加权胜率 {s['weighted_win_rate']:.1%} · "
            f"中位胜率 {s['median_win_rate']:.1%} · "
            f"单票≥80% {s['win_ge_80pct_tickers']} 只"
        )
        print("  Top10（按胜率，≥3笔）：")
        for r in s.get("top10") or []:
            print(
                f"    {r['代码']:5} n={int(r['笔数']):3d}  "
                f"胜率{r['胜率']:.1%}  年化{r['年化']:+.1%}  "
                f"回撤{r['最大回撤']:.1%}  ${r['成交额M']:.0f}M"
            )

    print(f"\n→ {OUT_JSON}")
    print(f"→ {OUT_CSV}")
    approved = doc.get("oos_approved") or {}
    if approved.get("tickers"):
        ps = approved.get("pool_stats") or {}
        print(
            f"→ {APPROVED_JSON}  "
            f"准入 {ps.get('count', 0)} 只 · "
            f"OOS加权胜率 {ps.get('weighted_win_rate', 0):.1%}"
        )
    print(f"{'=' * 88}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="S8U Ultra80 高流通票全市场回测")
    ap.add_argument("--ticker", nargs="+", default=None, help="指定标的；默认扫流动性池")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-dvol-m", type=float, default=50.0, help="最低日均成交额(百万美元)")
    ap.add_argument("--min-trades", type=int, default=3, help="汇总统计最少笔数")
    ap.add_argument("--alloc", type=float, default=0.25)
    ap.add_argument("--quick", action="store_true", help="缩小候选池加速")
    args = ap.parse_args()

    if args.ticker:
        tickers = [t.upper() for t in args.ticker]
    else:
        tickers = build_liquid_universe(min_dvol_m=args.min_dvol_m, quick=args.quick)

    doc = run_universe(
        tickers,
        start=args.start,
        end=args.end,
        min_dvol_m=args.min_dvol_m,
        min_trades=args.min_trades,
        alloc_pct=args.alloc,
    )
    print_report(doc)


if __name__ == "__main__":
    main()
