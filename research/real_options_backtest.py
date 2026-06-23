#!/usr/bin/env python3
"""真实期权回测：真实 EOD 期权链入场价 + 到期真实股价内在值结算（零 Black-Scholes）。

数据：
  · 期权入场价 = DoltHub post-no-preference/options 真实 EOD bid/ask
  · 到期结算   = 真实标的收盘价的内在价值（yfinance）
无任何模型估值；无数据的日期如实跳过并计入覆盖率。

用法：
  python research/real_options_backtest.py --symbols AAPL,MSFT,NVDA --start 2024-01-01 --end 2025-06-01 --step 5
  python research/real_options_backtest.py --structure csp --symbols AAPL,AMD
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.options_history import (  # noqa: E402
    fetch_eod_chain,
    pick_bear_call_eod,
    pick_bear_put_debit_eod,
    pick_csp_eod,
    settle_bear_call_at_expiry,
    settle_bear_put_debit_at_expiry,
    settle_csp_at_expiry,
)
from quant.providers.yahoo import YahooProvider  # noqa: E402
from research.gainer_daily_backtest import LIQUID100  # noqa: E402

OUT_DIR = ROOT / "research"
_ETF_SKIP = {"SPY", "QQQ", "GOOG", "XLE", "XLF", "IWM", "DIA"}
_LIQUID100 = [t for t in LIQUID100 if t not in _ETF_SKIP]
_LIQUID40 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "V",
    "UNH", "XOM", "LLY", "MA", "HD", "PG", "COST", "MRK", "ABBV", "KO",
    "PEP", "AMD", "NFLX", "CRM", "ORCL", "BAC", "WMT", "CSCO", "ACN", "MCD",
    "LIN", "TMO", "ABT", "DIS", "INTU", "QCOM", "IBM", "GE", "TXN", "AMAT",
]
POOLS = {
    "liquid40": _LIQUID40,
    "liquid100": _LIQUID100,
    "mega8": ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL"],
}


def _resolve_symbols(pool: str | None, symbols_arg: str) -> list[str]:
    if pool:
        if pool not in POOLS:
            raise ValueError(f"未知 pool {pool!r}，可选: {', '.join(POOLS)}")
        return list(POOLS[pool])
    if symbols_arg:
        return [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    return list(POOLS["liquid40"])


def _batch_prices(symbols: list[str], start: str, end: str) -> dict[str, pd.Series]:
    yahoo = YahooProvider()
    end_ext = (pd.Timestamp(end) + pd.Timedelta(days=70)).strftime("%Y-%m-%d")
    batch = yahoo.fetch_batch(symbols, start, end_ext)
    out: dict[str, pd.Series] = {}
    for sym, df in batch.items():
        s = df["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out[sym.upper()] = s
    return out


def _prefetch_chains(pairs: list[tuple[str, str]], workers: int) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    if not pairs:
        return out

    def _one(pair: tuple[str, str]) -> tuple[tuple[str, str], pd.DataFrame]:
        sym, day = pair
        return pair, fetch_eod_chain(sym, day)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one, p) for p in pairs]
        for fut in as_completed(futs):
            pair, chain = fut.result()
            out[pair] = chain
            done += 1
            if done % 100 == 0 or done == len(pairs):
                print(f"  期权链进度 {done}/{len(pairs)}", flush=True)
    return out


def _close_on_or_after(prices: pd.Series, day: pd.Timestamp) -> float | None:
    sub = prices[prices.index >= day]
    return float(sub.iloc[0]) if not sub.empty else None


def _prepare_dataset(
    symbols: list[str], start: str, end: str, *, step_days: int, workers: int,
) -> tuple[dict[str, pd.Series], dict[str, list[tuple[pd.Timestamp, str]]], dict[tuple[str, str], pd.DataFrame]]:
    print(f"批量拉取 {len(symbols)} 只标的股价 …", flush=True)
    price_map = _batch_prices(symbols, start, end)
    pairs: list[tuple[str, str]] = []
    sym_days: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for sym in symbols:
        prices = price_map.get(sym)
        if prices is None or prices.empty:
            continue
        cal = prices[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
        sampled = cal.index[::max(1, step_days)]
        sym_days[sym] = [(ts, ts.strftime("%Y-%m-%d")) for ts in sampled]
        for _, day in sym_days[sym]:
            pairs.append((sym, day))
    print(f"预取 {len(pairs)} 组 (标的×采样日) 真实期权链 …", flush=True)
    chain_map = _prefetch_chains(pairs, workers)
    return price_map, sym_days, chain_map


def run_real_backtest(
    symbols: list[str], start: str, end: str, *,
    step_days: int = 5, structure: str = "bear_call",
    otm: float = 0.08, width_pct: float = 0.10, min_dte: int = 2, max_dte: int = 45,
    workers: int = 6,
    _dataset: tuple | None = None,
) -> dict:
    rows: list[dict] = []
    attempted = 0
    no_data = 0
    no_struct = 0

    if _dataset is None:
        price_map, sym_days, chain_map = _prepare_dataset(symbols, start, end, step_days=step_days, workers=workers)
    else:
        price_map, sym_days, chain_map = _dataset

    for sym in symbols:
        prices = price_map.get(sym)
        if prices is None or prices.empty or sym not in sym_days:
            continue
        print(f"[{sym}] 处理 {len(sym_days[sym])} 个采样日 …", flush=True)
        for ts, day in sym_days[sym]:
            attempted += 1
            chain = chain_map.get((sym, day), pd.DataFrame())
            if chain.empty:
                no_data += 1
                continue
            spot = float(prices.loc[ts])
            if structure == "bear_call":
                plan, why = pick_bear_call_eod(chain, spot, day, otm=otm, width_pct=width_pct,
                                               min_dte=min_dte, max_dte=max_dte)
            elif structure == "csp":
                plan, why = pick_csp_eod(chain, spot, day, otm=max(otm, 0.10),
                                         min_dte=min_dte, max_dte=max_dte)
            elif structure == "bear_put_debit":
                plan, why = pick_bear_put_debit_eod(chain, spot, day, otm=otm, width_pct=width_pct,
                                                    min_dte=min_dte, max_dte=max_dte)
            else:
                raise ValueError(f"未知结构 {structure}")
            if plan is None:
                no_struct += 1
                continue
            exp_close = _close_on_or_after(prices, plan["expiration"])
            if exp_close is None:
                continue
            if structure == "bear_call":
                pnl = settle_bear_call_at_expiry(plan, exp_close)
                margin = plan["max_loss"] if plan["max_loss"] > 0 else plan["width"] * 100
                legs = f"卖C${plan['short_strike']:g}/买C${plan['long_strike']:g}"
            elif structure == "csp":
                pnl = settle_csp_at_expiry(plan, exp_close)
                margin = plan["collateral"]
                legs = f"卖P${plan['short_strike']:g}"
            else:
                pnl = settle_bear_put_debit_at_expiry(plan, exp_close)
                margin = plan["max_loss"] if plan["max_loss"] > 0 else plan["debit"] * 100
                legs = f"买P${plan['long_strike']:g}/卖P${plan['short_strike']:g}"
            ret_on_margin = pnl / margin if margin > 0 else 0.0
            rows.append({
                "代码": sym, "入场日": day, "现价": round(spot, 2),
                "结构": structure, "腿": legs,
                "到期": plan["expiration"].strftime("%Y-%m-%d"), "持有天数": plan["dte"],
                "入场IV%": round(plan.get("short_iv", 0) * 100, 1),
                "到期收盘": round(exp_close, 2),
                "盈亏$/张": round(pnl, 0), "保证金$": round(margin, 0),
                "回报/保证金%": round(ret_on_margin * 100, 2),
                "胜负": "胜" if pnl > 0 else "负",
            })

    detail = pd.DataFrame(rows)
    summary = {
        "结构": structure, "标的数": len(symbols),
        "尝试日数": attempted, "有真实数据": attempted - no_data,
        "数据覆盖率": f"{(attempted - no_data) / max(attempted, 1):.0%}",
        "无可成交结构": no_struct, "成交笔数": len(detail),
    }
    if not detail.empty:
        r = detail["回报/保证金%"] / 100.0
        summary.update({
            "胜率": round(float((detail["盈亏$/张"] > 0).mean()), 4),
            "总盈亏$/张合计": round(float(detail["盈亏$/张"].sum()), 0),
            "均盈亏$/张": round(float(detail["盈亏$/张"].mean()), 1),
            "中位盈亏$/张": round(float(detail["盈亏$/张"].median()), 1),
            "中位回报/保证金%": round(float(r.median()) * 100, 3),
            "均回报/保证金%(易受离群值影响)": round(float(r.mean()) * 100, 2),
            "10分位回报%": round(float(r.quantile(0.1)) * 100, 3),
            "最差单笔$/张": round(float(detail["盈亏$/张"].min()), 0),
            "盈亏比(总赢/总亏)": round(
                float(detail.loc[detail["盈亏$/张"] > 0, "盈亏$/张"].sum())
                / abs(float(detail.loc[detail["盈亏$/张"] < 0, "盈亏$/张"].sum()) or 1), 2),
        })
    return {"detail": detail, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=list(POOLS.keys()), default=None,
                   help="预设标的池：liquid40(默认)/liquid100/mega8")
    p.add_argument("--symbols", default="", help="逗号分隔标的（覆盖 --pool）")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-06-01")
    p.add_argument("--step", type=int, default=5, help="每隔 N 个交易日采样一次")
    p.add_argument("--workers", type=int, default=6, help="并行拉取期权链线程数")
    p.add_argument("--structure", choices=["bear_call", "csp", "bear_put_debit", "all"], default="bear_call")
    p.add_argument("--otm", type=float, default=0.08)
    p.add_argument("--width", type=float, default=0.10)
    args = p.parse_args()

    syms = _resolve_symbols(args.pool or ("liquid40" if not args.symbols else None), args.symbols)
    structures = ["bear_call", "csp", "bear_put_debit"] if args.structure == "all" else [args.structure]
    print(f"真实期权回测 · {len(syms)} 标的 · {args.start}~{args.end} · 每{args.step}日采样 · {args.workers}线程")
    print("数据：DoltHub 真实EOD期权 + 到期真实股价结算（零模型）\n")

    all_summaries = {}
    dataset = _prepare_dataset(syms, args.start, args.end, step_days=args.step, workers=args.workers)
    for structure in structures:
        print(f"\n{'='*60}\n结构: {structure}\n{'='*60}")
        res = run_real_backtest(syms, args.start, args.end, step_days=args.step,
                                structure=structure, otm=args.otm, width_pct=args.width,
                                workers=args.workers, _dataset=dataset)
        s = res["summary"]
        all_summaries[structure] = s
        print("\n【真实回测汇总】")
        for k, v in s.items():
            print(f"  {k}: {v}")
        detail = res["detail"]
        if not detail.empty:
            tag = f"{structure}_{args.start[:4]}_{args.end[:4]}_{len(syms)}sym"
            out = OUT_DIR / f"real_options_{tag}.csv"
            detail.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n明细 → {out}")
            by_sym = detail.groupby("代码").agg(
                笔数=("盈亏$/张", "count"),
                胜率=("胜负", lambda x: (x == "胜").mean()),
                中位盈亏=("盈亏$/张", "median"),
                总盈亏=("盈亏$/张", "sum"),
            ).round(2).sort_values("总盈亏", ascending=False)
            print("\n按标的（前15）:")
            print(by_sym.head(15).to_string())

    summary_path = OUT_DIR / f"real_options_summary_{args.start[:4]}_{args.end[:4]}_{len(syms)}sym.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 JSON → {summary_path}")


if __name__ == "__main__":
    main()
