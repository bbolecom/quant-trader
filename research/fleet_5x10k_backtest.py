#!/usr/bin/env python3
"""5×$10,000 舰队回测：多架构对比 + 年度切片 + 最优推荐。

每账户独立复利，舰队总账本 = 5 户权益之和（$50k 起点）。

架构：
  etf_iron      — 5 ETF 各开周铁鹰（SPY/QQQ/IWM/XLF/DIA）
  affordable_csp — 5 只 $10k 可负担 CSP（δ0.25 MA50 50%止盈）
  auto_tier_a   — CSP 放得下→CSP，否则→铁鹰（与 tier_a_csp 舰队逻辑一致）
  mixed_balanced — 每户：铁鹰25% + CSP20% + 现金55%
  etf3_csp2     — 3 ETF 铁鹰 + 2 廉价 CSP

用法：
    python research/fleet_5x10k_backtest.py
    python research/fleet_5x10k_backtest.py --year 2022
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import metrics as M
from quant.decline_income import (
    CSP_DTE_CAL,
    CSP_HOLD_TD,
    CSP_MA_WINDOW,
    CSP_STEP_TD,
    DEFAULT_VRP,
    equity_metrics_from_trades,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS, bs_put_price, realized_vol, strike_for_put_delta
from research.continued_search import ETF_CORE
from research.liquid_tier_a_scan import _avg_dollar_vol, build_candidate_pool
from research.market_pattern_scan import _weekly_iron_returns
from research.triple_target_scan import classify_tier, gap_score, set_scan_targets, targets_label

ACCOUNT = 10_000.0
N_ACCOUNTS = 5
FLEET_TOTAL = ACCOUNT * N_ACCOUNTS
IRON_MARGIN = 2_500.0
RESULTS_CSV = ROOT / "research" / "fleet_5x10k_results.csv"
BEST_JSON = ROOT / "research" / "fleet_5x10k_best.json"

DEFAULT_EXCLUDE = {"SNDK", "MSTR", "SOXL", "TQQQ", "UVIX", "UVXY", "VXX"}

# 廉价 CSP 候选（股价通常 <$80，$10k 可开 1–2 张）
AFFORDABLE_POOL = [
    "INTC", "BAC", "F", "SOFI", "HOOD", "PLTR", "AMD", "MU", "WDC", "COIN",
    "PFE", "T", "VZ", "KO", "CSCO", "NFLX", "PYPL", "RIVN", "DKNG",
]


def _csp_trades(
    close: pd.Series,
    *,
    account: float,
    delta: float = 0.25,
    take_profit: float = 0.5,
    max_margin_pct: float = 0.50,
    use_ma: bool = True,
) -> list[tuple[pd.Timestamp, float]]:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(CSP_MA_WINDOW).mean() if use_ma else None
    T = CSP_DTE_CAL / TRADING_DAYS
    cap = account * max_margin_pct
    out: list[tuple[pd.Timestamp, float]] = []
    i = max(25, CSP_MA_WINDOW if use_ma else 0)
    while i + CSP_HOLD_TD < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CSP_STEP_TD
            continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += CSP_STEP_TD
            continue
        iv = sigma * (1 + DEFAULT_VRP)
        K = strike_for_put_delta(S, T, iv, target_delta=delta)
        margin = K * 100
        if margin > cap or K <= 0:
            i += CSP_STEP_TD
            continue
        credit = bs_put_price(S, K, T, iv)
        alloc = min(margin / account, max_margin_pct)
        exited = False
        if take_profit > 0:
            path = close.iloc[i:i + CSP_HOLD_TD + 1]
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                remain = max(0.0, 1 - j / CSP_HOLD_TD)
                mark = max(0.0, K - Sj) + credit * remain * 0.5
                if credit - mark >= take_profit * credit:
                    out.append((pd.Timestamp(close.index[i]), (credit - mark) / K * alloc))
                    exited = True
                    break
        if not exited:
            ST = float(close.iloc[i + CSP_HOLD_TD])
            out.append((pd.Timestamp(close.index[i]), (credit - max(0.0, K - ST)) / K * alloc))
        i += CSP_STEP_TD
    return out


def _iron_trades(
    close: pd.Series,
    *,
    account: float,
    use_ma: bool = True,
) -> list[tuple[pd.Timestamp, float]]:
    close = close.astype(float).dropna()
    alloc = min(IRON_MARGIN / account, 0.25)
    out: list[tuple[pd.Timestamp, float]] = []
    hold = 5
    step = 5
    rv = realized_vol(close)
    ma = close.rolling(CSP_MA_WINDOW).mean() if use_ma else None
    from quant.decline_income import estimate_put_credit_spread, WEEKLY_SOUP_DELTA, WEEKLY_SOUP_WIDTH, WEEKLY_DTE

    i = max(25, CSP_MA_WINDOW)
    while i + hold < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += step
            continue
        if use_ma and ma is not None and not (S > float(ma.iloc[i])):
            i += step
            continue
        ks, kl, credit, _, _, _ = estimate_put_credit_spread(
            S, sigma, short_delta=WEEKLY_SOUP_DELTA, width=WEEKLY_SOUP_WIDTH,
            dte_days=WEEKLY_DTE, vrp=DEFAULT_VRP,
        )
        ST = float(close.iloc[i + hold])
        pnl = credit - (max(0.0, ks - ST) - max(0.0, kl - ST))
        out.append((pd.Timestamp(close.index[i]), pnl / WEEKLY_SOUP_WIDTH * alloc))
        i += step
    return out


def _mixed_trades(close: pd.Series, *, account: float) -> list[tuple[pd.Timestamp, float]]:
    iron = _iron_trades(close, account=account, use_ma=True)
    csp = _csp_trades(close, account=account, take_profit=0.5)
    combined = iron + csp
    combined.sort(key=lambda x: x[0])
    return combined


def _auto_trades(close: pd.Series, *, account: float, px_hint: float) -> list[tuple[pd.Timestamp, float]]:
    """CSP 放得下→CSP，否则铁鹰。"""
    cap = account * 0.50
    if px_hint * 100 * 0.9 <= cap:
        return _csp_trades(close, account=account, take_profit=0.5)
    return _iron_trades(close, account=account, use_ma=True)


def _account_equity(
    trades: list[tuple[pd.Timestamp, float]],
    *,
    year_start: str | None = None,
    year_end: str | None = None,
    initial: float = ACCOUNT,
) -> tuple[pd.Series, pd.Series]:
    if not trades:
        return pd.Series([initial], index=pd.DatetimeIndex([pd.Timestamp("1970-01-01")])), pd.Series(dtype=float)
    ser = pd.Series({ts: r for ts, r in trades}).sort_index()
    if year_start and year_end:
        ser = ser[(ser.index >= pd.Timestamp(year_start)) & (ser.index <= pd.Timestamp(year_end))]
    if ser.empty:
        return pd.Series([initial], index=pd.DatetimeIndex([pd.Timestamp(year_start or "1970-01-01")])), ser
    eq_vals = [initial]
    for r in ser:
        eq_vals.append(eq_vals[-1] * (1.0 + float(r)))
    eq = pd.Series(eq_vals[1:], index=pd.DatetimeIndex(ser.index))
    return eq, ser


def _stats(eq: pd.Series, rets: pd.Series) -> dict:
    if eq.empty or len(eq) < 1:
        return {"total": 0.0, "ann": 0.0, "max_dd": 0.0, "win_rate": 0.0, "trades": 0, "sharpe": 0.0}
    total = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) >= 1 else 0.0
    ann = M.cagr(eq) if isinstance(eq.index, pd.DatetimeIndex) and len(eq) >= 2 else total
    mdd = M.max_drawdown(eq) if len(eq) >= 2 else 0.0
    wr = float((rets > 0).mean()) if len(rets) else 0.0
    sharpe = M.sharpe_ratio(eq.pct_change().fillna(0.0)) if len(eq) >= 2 else 0.0
    return {"total": total, "ann": ann, "max_dd": mdd, "win_rate": wr, "trades": int(len(rets)), "sharpe": sharpe}


def _fleet_stats(account_eqs: list[pd.Series], account_rets: list[pd.Series]) -> dict:
    """5 户独立 → 舰队总权益 = 各户权益之和。"""
    if not account_eqs:
        return _stats(pd.Series(dtype=float), pd.Series(dtype=float))

    norm_eqs: list[pd.Series] = []
    for eq in account_eqs:
        if eq.empty:
            continue
        s = eq.copy()
        s.index = pd.DatetimeIndex(s.index)
        norm_eqs.append(s)
    if not norm_eqs:
        return _stats(pd.Series(dtype=float), pd.Series(dtype=float))

    all_idx = sorted(set().union(*[set(eq.index) for eq in norm_eqs]))
    fleet_vals = []
    for ts in all_idx:
        s = 0.0
        for eq in norm_eqs:
            sub = eq.loc[:ts]
            s += float(sub.iloc[-1]) if len(sub) else ACCOUNT
        fleet_vals.append(s)
    fleet_eq = pd.Series(fleet_vals, index=pd.DatetimeIndex(all_idx))
    all_rets = pd.concat(account_rets) if account_rets else pd.Series(dtype=float)
    st = _stats(fleet_eq, all_rets)
    st["fleet_start"] = FLEET_TOTAL
    st["fleet_end"] = float(fleet_eq.iloc[-1]) if len(fleet_eq) else FLEET_TOTAL
    st["per_account_ann"] = float(np.mean([
        _stats(eq, r)["ann"] for eq, r in zip(account_eqs, account_rets) if len(eq)
    ])) if account_eqs else 0.0
    return st


def _pick_affordable_csp(batch: dict, n: int = 5) -> list[str]:
    scored: list[tuple[str, float, float, float]] = []
    for tk in AFFORDABLE_POOL:
        if tk in DEFAULT_EXCLUDE:
            continue
        df = batch.get(tk)
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        px = float(close.iloc[-1])
        if px * 100 > ACCOUNT * 0.50:
            continue
        trades = _csp_trades(close, account=ACCOUNT, take_profit=0.5)
        if len(trades) < 30:
            continue
        eq, rets = _account_equity(trades)
        st = _stats(eq, rets)
        scored.append((tk, st["ann"], st["win_rate"], -st["max_dd"]))
    scored.sort(key=lambda x: (-x[1], -x[2], -x[3]))
    return [t[0] for t in scored[:n]]


def _run_fleet(
    name: str,
    slots: list[tuple[str, str]],
    batch: dict,
    *,
    year_start: str | None = None,
    year_end: str | None = None,
) -> dict:
    """slots: [(ticker, mode)] mode=iron|csp|auto|mixed"""
    account_eqs: list[pd.Series] = []
    account_rets: list[pd.Series] = []
    slot_detail = []

    for i, (tk, mode) in enumerate(slots[:N_ACCOUNTS]):
        df = batch.get(tk)
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        px = float(close.iloc[-1])
        if mode == "iron":
            trades = _iron_trades(close, account=ACCOUNT)
        elif mode == "csp":
            trades = _csp_trades(close, account=ACCOUNT, take_profit=0.5)
        elif mode == "mixed":
            trades = _mixed_trades(close, account=ACCOUNT)
        else:
            trades = _auto_trades(close, account=ACCOUNT, px_hint=px)

        eq, rets = _account_equity(trades, year_start=year_start, year_end=year_end)
        st = _stats(eq, rets)
        account_eqs.append(eq)
        account_rets.append(rets)
        slot_detail.append({
            "account": f"账户{i + 1}",
            "ticker": tk,
            "mode": mode,
            **st,
        })

    fleet = _fleet_stats(account_eqs, account_rets)
    tier = classify_tier(fleet["ann"], fleet["max_dd"], fleet["win_rate"])
    return {
        "fleet_id": name,
        "name": name,
        "slots": slots[:N_ACCOUNTS],
        "slot_detail": slot_detail,
        **fleet,
        "tier": tier,
        "gap_score": gap_score(fleet["ann"], fleet["max_dd"], fleet["win_rate"]),
    }


def run_all(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    end = end or date.today().isoformat()
    year_start = f"{year}-01-01" if year else None
    year_end = f"{year}-12-31" if year else None

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    pool = build_candidate_pool(use_broad=False)
    tickers = sorted(set(pool + ETF_CORE + AFFORDABLE_POOL + list(DEFAULT_EXCLUDE)))
    warmup = (pd.Timestamp(start) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    batch = yahoo.fetch_batch(tickers, warmup, end)

    affordable = _pick_affordable_csp(batch, n=5)
    etf5 = ETF_CORE[:5]

    architectures = {
        "etf_iron_5": [(t, "iron") for t in etf5],
        "affordable_csp_5": [(t, "csp") for t in affordable],
        "auto_tier_a_5": [(t, "auto") for t in (affordable[:3] + etf5[:2])],
        "mixed_balanced_5": [(t, "mixed") for t in (etf5[:3] + affordable[:2])],
        "etf3_csp2": [(t, "iron") for t in etf5[:3]] + [(t, "csp") for t in affordable[:2]],
    }

    rows = []
    for fid, slots in architectures.items():
        if len(slots) < N_ACCOUNTS:
            continue
        res = _run_fleet(fid, slots, batch, year_start=year_start, year_end=year_end)
        rows.append(res)

    df = pd.DataFrame([
        {
            "fleet_id": r["fleet_id"],
            "year": year or "full",
            "ann": r["ann"],
            "total": r["total"],
            "max_dd": r["max_dd"],
            "win_rate": r["win_rate"],
            "trades": r["trades"],
            "per_account_ann": r["per_account_ann"],
            "fleet_end": r.get("fleet_end", FLEET_TOTAL),
            "tier": r["tier"],
            "gap_score": r["gap_score"],
            "slots": json.dumps(r["slots"], ensure_ascii=False),
            "detail": json.dumps(r["slot_detail"], ensure_ascii=False),
        }
        for r in rows
    ])
    if not df.empty:
        def _bal(r: pd.Series) -> float:
            ann, dd, wr = max(r["ann"], 0), max(r["max_dd"], -1), r["win_rate"]
            return 0.35 * min(ann / 0.30, 1) + 0.35 * (1 + dd / 0.15) + 0.30 * wr

        df = df.assign(bal=df.apply(_bal, axis=1)).sort_values("bal", ascending=False)
    return df


def _print_report(df: pd.DataFrame, *, year: int | None = None) -> None:
    set_scan_targets(preset="moderate")
    label = f"{year} 年" if year else "全样本 2019+"
    print("=" * 68)
    print(f"5×$10,000 舰队回测 · {label} · 总本金 ${FLEET_TOTAL:,.0f}")
    print(f"目标参考：{targets_label()}")
    print("=" * 68)

    if df.empty:
        print("无结果")
        return

    best = df.iloc[0]
    print(f"\n★ 最优舰队：{best['fleet_id']}")
    print(f"  舰队全年/累计：{best['total']:.1%}  年化：{best['ann']:.1%}")
    print(f"  最大回撤：{best['max_dd']:.1%}  胜率：{best['win_rate']:.1%}")
    print(f"  期末权益：${best['fleet_end']:,.0f}  Tier={best['tier']}  gap={best['gap_score']:.3f}")
    slots = json.loads(best["slots"])
    print(f"  配置：")
    for i, (tk, mode) in enumerate(slots):
        print(f"    账户{i+1}: {tk} · {mode}")

    print(f"\n【全部架构排名】")
    for _, r in df.iterrows():
        print(
            f"  {r['fleet_id']}: 收益={r['total']:.1%} 年化={r['ann']:.1%} "
            f"回撤={r['max_dd']:.1%} 胜率={r['win_rate']:.1%} Tier={r['tier']}"
        )

    # SPY benchmark
    print(f"\n【单户平均年化】Top 舰队 per_account_ann = {best['per_account_ann']:.1%}")


def main() -> None:
    p = argparse.ArgumentParser(description="5×$10k 舰队回测")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--year", type=int, default=None, help="仅测某年，如 2022")
    p.add_argument("--all-years", action="store_true", help="输出 2019–2024 逐年表")
    args = p.parse_args()

    if args.all_years:
        print("逐年舰队最优（auto_tier_a 架构）…")
        for y in range(2019, 2025):
            df = run_all(start=args.start, end=args.end, year=y)
            if df.empty:
                continue
            b = df.iloc[0]
            print(f"  {y}: {b['fleet_id']} 收益={b['total']:+.1%} 回撤={b['max_dd']:.1%} 胜率={b['win_rate']:.0%}")
        return

    df = run_all(start=args.start, end=args.end, year=args.year)
    if not args.year:
        df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
        if not df.empty:
            best_row = df.iloc[0].to_dict()
            BEST_JSON.write_text(json.dumps(best_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_report(df, year=args.year)


if __name__ == "__main__":
    main()
