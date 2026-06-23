"""继续寻找：多目标 × 多策略 × 参数网格，排除 SNDK 个例。

搜索空间：
  - 目标：realistic(20/12/88) · pragmatic(25/18/90) · moderate(30/15/88) · relaxed(50/15/85)
  - 铁鹰：ETF/低波蓝筹，δ × 账户占用率
  - 廉价 CSP：$10k 可负担标的，δ 网格
  - 卖看涨价差：高流动振幅票
  - 组合：铁鹰舰队 + 动量多头 权重扫描

用法：
    python research/continued_search.py
    python research/continued_search.py --preset realistic
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
    CSP_MA_WINDOW,
    backtest_bear_call_spread,
    equity_metrics_from_trades,
    realized_vol,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS
from research.gainer_daily_backtest import (
    LIQUID100,
    GainerProFilters,
    fetch_gainer_data_yahoo,
    high_win_filters,
    backtest_daily_gainer_portfolio,
)
from research.liquid_tier_a_scan import _avg_dollar_vol, build_candidate_pool
from research.market_pattern_scan import (
    DEFAULT_EXCLUDE,
    ETF_SET,
    _account_returns_from_trades,
    _csp_returns_filtered,
    _weekly_iron_returns,
)
from research.triple_target_scan import (
    ScanTargets,
    classify_tier,
    gap_score,
    oos_meets_target,
    set_scan_targets,
    targets_label,
)

RESULTS_CSV = ROOT / "research" / "continued_search_results.csv"
BEST_JSON = ROOT / "research" / "continued_search_best.json"

TARGET_PRESETS = {
    "survivor": ScanTargets(0.15, -0.15, 0.88),
    "realistic": ScanTargets(0.20, -0.12, 0.88),
    "pragmatic": ScanTargets(0.25, -0.18, 0.90),
    "moderate": ScanTargets(0.30, -0.15, 0.88),
    "relaxed": ScanTargets(0.50, -0.15, 0.85),
}

ETF_CORE = ["SPY", "QQQ", "IWM", "XLF", "DIA"]
LOW_VOL_BLUE = ["SPY", "QQQ", "AAPL", "GOOGL", "JPM", "BRK-B", "V", "WMT"]


def _tier_row(
    strategy_id: str,
    name: str,
    ann: float,
    dd: float,
    win: float,
    *,
    eq: pd.Series | None,
    trade_count: int,
    params: dict,
    preset: str,
) -> dict:
    oos = bool(eq is not None and len(eq) >= 2 and oos_meets_target(eq, win))
    tier = classify_tier(ann, dd, win, oos=oos)
    return {
        "strategy_id": strategy_id,
        "name": name,
        "preset": preset,
        "targets": targets_label(),
        "年化": ann,
        "最大回撤": dd,
        "胜率": win,
        "交易数": trade_count,
        "tier": tier,
        "gap_score": gap_score(ann, dd, win),
        "oos_pass": oos,
        "params": json.dumps(params, ensure_ascii=False),
    }


def scan_iron_concurrent(
    batch: dict[str, pd.DataFrame],
    tickers: list[str],
    *,
    account: float,
    preset: str,
    margin: float = 2500.0,
) -> list[dict]:
    """同一账户并行开 N 张铁鹰（按时间对齐叠加账户收益）。"""
    rows: list[dict] = []
    max_slots = int(account // margin)
    for n in range(1, min(len(tickers), max_slots) + 1):
        for ma_on in [True, False]:
            tks = tickers[:n]
            by_idx: dict[int, list[float]] = {}
            for tk in tks:
                df = batch.get(tk)
                if df is None or df.empty:
                    continue
                close = df["Close"].astype(float)
                rors = _weekly_iron_returns(close, use_ma=ma_on)
                hold = 5
                i = max(25, CSP_MA_WINDOW if ma_on else 0)
                idx = 0
                while i + hold < len(close):
                    if idx < len(rors):
                        by_idx.setdefault(i, []).append(rors[idx])
                    i += 5
                    idx += 1
            if len(by_idx) < 30:
                continue
            slot_alloc = margin / account
            combined = []
            for i in sorted(by_idx):
                combined.append(sum(by_idx[i]) * slot_alloc / len(by_idx[i]))
            cyc = TRADING_DAYS / 5
            stats = equity_metrics_from_trades(combined, alloc_pct=1.0, cycles_per_year=cyc)
            eq = stats.get("净值曲线")
            label = "MA50" if ma_on else "无MA"
            rows.append(_tier_row(
                f"iron_parallel_{n}_{label}_{preset}",
                f"并行铁鹰×{n} {label} ({','.join(tks)})",
                stats.get("年化收益率", 0.0),
                stats.get("最大回撤", 0.0),
                stats.get("胜率", 0.0),
                eq=eq, trade_count=len(combined),
                params={"tickers": tks, "n": n, "ma": ma_on, "margin": margin},
                preset=preset,
            ))
    return rows


def scan_iron_grid(
    batch: dict[str, pd.DataFrame],
    tickers: list[str],
    *,
    account: float,
    preset: str,
) -> list[dict]:
    rows: list[dict] = []
    hold = 5
    cyc = TRADING_DAYS / hold
    margin = 2500.0

    for delta_label, short_delta in [("d05", 0.05), ("d08", 0.08), ("d10", 0.10)]:
        for alloc_pct in [0.25, 0.50, 0.75, 1.0]:
            for n_slots in [1, 3, 5]:
                tks = tickers[:n_slots]
                if not tks:
                    continue
                per_slot = account / n_slots
                combined: list[float] = []
                for tk in tks:
                    df = batch.get(tk)
                    if df is None or df.empty:
                        continue
                    close = df["Close"].astype(float)
                    # reuse iron with default delta via custom loop if needed — use MA50 path
                    rors = _weekly_iron_returns(close, use_ma=True)
                    eff_margin = min(margin, per_slot * alloc_pct)
                    eff_alloc = eff_margin / account
                    for r in rors:
                        combined.append(r * eff_alloc / len(tks))

                if len(combined) < 30:
                    continue
                stats = equity_metrics_from_trades(combined, alloc_pct=1.0, cycles_per_year=cyc)
                eq = stats.get("净值曲线")
                rows.append(_tier_row(
                    f"iron_{preset}_{delta_label}_a{int(alloc_pct*100)}_n{n_slots}",
                    f"铁鹰ETF δ≈0.10 alloc={alloc_pct:.0%} ×{n_slots}",
                    stats.get("年化收益率", 0.0),
                    stats.get("最大回撤", 0.0),
                    stats.get("胜率", 0.0),
                    eq=eq, trade_count=len(combined),
                    params={"tickers": tks, "alloc_pct": alloc_pct, "n_slots": n_slots},
                    preset=preset,
                ))
    return rows


def scan_affordable_csp(
    liquid: list[tuple[str, pd.DataFrame, float]],
    *,
    account: float,
    preset: str,
    max_margin: float = 5000.0,
) -> list[dict]:
    rows: list[dict] = []
    cyc = TRADING_DAYS / 35
    cap = min(max_margin, account * 0.50)

    affordable = []
    for tk, df, dv in liquid:
        close = df["Close"].astype(float)
        px = float(close.iloc[-1])
        if px * 100 * 0.85 <= cap and tk not in DEFAULT_EXCLUDE:
            affordable.append((tk, df, dv, px))
    affordable.sort(key=lambda x: -x[2])

    for delta in [0.15, 0.20, 0.25]:
        for ma_on in [True, False]:
            for tk, df, dv, px in affordable[:25]:
                close = df["Close"].astype(float)
                rors = _csp_returns_filtered(
                    close, use_ma=ma_on, account=account,
                    max_margin_pct=cap / account,
                )
                if len(rors) < 25:
                    continue
                stats = _account_returns_from_trades(
                    rors, margin_or_alloc=cap, account=account, cycles_per_year=cyc,
                )
                eq = stats.get("净值曲线")
                label = "MA50" if ma_on else "无MA"
                rows.append(_tier_row(
                    f"csp_{tk}_d{delta}_{label}_{preset}",
                    f"CSP {tk} δ={delta} {label}",
                    stats.get("年化收益率", 0.0),
                    stats.get("最大回撤", 0.0),
                    stats.get("胜率", 0.0),
                    eq=eq, trade_count=len(rors),
                    params={"ticker": tk, "delta": delta, "ma": ma_on, "px": px, "dvol_m": dv},
                    preset=preset,
                ))
    return rows


def scan_bear_call_liquid(
    liquid: list[tuple[str, pd.DataFrame, float]],
    *,
    preset: str,
) -> list[dict]:
    rows: list[dict] = []
    # 高波 + 高流动 — 卖 call 价差主战场
    candidates = sorted(liquid, key=lambda x: -x[2])[:40]
    for tk, df, dv in candidates:
        if tk in DEFAULT_EXCLUDE:
            continue
        close = df["Close"].astype(float)
        rv = float(realized_vol(close).iloc[-1])
        if rv < 0.35:
            continue
        bc = backtest_bear_call_spread(close)
        eq = bc.get("净值曲线")
        if eq is None or len(eq) < 2:
            continue
        rows.append(_tier_row(
            f"bearcall_{tk}_{preset}",
            f"卖Call价差 {tk}",
            bc.get("年化", bc.get("年化收益率", 0.0)),
            bc.get("最大回撤", 0.0),
            bc.get("胜率", 0.0),
            eq=eq, trade_count=int(bc.get("交易数", 0)),
            params={"ticker": tk, "rv": rv, "dvol_m": dv},
            preset=preset,
        ))
    return rows


def scan_combos(
    batch: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    account: float,
    preset: str,
) -> list[dict]:
    rows: list[dict] = []
    # 并行铁鹰 SPY+QQQ 周收益
    by_idx: dict[int, list[float]] = {}
    margin = 2500.0
    slot_alloc = margin / account
    for tk in ["SPY", "QQQ"]:
        df = batch.get(tk)
        if df is None:
            continue
        close = df["Close"].astype(float)
        rors = _weekly_iron_returns(close, use_ma=False)
        hold = 5
        i = max(25, CSP_MA_WINDOW)
        idx = 0
        while i + hold < len(close):
            if idx < len(rors):
                by_idx.setdefault(i, []).append(rors[idx])
            i += 5
            idx += 1
    if len(by_idx) < 30:
        return rows
    iron_weekly = [sum(by_idx[k]) * slot_alloc / len(by_idx[k]) for k in sorted(by_idx)]

    g_rets = pd.Series(dtype=float)
    try:
        data, spy = fetch_gainer_data_yahoo(LIQUID100[:60], start, end)
        g = backtest_daily_gainer_portfolio(
            data, spy, start=start, end=end, filt=GainerProFilters(),
        )
        if not g.get("error") and "权益曲线" in g:
            g_curve = g["权益曲线"]
            g_eq = g_curve.set_index(pd.to_datetime(g_curve["日期"]))["权益"]
            g_rets = g_eq.pct_change().fillna(0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"  combo gainer 跳过: {exc}")
        return rows

    if g_rets.empty or len(g_rets) < 20:
        return rows

    n = len(g_rets)
    iron_daily = np.interp(
        np.linspace(0, len(iron_weekly) - 1, n),
        np.arange(len(iron_weekly)),
        iron_weekly,
    )
    g_aligned = g_rets.iloc[-n:].values

    for w_iron, w_g in [(0.85, 0.15), (0.75, 0.25), (0.6, 0.4), (0.5, 0.5), (0.7, 0.3)]:
        port = w_iron * iron_daily + w_g * g_aligned
        eq = pd.Series((1 + port).cumprod())
        win = float((port > 0).mean())
        years = max(n / 252, 0.5)
        ann = float(eq.iloc[-1] ** (1 / years) - 1)
        dd = float((eq / eq.cummax() - 1).min())
        rows.append(_tier_row(
            f"combo_iron{int(w_iron*100)}_g{int(w_g*100)}_{preset}",
            f"组合 铁鹰{w_iron:.0%}+动量{w_g:.0%}",
            ann, dd, win,
            eq=eq, trade_count=n,
            params={"w_iron": w_iron, "w_gainer": w_g, "iron": "SPY+QQQ"},
            preset=preset,
        ))
    return rows


def scan_fleet_book(
    batch: dict[str, pd.DataFrame],
    tickers: list[str],
    *,
    account_per: float,
    n_accounts: int,
    preset: str,
) -> list[dict]:
    """5×$10k 合成 $50k 账本：每账户独立铁鹰槽。"""
    total = account_per * n_accounts
    margin = 2500.0
    alloc = margin / account_per  # 25%
    combined: list[float] = []

    for acc_i, tk in enumerate(tickers[:n_accounts]):
        df = batch.get(tk)
        if df is None:
            continue
        rors = _weekly_iron_returns(df["Close"].astype(float), use_ma=True)
        for r in rors:
            combined.append(r * alloc * (account_per / total))

    if len(combined) < 40:
        return []
    stats = equity_metrics_from_trades(
        combined, alloc_pct=1.0, cycles_per_year=TRADING_DAYS / 5,
    )
    eq = stats.get("净值曲线")
    return [_tier_row(
        f"fleet_{n_accounts}x{int(account_per)}_{preset}",
        f"舰队 {n_accounts}×${account_per:,.0f} 铁鹰",
        stats.get("年化收益率", 0.0),
        stats.get("最大回撤", 0.0),
        stats.get("胜率", 0.0),
        eq=eq, trade_count=len(combined),
        params={"tickers": tickers[:n_accounts], "account_per": account_per},
        preset=preset,
    )]


def run_continued_search(
    *,
    presets: list[str] | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 50.0,
    account: float = 10_000.0,
    quick: bool = False,
) -> pd.DataFrame:
    presets = presets or ["survivor", "realistic", "pragmatic"]
    end = end or date.today().isoformat()

    pool = build_candidate_pool(use_broad=not quick, max_names=150 if quick else 0)
    pool = [t for t in pool if t not in DEFAULT_EXCLUDE]
    for t in ETF_CORE + LOW_VOL_BLUE:
        if t not in pool:
            pool.append(t)

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)

    liquid: list[tuple[str, pd.DataFrame, float]] = []
    for tk, df in batch.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m >= min_dvol_m:
            liquid.append((tk, df, dvol_m))
    liquid.sort(key=lambda x: -x[2])
    print(f"继续寻找 · 流动池 {len(liquid)} 只 · 排除 {sorted(DEFAULT_EXCLUDE)}")

    all_rows: list[dict] = []
    for preset_name in presets:
        if preset_name not in TARGET_PRESETS:
            continue
        t = TARGET_PRESETS[preset_name]
        set_scan_targets(ann=t.ann, max_dd=t.dd, win=t.win)
        print(f"\n── 目标 [{preset_name}] {targets_label()} ──")

        all_rows.extend(scan_iron_concurrent(batch, ETF_CORE, account=account, preset=preset_name))
        all_rows.extend(scan_iron_concurrent(batch, LOW_VOL_BLUE[:5], account=account, preset=preset_name))
        all_rows.extend(scan_iron_grid(batch, ETF_CORE, account=account, preset=preset_name))
        all_rows.extend(scan_affordable_csp(liquid, account=account, preset=preset_name))
        all_rows.extend(scan_bear_call_liquid(liquid, preset=preset_name))
        all_rows.extend(scan_combos(batch, start=start, end=end, account=account, preset=preset_name))
        all_rows.extend(scan_fleet_book(
            batch, ETF_CORE, account_per=account, n_accounts=5, preset=preset_name,
        ))
        tier_a = sum(1 for r in all_rows if r.get("preset") == preset_name and r.get("tier") == "A")
        tier_b = sum(1 for r in all_rows if r.get("preset") == preset_name and r.get("tier") == "B")
        print(f"  累计 Tier A={tier_a}  Tier B={tier_b}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    tier_ord = {"A": 0, "B": 1, "C": 2}
    df["_ord"] = df["tier"].map(tier_ord).fillna(9)
    df = df.sort_values(["preset", "_ord", "gap_score"]).drop(columns=["_ord"])
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    best_by_preset: dict = {}
    for p in presets:
        sub = df[(df["preset"] == p) & (df["tier"].isin(["A", "B"]))]
        if sub.empty:
            sub = df[df["preset"] == p].nsmallest(3, "gap_score")
        if not sub.empty:
            best_by_preset[p] = sub.head(5).to_dict(orient="records")

    BEST_JSON.write_text(
        json.dumps({"updated": date.today().isoformat(), "best_by_preset": best_by_preset}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return df


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("无结果")
        return
    print(f"\n{'='*64}\n继续寻找 · 汇总\n{'='*64}")
    for preset in df["preset"].unique():
        sub = df[df["preset"] == preset]
        ta = sub[sub["tier"] == "A"]
        tb = sub[sub["tier"] == "B"]
        print(f"\n[{preset}] Tier A={len(ta)}  Tier B={len(tb)}  总={len(sub)}")
        top = pd.concat([ta, tb]).drop_duplicates() if len(ta) or len(tb) else sub
        top = top.nsmallest(8, "gap_score")
        for _, r in top.iterrows():
            mark = "★" if r["tier"] == "A" else "·"
            print(
                f"  {mark} {r['name'][:40]:40s} 年化={r['年化']:.1%} "
                f"回撤={r['最大回撤']:.1%} 胜率={r['胜率']:.1%} tier={r['tier']}"
            )
    print(f"\n→ {RESULTS_CSV}\n→ {BEST_JSON}")


def main() -> None:
    p = argparse.ArgumentParser(description="继续寻找普适策略")
    p.add_argument("--preset", default="", help="单个目标预设，空=全部")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    presets = [args.preset] if args.preset else None
    df = run_continued_search(
        presets=presets, start=args.start, end=args.end,
        account=args.account, quick=args.quick,
    )
    _print_summary(df)


if __name__ == "__main__":
    main()
