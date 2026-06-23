"""全市场高流动性 Tier A 扫描：不限 SNDK，按成交额过滤后回测 CSP / 周铁鹰。

用法：
    python research/liquid_tier_a_scan.py --quick
    python research/liquid_tier_a_scan.py --min-dvol-m 100 --top 20
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

from quant.decline_income import (
    CSP_DTE_CAL,
    CSP_HOLD_TD,
    CSP_MA_WINDOW,
    CSP_STEP_TD,
    DEFAULT_VRP,
    backtest_weekly_put_spread,
    equity_metrics_from_trades,
    realized_vol,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS, bs_put_price, strike_for_put_delta
from research.gainer_daily_backtest import LIQUID100, GAINER_MOMENTUM
from research.triple_target_scan import (
    classify_tier,
    gap_score,
    oos_meets_target,
    set_scan_targets,
    targets_label,
)
from quant.screener import fetch_broad_universe

RESULTS_CSV = ROOT / "research" / "liquid_tier_a_results.csv"
FLEET_PICKS_JSON = ROOT / "research" / "liquid_fleet_picks.json"

# 与 scan_csp_universe 对齐的硬编码大盘池
MEGA_LIQUID = sorted(set(LIQUID100 + GAINER_MOMENTUM[:80] + [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "MU",
    "WDC", "STX", "SNDK", "PLTR", "COIN", "SMCI", "MSTR", "HOOD", "SOFI", "UBER",
    "JPM", "BAC", "V", "MA", "XOM", "LLY", "UNH", "WMT", "COST", "HD",
    "SPY", "QQQ", "IWM", "TQQQ", "SOXL", "TSM", "AVGO", "ORCL", "CRM", "NFLX",
    "BABA", "PYPL", "RIVN", "DKNG", "CRWD", "SNOW", "PANW", "NET", "SHOP",
]))


def build_candidate_pool(*, use_broad: bool = True, max_names: int = 0) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in MEGA_LIQUID:
        u = str(t).strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    if use_broad:
        try:
            for t in fetch_broad_universe(screen_count=200, extra=LIQUID100):
                u = str(t).strip().upper()
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
        except Exception:  # noqa: BLE001
            pass
    if max_names > 0:
        out = out[:max_names]
    return out


def _avg_dollar_vol(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    c = close.astype(float)
    v = volume.astype(float)
    return float((c * v).tail(window).mean())


def _csp_trade_returns(
    close: pd.Series,
    *,
    delta: float = 0.25,
    ma_window: int = CSP_MA_WINDOW,
    vrp: float = DEFAULT_VRP,
) -> list[float]:
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(ma_window).mean() if ma_window else None
    T = CSP_DTE_CAL / TRADING_DAYS
    rors: list[float] = []
    i = max(25, ma_window)
    while i + CSP_HOLD_TD < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += CSP_STEP_TD
            continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += CSP_STEP_TD
            continue
        iv = sigma * (1 + vrp)
        K = strike_for_put_delta(S, T, iv, target_delta=delta)
        credit = bs_put_price(S, K, T, iv)
        if K <= 0:
            i += CSP_STEP_TD
            continue
        ST = float(close.iloc[i + CSP_HOLD_TD])
        rors.append((credit - max(0.0, K - ST)) / K)
        i += CSP_STEP_TD
    return rors


def score_ticker(
    ticker: str,
    df: pd.DataFrame,
    *,
    dvol_m: float,
    allocs: list[float] | None = None,
    min_trades: int = 30,
    account_size: float = 10_000.0,
) -> dict | None:
    if df is None or df.empty or len(df) < CSP_MA_WINDOW + 60:
        return None
    close = df["Close"].astype(float)
    px = float(close.iloc[-1])
    rv_pct = float(realized_vol(close).iloc[-1]) * 100
    ma50 = float(close.rolling(CSP_MA_WINDOW).mean().iloc[-1])
    above_ma = px > ma50
    allocs = allocs or [0.20, 0.35, 0.50]

    rors = _csp_trade_returns(close, delta=0.25, ma_window=CSP_MA_WINDOW)
    if len(rors) < min_trades:
        return None

    best_csp: dict | None = None
    cyc_yr = TRADING_DAYS / CSP_HOLD_TD
    capital_per_contract = 0.0
    try:
        from quant.decline_income import csp_income_plan
        plan = csp_income_plan(ticker, df, delta=0.25, ma_window=CSP_MA_WINDOW)
        if plan:
            capital_per_contract = plan.capital_per_contract
    except Exception:  # noqa: BLE001
        capital_per_contract = px * 100 * 0.85

    for alloc in allocs:
        stats = equity_metrics_from_trades(rors, alloc_pct=alloc, cycles_per_year=cyc_yr)
        eq = stats.get("净值曲线")
        if eq is None or len(eq) < 2:
            continue
        ann = stats["年化收益率"]
        dd = stats["最大回撤"]
        win = stats["胜率"]
        tier = classify_tier(ann, dd, win, oos=oos_meets_target(eq, win))
        row = {
            "代码": ticker,
            "现价": round(px, 2),
            "成交额M": round(dvol_m, 1),
            "RV%": round(rv_pct, 1),
            "站上MA50": above_ma,
            "策略": "CSP",
            "delta": 0.25,
            "alloc": alloc,
            "年化": ann,
            "最大回撤": dd,
            "胜率": win,
            "交易数": len(rors),
            "tier": tier,
            "gap_score": gap_score(ann, dd, win),
            "oos_pass": oos_meets_target(eq, win),
            "单张担保金$": round(capital_per_contract, 0),
            "fits_10k": capital_per_contract <= account_size * 0.50,
        }
        if best_csp is None or row["gap_score"] < best_csp["gap_score"]:
            best_csp = row

    # 周铁鹰（$10k 账户）
    wk = backtest_weekly_put_spread(close)
    wk_row: dict | None = None
    if wk and wk.get("净值曲线") is not None:
        eq = wk["净值曲线"]
        ann = wk.get("年化", 0.0)
        dd = wk.get("最大回撤", 0.0)
        win = wk.get("胜率", 0.0)
        tier = classify_tier(ann, dd, win, oos=oos_meets_target(eq, win))
        margin = 2500.0
        fits_10k = margin <= account_size * 0.25
        wk_row = {
            "代码": ticker,
            "现价": round(px, 2),
            "成交额M": round(dvol_m, 1),
            "RV%": round(rv_pct, 1),
            "站上MA50": above_ma,
            "策略": "偏斜铁鹰",
            "delta": 0.10,
            "alloc": 0.25,
            "年化": ann,
            "最大回撤": dd,
            "胜率": win,
            "交易数": wk.get("交易数", 0),
            "tier": tier,
            "gap_score": gap_score(ann, dd, win),
            "oos_pass": oos_meets_target(eq, win),
            "单张担保金$": margin,
            "fits_10k": fits_10k,
        }

    if best_csp is None and wk_row is None:
        return None

    # 选 gap 更小的主策略
    candidates = [x for x in [best_csp, wk_row] if x is not None]
    primary = min(candidates, key=lambda x: x["gap_score"])
    if prefer := (account_size < capital_per_contract * 0.5 and wk_row):
        if wk_row.get("fits_10k") and wk_row["gap_score"] <= primary["gap_score"] * 1.5:
            primary = wk_row
    return primary


def scan_liquid_universe(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 50.0,
    min_trades: int = 30,
    quick: bool = False,
    account_size: float = 10_000.0,
) -> pd.DataFrame:
    set_scan_targets(preset="relaxed")
    end = end or date.today().isoformat()
    pool = build_candidate_pool(use_broad=not quick, max_names=120 if quick else 0)
    print(f"候选池 {len(pool)} 只 · 流动性门槛 ≥ ${min_dvol_m:.0f}M/日 · 目标 {targets_label()}")

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
    print(f"通过流动性 {len(liquid)} 只")

    rows: list[dict] = []
    for i, (tk, df, dvol_m) in enumerate(liquid):
        if i % 20 == 0:
            print(f"  回测 {i + 1}/{len(liquid)} …")
        scored = score_ticker(tk, df, dvol_m=dvol_m, min_trades=min_trades, account_size=account_size)
        if scored:
            rows.append(scored)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    df_out = df_out.sort_values(["tier", "gap_score"], ascending=[True, True])
    tier_order = {"A": 0, "B": 1, "C": 2}
    df_out["_tier_ord"] = df_out["tier"].map(tier_order).fillna(9)
    df_out = df_out.sort_values(["_tier_ord", "gap_score"]).drop(columns=["_tier_ord"])
    df_out.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    return df_out


def pick_fleet_from_patterns(n: int = 5, *, account: float = 10_000.0) -> list[str]:
    """从规律扫描结果选舰队：优先 ETF + 低波铁鹰，排除 Tier-A 单票异常。"""
    rules_path = ROOT / "research" / "market_pattern_rules.json"
    ticker_path = ROOT / "research" / "market_pattern_tickers.csv"
    if rules_path.exists():
        rules = json.loads(rules_path.read_text())
        bp = rules.get("best_portfolio") or {}
        tks = bp.get("tickers") or []
        if len(tks) >= n:
            return tks[:n]
    if ticker_path.exists():
        tdf = pd.read_csv(ticker_path)
        sub = tdf[
            (tdf["策略"] == "铁鹰+MA50")
            & (tdf["ETF"] == True)  # noqa: E712
            & (tdf["RV%"] < 35)
        ].sort_values(["胜率", "成交额M"], ascending=[False, False])
        picks = sub["代码"].drop_duplicates().tolist()
        if len(picks) >= n:
            return picks[:n]
        sub2 = tdf[
            (tdf["策略"] == "铁鹰+MA50")
            & (tdf["RV%"] < 30)
            & (tdf["成交额M"] >= 500)
        ].sort_values(["胜率", "成交额M"], ascending=[False, False])
        picks = sub2["代码"].drop_duplicates().tolist()
        if len(picks) >= n:
            return picks[:n]
    return ["SPY", "QQQ", "IWM", "XLF", "JPM"][:n]


def pick_fleet_tickers(
    n: int = 5,
    *,
    account_size: float = 10_000.0,
    min_dvol_m: float = 50.0,
    prefer_weekly_for_small: bool = True,
    results_path: Path | None = None,
    use_patterns: bool = True,
) -> list[str]:
    """从扫描结果或规律模型选取 n 只标的。"""
    if use_patterns:
        cs_path = ROOT / "research" / "continued_search_best.json"
        if cs_path.exists():
            data = json.loads(cs_path.read_text())
            for preset in ("realistic", "survivor", "pragmatic"):
                rows = (data.get("best_by_preset") or {}).get(preset) or []
                for row in rows:
                    if row.get("tier") in ("A", "B") and "并行铁鹰" in str(row.get("name", "")):
                        import re
                        m = re.search(r"\(([^)]+)\)", row["name"])
                        if m:
                            tks = [x.strip() for x in m.group(1).split(",")]
                            picks: list[str] = []
                            for tk in tks:
                                if tk not in picks:
                                    picks.append(tk)
                            while len(picks) < n:
                                for d in ["IWM", "XLF", "DIA", "JPM"]:
                                    if d not in picks:
                                        picks.append(d)
                                    if len(picks) >= n:
                                        break
                            return picks[:n]
        picks = pick_fleet_from_patterns(n, account=account_size)
        if len(picks) >= n:
            return picks

    path = results_path or RESULTS_CSV
    if path.exists():
        df = pd.read_csv(path)
    elif FLEET_PICKS_JSON.exists():
        data = json.loads(FLEET_PICKS_JSON.read_text())
        return data.get("tickers", [])[:n]
    else:
        df = scan_liquid_universe(min_dvol_m=min_dvol_m, quick=True, account_size=account_size)

    if df.empty:
        return ["INTC", "MU", "AMD", "WDC", "NVDA"][:n]

    picks: list[str] = []
    sub = df.copy()
    if prefer_weekly_for_small:
        if "fits_10k" in sub.columns:
            fit = sub[sub["fits_10k"] == True]  # noqa: E712
            if len(fit) >= n:
                sub = fit
        else:
            fit = sub[sub["策略"] == "偏斜铁鹰"]
            if len(fit) >= n:
                sub = fit
    tier_a = sub[sub["tier"] == "A"]
    if len(tier_a) >= n:
        sub = tier_a
    elif len(sub[sub["tier"] == "B"]) >= n:
        sub = sub[sub["tier"] == "B"]

    sub = sub.sort_values(["成交额M", "gap_score"], ascending=[False, True])
    for tk in sub["代码"]:
        if tk not in picks:
            picks.append(str(tk))
        if len(picks) >= n:
            break
    return picks


def save_fleet_picks(tickers: list[str], *, min_dvol_m: float) -> None:
    FLEET_PICKS_JSON.write_text(
        json.dumps({"tickers": tickers, "min_dvol_m": min_dvol_m, "updated": date.today().isoformat()}, indent=2),
        encoding="utf-8",
    )


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}\n全市场流动性扫描 · {targets_label()}\n{'='*60}")
    if df.empty:
        print("无结果")
        return
    for tier in ["A", "B", "C"]:
        print(f"  Tier {tier}: {int((df['tier'] == tier).sum())}")
    print(f"\nTop 10（按 gap_score）：")
    for _, r in df.head(10).iterrows():
        print(
            f"  {r['代码']:6s} {r['策略']:8s} 年化={r['年化']:.1%} 回撤={r['最大回撤']:.1%} "
            f"胜率={r['胜率']:.1%} 成交额=${r['成交额M']:.0f}M tier={r['tier']}"
        )
    tier_a = df[df["tier"] == "A"]
    if not tier_a.empty:
        print(f"\n✅ Tier-A 标的（{len(tier_a)}）：{', '.join(tier_a['代码'].tolist())}")
    print(f"\n→ {RESULTS_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description="全市场高流动性 Tier A 扫描")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--min-dvol-m", type=float, default=50.0, help="最低日均成交额(百万USD)")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--quick", action="store_true", help="缩小候选池加速")
    p.add_argument("--pick-fleet", type=int, default=0, help="输出 Top-N 舰队标的并写入 JSON")
    args = p.parse_args()

    df = scan_liquid_universe(
        start=args.start, end=args.end,
        min_dvol_m=args.min_dvol_m, min_trades=args.min_trades,
        quick=args.quick, account_size=args.account,
    )
    _print_summary(df)
    if args.pick_fleet > 0:
        picks = pick_fleet_tickers(args.pick_fleet, account_size=args.account, min_dvol_m=args.min_dvol_m)
        save_fleet_picks(picks, min_dvol_m=args.min_dvol_m)
        print(f"\n舰队推荐 {args.pick_fleet} 只：{', '.join(picks)}")
        print(f"→ {FLEET_PICKS_JSON}")


if __name__ == "__main__":
    main()
