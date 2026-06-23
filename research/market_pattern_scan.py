"""全市场规律挖掘：排除单票异常（如 SNDK），寻找可复制的普遍性。

分析维度：
  1. 特征分桶 — RV%、成交额、是否 ETF、是否站上 MA50
  2. 双策略横截面 — CSP vs 偏斜铁鹰，统一 $10k 账户口径
  3. 组合策略 — ETF 篮子、低波 CSP 轮动、铁鹰等权舰队
  4. 规则提炼 — 哪些条件在全市场统计上稳定成立

用法：
    python research/market_pattern_scan.py
    python research/market_pattern_scan.py --exclude SNDK,MSTR,SOXL
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

from quant.decline_income import (
    CSP_DTE_CAL,
    CSP_HOLD_TD,
    CSP_MA_WINDOW,
    CSP_STEP_TD,
    DEFAULT_VRP,
    equity_metrics_from_trades,
    realized_vol,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.vol_decay import TRADING_DAYS, bs_put_price, strike_for_put_delta
from research.liquid_tier_a_scan import (
    _avg_dollar_vol,
    build_candidate_pool,
)
from research.triple_target_scan import (
    classify_tier,
    gap_score,
    oos_meets_target,
    set_scan_targets,
    targets_label,
)

BUCKET_CSV = ROOT / "research" / "market_pattern_buckets.csv"
TICKER_CSV = ROOT / "research" / "market_pattern_tickers.csv"
PORTFOLIO_CSV = ROOT / "research" / "market_pattern_portfolios.csv"
RULES_JSON = ROOT / "research" / "market_pattern_rules.json"

ETF_SET = {
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "TQQQ", "SOXL", "ARKK", "HYG", "TLT", "GLD", "SLV", "EEM", "FXI",
}

DEFAULT_EXCLUDE = {"SNDK", "MSTR", "SOXL", "TQQQ"}  # 极端个例 / 杠杆 ETF


def _rv_bucket(rv_pct: float) -> str:
    if rv_pct < 25:
        return "低波<25%"
    if rv_pct < 45:
        return "中波25-45%"
    if rv_pct < 70:
        return "高波45-70%"
    return "极端>70%"


def _dvol_bucket(dvol_m: float) -> str:
    if dvol_m >= 5000:
        return "超大盘≥5B"
    if dvol_m >= 1000:
        return "大盘1-5B"
    if dvol_m >= 200:
        return "中盘200M-1B"
    return "小盘<200M"


def _weekly_iron_returns(close: pd.Series, *, use_ma: bool = True) -> list[float]:
    """与 decline_income.backtest_weekly_put_spread 一致（含 50% 止盈）。"""
    from quant.decline_income import (
        WEEKLY_DTE,
        WEEKLY_SOUP_DELTA,
        WEEKLY_SOUP_TAKE_PROFIT,
        WEEKLY_SOUP_WIDTH,
        estimate_put_credit_spread,
    )

    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(CSP_MA_WINDOW).mean() if use_ma else None
    hold = max(1, int(WEEKLY_DTE * TRADING_DAYS / 7))
    width = WEEKLY_SOUP_WIDTH
    rors: list[float] = []
    i = max(25, CSP_MA_WINDOW if use_ma else 0)
    while i + hold < len(close):
        S = float(close.iloc[i])
        sigma = float(rv.iloc[i])
        if not np.isfinite(sigma) or sigma <= 0:
            i += 5
            continue
        if ma is not None and not (S > float(ma.iloc[i])):
            i += 5
            continue
        ks, kl, credit, mrg, _, _ = estimate_put_credit_spread(
            S, sigma, short_delta=WEEKLY_SOUP_DELTA, width=width,
            dte_days=WEEKLY_DTE, vrp=DEFAULT_VRP,
        )
        if mrg <= 0:
            i += 5
            continue
        exited = False
        if WEEKLY_SOUP_TAKE_PROFIT > 0:
            path = close.iloc[i:i + hold + 1]
            for j in range(1, len(path)):
                Sj = float(path.iloc[j])
                remain = max(0.0, 1 - j / hold)
                short_loss = max(0.0, ks - Sj) - max(0.0, kl - Sj)
                mark = short_loss + credit * remain * 0.5
                if credit - mark >= WEEKLY_SOUP_TAKE_PROFIT * credit:
                    rors.append((credit - mark) / width)
                    exited = True
                    break
        if not exited:
            ST = float(close.iloc[i + hold])
            pnl = credit - (max(0.0, ks - ST) - max(0.0, kl - ST))
            rors.append(pnl / width)
        i += 5
    return rors


def _csp_returns_filtered(
    close: pd.Series,
    *,
    use_ma: bool,
    account: float,
    max_margin_pct: float = 0.50,
) -> list[float]:
    """CSP 回测：保证金超过账户上限的周期跳过（小账户现实约束）。"""
    close = close.astype(float).dropna()
    rv = realized_vol(close)
    ma = close.rolling(CSP_MA_WINDOW).mean() if use_ma else None
    T = CSP_DTE_CAL / TRADING_DAYS
    cap = account * max_margin_pct
    rors: list[float] = []
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
        K = strike_for_put_delta(S, T, iv, target_delta=0.25)
        margin = K * 100
        if margin > cap:
            i += CSP_STEP_TD
            continue
        credit = bs_put_price(S, K, T, iv)
        if K <= 0:
            i += CSP_STEP_TD
            continue
        ST = float(close.iloc[i + CSP_HOLD_TD])
        rors.append((credit - max(0.0, K - ST)) / K)
        i += CSP_STEP_TD
    return rors


def _account_returns_from_trades(
    trade_returns: list[float],
    *,
    margin_or_alloc: float,
    account: float = 10_000.0,
    cycles_per_year: float,
) -> dict:
    """把单笔收益换算成账户收益率（margin/account × trade_return）。"""
    alloc = min(margin_or_alloc / account, 1.0)
    return equity_metrics_from_trades(
        trade_returns, alloc_pct=alloc, cycles_per_year=cycles_per_year,
    )


def score_ticker_both(
    ticker: str,
    df: pd.DataFrame,
    *,
    dvol_m: float,
    account: float = 10_000.0,
    min_trades: int = 40,
) -> list[dict]:
    """同一标的 CSP + 铁鹰，统一账户口径。"""
    if df is None or df.empty or len(df) < CSP_MA_WINDOW + 80:
        return []
    close = df["Close"].astype(float)
    px = float(close.iloc[-1])
    rv_pct = float(realized_vol(close).iloc[-1]) * 100
    ma50 = float(close.rolling(CSP_MA_WINDOW).mean().iloc[-1])
    above = px > ma50
    is_etf = ticker in ETF_SET
    cyc_csp = TRADING_DAYS / CSP_HOLD_TD

    rows: list[dict] = []
    wk_margin = 2500.0
    cyc_wk = TRADING_DAYS / max(1, int(7 * TRADING_DAYS / 7))

    for ma_on, label in [(True, "CSP+MA50"), (False, "CSP无MA")]:
        rors = _csp_returns_filtered(close, use_ma=ma_on, account=account)
        if len(rors) < min_trades // 2:
            continue
        stats = _account_returns_from_trades(
            rors, margin_or_alloc=account * 0.50, account=account, cycles_per_year=cyc_csp,
        )
        eq = stats.get("净值曲线")
        if eq is None or len(eq) < 2:
            continue
        ann, dd, win = stats["年化收益率"], stats["最大回撤"], stats["胜率"]
        rows.append({
            "代码": ticker, "策略": label, "现价": px, "RV%": rv_pct,
            "成交额M": dvol_m, "RV桶": _rv_bucket(rv_pct), "流动性桶": _dvol_bucket(dvol_m),
            "ETF": is_etf, "站上MA50": above,
            "年化": ann, "最大回撤": dd, "胜率": win, "交易数": len(rors),
            "tier": classify_tier(ann, dd, win, oos=oos_meets_target(eq, win)),
            "gap_score": gap_score(ann, dd, win),
            "账户口径$": account, "占用$": account * 0.50,
        })

    for ma_on, label in [(True, "铁鹰+MA50"), (False, "铁鹰无MA")]:
        rors = _weekly_iron_returns(close, use_ma=ma_on)
        if len(rors) < min_trades // 3:
            continue
        stats = _account_returns_from_trades(
            rors, margin_or_alloc=wk_margin, account=account, cycles_per_year=cyc_wk,
        )
        eq = stats.get("净值曲线")
        if eq is None or len(eq) < 2:
            continue
        ann, dd, win = stats["年化收益率"], stats["最大回撤"], stats["胜率"]
        rows.append({
            "代码": ticker, "策略": label, "现价": px, "RV%": rv_pct,
            "成交额M": dvol_m, "RV桶": _rv_bucket(rv_pct), "流动性桶": _dvol_bucket(dvol_m),
            "ETF": is_etf, "站上MA50": above,
            "年化": ann, "最大回撤": dd, "胜率": win, "交易数": len(rors),
            "tier": classify_tier(ann, dd, win, oos=oos_meets_target(eq, win)),
            "gap_score": gap_score(ann, dd, win),
            "账户口径$": account, "占用$": wk_margin,
        })
    return rows


def aggregate_buckets(tdf: pd.DataFrame) -> pd.DataFrame:
    """按 RV桶 × 策略 × ETF 聚合中位数。"""
    if tdf.empty:
        return tdf
    grp = tdf.groupby(["RV桶", "策略", "ETF"], observed=True)
    out = grp.agg(
        样本数=("代码", "count"),
        年化中位=("年化", "median"),
        年化均值=("年化", "mean"),
        回撤中位=("最大回撤", "median"),
        胜率中位=("胜率", "median"),
        TierA数=("tier", lambda s: int((s == "A").sum())),
        TierB数=("tier", lambda s: int((s == "B").sum())),
    ).reset_index()
    out["达标率TierAB"] = (
        out["TierA数"] + out["TierB数"]
    ) / out["样本数"].clip(lower=1)
    return out.sort_values(["RV桶", "策略"])


def backtest_equal_basket(
    batch: dict[str, pd.DataFrame],
    tickers: list[str],
    *,
    strategy: str = "iron_ma50",
    account: float = 10_000.0,
    slots: int = 5,
) -> dict:
    """等权组合：每周期对 slots 只标的各开 1 张，账户收益 = 平均。"""
    per_slot = account / slots
    margin = 2500.0 if "铁鹰" in strategy else None
    all_trade_dates: list[tuple[int, float]] = []  # (time_idx, account_return)

    for tk in tickers:
        df = batch.get(tk)
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        rv = realized_vol(close)
        ma = close.rolling(CSP_MA_WINDOW).mean()
        hold = 5 if "铁鹰" in strategy else CSP_HOLD_TD
        step = 5
        i = max(25, CSP_MA_WINDOW)
        while i + hold < len(close):
            S = float(close.iloc[i])
            sigma = float(rv.iloc[i])
            if not np.isfinite(sigma) or sigma <= 0:
                i += step
                continue
            use_ma = strategy in ("iron_ma50", "csp_ma50")
            if use_ma and not (S > float(ma.iloc[i])):
                i += step
                continue
            if "铁鹰" in strategy:
                from quant.decline_income import estimate_put_credit_spread, WEEKLY_SOUP_DELTA, WEEKLY_SOUP_WIDTH, WEEKLY_DTE
                ks, kl, credit, mrg, _, _ = estimate_put_credit_spread(
                    S, sigma, short_delta=WEEKLY_SOUP_DELTA, width=WEEKLY_SOUP_WIDTH,
                    dte_days=WEEKLY_DTE, vrp=DEFAULT_VRP,
                )
                ST = float(close.iloc[i + hold])
                pnl = credit - (max(0.0, ks - ST) - max(0.0, kl - ST))
                tr = pnl / WEEKLY_SOUP_WIDTH
                alloc = min(margin / per_slot, 1.0) if margin else 0.25
            else:
                iv = sigma * (1 + DEFAULT_VRP)
                T = CSP_DTE_CAL / TRADING_DAYS
                K = strike_for_put_delta(S, T, iv, target_delta=0.25)
                credit = bs_put_price(S, K, T, iv)
                ST = float(close.iloc[i + hold])
                tr = (credit - max(0.0, K - ST)) / K if K > 0 else 0.0
                mrg = K * 100
                alloc = min(mrg / per_slot, 1.0)
            all_trade_dates.append((i, tr * alloc / len(tickers)))
            i += step

    if not all_trade_dates:
        return {}
    all_trade_dates.sort(key=lambda x: x[0])
    rors = [r for _, r in all_trade_dates]
    cyc = TRADING_DAYS / 5
    stats = equity_metrics_from_trades(rors, alloc_pct=1.0, cycles_per_year=cyc)
    eq = stats.get("净值曲线")
    return {
        "name": strategy,
        "tickers": tickers,
        "slots": slots,
        "交易数": len(rors),
        "年化": stats.get("年化收益率", 0.0),
        "最大回撤": stats.get("最大回撤", 0.0),
        "胜率": stats.get("胜率", 0.0),
        "tier": classify_tier(
            stats.get("年化收益率", 0.0),
            stats.get("最大回撤", 0.0),
            stats.get("胜率", 0.0),
            oos=bool(eq is not None and oos_meets_target(eq, stats.get("胜率", 0.0))),
        ),
        "gap_score": gap_score(
            stats.get("年化收益率", 0.0),
            stats.get("最大回撤", 0.0),
            stats.get("胜率", 0.0),
        ),
    }


def derive_rules(tdf: pd.DataFrame, bdf: pd.DataFrame, portfolios: list[dict]) -> dict:
    """从统计结果提炼可执行规则。"""
    rules: list[dict] = []

    if not bdf.empty:
        # 铁鹰 + 低波
        ic = bdf[(bdf["策略"].str.contains("铁鹰")) & (bdf["RV桶"] == "低波<25%")]
        if not ic.empty:
            row = ic.sort_values("胜率中位", ascending=False).iloc[0]
            rules.append({
                "id": "iron_low_vol",
                "pattern": "低波(RV<25%) + 铁鹰+MA50",
                "median_win": round(float(row["胜率中位"]), 3),
                "median_ann": round(float(row["年化中位"]), 3),
                "median_dd": round(float(row["回撤中位"]), 3),
                "n": int(row["样本数"]),
                "action": "全市场最稳：超大盘低波名/ETF 做铁鹰，MA50 过滤",
            })

        csp_mid = bdf[(bdf["策略"].str.contains("CSP")) & (bdf["RV桶"] == "中波25-45%")]
        if not csp_mid.empty:
            row = csp_mid.sort_values("年化中位", ascending=False).iloc[0]
            rules.append({
                "id": "csp_mid_vol",
                "pattern": "中波(RV 25-45%) + CSP+MA50",
                "median_win": round(float(row["胜率中位"]), 3),
                "median_ann": round(float(row["年化中位"]), 3),
                "median_dd": round(float(row["回撤中位"]), 3),
                "n": int(row["样本数"]),
                "action": "收益/风险均衡带：中波蓝筹 CSP，必须 MA50",
            })

    ma50_uplift: dict = {}
    if not tdf.empty:
        etf_ic = tdf[(tdf["ETF"]) & (tdf["策略"] == "铁鹰+MA50")]
        ma50_uplift = {
            "note": "MA50 在全市场的胜率提升（中位数差）",
            "csp": round(float(
                tdf[tdf["策略"] == "CSP+MA50"]["胜率"].median()
                - tdf[tdf["策略"] == "CSP无MA"]["胜率"].median()
            ), 4) if len(tdf[tdf["策略"] == "CSP+MA50"]) else 0,
            "iron": round(float(
                tdf[tdf["策略"] == "铁鹰+MA50"]["胜率"].median()
                - tdf[tdf["策略"] == "铁鹰无MA"]["胜率"].median()
            ), 4) if len(tdf[tdf["策略"] == "铁鹰+MA50"]) else 0,
        }
        if len(etf_ic) >= 3:
            rules.append({
                "id": "etf_iron_fleet",
                "pattern": "ETF 铁鹰舰队",
                "median_win": round(float(etf_ic["胜率"].median()), 3),
                "median_ann": round(float(etf_ic["年化"].median()), 3),
                "median_dd": round(float(etf_ic["最大回撤"].median()), 3),
                "n": len(etf_ic),
                "action": "5×$10k 优先 SPY/QQQ/IWM/XLF 等 ETF，不押单票",
            })

    tier_a = tdf[tdf["tier"] == "A"] if not tdf.empty else pd.DataFrame()
    rules.append({
        "id": "avoid_outliers",
        "pattern": "排除极端个例",
        "note": f"Tier-A 单票仅 {len(tier_a)} 只（多为数据/波动异常），不可外推",
        "action": "默认排除 SNDK/杠杆ETF；以组合+分桶中位数做决策",
    })

    best_port = min(portfolios, key=lambda x: x.get("gap_score", 99)) if portfolios else {}
    return {
        "targets": targets_label(),
        "rules": rules,
        "ma50_uplift": ma50_uplift,
        "best_portfolio": best_port,
        "tier_a_count": int((tdf["tier"] == "A").sum()) if not tdf.empty else 0,
        "tier_a_tickers": tier_a["代码"].unique().tolist() if not tier_a.empty else [],
    }


def run_scan(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 50.0,
    exclude: set[str] | None = None,
    account: float = 10_000.0,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    set_scan_targets(preset="relaxed")
    end = end or date.today().isoformat()
    exclude = exclude or DEFAULT_EXCLUDE

    pool = build_candidate_pool(use_broad=not quick, max_names=120 if quick else 0)
    pool = [t for t in pool if t not in exclude]
    print(f"规律扫描 · 候选 {len(pool)} 只 · 排除 {sorted(exclude)} · {targets_label()}")

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)

    all_rows: list[dict] = []
    liquid: list[tuple[str, pd.DataFrame, float]] = []
    for tk, df in batch.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m >= min_dvol_m:
            liquid.append((tk, df, dvol_m))
    liquid.sort(key=lambda x: -x[2])
    print(f"流动性通过 {len(liquid)} 只")

    for i, (tk, df, dvol_m) in enumerate(liquid):
        if i % 25 == 0:
            print(f"  分析 {i + 1}/{len(liquid)} …")
        all_rows.extend(score_ticker_both(tk, df, dvol_m=dvol_m, account=account))

    tdf = pd.DataFrame(all_rows)
    if not tdf.empty:
        tdf.to_csv(TICKER_CSV, index=False, encoding="utf-8-sig")

    bdf = aggregate_buckets(tdf)
    if not bdf.empty:
        bdf.to_csv(BUCKET_CSV, index=False, encoding="utf-8-sig")

    # 组合回测
    batch_dict = {tk: df for tk, df, _ in liquid}
    mega = [tk for tk, _, dv in liquid if dv >= 1000][:10]
    etfs = [tk for tk in ["SPY", "QQQ", "IWM", "XLF", "DIA"] if tk in batch_dict]
    low_rv = []
    for tk, df, dv in liquid:
        if tk in batch_dict:
            rv = float(realized_vol(df["Close"].astype(float)).iloc[-1]) * 100
            if rv < 30 and dv >= 500:
                low_rv.append(tk)
    low_rv = low_rv[:8]

    portfolios = []
    scenarios = [
        ("ETF铁鹰舰队5", "iron_ma50", etfs[:5]),
        ("低波ETF铁鹰5", "iron_ma50", [t for t in etfs if t in batch_dict][:5] or etfs[:4]),
        ("低波蓝筹铁鹰5", "iron_ma50", low_rv[:5]),
    ]
    for name, strat, tks in scenarios:
        if len(tks) < 2:
            continue
        r = backtest_equal_basket(batch_dict, tks, strategy=strat, account=account, slots=len(tks))
        if r:
            r["scenario"] = name
            portfolios.append(r)

    pdf = pd.DataFrame(portfolios)
    if not pdf.empty:
        pdf.to_csv(PORTFOLIO_CSV, index=False, encoding="utf-8-sig")

    rules = derive_rules(tdf, bdf, portfolios)
    RULES_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return tdf, bdf, portfolios, rules


def _print_report(tdf: pd.DataFrame, bdf: pd.DataFrame, portfolios: list[dict], rules: dict) -> None:
    print(f"\n{'='*64}")
    print("全市场规律报告（排除极端个例）")
    print(f"{'='*64}")

    if not bdf.empty:
        print("\n【分桶中位数】RV × 策略")
        for _, r in bdf.sort_values(["RV桶", "策略"]).iterrows():
            etf = "ETF" if r["ETF"] else "个股"
            print(
                f"  {r['RV桶']:12s} {r['策略']:10s} {etf} "
                f"n={int(r['样本数']):3d} 年化中位={r['年化中位']:.1%} "
                f"回撤中位={r['回撤中位']:.1%} 胜率中位={r['胜率中位']:.1%} "
                f"TierAB率={r['达标率TierAB']:.0%}"
            )

    if portfolios:
        print("\n【组合策略】等权舰队（$10k × N 槽）")
        for p in sorted(portfolios, key=lambda x: x["gap_score"]):
            print(
                f"  {p['scenario']:16s} {','.join(p['tickers'][:5]):30s} "
                f"年化={p['年化']:.1%} 回撤={p['最大回撤']:.1%} "
                f"胜率={p['胜率']:.1%} tier={p['tier']}"
            )

    print("\n【提炼规则】")
    for rule in rules.get("rules", []):
        if isinstance(rule, dict):
            print(f"  · {rule.get('pattern', rule.get('id'))}: {rule.get('action', rule.get('note', ''))}")
    mu = rules.get("ma50_uplift") or {}
    if mu:
        print(f"  · MA50 普适效应: CSP 胜率 +{mu.get('csp', 0):.1%} · 铁鹰胜率 +{mu.get('iron', 0):.1%}")

    if rules.get("best_portfolio"):
        bp = rules["best_portfolio"]
        print(f"\n★ 最接近三重目标的组合：{bp.get('scenario', bp.get('name'))} "
              f"→ 年化={bp.get('年化', 0):.1%} 回撤={bp.get('最大回撤', 0):.1%} "
              f"胜率={bp.get('胜率', 0):.1%}")

    print(f"\n→ {TICKER_CSV}\n→ {BUCKET_CSV}\n→ {PORTFOLIO_CSV}\n→ {RULES_JSON}")


def main() -> None:
    p = argparse.ArgumentParser(description="全市场规律挖掘")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--min-dvol-m", type=float, default=50.0)
    p.add_argument("--exclude", default=",".join(sorted(DEFAULT_EXCLUDE)))
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    excl = {x.strip().upper() for x in args.exclude.split(",") if x.strip()}

    tdf, bdf, portfolios, rules = run_scan(
        start=args.start, end=args.end, min_dvol_m=args.min_dvol_m,
        exclude=excl, account=args.account, quick=args.quick,
    )
    _print_report(tdf, bdf, portfolios, rules)


if __name__ == "__main__":
    main()
