"""5×每日选股 / 圣杯舰队：策略定义、历史回测统计、今日预测选股。"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import screen_strategies, screener

ROOT = Path(__file__).resolve().parent.parent
FLEET_CONFIG = ROOT / "daily_screen_config.json"
STATS_JSON = ROOT / "research" / "screen_fleet_stats.json"
MARKET_SCAN_CSV = ROOT / "research" / "market_triple_scan_results.csv"
MARKET_FLEET_JSON = ROOT / "research" / "market_triple_fleet.json"

DEFAULT_TARGET_PROFILE = {
    "label": "三标达标",
    "ann_return": 0.80,
    "max_dd": -0.10,
    "win_rate": 0.85,
}


def load_market_scan_results(path: Path | None = None) -> pd.DataFrame:
    p = path or MARKET_SCAN_CSV
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def market_scan_summary_table(df: pd.DataFrame, targets: dict | None = None) -> pd.DataFrame:
    """Tier A 达标组合摘要表。"""
    if df.empty:
        return df
    t = targets or DEFAULT_TARGET_PROFILE
    tier_a = df[df["tier"] == "A"].copy()
    if tier_a.empty:
        tier_a = df.nsmallest(10, "gap_score")
    tier_a["三标达标"] = tier_a.apply(
        lambda r: "✅"
        if meets_target_profile(
            {"ann_return": r["年化"], "max_dd": r["最大回撤"], "trade_win_rate": r["胜率"]},
            t,
        )
        else "❌",
        axis=1,
    )
    cols = ["三标达标", "代码", "策略", "delta", "ma_window", "alloc", "年化", "最大回撤", "胜率", "成交额M", "tier"]
    return tier_a[[c for c in cols if c in tier_a.columns]].drop_duplicates(
        subset=[c for c in ["代码", "delta", "ma_window", "alloc"] if c in tier_a.columns]
    )


def load_fleet_config(path: Path | None = None) -> dict:
    p = path or FLEET_CONFIG
    if not p.exists():
        return {"accounts": [], "labels": [], "backtest_years": 5}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_fleet_stats(stats: dict, path: Path | None = None) -> None:
    p = path or STATS_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_fleet_stats(path: Path | None = None) -> dict:
    p = path or STATS_JSON
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def fleet_accounts(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_fleet_config()
    return list(cfg.get("accounts") or [])


def load_target_profile(cfg: dict | None = None) -> dict[str, float | str]:
    cfg = cfg or load_fleet_config()
    raw = cfg.get("target_profile") or {}
    out = dict(DEFAULT_TARGET_PROFILE)
    for k in ("ann_return", "max_dd", "win_rate", "label", "note"):
        if k in raw and raw[k] is not None:
            out[k] = raw[k]
    return out


def is_csp_account(acct: dict) -> bool:
    return str(acct.get("strategy_type", "screen")).lower() == "csp"


def account_strategy_label(acct: dict) -> str:
    if is_csp_account(acct):
        p = acct.get("csp_params") or {}
        ma = int(p.get("ma_window", 0))
        ma_s = f"MA{ma}" if ma > 0 else "无MA"
        return f"CSP {acct.get('ticker', 'SNDK')} δ={p.get('delta', 0.25)} {ma_s}"
    return preset_for_account(acct).name


def meets_target_profile(stats: dict[str, Any], targets: dict | None = None) -> bool:
    t = targets or DEFAULT_TARGET_PROFILE
    ann = float(stats.get("ann_return", 0) or 0)
    dd = float(stats.get("max_dd", 0) or 0)
    wr = float(stats.get("trade_win_rate", stats.get("win_rate", 0)) or 0)
    return ann > float(t["ann_return"]) and dd > float(t["max_dd"]) and wr > float(t["win_rate"])


def target_gap_summary(stats: dict[str, Any], targets: dict | None = None) -> str:
    """未达标时返回差距摘要。"""
    t = targets or DEFAULT_TARGET_PROFILE
    gaps: list[str] = []
    ann = float(stats.get("ann_return", 0) or 0)
    dd = float(stats.get("max_dd", 0) or 0)
    wr = float(stats.get("trade_win_rate", stats.get("win_rate", 0)) or 0)
    if ann <= float(t["ann_return"]):
        gaps.append(f"年化差 {float(t['ann_return']) - ann:.1%}")
    if dd <= float(t["max_dd"]):
        gaps.append(f"回撤超 {abs(dd - float(t['max_dd'])):.1%}")
    if wr <= float(t["win_rate"]):
        gaps.append(f"胜率差 {float(t['win_rate']) - wr:.1%}")
    return "；".join(gaps) if gaps else "达标"


def _stats_from_anchor(acct: dict, *, years: float) -> dict[str, Any]:
    a = acct.get("anchor_stats") or {}
    wr = float(a.get("win_rate", a.get("trade_win_rate", 0)) or 0)
    return {
        "years": years,
        "ann_return": float(a.get("ann_return", 0) or 0),
        "total_return": float(a.get("total_return", 0) or 0),
        "max_dd": float(a.get("max_dd", 0) or 0),
        "sharpe": float(a.get("sharpe", 0) or 0),
        "period_win_rate": wr,
        "trade_win_rate": wr,
        "rebalance_count": int(a.get("trade_count", 0) or 0),
        "final_equity": float(acct.get("account_size", 10_000) or 10_000),
        "updated": date.today().isoformat(),
        "source": a.get("source", "anchor"),
    }


def preset_for_account(acct: dict) -> screen_strategies.ScreenStrategyPreset:
    if is_csp_account(acct):
        raise ValueError(f"账户 {acct.get('id')} 为 CSP 策略，无 screen preset")
    return screen_strategies.get_preset(acct["preset_id"])


def tickers_for_preset(
    preset: screen_strategies.ScreenStrategyPreset,
    acct: dict | None = None,
) -> list[str]:
    if acct and acct.get("tickers"):
        return [str(t).upper() for t in acct["tickers"]]
    if preset.pool == "custom":
        return list(preset.custom_tickers)
    if preset.pool == "sp500":
        try:
            return screener.fetch_sp500_tickers()[: preset.pool_size]
        except Exception:  # noqa: BLE001
            pass
    # 涨幅榜/活跃榜历史回测：用高流动自选代理，避免拉全量标普
    proxy = list(getattr(screener, "_FALLBACK_TICKERS", [])) or preset.custom_tickers
    return proxy[: min(preset.pool_size, 40)]


def summarize_backtest_result(bt: dict[str, Any], *, years: float) -> dict[str, Any]:
    """从 backtest_screen_preset 结果提取展示用指标。"""
    picks = bt.get("选股明细")
    trade_wr = float("nan")
    if isinstance(picks, pd.DataFrame) and not picks.empty and "策略后向收益" in picks.columns:
        s = pd.to_numeric(picks["策略后向收益"], errors="coerce")
        if s.notna().any():
            trade_wr = float((s > 0).mean())

    return {
        "years": years,
        "ann_return": float(bt.get("年化收益率", 0)),
        "total_return": float(bt.get("累计收益率", 0)),
        "max_dd": float(bt.get("最大回撤", 0)),
        "sharpe": float(bt.get("夏普比率", 0)),
        "period_win_rate": float(bt.get("盈利周期占比", 0)),
        "trade_win_rate": trade_wr,
        "rebalance_count": int(bt.get("调仓次数", 0)),
        "final_equity": float(bt.get("期末权益", 0)),
        "updated": date.today().isoformat(),
    }


def _stats_from_historical_picks(
    daily: pd.DataFrame,
    *,
    forward_days: int,
    years: float,
    initial_capital: float,
) -> dict[str, Any]:
    """从历史每日选股明细计算组合级指标（买入持有后 N 日口径）。"""
    if daily.empty:
        return {
            "years": years,
            "ann_return": 0.0,
            "total_return": 0.0,
            "max_dd": 0.0,
            "sharpe": 0.0,
            "period_win_rate": 0.0,
            "trade_win_rate": 0.0,
            "rebalance_count": 0,
            "final_equity": initial_capital,
            "updated": date.today().isoformat(),
        }

    fwd_col = f"后{forward_days}日收益"
    strat_col = "策略后向收益"
    fwd = pd.to_numeric(daily.get(fwd_col), errors="coerce")
    strat = pd.to_numeric(daily.get(strat_col), errors="coerce") if strat_col in daily.columns else pd.Series(dtype=float)

    trade_wr = float((fwd > 0).mean()) if fwd.notna().any() else 0.0
    by_day = daily.groupby("选股日期", sort=True)
    day_ret = by_day[fwd_col].mean() if fwd_col in daily.columns else pd.Series(dtype=float)
    day_ret = pd.to_numeric(day_ret, errors="coerce").dropna()

    equity = initial_capital
    eq_rows: list[dict] = []
    for dt, r in day_ret.items():
        equity *= 1.0 + float(r)
        eq_rows.append({"日期": dt, "权益": equity})

    final_eq = equity
    total_ret = final_eq / initial_capital - 1.0
    ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 0.1)) - 1.0 if total_ret > -1 else -1.0
    period_wr = float((day_ret > 0).mean()) if len(day_ret) else 0.0

    if len(eq_rows) > 1:
        eq_s = pd.Series([r["权益"] for r in eq_rows])
        max_dd = float((eq_s / eq_s.cummax() - 1.0).min())
        sharpe = float(day_ret.mean() / day_ret.std() * np.sqrt(252 / max(len(day_ret), 1))) if day_ret.std() > 0 else 0.0
    else:
        max_dd = 0.0
        sharpe = 0.0

    if strat.notna().any():
        trade_wr = float((strat > 0).mean())

    return {
        "years": years,
        "ann_return": ann_ret,
        "total_return": total_ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "period_win_rate": period_wr,
        "trade_win_rate": trade_wr,
        "rebalance_count": int(daily["选股日期"].nunique()) if "选股日期" in daily.columns else 0,
        "final_equity": final_eq,
        "updated": date.today().isoformat(),
        "equity_curve": pd.DataFrame(eq_rows),
    }


def backtest_account(
    acct: dict,
    data: dict[str, pd.DataFrame],
    *,
    years: float = 5.0,
    initial_capital: float = 10_000.0,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    if is_csp_account(acct):
        stats = _stats_from_anchor(acct, years=years)
        stats["final_equity"] = initial_capital * (1.0 + stats["ann_return"]) ** years
        return {
            "account_id": acct["id"],
            "label": acct.get("label", acct["id"]),
            "role": acct.get("role", ""),
            "description": acct.get("description", ""),
            "preset_id": "csp",
            "preset_name": account_strategy_label(acct),
            "strategy_type": "csp",
            "stats": stats,
            "backtest": {"权益曲线": pd.DataFrame(), "选股明细": pd.DataFrame()},
        }

    preset = preset_for_account(acct)
    if not data:
        return {"error": "无可用行情", "account_id": acct["id"]}

    best = max(data.keys(), key=lambda t: len(data[t]))
    end_ts = data[best].index[-1]
    start_ts = end_ts - pd.DateOffset(years=years)
    fwd = max(int(preset.forward_eval_days), 1)

    hres = screener.run_historical_daily_screen(
        data, preset.filters,
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
        rebalance_days=preset.rebalance_days,
        top_picks=preset.top_picks,
        forward_days=fwd,
        backward_days=preset.filters.lookback_days,
        strategy_name=preset.trading_strategy,
        params=preset.trading_params,
        allow_short=allow_short,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    if hres.get("error"):
        return {"error": hres["error"], "account_id": acct["id"]}

    daily = hres.get("daily_picks", pd.DataFrame())
    if daily.empty:
        return {"error": "回测期内无有效选股", "account_id": acct["id"]}

    stats = _stats_from_historical_picks(
        daily, forward_days=fwd, years=years, initial_capital=initial_capital,
    )
    eq_df = stats.pop("equity_curve", pd.DataFrame())

    bt = {
        "权益曲线": eq_df,
        "选股明细": daily,
        "累计收益率": stats["total_return"],
        "年化收益率": stats["ann_return"],
        "最大回撤": stats["max_dd"],
        "盈利周期占比": stats["period_win_rate"],
        "夏普比率": stats["sharpe"],
        "调仓次数": stats["rebalance_count"],
        "期末权益": stats["final_equity"],
    }

    return {
        "account_id": acct["id"],
        "label": acct.get("label", acct["id"]),
        "role": acct.get("role", ""),
        "description": acct.get("description", preset.rationale),
        "preset_id": acct["preset_id"],
        "preset_name": preset.name,
        "stats": stats,
        "backtest": bt,
    }


def today_picks_for_account(
    acct: dict,
    data: dict[str, pd.DataFrame],
    as_of: str | date | None = None,
    *,
    capital: float = 10_000.0,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    live_chain: dict | None = None,
) -> pd.DataFrame:
    """今日/指定日预测选股（无未来数据）。"""
    if is_csp_account(acct):
        return today_csp_signal_for_account(acct, capital=capital, live_chain=live_chain)

    preset = preset_for_account(acct)
    as_of = as_of or date.today().isoformat()
    return screen_strategies.trade_plan_at_date(
        preset, data, as_of,
        capital=capital, allow_short=allow_short,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
    )


def apply_live_csp_chain(
    row: dict,
    *,
    sym: str,
    spot: float,
    capital: float,
    live_chain: dict | None,
    model_can_open: bool,
    backtest_note: str = "",
) -> dict:
    """用真实期权链替换 BS 模型报价；无链则观望，不展示虚构行权价/权利金。"""
    lc = live_chain or {}
    out = dict(row)
    if not lc.get("enabled", True):
        out["数据源"] = "模型估算"
        out["可开仓"] = "⏸"
        out["方向"] = "观望"
        out["卖Put行权价"] = ""
        out["权利金$"] = ""
        out["建议张数"] = ""
        out["选股理由"] = (
            f"⚠️ **Black-Scholes 模型估算，非真实报价，不可直接下单。** "
            f"{sym} ${spot:.2f} · {backtest_note or '请本地运行 daily_pick.py 或开启 live_chain'}"
        )
        return out

    from quant.option_chain import build_csp

    cplan, why = build_csp(
        sym, spot, capital,
        otm=float(lc.get("csp_otm", 0.10)),
        min_dte=int(lc.get("min_dte", 2)),
        max_dte=int(lc.get("max_dte", 45)),
        min_oi=int(lc.get("min_open_interest", 25)),
        max_spread_pct=float(lc.get("max_spread_pct", 0.60)),
    )
    if cplan is None:
        out["数据源"] = "真实链不可用"
        out["可开仓"] = "⏸"
        out["方向"] = "观望"
        out["卖Put行权价"] = ""
        out["权利金$"] = ""
        out["建议张数"] = ""
        out["选股理由"] = (
            f"{sym} ${spot:.2f} · **真实链不可用**：{why} → **观望**"
            + (f" · {backtest_note}" if backtest_note else "")
        )
        return out

    nc = cplan.contracts
    trend_ok = model_can_open
    can = trend_ok and nc >= 1
    out["数据源"] = "真实链"
    out["卖Put行权价"] = cplan.legs[0].strike if cplan.legs else out.get("卖Put行权价")
    out["权利金$"] = round(cplan.net_per_contract, 0)
    out["建议张数"] = nc if nc >= 1 else 0
    out["可开仓"] = "✅" if can else "⏸"
    out["方向"] = "卖Put" if can else "观望"
    parts = [
        f"{sym} ${spot:.2f} · **真实链** {cplan.legs_label()} @{cplan.expiry}({cplan.dte}d)",
        f"收${cplan.net_per_contract:.0f}/张 · 占用${cplan.collateral:.0f}/张",
    ]
    if nc >= 1:
        parts.append(f"× {nc}张")
    elif trend_ok:
        parts.append("现金不够1张")
    if not trend_ok:
        parts.append("趋势过滤未通过（如跌破MA50）")
    if backtest_note:
        parts.append(backtest_note)
    out["选股理由"] = " · ".join(parts)
    return out


def today_csp_signal_for_account(
    acct: dict,
    *,
    capital: float = 10_000.0,
    live_chain: dict | None = None,
) -> pd.DataFrame:
    """CSP 账户今日开仓信号（卖 Put 计划）。"""
    from research.tier_a_csp import _scan_csp_row
    from quant.data import fetch_history

    sym = str(acct.get("ticker", "SNDK")).upper()
    p = acct.get("csp_params") or {}
    lookback = int(p.get("lookback_days", 550))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback)).isoformat()
    try:
        df = fetch_history(sym, start=start, end=end)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    row = _scan_csp_row(
        sym, df, capital, acct.get("label", acct["id"]),
        delta=float(p.get("delta", 0.25)),
        ma_window=int(p.get("ma_window", 50)),
        dte_days=int(p.get("dte_days", 35)),
        alloc_pct=float(p.get("alloc_pct", 0.75)),
        max_single=float(p.get("max_single_ticker_pct", 0.75)),
    )
    if row is None:
        return pd.DataFrame()

    reason_parts = [
        acct.get("description", account_strategy_label(acct)),
        f"现价 ${row.close:.2f}",
        f"MA50 {'上方' if row.above_ma else '下方'}",
    ]
    if row.can_open:
        reason_parts.append(
            f"建议卖 {row.put_strike} Put × {row.suggested_contracts} 张，"
            f"权利金约 ${row.premium_per_contract:.0f}/张"
        )
    else:
        reason_parts.append("今日条件未满足，暂不开仓")
    if row.flags:
        reason_parts.append("；".join(row.flags))

    n = row.suggested_contracts
    model_open = bool(row.can_open and n > 0)
    bt_note = ""
    if row.bt_win_rate is not None:
        bt_note = (
            f"历史回测 胜率{row.bt_win_rate:.0%}"
            + (f" · 年化{row.bt_annual:.0%}" if row.bt_annual else "")
            + (f" · 回撤{row.bt_max_dd:.0%}" if row.bt_max_dd else "")
        )

    base = {
        "选股日期": end,
        "代码": sym,
        "名称": sym,
        "现价": round(row.close, 2),
        "方向": "卖Put" if model_open else "观望",
        "仓位%": round(row.alloc_pct * 100, 0) if row.alloc_pct else "",
        "选股理由": " · ".join(reason_parts),
        "卖Put行权价": row.put_strike,
        "权利金$": row.premium_per_contract,
        "建议张数": n if model_open else 0,
        "可开仓": "✅" if model_open else "⏸",
        "回测胜率": row.bt_win_rate,
        "回测年化": row.bt_annual,
        "回测回撤": row.bt_max_dd,
        "数据源": "模型估算",
    }
    enriched = apply_live_csp_chain(
        base,
        sym=sym,
        spot=float(row.close),
        capital=capital,
        live_chain=live_chain,
        model_can_open=model_open,
        backtest_note=bt_note,
    )
    return pd.DataFrame([enriched])


def fleet_stats_table(stats_doc: dict, targets: dict | None = None) -> pd.DataFrame:
    """将 stats JSON 转为对比表。"""
    t = targets or load_target_profile(stats_doc)
    rows: list[dict] = []
    for acct in stats_doc.get("accounts") or []:
        s = acct.get("stats") or {}
        ok = meets_target_profile(s, t) if s else False
        rows.append({
            "账户": acct.get("label", ""),
            "策略": acct.get("preset_name", acct.get("preset_id", "")),
            "角色": acct.get("role", ""),
            "三标达标": "✅" if ok else "❌",
            "年化收益": s.get("ann_return"),
            "最大回撤": s.get("max_dd"),
            "胜率": s.get("trade_win_rate"),
            "选股日胜率": s.get("period_win_rate"),
            "夏普": s.get("sharpe"),
            "交易/调仓": s.get("rebalance_count"),
            "回测年数": s.get("years"),
        })
    return pd.DataFrame(rows)
