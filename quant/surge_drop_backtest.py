"""暴涨/暴跌池 · 分策略回测（次日开盘 + 止盈止损 + 等权组合）。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from typing import Any

import pandas as pd

from research.extreme15_pattern import (
    TRAIN_END,
    Rule,
    _portfolio,
    _select,
    _trade_returns,
    backtest_rule,
    build_event_panel,
)
from research.gainer_daily_backtest import fetch_gainer_data_yahoo


def list_strategy_presets() -> dict[str, str]:
    return {
        "drop_rebound": "暴跌反弹做多（量比≥2）",
        "drop_panic": "暴跌恐慌做多（深跌+放量）",
        "drop_gap_up": "暴跌后向上跳空做多",
        "surge_chase": "暴涨延续做多（强收盘+大盘多）",
        "surge_fade": "暴涨衰竭做空（过热）",
        "surge_weak": "暴涨弱收盘做空",
    }


def get_strategy_rule(preset: str) -> Rule:
    presets: dict[str, Rule] = {
        "drop_rebound": Rule(
            "暴跌反弹做多", "drop", "long", hold_days=3, stop=0.07, tp=0.12,
            min_close_strength=0.4, min_vol_ratio=2.0,
        ),
        "drop_panic": Rule(
            "暴跌恐慌做多", "drop", "long", hold_days=5, stop=0.08, tp=0.15,
            min_vol_ratio=2.5, max_pre20=-0.20,
        ),
        "drop_gap_up": Rule(
            "暴跌向上跳空", "drop", "long", hold_days=3, stop=0.07, tp=0.12, min_gap=0.03,
        ),
        "surge_chase": Rule(
            "暴涨延续做多", "surge", "long", hold_days=3, stop=0.06, tp=0.12,
            min_close_strength=0.8, min_vol_ratio=1.5, require_spy_bull=True,
        ),
        "surge_fade": Rule(
            "暴涨衰竭做空", "surge", "short", hold_days=3, stop=0.08, tp=0.10,
            min_vol_ratio=3.0,
        ),
        "surge_weak": Rule(
            "暴涨弱收盘空", "surge", "short", hold_days=3, stop=0.08, tp=0.10,
            max_close_strength=0.5, min_vol_ratio=2.0,
        ),
    }
    if preset not in presets:
        raise KeyError(f"未知策略: {preset}")
    return presets[preset]


def equity_curve_from_trades(trades: pd.DataFrame, *, max_per_day: int = 3) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["入场日"] = pd.to_datetime(t["入场日"])
    t = t.sort_values(["入场日"]).groupby("入场日").head(max_per_day)
    daily = t.groupby("入场日")["净收益"].mean().sort_index()
    equity = (1 + daily).cumprod()
    return pd.DataFrame({"日期": equity.index, "权益": equity.values})


def summarize_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    rows: list[dict] = []
    for tk, sub in t.groupby("代码"):
        r = sub["净收益"].astype(float)
        rows.append({
            "代码": tk,
            "笔数": int(len(sub)),
            "胜率": round(float((r > 0).mean()), 4),
            "均收益%": round(float(r.mean()) * 100, 2),
            "累计%": round(float((1 + r).prod() - 1) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("笔数", ascending=False).reset_index(drop=True)


def run_strategy_backtest(
    data: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    preset: str,
    *,
    fee_bps: float = 5.0,
    slip_bps: float = 15.0,
) -> dict[str, Any]:
    """单策略回测，含全样本/IS/OOS 与分票统计。"""
    rule = get_strategy_rule(preset)
    res = backtest_rule(data, events, rule, fee_bps=fee_bps, slip_bps=slip_bps)
    trades = res.pop("_trades", pd.DataFrame())
    out: dict[str, Any] = {
        "preset": preset,
        "label": list_strategy_presets()[preset],
        "rule": asdict(rule),
        **res,
        "by_ticker": summarize_by_ticker(trades).to_dict(orient="records"),
    }
    if not trades.empty:
        out["trades"] = trades.to_dict(orient="records")
    return out


def run_pool_backtest_suite(
    tickers: list[str],
    *,
    years: int = 5,
    threshold_pct: float = 15.0,
    min_dvol_m: float = 50.0,
    presets: list[str] | None = None,
    fee_bps: float = 5.0,
    slip_bps: float = 15.0,
    end: str | None = None,
) -> dict[str, Any]:
    """对暴涨/暴跌池跑多策略回测套件。"""
    end_d = end or date.today().isoformat()
    start_d = (date.fromisoformat(end_d) - timedelta(days=years * 365 + 120)).isoformat()
    presets = presets or list(list_strategy_presets().keys())

    data, spy = fetch_gainer_data_yahoo(tickers + ["SPY"], start_d, end_d)
    spy_close = spy["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy.index)

    events = build_event_panel(
        data,
        spy_close,
        threshold_pct=threshold_pct,
        min_price=3.0,
        min_dvol_m=min_dvol_m,
    )

    strategies: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    for p in presets:
        try:
            res = run_strategy_backtest(data, events, p, fee_bps=fee_bps, slip_bps=slip_bps)
        except KeyError:
            continue
        tr = res.pop("trades", None)
        if tr:
            df = pd.DataFrame(tr)
            df["策略"] = p
            all_trades.append(df)
        strategies.append(res)

    return {
        "period": {"start": start_d, "end": end_d, "train_end": TRAIN_END},
        "universe": len(tickers),
        "threshold_pct": threshold_pct,
        "events": {
            "total": int(len(events)),
            "surge": int((events["direction"] == "surge").sum()) if not events.empty else 0,
            "drop": int((events["direction"] == "drop").sum()) if not events.empty else 0,
        },
        "strategies": strategies,
        "all_trades": pd.concat(all_trades, ignore_index=True).to_dict(orient="records") if all_trades else [],
    }
