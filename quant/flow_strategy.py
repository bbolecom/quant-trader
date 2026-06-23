"""资金流向可操作策略 · 信号、选股、组合回测。

策略逻辑（flow_action_v1）：
  做多：SPY>MA50 + 命中 U_S1/U_S2/U_A2 → 收盘买入，次日收盘平仓
  做空：命中 D_S2/D_A3/D_B3 或 前日涨>30%（融资砸盘代理）→ 次日做空收益
  观望：D_B2 极端波动、无信号、多空冲突 → 不交易

组合：做多池与做空池等权；无信号日空仓。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant.capital_flow import (
    _match_down_patterns,
    _match_up_patterns,
    _tier_score,
    build_flow_history,
)

Signal = Literal["long", "short", "flat"]

DEFAULT_LONG_PATTERNS = frozenset({"U_S1", "U_S2", "U_A2"})
DEFAULT_SHORT_PATTERNS = frozenset({"D_S2", "D_A3", "D_B3", "D_OFFERING"})
DEFAULT_OFFERING_PROXY_PCT = 30.0


@dataclass
class FlowStrategyParams:
    """可操作策略参数。"""

    name: str = "flow_action_v1"
    long_patterns: frozenset[str] = DEFAULT_LONG_PATTERNS
    short_patterns: frozenset[str] = DEFAULT_SHORT_PATTERNS
    offering_proxy_pct: float = DEFAULT_OFFERING_PROXY_PCT
    long_top_n: int = 3
    short_top_n: int = 2
    min_dvol_m: float = 30.0
    min_price: float = 3.0
    long_weight: float = 0.5
    short_weight: float = 0.5
    fee_bps: float = 5.0
    require_spy_bull_for_long: bool = True
    # U_S2 额外过滤（提高胜率）
    long_s2_min_gain_pct: float = 0.0
    long_s2_max_gain_pct: float = 999.0
    long_min_close_strength: float = 0.0
    long_min_vol_ratio: float = 0.0
    long_max_vol_ratio: float = 999.0
    long_min_gain_5d_pct: float = 0.0
    long_max_gain_5d_pct: float = 999.0
    require_spy_positive_1d: bool = False
    min_spy_1d_pct: float = 0.0
    # 做空额外过滤（提高胜率）
    short_min_prev_pct: float = 0.0
    short_max_today_gain_pct: float = 999.0
    require_spy_bear_for_short: bool = False
    max_spy_1d_pct_for_short: float = 999.0
    # 形态滚动胜率（无未来函数）
    min_recent_setup_win_rate: float = 0.0
    min_recent_setup_samples: int = 3
    use_recent_setup_win: bool = True

    @classmethod
    def from_dict(cls, raw: dict | None) -> FlowStrategyParams:
        raw = raw or {}
        lp = raw.get("long_patterns")
        sp = raw.get("short_patterns")
        return cls(
            name=str(raw.get("name", "flow_action_v1")),
            long_patterns=frozenset(lp) if lp else DEFAULT_LONG_PATTERNS,
            short_patterns=frozenset(sp) if sp else DEFAULT_SHORT_PATTERNS,
            offering_proxy_pct=float(raw.get("offering_proxy_pct", DEFAULT_OFFERING_PROXY_PCT)),
            long_top_n=int(raw.get("long_top_n", 3)),
            short_top_n=int(raw.get("short_top_n", 2)),
            min_dvol_m=float(raw.get("min_dvol_m", 30.0)),
            min_price=float(raw.get("min_price", 3.0)),
            long_weight=float(raw.get("long_weight", 0.5)),
            short_weight=float(raw.get("short_weight", 0.5)),
            fee_bps=float(raw.get("fee_bps", 5.0)),
            require_spy_bull_for_long=bool(raw.get("require_spy_bull_for_long", True)),
            long_s2_min_gain_pct=float(raw.get("long_s2_min_gain_pct", 0.0)),
            long_s2_max_gain_pct=float(raw.get("long_s2_max_gain_pct", 999.0)),
            long_min_close_strength=float(raw.get("long_min_close_strength", 0.0)),
            long_min_vol_ratio=float(raw.get("long_min_vol_ratio", 0.0)),
            long_max_vol_ratio=float(raw.get("long_max_vol_ratio", 999.0)),
            long_min_gain_5d_pct=float(raw.get("long_min_gain_5d_pct", 0.0)),
            long_max_gain_5d_pct=float(raw.get("long_max_gain_5d_pct", 999.0)),
            require_spy_positive_1d=bool(raw.get("require_spy_positive_1d", False)),
            min_spy_1d_pct=float(raw.get("min_spy_1d_pct", 0.0)),
            short_min_prev_pct=float(raw.get("short_min_prev_pct", 0.0)),
            short_max_today_gain_pct=float(raw.get("short_max_today_gain_pct", 999.0)),
            require_spy_bear_for_short=bool(raw.get("require_spy_bear_for_short", False)),
            max_spy_1d_pct_for_short=float(raw.get("max_spy_1d_pct_for_short", 999.0)),
            min_recent_setup_win_rate=float(raw.get("min_recent_setup_win_rate", 0.0)),
            min_recent_setup_samples=int(raw.get("min_recent_setup_samples", 3)),
            use_recent_setup_win=bool(raw.get("use_recent_setup_win", True)),
        )


def _pattern_ids_up(r: dict, spy_bull: bool) -> set[str]:
    return {h["规律ID"] for h in _match_up_patterns(r, spy_bull=spy_bull)}


def _pattern_ids_down(r: dict, spy_bull: bool, offering_proxy_pct: float = DEFAULT_OFFERING_PROXY_PCT) -> set[str]:
    ids = {h["规律ID"] for h in _match_down_patterns(r, spy_bull=spy_bull)}
    prev = float(r.get("前日涨幅%", r.get("prev_gain_pct", 0)) or 0)
    if prev > offering_proxy_pct:
        ids.add("D_OFFERING")
    return ids


def _filter_long_ids(
    long_ids: set[str],
    r: dict[str, Any],
    params: FlowStrategyParams,
) -> set[str]:
    """按规律细分过滤，提升胜率。"""
    out: set[str] = set()
    g = float(r.get("涨幅%", 0) or 0)
    g5 = float(r.get("涨幅5d%", 0) or 0)
    vr = float(r.get("量比", r.get("vol_ratio", 0)) or 0)
    cs = float(r.get("close_strength", r.get("收盘强度", 0.5)) or 0.5)

    if "U_A2" in long_ids:
        out.add("U_A2")
    if "U_S2" in long_ids:
        if params.long_s2_min_gain_pct <= g <= params.long_s2_max_gain_pct:
            if cs >= params.long_min_close_strength:
                if params.long_min_vol_ratio <= vr <= params.long_max_vol_ratio:
                    if params.long_min_gain_5d_pct <= g5 <= params.long_max_gain_5d_pct:
                        out.add("U_S2")
    # 其他做多规律（如 U_S1）仅做全局量比/收强过滤
    for pid in long_ids - {"U_A2", "U_S2"}:
        if params.long_min_vol_ratio <= vr <= params.long_max_vol_ratio:
            if cs >= params.long_min_close_strength:
                out.add(pid)
    return out


def _filter_short_ids(
    short_ids: set[str],
    r: dict[str, Any],
    params: FlowStrategyParams,
    *,
    spy_bull: bool = True,
    spy_1d_pct: float | None = None,
) -> set[str]:
    """按规律细分过滤做空信号。"""
    if not short_ids:
        return set()
    prev = float(r.get("前日涨幅%", r.get("prev_gain_pct", 0)) or 0)
    g = float(r.get("涨幅%", 0) or 0)
    if prev < params.short_min_prev_pct:
        return set()
    if g > params.short_max_today_gain_pct:
        return set()
    if params.require_spy_bear_for_short and spy_bull:
        return set()
    if params.max_spy_1d_pct_for_short < 999.0:
        s1 = spy_1d_pct if spy_1d_pct is not None else float(r.get("spy_1d_pct", 0) or 0)
        if not np.isfinite(s1) or s1 > params.max_spy_1d_pct_for_short:
            return set()
    return short_ids


def _setup_win_ok(row: dict[str, Any], params: FlowStrategyParams) -> bool:
    if params.min_recent_setup_win_rate <= 0:
        return True
    wr_col = "近8次胜率" if params.use_recent_setup_win else "历史胜率"
    n_col = "近8次样本" if params.use_recent_setup_win else "历史样本"
    wr = float(row.get(wr_col, np.nan))
    n = int(row.get(n_col, 0) or 0)
    if not np.isfinite(wr) or n < params.min_recent_setup_samples:
        return False
    return wr >= params.min_recent_setup_win_rate


def enrich_panel_setup_win(panel: pd.DataFrame) -> pd.DataFrame:
    """按代码×规律滚动统计历史次日胜率（不含当日）。"""
    if panel.empty:
        return panel
    out = panel.copy()
    out["历史胜率"] = np.nan
    out["历史样本"] = 0
    out["近8次胜率"] = np.nan
    out["近8次样本"] = 0
    out = out.sort_values(["代码", "规律", "日期"])
    for _, grp in out.groupby(["代码", "规律"], sort=False):
        wins: list[float] = []
        for idx in grp.index:
            if wins:
                out.at[idx, "历史胜率"] = float(np.mean(wins))
                out.at[idx, "历史样本"] = len(wins)
                recent = wins[-8:]
                out.at[idx, "近8次胜率"] = float(np.mean(recent))
                out.at[idx, "近8次样本"] = len(recent)
            row = out.loc[idx]
            side = row["signal"]
            raw = float(row["fwd_1d"])
            win = raw > 0 if side == "long" else raw < 0
            wins.append(float(win))
    return out.sort_values(["日期", "score"], ascending=[True, False])


def evaluate_actionable_signal(
    row: dict[str, Any],
    params: FlowStrategyParams,
    *,
    spy_bull: bool = True,
    spy_1d_pct: float | None = None,
) -> dict[str, Any]:
    """单条特征 → 可操作策略信号（long/short/flat）。"""
    r = dict(row)
    if float(r.get("dvol_m", 0) or 0) < params.min_dvol_m:
        return _flat("成交额不足")
    if float(r.get("现价", r.get("收盘价", 0)) or 0) < params.min_price:
        return _flat("价格过低")

    up_ids = _pattern_ids_up(r, spy_bull)
    down_ids = _pattern_ids_down(r, spy_bull, params.offering_proxy_pct)
    prev = float(r.get("前日涨幅%", 0) or 0)

    if "D_B2" in down_ids and "D_B2" not in params.short_patterns:
        return _flat("D_B2极端波动观望")

    short_ids = _filter_short_ids(
        down_ids & params.short_patterns, r, params,
        spy_bull=spy_bull, spy_1d_pct=spy_1d_pct,
    )
    long_ids = _filter_long_ids(up_ids & params.long_patterns, r, params)
    if params.require_spy_bull_for_long and not spy_bull:
        long_ids = set()
    if params.require_spy_positive_1d:
        s1 = spy_1d_pct if spy_1d_pct is not None else float(r.get("spy_1d_pct", 0) or 0)
        if not np.isfinite(s1) or s1 < params.min_spy_1d_pct:
            long_ids = set()

    if short_ids and long_ids:
        return _flat("多空规律冲突")

    if short_ids:
        score = max(_tier_score_for_ids(short_ids)) + abs(prev) / 100.0
        return {
            "signal": "short",
            "方向": "做空",
            "策略动作": "买Put价差/做空1日",
            "规律": "、".join(sorted(short_ids)),
            "score": score,
            "选股理由": _reason(short_ids, r),
        }
    if long_ids:
        cs = float(r.get("close_strength", r.get("收盘强度", 0.5)) or 0.5)
        score = max(_tier_score_for_ids(long_ids)) + cs
        return {
            "signal": "long",
            "方向": "做多",
            "策略动作": "次日做多",
            "规律": "、".join(sorted(long_ids)),
            "score": score,
            "选股理由": _reason(long_ids, r),
        }
    return _flat("未命中策略规律")


def _tier_score_for_ids(ids: set[str]) -> list[int]:
    from quant.capital_flow import FLOW_CATALOG
    tier_map = {p.id: p.tier for p in FLOW_CATALOG}
    return [_tier_score(tier_map.get(i, "C")) for i in ids] or [0]


def _reason(ids: set[str], r: dict) -> str:
    g = float(r.get("涨幅%", 0) or 0)
    vr = float(r.get("量比", r.get("vol_ratio", 0)) or 0)
    prev = float(r.get("前日涨幅%", 0) or 0)
    return f"{'、'.join(sorted(ids))} · 涨{g:.1f}% 量比{vr:.2f} 前日{prev:.1f}%"


def _flat(reason: str) -> dict[str, Any]:
    return {
        "signal": "flat",
        "方向": "观望",
        "策略动作": "观望",
        "规律": "—",
        "score": 0.0,
        "选股理由": reason,
    }


def build_signal_panel(
    data: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    params: FlowStrategyParams,
) -> pd.DataFrame:
    """全历史信号面板（含 fwd_1d）。"""
    spy_ma50 = spy_close.rolling(50, min_periods=25).mean()
    spy_1d = spy_close.pct_change()
    rows: list[dict] = []
    for tk, df in data.items():
        hist = build_flow_history(df, spy_close)
        if hist.empty:
            continue
        for _, row in hist.iterrows():
            d = row.get("日期")
            try:
                spy_bull = float(spy_close.loc[d]) > float(spy_ma50.loc[d])
                spy_1d_pct = float(spy_1d.loc[d]) * 100
            except Exception:  # noqa: BLE001
                spy_bull = True
                spy_1d_pct = 0.0
            r = row.to_dict()
            sig = evaluate_actionable_signal(
                r, params, spy_bull=spy_bull, spy_1d_pct=spy_1d_pct,
            )
            if sig["signal"] == "flat":
                continue
            rows.append({
                "日期": pd.Timestamp(d),
                "代码": tk,
                "现价": r.get("现价"),
                "fwd_1d": float(r.get("fwd_1d", 0)),
                "signal": sig["signal"],
                "方向": sig["方向"],
                "策略动作": sig["策略动作"],
                "规律": sig["规律"],
                "score": sig["score"],
                "选股理由": sig["选股理由"],
                "涨幅%": r.get("涨幅%"),
                "量比": r.get("量比"),
                "收盘强度": r.get("close_strength"),
                "5日涨%": r.get("涨幅5d%"),
                "前日涨%": r.get("前日涨幅%"),
                "spy_bull": spy_bull,
                "spy_1d%": spy_1d_pct,
            })
    if not rows:
        return pd.DataFrame()
    panel = pd.DataFrame(rows).sort_values(["日期", "score"], ascending=[True, False])
    return enrich_panel_setup_win(panel)


def _row_matches_params(row: dict[str, Any], params: FlowStrategyParams) -> bool:
    """面板行二次过滤（与 evaluate 规则一致，便于宽面板 + 参数寻优）。"""
    side = row.get("signal")
    if side not in ("long", "short"):
        return False

    def _num(key: str, alt: str = "") -> float | None:
        v = row.get(key)
        if v is None and alt:
            v = row.get(alt)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return float(v)

    g = _num("涨幅%")
    prev = _num("前日涨%", "前日涨幅%")
    g5 = _num("5日涨%", "涨幅5d%")
    vr = _num("量比")
    cs = _num("收盘强度", "close_strength")
    spy_bull = row.get("spy_bull")
    spy_1d = _num("spy_1d%")
    pat = str(row.get("规律", "") or "")

    if side == "short":
        if prev is not None and prev < params.short_min_prev_pct:
            return False
        if g is not None and g > params.short_max_today_gain_pct:
            return False
        if params.require_spy_bear_for_short and spy_bull is True:
            return False
        if params.max_spy_1d_pct_for_short < 999.0 and spy_1d is not None:
            if spy_1d > params.max_spy_1d_pct_for_short:
                return False
        if pat:
            short_ids = {p.strip() for p in pat.replace("、", ",").split(",") if p.strip()}
            if not short_ids & params.short_patterns:
                return False
    else:
        if params.require_spy_bull_for_long and spy_bull is False:
            return False
        if params.require_spy_positive_1d and spy_1d is not None and spy_1d < params.min_spy_1d_pct:
            return False
        if pat:
            long_ids = {p.strip() for p in pat.replace("、", ",").split(",") if p.strip()}
            if not long_ids & params.long_patterns:
                return False
            if "U_S2" in long_ids:
                if g is not None and not (params.long_s2_min_gain_pct <= g <= params.long_s2_max_gain_pct):
                    return False
                if cs is not None and cs < params.long_min_close_strength:
                    return False
                if vr is not None and not (params.long_min_vol_ratio <= vr <= params.long_max_vol_ratio):
                    return False
                if g5 is not None and not (params.long_min_gain_5d_pct <= g5 <= params.long_max_gain_5d_pct):
                    return False
    return _setup_win_ok(row, params)


def select_picks_for_date(panel: pd.DataFrame, as_of: pd.Timestamp, params: FlowStrategyParams) -> pd.DataFrame:
    day = panel[panel["日期"] == as_of]
    if day.empty:
        return pd.DataFrame()
    day = day[day.apply(lambda r: _row_matches_params(r.to_dict(), params), axis=1)]
    longs = day[day["signal"] == "long"].head(max(params.long_top_n, 0))
    shorts = day[day["signal"] == "short"].head(max(params.short_top_n, 0))
    parts = [p for p in [longs, shorts] if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_portfolio_backtest(
    panel: pd.DataFrame,
    params: FlowStrategyParams,
    *,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """逐日组合回测：做多/做空池等权，持有 1 日。"""
    if panel.empty:
        return {"error": "无信号面板"}

    fee = params.fee_bps / 10_000.0
    dates = sorted(panel["日期"].unique())
    equity = initial_capital
    curve: list[dict] = []
    trades: list[dict] = []

    for d in dates:
        picks = select_picks_for_date(panel, d, params)
        if picks.empty:
            curve.append({"日期": d, "权益": equity, "日收益%": 0.0, "交易数": 0})
            continue

        long_p = picks[picks["signal"] == "long"]
        short_p = picks[picks["signal"] == "short"]
        long_ret = float(long_p["fwd_1d"].mean()) if not long_p.empty else 0.0
        short_ret = float(-short_p["fwd_1d"].mean()) if not short_p.empty else 0.0

        n_long, n_short = len(long_p), len(short_p)
        if n_long and n_short:
            port_ret = params.long_weight * long_ret + params.short_weight * short_ret
        elif n_long:
            port_ret = long_ret
        else:
            port_ret = short_ret

        trade_count = n_long + n_short
        port_ret -= 2 * fee * trade_count / max(trade_count, 1)

        pnl = equity * port_ret
        equity += pnl
        curve.append({
            "日期": d,
            "权益": equity,
            "日收益%": port_ret * 100,
            "交易数": trade_count,
            "做多数": n_long,
            "做空数": n_short,
        })

        for _, r in picks.iterrows():
            side = r["signal"]
            raw = float(r["fwd_1d"])
            adj = raw if side == "long" else -raw
            trades.append({
                "日期": d.strftime("%Y-%m-%d"),
                "代码": r["代码"],
                "方向": side,
                "规律": r.get("规律", ""),
                "涨幅%": r.get("涨幅%"),
                "量比": r.get("量比"),
                "次日收益%": raw * 100,
                "策略收益%": adj * 100,
                "选股理由": r.get("选股理由", ""),
            })

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    rets = curve_df["日收益%"] / 100.0
    active = rets[rets != 0]
    win = float((active > 0).mean()) if len(active) else 0.0
    total = equity / initial_capital - 1.0
    years = max(len(dates) / 252.0, 0.1)
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else total
    sharpe = float(active.mean() / active.std() * np.sqrt(252)) if len(active) > 1 and active.std() > 0 else 0.0
    eq = curve_df["权益"]
    max_dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0

    trade_wins = trades_df["策略收益%"] > 0 if not trades_df.empty else pd.Series(dtype=bool)
    trade_wr = float(trade_wins.mean()) if len(trade_wins) else 0.0

    return {
        "params": params,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "累计收益率": total,
        "年化收益率": ann,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "日胜率": win,
        "笔胜率": trade_wr,
        "交易天数": int(len(active)),
        "总笔数": len(trades_df),
        "权益曲线": curve_df,
        "交易明细": trades_df,
    }


def today_actionable_picks(
    batch: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    params: FlowStrategyParams,
) -> pd.DataFrame:
    """今日可操作策略选股（与回测同一套规则）。"""
    spy_bull = float(spy_close.iloc[-1]) > float(spy_close.rolling(50).mean().iloc[-1])
    rows: list[dict] = []
    for tk, df in batch.items():
        if df is None or len(df) < 25:
            continue
        from quant.capital_flow import enrich_flow_row
        feat = enrich_flow_row(df, spy_close)
        if not feat:
            continue
        sig = evaluate_actionable_signal(
            feat, params, spy_bull=spy_bull,
            spy_1d_pct=float(spy_close.pct_change().iloc[-1] * 100) if len(spy_close) >= 2 else 0,
        )
        if sig["signal"] == "flat":
            continue
        rows.append({
            "代码": tk,
            "现价": round(float(feat["现价"]), 2),
            "方向": sig["方向"],
            "策略动作": sig["策略动作"],
            "规律": sig["规律"],
            "score": sig["score"],
            "选股理由": sig["选股理由"],
            "涨幅%": feat.get("涨幅%"),
            "量比": round(float(feat.get("量比", 0)), 2),
            "收盘强度": round(float(feat.get("close_strength", 0)), 2),
            "成交额M": round(float(feat.get("dvol_m", 0)), 1),
            "5日涨%": round(float(feat.get("涨幅5d%", 0)), 1),
            "MA50": "上" if feat.get("above_ma50") else "下",
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    if params.min_recent_setup_win_rate > 0:
        enriched: list[dict] = []
        for tk, gdf in batch.items():
            if gdf is None or len(gdf) < 55:
                continue
            mini = build_signal_panel({tk: gdf}, spy_close, params)
            if mini.empty:
                continue
            last = mini.iloc[-1]
            enriched.append({
                "代码": tk,
                "近8次胜率": float(last.get("近8次胜率", np.nan)),
                "近8次样本": int(last.get("近8次样本", 0) or 0),
            })
        if enriched:
            wr_map = {r["代码"]: r for r in enriched}
            def ok(row):
                m = wr_map.get(row["代码"])
                if not m:
                    return False
                fake = {"近8次胜率": m["近8次胜率"], "近8次样本": m["近8次样本"]}
                return _setup_win_ok(fake, params)
            df = df[df.apply(ok, axis=1)]
    longs = df[df["方向"] == "做多"].head(max(params.long_top_n, 0))
    shorts = df[df["方向"] == "做空"].head(max(params.short_top_n, 0))
    parts = [p for p in [longs, shorts] if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_strategy_config(path: Path | None = None) -> FlowStrategyParams:
    p = path or Path(__file__).resolve().parents[1] / "flow_strategy_config.json"
    if p.exists():
        return FlowStrategyParams.from_dict(json.loads(p.read_text(encoding="utf-8")))
    return FlowStrategyParams()
