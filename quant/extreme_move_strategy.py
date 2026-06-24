"""Extreme up/down move event strategy.

The module keeps the research logic data-source agnostic: callers pass normalized
OHLCV frames, and the strategy returns event tables, simulated trades, and
summary statistics. It is intentionally conservative about liquidity because
large gap moves often look better in historical bars than they are tradable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant import metrics as M

ExtremeEventKind = Literal["surge_continuation", "drop_rebound"]
ExtremeMode = Literal["both", "surge", "drop"]


EVENT_LABELS: dict[ExtremeEventKind, str] = {
    "surge_continuation": "暴涨延续",
    "drop_rebound": "暴跌反弹",
}


@dataclass
class ExtremeMoveConfig:
    """Parameters for liquid extreme-move research."""

    event_threshold_pct: float = 10.0
    min_price: float = 5.0
    min_dollar_vol_m: float = 100.0
    min_vol_ratio: float = 2.0
    max_vol_ratio: float = 20.0
    surge_min_close_strength: float = 0.72
    drop_min_close_strength: float = 0.35
    max_atr_pct: float = 18.0
    max_pre_20d_abs_pct: float = 120.0
    hold_days: int = 3
    stop_loss_pct: float = 0.06
    take_profit_pct: float = 0.12
    max_positions_per_day: int = 3
    mode: ExtremeMode = "both"


@dataclass
class ExtremeMoveEvent:
    代码: str
    日期: str
    类型: ExtremeEventKind
    类型名: str
    方向: str
    涨跌幅_pct: float
    成交额M: float
    量比: float
    收盘强度: float
    跳空_pct: float
    ret_5d_pct: float
    ret_20d_pct: float
    ATR_pct: float
    post_1d_pct: float
    post_3d_pct: float
    post_5d_pct: float
    综合分: float
    说明: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["5日涨跌_pct"] = row.pop("ret_5d_pct")
        row["20日涨跌_pct"] = row.pop("ret_20d_pct")
        row["后1日_pct"] = row.pop("post_1d_pct")
        row["后3日_pct"] = row.pop("post_3d_pct")
        row["后5日_pct"] = row.pop("post_5d_pct")
        return row


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data missing columns: {missing}")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def compute_extreme_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute event features using only information known at each bar close."""

    _require_columns(df)
    if df.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=pd.to_datetime(df.index))
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    prev_close = close.shift(1)

    high_low = (high - low).replace(0, np.nan)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["涨跌幅_pct"] = close.pct_change() * 100.0
    out["成交额USD"] = close * vol
    out["量比"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    out["收盘强度"] = ((close - low) / high_low).clip(0.0, 1.0).fillna(0.5)
    out["跳空_pct"] = (open_ / prev_close - 1.0) * 100.0
    out["振幅_pct"] = ((high - low) / prev_close.replace(0, np.nan)) * 100.0
    out["5日涨跌_pct"] = (close.shift(1) / close.shift(6) - 1.0) * 100.0
    out["20日涨跌_pct"] = (close.shift(1) / close.shift(21) - 1.0) * 100.0
    out["ATR_pct"] = tr.rolling(14).mean() / close.replace(0, np.nan) * 100.0
    out["创20日高"] = close >= close.shift(1).rolling(20).max()
    out["创20日低"] = close <= close.shift(1).rolling(20).min()
    out["后1日_pct"] = (close.shift(-1) / close - 1.0) * 100.0
    out["后3日_pct"] = (close.shift(-3) / close - 1.0) * 100.0
    out["后5日_pct"] = (close.shift(-5) / close - 1.0) * 100.0
    out["Close"] = close
    out["Open"] = open_
    out["High"] = high
    out["Low"] = low
    return out


def classify_extreme_row(
    row: pd.Series,
    cfg: ExtremeMoveConfig | None = None,
) -> tuple[ExtremeEventKind | None, float, str]:
    """Classify one feature row into a tradable extreme-move setup."""

    cfg = cfg or ExtremeMoveConfig()
    ret = _finite(row.get("涨跌幅_pct"))
    close = _finite(row.get("Close"))
    dvol_m = _finite(row.get("成交额USD")) / 1e6
    vol_ratio = _finite(row.get("量比"))
    close_strength = _finite(row.get("收盘强度"), 0.5)
    pre20 = _finite(row.get("20日涨跌_pct"))
    atr = _finite(row.get("ATR_pct"))

    if close < cfg.min_price or dvol_m < cfg.min_dollar_vol_m:
        return None, 0.0, ""
    if vol_ratio < cfg.min_vol_ratio or vol_ratio > cfg.max_vol_ratio:
        return None, 0.0, ""
    if atr > cfg.max_atr_pct or abs(pre20) > cfg.max_pre_20d_abs_pct:
        return None, 0.0, ""

    dvol_score = min(1.0, dvol_m / max(cfg.min_dollar_vol_m * 5.0, 1.0))
    vol_score = min(1.0, vol_ratio / 8.0)

    if cfg.mode in ("both", "surge") and ret >= cfg.event_threshold_pct:
        if close_strength < cfg.surge_min_close_strength:
            return None, 0.0, ""
        breakout_bonus = 1.0 if bool(row.get("创20日高", False)) else 0.0
        score = 0.35 * close_strength + 0.25 * vol_score + 0.25 * dvol_score + 0.15 * breakout_bonus
        note = (
            f"涨{ret:.1f}% · 量比{vol_ratio:.1f} · 成交额${dvol_m:.0f}M · "
            f"收盘强度{close_strength:.2f}"
        )
        return "surge_continuation", float(min(score, 1.0)), note

    if cfg.mode in ("both", "drop") and ret <= -cfg.event_threshold_pct:
        if close_strength < cfg.drop_min_close_strength:
            return None, 0.0, ""
        low_bonus = 0.0 if bool(row.get("创20日低", False)) else 0.15
        score = 0.30 * close_strength + 0.25 * vol_score + 0.30 * dvol_score + low_bonus
        note = (
            f"跌{abs(ret):.1f}% · 量比{vol_ratio:.1f} · 成交额${dvol_m:.0f}M · "
            f"收盘未贴低{close_strength:.2f}"
        )
        return "drop_rebound", float(min(score, 1.0)), note

    return None, 0.0, ""


def scan_ticker_events(
    ticker: str,
    df: pd.DataFrame,
    cfg: ExtremeMoveConfig | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[ExtremeMoveEvent]:
    """Scan one ticker for liquid +/-10% events."""

    cfg = cfg or ExtremeMoveConfig()
    if df is None or len(df) < 60:
        return []

    feats = compute_extreme_features(df)
    if start:
        feats = feats.loc[feats.index >= pd.Timestamp(start)]
    if end:
        feats = feats.loc[feats.index <= pd.Timestamp(end)]

    events: list[ExtremeMoveEvent] = []
    for ts, row in feats.iterrows():
        kind, score, note = classify_extreme_row(row, cfg)
        if kind is None:
            continue
        events.append(
            ExtremeMoveEvent(
                代码=ticker.upper(),
                日期=ts.strftime("%Y-%m-%d"),
                类型=kind,
                类型名=EVENT_LABELS[kind],
                方向="做多",
                涨跌幅_pct=round(_finite(row.get("涨跌幅_pct")), 2),
                成交额M=round(_finite(row.get("成交额USD")) / 1e6, 1),
                量比=round(_finite(row.get("量比")), 2),
                收盘强度=round(_finite(row.get("收盘强度"), 0.5), 2),
                跳空_pct=round(_finite(row.get("跳空_pct")), 2),
                ret_5d_pct=round(_finite(row.get("5日涨跌_pct")), 2),
                ret_20d_pct=round(_finite(row.get("20日涨跌_pct")), 2),
                ATR_pct=round(_finite(row.get("ATR_pct")), 2),
                post_1d_pct=round(_finite(row.get("后1日_pct"), np.nan), 2),
                post_3d_pct=round(_finite(row.get("后3日_pct"), np.nan), 2),
                post_5d_pct=round(_finite(row.get("后5日_pct"), np.nan), 2),
                综合分=round(score, 3),
                说明=note,
            )
        )
    return events


def scan_universe_events(
    data: dict[str, pd.DataFrame],
    cfg: ExtremeMoveConfig | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Scan many tickers and return a sorted event table."""

    rows: list[dict[str, Any]] = []
    for ticker, df in data.items():
        if ticker.upper() == "SPY":
            continue
        for event in scan_ticker_events(ticker, df, cfg, start=start, end=end):
            rows.append(event.to_dict())
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["日期"] = pd.to_datetime(out["日期"])
    return out.sort_values(["日期", "综合分", "成交额M"], ascending=[True, False, False]).reset_index(drop=True)


def _trade_from_event(
    ticker: str,
    df: pd.DataFrame,
    event: pd.Series,
    cfg: ExtremeMoveConfig,
    *,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any] | None:
    event_date = pd.Timestamp(event["日期"])
    hist = df.sort_index().copy()
    hist.index = pd.to_datetime(hist.index)
    dates = hist.index[hist.index > event_date]
    if len(dates) == 0:
        return None

    entry_date = dates[0]
    entry = _finite(hist.loc[entry_date, "Open"], _finite(hist.loc[entry_date, "Close"]))
    if entry <= 0:
        return None

    planned_exit_idx = min(cfg.hold_days - 1, len(dates) - 1)
    planned_exit_date = dates[planned_exit_idx]
    stop_price = entry * (1.0 - cfg.stop_loss_pct)
    target_price = entry * (1.0 + cfg.take_profit_pct)
    exit_date = planned_exit_date
    exit_price = _finite(hist.loc[planned_exit_date, "Close"])
    exit_reason = "持有到期"

    for dt in dates[: planned_exit_idx + 1]:
        day = hist.loc[dt]
        if _finite(day["Low"]) <= stop_price:
            exit_date = dt
            exit_price = stop_price
            exit_reason = "止损"
            break
        if _finite(day["High"]) >= target_price:
            exit_date = dt
            exit_price = target_price
            exit_reason = "止盈"
            break

    gross_ret = exit_price / entry - 1.0
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    net_ret = gross_ret - round_trip_cost
    return {
        "选股日期": event_date.strftime("%Y-%m-%d"),
        "代码": ticker,
        "类型名": event["类型名"],
        "入场日期": entry_date.strftime("%Y-%m-%d"),
        "出场日期": pd.Timestamp(exit_date).strftime("%Y-%m-%d"),
        "入场价": round(entry, 4),
        "出场价": round(exit_price, 4),
        "毛收益率": gross_ret,
        "净收益率": net_ret,
        "退出原因": exit_reason,
        "综合分": _finite(event.get("综合分")),
        "事件涨跌幅_pct": _finite(event.get("涨跌幅_pct")),
        "成交额M": _finite(event.get("成交额M")),
        "量比": _finite(event.get("量比")),
        "说明": event.get("说明", ""),
    }


def simulate_event_trades(
    data: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    cfg: ExtremeMoveConfig | None = None,
    *,
    fee_bps: float = 5.0,
    slippage_bps: float = 15.0,
) -> pd.DataFrame:
    """Convert event rows into next-session trades with stop/target logic."""

    cfg = cfg or ExtremeMoveConfig()
    if events.empty:
        return pd.DataFrame()

    picked = []
    events = events.copy()
    events["日期"] = pd.to_datetime(events["日期"])
    for _, day in events.sort_values(["日期", "综合分"], ascending=[True, False]).groupby("日期"):
        picked.append(day.head(max(1, cfg.max_positions_per_day)))
    selected = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()

    trades: list[dict[str, Any]] = []
    for _, event in selected.iterrows():
        ticker = str(event["代码"]).upper()
        df = data.get(ticker)
        if df is None or df.empty:
            continue
        trade = _trade_from_event(ticker, df, event, cfg, fee_bps=fee_bps, slippage_bps=slippage_bps)
        if trade is not None:
            trades.append(trade)
    return pd.DataFrame(trades)


def summarize_event_strategy(
    trades: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    target_cagr: float = 1.0,
    target_win_rate: float = 0.90,
    target_max_drawdown: float = -0.10,
) -> dict[str, Any]:
    """Summarize trades against the user's aspirational constraints."""

    if trades.empty:
        return {"error": "没有满足条件的交易。"}

    daily = (
        trades.assign(入场日期=pd.to_datetime(trades["入场日期"]))
        .groupby("入场日期")["净收益率"]
        .mean()
        .sort_index()
    )
    equity = (1.0 + daily).cumprod()
    trade_ret = trades["净收益率"].astype(float)
    wins = trade_ret[trade_ret > 0]
    losses = trade_ret[trade_ret <= 0]
    avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
    payoff = float(wins.mean()) / avg_loss if len(wins) and avg_loss > 0 else 0.0

    if start and end:
        years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1 / 365.25)
        cagr = float(equity.iloc[-1] ** (1 / years) - 1.0)
    else:
        cagr = M.cagr(equity)

    max_dd = M.max_drawdown(equity)
    win_rate = float((trade_ret > 0).mean())
    return {
        "交易次数": int(len(trades)),
        "交易日数": int(len(daily)),
        "累计收益率": float(equity.iloc[-1] - 1.0),
        "年化收益率": cagr,
        "最大回撤": max_dd,
        "胜率": win_rate,
        "平均单笔收益": float(trade_ret.mean()),
        "盈亏比": payoff,
        "期末权益": float(equity.iloc[-1]),
        "目标年化>=100%": bool(cagr >= target_cagr),
        "目标胜率>=90%": bool(win_rate >= target_win_rate),
        "目标回撤>-10%": bool(max_dd >= target_max_drawdown),
    }
