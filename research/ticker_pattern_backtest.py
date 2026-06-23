#!/usr/bin/env python3
"""单标的规律策略 · 事件驱动回测（年化 / 回撤 / 胜率）。

基于 ticker_pattern_mine 挖掘的规律，将信号转为可交易规则并回测：
  S1 动量顺势做多   · 量比+5日涨+MA50+（可选SPY过滤）
  S2 深跌惯性做空   · 5日跌>10%
  S3 缩量顶做空     · 5日涨>15% + 量比<1
  S4 过热回避做空   · 20日涨>40%
  S5 组合轮动       · S1 多 / S2·S3·S4 空，弱市不做多

用法：
    python research/ticker_pattern_backtest.py
    python research/ticker_pattern_backtest.py --ticker MSTR SMCI COIN
    python research/ticker_pattern_backtest.py --start 2019-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.decline_income import equity_metrics_from_trades
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.ticker_pattern_strategy import (
    detect_long_signal as _detect_long_signal,
    long_momentum as _long_momentum,
    long_signal_highwin,
    long_signal_ultra,
    long_trend as _long_trend,
    short_deep_drop as _short_deep_drop,
    short_fade as _short_fade,
    short_overheat as _short_overheat,
    short_shrink_top as _short_shrink_top,
    trade_return_bracket,
)
from research.ticker_pattern_mine import TRAIN_END, build_ticker_panel

FEE = 10 / 10_000  # 单边 5bp，往返 10bp
DEFAULT_TICKERS = ["MSTR", "SMCI", "COIN"]
OUT_JSON = ROOT / "research" / "ticker_pattern_backtest.json"


def _combo_signal(row: pd.Series, spy_bull: bool) -> tuple[str | None, str]:
    """组合：空信号优先；弱市不做多。"""
    if _short_fade(row, spy_bull):
        return "short", "S6超涨回吐"
    if _short_shrink_top(row):
        return "short", "S3缩量顶"
    if _short_deep_drop(row):
        return "short", "S2深跌惯性"
    if _short_overheat(row):
        return "short", "S4过热"
    sid, name = _detect_long_signal(row, spy_bull, high_win=False)
    if sid == "S1":
        return "long", "S1动量"
    if sid == "S7":
        return "long", "S7趋势中继"
    return None, ""


@dataclass(frozen=True)
class StrategySpec:
    id: str
    name: str
    side: str  # long | short
    hold_days: int
    signal_fn: Callable[[pd.Series, bool], bool]
    description: str
    entry_lag: int = 1  # 0=信号日收盘  1=次日收盘（默认）


STRATEGIES: list[StrategySpec] = [
    StrategySpec("S1", "动量顺势做多", "long", 5, _long_momentum,
                 "量比+5日涨≥5%+MA50+换手≥2%", entry_lag=1),
    StrategySpec("S7", "趋势中继做多", "long", 5, _long_trend,
                 "20日涨≥30%+MA50+SPY牛市", entry_lag=1),
    StrategySpec("S2", "深跌惯性做空", "short", 3, lambda r, sb: _short_deep_drop(r),
                 "5日跌>10%，持有3日", entry_lag=0),
    StrategySpec("S3", "缩量顶做空", "short", 5, lambda r, sb: _short_shrink_top(r),
                 "5日涨>15%+量比<1", entry_lag=0),
    StrategySpec("S4", "过热回避做空", "short", 5, lambda r, sb: _short_overheat(r),
                 "20日涨>40%", entry_lag=0),
    StrategySpec("S6", "超涨回吐做空", "short", 1, _short_fade,
                 "弱市涨7~14%+收弱", entry_lag=1),
]


def _attach_spy(panel: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    out = panel.copy()
    out["日期"] = pd.to_datetime(out["日期"])
    spy = spy_close.copy()
    spy.index = pd.to_datetime(spy.index)
    spy_ma50 = spy.rolling(50, min_periods=25).mean()
    spy_map = pd.DataFrame({
        "日期": spy.index,
        "spy_close": spy.values,
        "spy_ma50": spy_ma50.values,
    })
    out = out.merge(spy_map, on="日期", how="left")
    out["spy_bull"] = out["spy_close"] > out["spy_ma50"]
    return out


def _trade_return(close: pd.Series, entry_i: int, hold: int, side: str) -> float | None:
    exit_i = entry_i + hold
    if exit_i >= len(close):
        return None
    r = float(close.iloc[exit_i] / close.iloc[entry_i] - 1.0)
    if side == "short":
        r = -r
    return r - FEE


def backtest_strategy(
    panel: pd.DataFrame,
    close: pd.Series,
    spec: StrategySpec,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """非重叠持仓：信号日 → entry_lag 后入场 → 持有 hold_days。"""
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    if start:
        p = p[p["日期"] >= pd.Timestamp(start)]
    if end:
        p = p[p["日期"] <= pd.Timestamp(end)]
    close = close.copy()
    close.index = pd.to_datetime(close.index)

    date_to_i = {d: i for i, d in enumerate(close.index)}
    trades: list[dict] = []
    next_free = 0

    for _, row in p.iterrows():
        d = row["日期"]
        i = date_to_i.get(d)
        if i is None or i < next_free:
            continue
        spy_bull = bool(row.get("spy_bull", True))
        if not spec.signal_fn(row, spy_bull):
            continue
        entry_i = i + spec.entry_lag
        if entry_i >= len(close):
            continue
        ret = _trade_return(close, entry_i, spec.hold_days, spec.side)
        if ret is None:
            continue
        exit_i = entry_i + spec.hold_days
        trades.append({
            "信号日": d.strftime("%Y-%m-%d"),
            "入场日": close.index[entry_i].strftime("%Y-%m-%d"),
            "出场日": close.index[exit_i].strftime("%Y-%m-%d"),
            "方向": spec.side,
            "收益": ret,
            "收益%": ret * 100,
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


def backtest_combo(
    panel: pd.DataFrame,
    close: pd.Series,
    *,
    start: str | None = None,
    end: str | None = None,
    hold_days: int = 5,
) -> pd.DataFrame:
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    if start:
        p = p[p["日期"] >= pd.Timestamp(start)]
    if end:
        p = p[p["日期"] <= pd.Timestamp(end)]
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    date_to_i = {d: i for i, d in enumerate(close.index)}

    trades: list[dict] = []
    next_free = 0
    for _, row in p.iterrows():
        d = row["日期"]
        i = date_to_i.get(d)
        if i is None or i < next_free:
            continue
        side, tag = _combo_signal(row, bool(row.get("spy_bull", True)))
        if side is None:
            continue
        lag = 1 if side == "long" else 0
        entry_i = i + lag
        if entry_i >= len(close):
            continue
        hd = 5 if side == "long" else (1 if tag == "S6超涨回吐" else 3)
        ret = _trade_return(close, entry_i, hd, side)
        if ret is None:
            continue
        exit_i = entry_i + hd
        trades.append({
            "信号日": d.strftime("%Y-%m-%d"),
            "入场日": close.index[entry_i].strftime("%Y-%m-%d"),
            "出场日": close.index[exit_i].strftime("%Y-%m-%d"),
            "方向": side,
            "子策略": tag,
            "收益": ret,
            "收益%": ret * 100,
        })
        next_free = exit_i + 1
    return pd.DataFrame(trades)


def _backtest_long_only(
    panel: pd.DataFrame,
    close: pd.Series,
    df: pd.DataFrame | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    high_win: bool = False,
    ultra: bool = False,
    use_bracket: bool = False,
    take_profit: float = 0.08,
    stop_loss: float = 0.05,
) -> pd.DataFrame:
    """S8 / S8H / S8U：仅做多，非重叠。"""
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    if start:
        p = p[p["日期"] >= pd.Timestamp(start)]
    if end:
        p = p[p["日期"] <= pd.Timestamp(end)]
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    high = df["High"].astype(float) if df is not None else close
    low = df["Low"].astype(float) if df is not None else close
    if df is not None:
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
    date_to_i = {d: i for i, d in enumerate(close.index)}

    trades: list[dict] = []
    next_free = 0
    hold = 5
    for _, row in p.iterrows():
        d = row["日期"]
        i = date_to_i.get(d)
        if i is None or i < next_free:
            continue
        spy_bull = bool(row.get("spy_bull", True))
        if ultra:
            sid, tag = long_signal_ultra(row, spy_bull)
            if sid is None:
                continue
            tag = f"{sid}{tag}"
        elif high_win:
            sid, tag = long_signal_highwin(row, spy_bull)
            if sid is None:
                continue
            tag = f"{sid}{tag}"
        else:
            tag = None
            if _long_momentum(row, spy_bull):
                tag = "S1动量"
            elif _long_trend(row, spy_bull):
                tag = "S7趋势"
            if tag is None:
                continue
        entry_i = i + 1
        if entry_i >= len(close):
            continue
        path_bracket = ultra or (use_bracket and df is not None)
        tp = 0.02 if ultra else take_profit
        sl = 0.05 if ultra else stop_loss
        if path_bracket and df is not None:
            ret, held = trade_return_bracket(
                close, high, low, entry_i, hold,
                take_profit=tp, stop_loss=sl, fee=FEE,
            )
        else:
            ret = _trade_return(close, entry_i, hold, "long")
            held = hold
        if ret is None:
            continue
        exit_i = min(entry_i + held, len(close) - 1)
        trades.append({
            "信号日": d.strftime("%Y-%m-%d"),
            "入场日": close.index[entry_i].strftime("%Y-%m-%d"),
            "出场日": close.index[exit_i].strftime("%Y-%m-%d"),
            "方向": "long",
            "子策略": tag,
            "收益": ret,
            "收益%": ret * 100,
        })
        next_free = exit_i + 1
    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame, *, alloc_pct: float = 0.25) -> dict:
    if trades.empty:
        return {
            "笔数": 0, "胜率": 0.0, "年化": 0.0, "最大回撤": 0.0,
            "累计收益": 0.0, "夏普": 0.0, "均单笔%": 0.0,
        }
    rets = trades["收益"].astype(float)
    dates = pd.to_datetime(trades["出场日"])
    stats = equity_metrics_from_trades(rets, alloc_pct=alloc_pct, dates=dates)
    return {
        "笔数": int(len(trades)),
        "胜率": round(float(stats.get("胜率", 0)), 4),
        "年化": round(float(stats.get("年化收益率", 0)), 4),
        "最大回撤": round(float(stats.get("最大回撤", 0)), 4),
        "累计收益": round(float(stats.get("累计收益率", 0)), 4),
        "夏普": round(float(stats.get("夏普比率", 0)), 2),
        "均单笔%": round(float(rets.mean() * 100), 2),
        "盈亏比": round(
            float(rets[rets > 0].mean() / abs(rets[rets < 0].mean()))
            if (rets < 0).any() and (rets > 0).any() else 0.0,
            2,
        ),
    }


def run_ticker(
    ticker: str,
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    alloc_pct: float = 0.25,
) -> dict:
    tk = ticker.upper()
    end = end or date.today().isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    df = yahoo.fetch_history(tk, start, end)
    if df is None or df.empty:
        return {"ticker": tk, "error": "无行情"}

    panel = build_ticker_panel(tk, start=start, end=end)
    if panel.empty:
        return {"ticker": tk, "error": "无法构建特征面板"}

    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    panel = _attach_spy(panel, spy)
    panel["代码"] = tk
    close = df["Close"].astype(float)

    cut = pd.Timestamp(TRAIN_END)
    periods = [
        ("全样本", None, None),
        ("样本内2019-2023", None, TRAIN_END),
        ("样本外2024+", "2024-01-01", None),
    ]

    rows: list[dict] = []
    for spec in STRATEGIES:
        for label, p_start, p_end in periods:
            ps = start if p_start is None else max(start, p_start)
            pe = end if p_end is None else min(end, p_end)
            if pd.Timestamp(ps) > pd.Timestamp(pe):
                continue
            tr = backtest_strategy(panel, close, spec, start=ps, end=pe)
            sm = summarize_trades(tr, alloc_pct=alloc_pct)
            rows.append({
                "代码": tk,
                "策略": spec.id,
                "策略名": spec.name,
                "方向": spec.side,
                "持有": f"{spec.hold_days}日",
                "区间": label,
                **sm,
            })

    for label, p_start, p_end in periods:
        ps = start if p_start is None else max(start, p_start)
        pe = end if p_end is None else min(end, p_end)
        if pd.Timestamp(ps) > pd.Timestamp(pe):
            continue
        tr = backtest_combo(panel, close, start=ps, end=pe)
        sm = summarize_trades(tr, alloc_pct=alloc_pct)
        rows.append({
            "代码": tk,
            "策略": "S5",
            "策略名": "组合轮动",
            "方向": "多/空",
            "持有": "混合",
            "区间": label,
            **sm,
        })
        # S8 纯多头：仅 S1 + S7
        tr8 = _backtest_long_only(panel, close, start=ps, end=pe)
        sm8 = summarize_trades(tr8, alloc_pct=alloc_pct)
        rows.append({
            "代码": tk,
            "策略": "S8",
            "策略名": "纯多头组合",
            "方向": "long",
            "持有": "5日",
            "区间": label,
            **sm8,
        })
        tr8h = _backtest_long_only(
            panel, close, df, start=ps, end=pe,
            high_win=True, use_bracket=False,
        )
        sm8h = summarize_trades(tr8h, alloc_pct=alloc_pct)
        rows.append({
            "代码": tk,
            "策略": "S8H",
            "策略名": "高胜率纯多头",
            "方向": "long",
            "持有": "5日",
            "区间": label,
            **sm8h,
        })
        tr8u = _backtest_long_only(
            panel, close, df, start=ps, end=pe,
            ultra=True,
        )
        sm8u = summarize_trades(tr8u, alloc_pct=alloc_pct)
        rows.append({
            "代码": tk,
            "策略": "S8U",
            "策略名": "Ultra80路径止盈",
            "方向": "long",
            "持有": "5日+2%TP/5%SL",
            "区间": label,
            **sm8u,
        })

    # 全样本组合交易明细
    combo_all = backtest_combo(panel, close, start=start, end=end)
    return {
        "ticker": tk,
        "start": start,
        "end": end,
        "alloc_pct": alloc_pct,
        "fee_bps_roundtrip": FEE * 10000,
        "results": rows,
        "combo_trades": combo_all.to_dict(orient="records") if not combo_all.empty else [],
    }


def run_all(
    tickers: list[str],
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    alloc_pct: float = 0.25,
) -> dict:
    docs = [run_ticker(t, start=start, end=end, alloc_pct=alloc_pct) for t in tickers]
    flat = []
    for d in docs:
        flat.extend(d.get("results") or [])
    summary_df = pd.DataFrame(flat)
    doc = {
        "generated": date.today().isoformat(),
        "start": start,
        "end": end or date.today().isoformat(),
        "alloc_pct_per_trade": alloc_pct,
        "tickers": docs,
        "summary_table": flat,
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def print_report(doc: dict) -> None:
    print(f"\n{'=' * 88}")
    print(f"规律策略回测 · {doc.get('generated')} · {doc.get('start')} ~ {doc.get('end')}")
    print(f"单笔仓位 {doc.get('alloc_pct_per_trade', 0.25):.0%} · 往返成本 {FEE*100:.2f}%")
    print(f"{'=' * 88}")

    df = pd.DataFrame(doc.get("summary_table") or [])
    if df.empty:
        print("无结果")
        return

    for tk in df["代码"].unique():
        sub = df[df["代码"] == tk]
        print(f"\n【{tk}】")
        show = sub[sub["区间"] == "全样本"].copy()
        if show.empty:
            show = sub
        cols = ["策略", "策略名", "笔数", "胜率", "年化", "最大回撤", "夏普", "均单笔%"]
        for _, r in show.sort_values("策略").iterrows():
            if r["策略"] not in ("S8", "S8H", "S8U", "S1", "S7", "S5"):
                continue
            print(
                f"  {r['策略']:3} {r['策略名']:12} "
                f"n={int(r['笔数']):3d}  胜率{r['胜率']:.1%}  "
                f"年化{r['年化']:+.1%}  回撤{r['最大回撤']:.1%}  "
                f"夏普{r['夏普']:+.2f}  均笔{r['均单笔%']:+.2f}%"
            )

        oos = sub[sub["区间"] == "样本外2024+"]
        if not oos.empty:
            print("  --- OOS 2024+ ---")
            for _, r in oos.sort_values("策略").iterrows():
                if r["策略"] not in ("S8", "S8H", "S8U", "S1", "S7"):
                    continue
                print(
                    f"  {r['策略']:3} {r['策略名']:12} "
                    f"n={int(r['笔数']):3d}  胜率{r['胜率']:.1%}  "
                    f"年化{r['年化']:+.1%}  回撤{r['最大回撤']:.1%}"
                )

    print(f"\n→ {OUT_JSON}")
    print(f"{'=' * 88}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="单标的规律策略回测")
    ap.add_argument("--ticker", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--alloc", type=float, default=0.25, help="每笔交易账户占比")
    args = ap.parse_args()
    doc = run_all(args.ticker, start=args.start, end=args.end, alloc_pct=args.alloc)
    print_report(doc)


if __name__ == "__main__":
    main()
