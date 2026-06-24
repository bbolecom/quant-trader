"""流动性动量延续规律 → 策略 + 回测（含样本外 OOS）。

目标对照：用户要求「年化>100% · 胜率>90% · 回撤<10% · 流动性好」。
本脚本诚实地挖掘 + 回测，并把真实绩效与该目标对照（大概率无法同时满足——
若能稳健满足，早被套利抹平）。

规律（事件驱动，无未来函数；信号仅用 t 日及之前信息，t+1 开盘成交）：
  进场（t 日收盘判定）：
    · 5 日动量 ret_5d ≥ ret5_min（强势）
    · 量比 vol_ratio ≥ vr_min（资金确认）
    · 收盘强度 close_strength ≥ cs_min（收在日内高位）
    · 站上 MA20 且站上 MA50（顺势）
    · 日成交额 ≥ dvol_min_m（流动性闸门）
  离场（真实 High/Low 路径，t+1 开盘为入场基准）：
    · H 日内 high ≥ 入场×(1+tp) → 止盈
    · H 日内 low ≤ 入场×(1-sl) → 止损（同日两触发按止损，保守）
    · 否则 H 日收盘平仓
  组合：K 个并发槽位，等权、按实现盈亏复利；满仓时跳过新信号 → 日频权益曲线求最大回撤。

用法：
    python research/liquid_momentum_pattern.py            # 默认参数 + 网格寻优 + OOS
    python research/liquid_momentum_pattern.py --no-grid  # 仅默认参数
    python research/liquid_momentum_pattern.py --years 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gainer_daily_backtest import GAINER_MOMENTUM  # 流动性宇宙

CACHE = ROOT / "research" / "_liquid_momentum_prices.pkl"
COST_BPS = 15.0  # 单边手续费+滑点（基点），往返按 2× 计


# --------------------------- 数据 ---------------------------

def fetch_prices(tickers: list[str], years: int, *, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    if use_cache and CACHE.exists():
        try:
            obj = pickle.loads(CACHE.read_bytes())
            if obj.get("years") == years and set(obj.get("tickers", [])) >= set(tickers):
                return obj["data"]
        except Exception:  # noqa: BLE001
            pass
    import yfinance as yf

    period = f"{years}y"
    raw = yf.download(
        tickers, period=period, interval="1d",
        group_by="ticker", auto_adjust=True, threads=True, progress=False,
    )
    data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(subset=["Close"]).copy()
        if len(df) < 120:
            continue
        df.index = pd.to_datetime(df.index)
        data[t] = df[["Open", "High", "Low", "Close", "Volume"]]
    CACHE.write_bytes(pickle.dumps({"years": years, "tickers": list(data.keys()), "data": data}))
    return data


# --------------------------- 信号 + 交易 ---------------------------

@dataclass
class Params:
    mode: str = "momentum"        # momentum（追强）| pullback（顺势回调，高胜率）
    ret5_min: float = 0.10
    vr_min: float = 1.5
    cs_min: float = 0.60
    dvol_min_m: float = 50.0
    tp: float = 0.10
    sl: float = 0.06
    horizon: int = 5
    slots: int = 8
    rsi_buy: float = 10.0         # pullback：RSI2 低于此值视为超跌


def _rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _signal_mask(df: pd.DataFrame, p: Params) -> pd.Series:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    ret5 = close.pct_change(5)
    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    vol_ratio = vol / vol_ma20.replace(0, np.nan)
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=25).mean()
    hl = (high - low).replace(0, np.nan)
    cs = ((close - low) / hl).clip(0, 1)
    dvol_m = close * vol / 1e6

    return (
        (ret5 >= p.ret5_min)
        & (vol_ratio >= p.vr_min)
        & (cs >= p.cs_min)
        & (close > ma20)
        & (close > ma50)
        & (dvol_m >= p.dvol_min_m)
    ).fillna(False)


def gen_trades(data: dict[str, pd.DataFrame], p: Params) -> pd.DataFrame:
    """生成所有交易：entry_date, exit_date, ret（已扣往返成本）。"""
    cost = 2 * COST_BPS / 1e4
    rows = []
    for tk, df in data.items():
        mask = _signal_mask(df, p)
        idx = np.where(mask.values)[0]
        n = len(df)
        opens = df["Open"].astype(float).values
        highs = df["High"].astype(float).values
        lows = df["Low"].astype(float).values
        closes = df["Close"].astype(float).values
        dates = df.index
        for i in idx:
            j = i + 1  # t+1 开盘进场
            if j >= n:
                continue
            entry = opens[j]
            if entry <= 0:
                continue
            tp_px = entry * (1 + p.tp)
            sl_px = entry * (1 - p.sl)
            exit_ret = None
            exit_k = min(n - 1, j + p.horizon - 1)
            for k in range(j, min(n, j + p.horizon)):
                if lows[k] <= sl_px:        # 同日双触发按止损（保守）
                    exit_ret = -p.sl
                    exit_k = k
                    break
                if highs[k] >= tp_px:
                    exit_ret = p.tp
                    exit_k = k
                    break
            if exit_ret is None:
                exit_ret = closes[exit_k] / entry - 1.0
            rows.append({
                "ticker": tk,
                "entry_date": dates[j],
                "exit_date": dates[exit_k],
                "ret": exit_ret - cost,
            })
    if not rows:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "ret"])
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


# --------------------------- 组合模拟 ---------------------------

def simulate(trades: pd.DataFrame, slots: int) -> dict:
    """K 槽位等权复利组合：返回 win_rate / 年化 / 最大回撤 / 交易数 等。"""
    if trades.empty:
        return {"n_trades": 0}
    taken = []
    open_pos = []  # (exit_date, alloc, ret)
    equity = 1.0
    free = slots
    all_dates = pd.DatetimeIndex(sorted(set(trades["entry_date"]) | set(trades["exit_date"])))
    by_entry = {d: g for d, g in trades.groupby("entry_date")}
    curve = []
    for d in all_dates:
        # 先平到期仓
        still = []
        for exit_d, alloc, ret in open_pos:
            if exit_d == d:
                equity += alloc * ret
                free += 1
            else:
                still.append((exit_d, alloc, ret))
        open_pos = still
        # 再开当日新仓（满仓则跳过）
        if d in by_entry:
            for _, tr in by_entry[d].iterrows():
                if free <= 0:
                    continue
                alloc = equity / slots
                if tr["exit_date"] == d:
                    # 当日即触发止盈/止损 → 立即了结，不占用后续槽位（否则会漏槽）。
                    equity += alloc * tr["ret"]
                else:
                    open_pos.append((tr["exit_date"], alloc, tr["ret"]))
                    free -= 1
                taken.append(tr["ret"])
        curve.append((d, equity))
    eq = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    taken_arr = np.array(taken) if taken else np.array([0.0])
    span_days = max((all_dates[-1] - all_dates[0]).days, 1)
    years = span_days / 365.25
    cagr = equity ** (1 / years) - 1 if years > 0 and equity > 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "n_trades": int(len(taken)),
        "win_rate": float((taken_arr > 0).mean()),
        "avg_ret": float(taken_arr.mean()),
        "final_equity": float(equity),
        "cagr": float(cagr),
        "max_dd": float(dd),
    }


def _fmt(m: dict) -> str:
    if not m.get("n_trades"):
        return "无交易"
    return (
        f"交易{m['n_trades']:>4} · 胜率{m['win_rate']*100:5.1f}% · "
        f"单笔均值{m['avg_ret']*100:5.2f}% · 年化{m['cagr']*100:7.1f}% · "
        f"最大回撤{m['max_dd']*100:6.1f}%"
    )


# --------------------------- 主流程 ---------------------------

def split_oos(trades: pd.DataFrame, ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades, trades
    cut = trades["entry_date"].quantile(ratio)
    return trades[trades["entry_date"] <= cut], trades[trades["entry_date"] > cut]


def evaluate(data, p: Params) -> dict:
    trades = gen_trades(data, p)
    full = simulate(trades, p.slots)
    is_tr, oos_tr = split_oos(trades)
    return {
        "params": asdict(p),
        "full": full,
        "is": simulate(is_tr, p.slots),
        "oos": simulate(oos_tr, p.slots),
    }


def grid_search(data) -> list[dict]:
    grid = {
        "ret5_min": [0.08, 0.12, 0.18],
        "vr_min": [1.5, 2.0],
        "cs_min": [0.60, 0.70],
        "tp": [0.08, 0.12],
        "sl": [0.05, 0.07],
        "horizon": [5, 10],
    }
    keys = list(grid)
    out = []
    for combo in itertools.product(*grid.values()):
        p = Params(**dict(zip(keys, combo)))
        trades = gen_trades(data, p)
        is_tr, oos_tr = split_oos(trades)
        m_is = simulate(is_tr, p.slots)
        if m_is.get("n_trades", 0) < 50:
            continue
        m_oos = simulate(oos_tr, p.slots)
        out.append({"params": asdict(p), "is": m_is, "oos": m_oos})
    # 按样本内 年化/回撤 排序（要求回撤非零避免除零）
    out.sort(key=lambda r: r["is"]["cagr"] / max(abs(r["is"]["max_dd"]), 0.02), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    universe = list(dict.fromkeys(GAINER_MOMENTUM))
    print(f"宇宙 {len(universe)} 只流动性标的 · 拉取 {args.years} 年日线…")
    data = fetch_prices(universe, args.years, use_cache=not args.no_cache)
    print(f"成功载入 {len(data)} 只。\n")

    print("=" * 78)
    print("默认参数（先验，不调参）：", asdict(Params()))
    base = evaluate(data, Params())
    print("  全样本 ", _fmt(base["full"]))
    print("  样本内 ", _fmt(base["is"]))
    print("  样本外 ", _fmt(base["oos"]))
    print("=" * 78)

    if not args.no_grid:
        print("\n网格寻优（按样本内 年化/回撤 排序），展示前 8 + 各自样本外：\n")
        results = grid_search(data)
        for r in results[:8]:
            pr = r["params"]
            tag = (f"ret5≥{pr['ret5_min']:.0%} vr≥{pr['vr_min']} cs≥{pr['cs_min']:.0%} "
                   f"tp{pr['tp']:.0%}/sl{pr['sl']:.0%} H{pr['horizon']}")
            print(f"[{tag}]")
            print("   IS : ", _fmt(r["is"]))
            print("   OOS: ", _fmt(r["oos"]))
        if results:
            best = results[0]
            (ROOT / "research" / "liquid_momentum_best.json").write_text(
                json.dumps(best, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print("\n最佳（样本内）已存 research/liquid_momentum_best.json")


if __name__ == "__main__":
    main()
