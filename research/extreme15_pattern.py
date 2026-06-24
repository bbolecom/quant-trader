#!/usr/bin/env python3
"""最近5年「单日异动 >阈值%（涨幅榜/跌幅榜双向）」后续规律发现 + 多空策略回测。

两段式：
  A. 规律发现：扫描所有 |单日涨跌| ≥ 阈值（默认15%）的流动性事件，
     统计「暴涨后 / 暴跌后」未来 1/3/5/10 日的收益分布与胜率，
     并按 收盘强度 / 量比 / 跳空 / 前期20日趋势 / 大盘环境 分档，
     找出「顺势 or 反转」「做多 or 做空」的方向性边缘。
  B. 策略回测：用发现的最优规则，次日开盘真实入场 + 止盈止损，
     等权多仓组合、按日复利，输出 年化/回撤/胜率/夏普/卡尔玛，
     IS(样本内) / OOS(样本外) 拆分 + 目标检验（年化>100% 回撤<10% 胜率>90%）。

流动性优先：价格≥$5、单日成交额≥$100M（可调）。

用法：
    python3 research/extreme15_pattern.py --pool momentum --threshold 15
    python3 research/extreme15_pattern.py --pool broad --threshold 15 --discover-only
    python3 research/extreme15_pattern.py --threshold 20 --hold 3 --stop 0.06 --tp 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant import metrics as M
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

TRAIN_END = "2024-06-30"
OUT_JSON = ROOT / "research" / "extreme15_pattern.json"
EVENTS_CSV = ROOT / "research" / "extreme15_events.csv"
TRADES_CSV = ROOT / "research" / "extreme15_trades.csv"


# ───────────────────────── 事件面板 ─────────────────────────
def build_event_panel(
    data: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    *,
    threshold_pct: float,
    min_price: float,
    min_dvol_m: float,
) -> pd.DataFrame:
    """向量化扫描所有 |单日涨跌| ≥ 阈值 的流动性事件，附带特征与多/空前向收益。"""
    spy = spy_close.astype(float)
    spy = spy[~spy.index.duplicated(keep="last")]
    spy_ret5 = spy.pct_change(5)
    spy_ma20 = spy.rolling(20).mean()
    spy_bull = (spy > spy_ma20)

    frames: list[pd.DataFrame] = []
    for tk, df in data.items():
        if tk.upper() == "SPY" or df is None or len(df) < 60:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        close = df["Close"].astype(float)
        open_ = df["Open"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        vol = df["Volume"].astype(float)
        prev = close.shift(1)

        ret1 = close / prev - 1.0
        dvol = close * vol
        vol_ratio = vol / vol.rolling(20).mean().replace(0, np.nan)
        hl = (high - low).replace(0, np.nan)
        close_strength = ((close - low) / hl).clip(0, 1)
        gap = open_ / prev - 1.0
        pre5 = prev / close.shift(6) - 1.0
        pre20 = prev / close.shift(21) - 1.0
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr_pct = tr.rolling(14).mean() / close.replace(0, np.nan)
        high20 = close >= close.shift(1).rolling(20).max()
        low20 = close <= close.shift(1).rolling(20).min()

        # 次日开盘入场 → 收盘出场（持有 1/3/5/10 日）的多头收益（close-to-open 基准）
        nxt_open = open_.shift(-1)
        fwd = {}
        for k in (1, 3, 5, 10):
            exit_close = close.shift(-k)
            fwd[f"long_open_{k}d"] = exit_close / nxt_open - 1.0
            fwd[f"long_cc_{k}d"] = close.shift(-k) / close - 1.0

        tmp = pd.DataFrame({
            "代码": tk.upper(),
            "ret1": ret1,
            "dvol_m": dvol / 1e6,
            "vol_ratio": vol_ratio,
            "close_strength": close_strength,
            "gap": gap,
            "pre5": pre5,
            "pre20": pre20,
            "atr_pct": atr_pct,
            "high20": high20,
            "low20": low20,
            "close": close,
            "nxt_open_exists": nxt_open.notna(),
            **fwd,
        }, index=close.index)
        tmp["spy_bull"] = spy_bull.reindex(close.index).ffill()
        tmp["spy_ret5"] = spy_ret5.reindex(close.index).ffill()
        tmp = tmp.reset_index().rename(columns={"index": "日期"})
        if tmp.columns[0] != "日期":
            tmp = tmp.rename(columns={tmp.columns[0]: "日期"})
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["日期"] = pd.to_datetime(panel["日期"])

    th = threshold_pct / 100.0
    ev = panel[
        (panel["ret1"].abs() >= th)
        & (panel["close"] >= min_price)
        & (panel["dvol_m"] >= min_dvol_m)
        & (panel["nxt_open_exists"])
    ].copy()
    ev["direction"] = np.where(ev["ret1"] > 0, "surge", "drop")
    return ev.dropna(subset=["vol_ratio", "close_strength"]).reset_index(drop=True)


# ───────────────────────── 规律发现 ─────────────────────────
def _wr(s: pd.Series) -> float:
    s = s.dropna()
    return float((s > 0).mean()) if len(s) else float("nan")


def _stat(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean%": round(float(s.mean()) * 100, 2),
        "median%": round(float(s.median()) * 100, 2),
        "win%": round(float((s > 0).mean()) * 100, 1),
    }


def discover(ev: pd.DataFrame) -> dict:
    """暴涨后/暴跌后，多空两侧的前向收益分布 + 条件分档。"""
    out: dict = {}
    for d, label in [("surge", "暴涨后"), ("drop", "暴跌后")]:
        sub = ev[ev["direction"] == d]
        if sub.empty:
            continue
        block: dict = {"事件数": int(len(sub)), "区间": "全样本"}
        # 多头（次日开盘入场）各持有期；做空收益 = -做多
        for k in (1, 3, 5, 10):
            lo = sub[f"long_open_{k}d"]
            block[f"做多{k}日"] = _stat(lo)
            block[f"做空{k}日"] = _stat(-lo)
        # 条件分档（看 3 日做多 open 入场）
        buckets: list[dict] = []
        col = "long_open_3d"
        for name, mask in [
            ("收盘强度≥0.8", sub["close_strength"] >= 0.8),
            ("收盘强度≤0.3", sub["close_strength"] <= 0.3),
            ("量比≥3", sub["vol_ratio"] >= 3),
            ("量比1.5~3", sub["vol_ratio"].between(1.5, 3)),
            ("跳空≥3%", sub["gap"] >= 0.03),
            ("跳空≤-3%", sub["gap"] <= -0.03),
            ("前20日已涨>30%", sub["pre20"] > 0.30),
            ("前20日已跌>30%", sub["pre20"] < -0.30),
            ("大盘多头", sub["spy_bull"] == True),  # noqa: E712
            ("大盘空头", sub["spy_bull"] == False),  # noqa: E712
            ("创20日高", sub["high20"] == True),  # noqa: E712
            ("创20日低", sub["low20"] == True),  # noqa: E712
        ]:
            seg = sub[mask]
            if len(seg) < 20:
                continue
            buckets.append({
                "条件": name,
                "做多3日": _stat(seg[col]),
                "做空3日": _stat(-seg[col]),
            })
        block["条件分档"] = buckets
        out[label] = block
    return out


# ───────────────────────── 策略回测 ─────────────────────────
@dataclass
class Rule:
    name: str
    direction: str        # surge | drop  （事件类型）
    side: str             # long | short （交易方向）
    hold_days: int = 3
    stop: float = 0.06
    tp: float = 0.12
    min_close_strength: float = 0.0
    max_close_strength: float = 1.0
    min_vol_ratio: float = 0.0
    max_vol_ratio: float = 99.0
    min_gap: float = -9.0
    max_gap: float = 9.0
    pre20_max_abs: float = 9.0
    min_pre20: float = -9.0
    max_pre20: float = 9.0
    require_spy_bull: bool = False
    require_spy_bear: bool = False
    require_high20: bool = False
    require_low20: bool = False
    max_per_day: int = 3


def _select(ev: pd.DataFrame, r: Rule) -> pd.DataFrame:
    s = ev[ev["direction"] == r.direction].copy()
    s = s[
        s["close_strength"].between(r.min_close_strength, r.max_close_strength)
        & s["vol_ratio"].between(r.min_vol_ratio, r.max_vol_ratio)
        & s["gap"].between(r.min_gap, r.max_gap)
        & (s["pre20"].abs() <= r.pre20_max_abs)
        & s["pre20"].between(r.min_pre20, r.max_pre20)
    ]
    if r.require_spy_bull:
        s = s[s["spy_bull"] == True]  # noqa: E712
    if r.require_spy_bear:
        s = s[s["spy_bull"] == False]  # noqa: E712
    if r.require_high20:
        s = s[s["high20"] == True]  # noqa: E712
    if r.require_low20:
        s = s[s["low20"] == True]  # noqa: E712
    return s


def _trade_returns(
    data: dict[str, pd.DataFrame],
    sel: pd.DataFrame,
    r: Rule,
    *,
    fee_bps: float,
    slip_bps: float,
) -> pd.DataFrame:
    """次日开盘入场 + 日内止盈/止损 + 持有到期；支持多空。"""
    cost = 2.0 * (fee_bps + slip_bps) / 10_000.0
    rows: list[dict] = []
    for _, e in sel.iterrows():
        tk = str(e["代码"]).upper()
        df = data.get(tk)
        if df is None or df.empty:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        idx = df.index[df.index > pd.Timestamp(e["日期"])]
        if len(idx) == 0:
            continue
        entry = float(df.loc[idx[0], "Open"])
        if entry <= 0:
            continue
        horizon = idx[: min(r.hold_days, len(idx))]
        if r.side == "long":
            stop_p = entry * (1 - r.stop)
            tp_p = entry * (1 + r.tp)
        else:
            stop_p = entry * (1 + r.stop)
            tp_p = entry * (1 - r.tp)
        exit_p = float(df.loc[horizon[-1], "Close"])
        reason = "到期"
        for dt in horizon:
            hi = float(df.loc[dt, "High"])
            lo = float(df.loc[dt, "Low"])
            if r.side == "long":
                if lo <= stop_p:
                    exit_p, reason = stop_p, "止损"; break
                if hi >= tp_p:
                    exit_p, reason = tp_p, "止盈"; break
            else:
                if hi >= stop_p:
                    exit_p, reason = stop_p, "止损"; break
                if lo <= tp_p:
                    exit_p, reason = tp_p, "止盈"; break
        gross = (exit_p / entry - 1.0) if r.side == "long" else (entry / exit_p - 1.0)
        rows.append({
            "事件日": pd.Timestamp(e["日期"]).strftime("%Y-%m-%d"),
            "入场日": idx[0].strftime("%Y-%m-%d"),
            "代码": tk,
            "事件涨跌%": round(float(e["ret1"]) * 100, 1),
            "净收益": gross - cost,
            "退出": reason,
        })
    return pd.DataFrame(rows)


def _portfolio(trades: pd.DataFrame, r: Rule) -> dict:
    if trades.empty:
        return {"error": "无交易"}
    t = trades.copy()
    t["入场日"] = pd.to_datetime(t["入场日"])
    # 每日最多 max_per_day 笔，等权
    t = t.sort_values(["入场日"]).groupby("入场日").head(r.max_per_day)
    daily = t.groupby("入场日")["净收益"].mean().sort_index()
    equity = (1 + daily).cumprod()
    tr = t["净收益"].astype(float)
    wins = tr[tr > 0]
    losses = tr[tr <= 0]
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
    return {
        "交易次数": int(len(t)),
        "交易日数": int(len(daily)),
        "累计收益": float(equity.iloc[-1] - 1),
        "年化": float(M.cagr(equity)),
        "最大回撤": float(M.max_drawdown(equity)),
        "胜率": float((tr > 0).mean()),
        "夏普": float(M.sharpe_ratio(daily)),
        "卡尔玛": float(M.calmar_ratio(equity)),
        "盈亏比": float(payoff) if payoff == payoff else 0.0,
        "平均单笔%": float(tr.mean() * 100),
    }


def backtest_rule(
    data: dict[str, pd.DataFrame],
    ev: pd.DataFrame,
    r: Rule,
    *,
    fee_bps: float,
    slip_bps: float,
) -> dict:
    sel = _select(ev, r)
    if sel.empty:
        return {"rule": r.name, "error": "无事件"}
    sel = sel.copy()
    sel["dt"] = pd.to_datetime(sel["日期"])
    trades = _trade_returns(data, sel, r, fee_bps=fee_bps, slip_bps=slip_bps)
    if trades.empty:
        return {"rule": r.name, "error": "无交易"}
    trades["dt"] = pd.to_datetime(trades["入场日"])
    cut = pd.Timestamp(TRAIN_END)
    res = {
        "rule": r.name,
        "事件类型": r.direction,
        "交易方向": r.side,
        "持有": r.hold_days,
        "止损": r.stop,
        "止盈": r.tp,
        "全样本": _portfolio(trades, r),
        "样本内": _portfolio(trades[trades["dt"] <= cut], r),
        "样本外": _portfolio(trades[trades["dt"] > cut], r),
        "_trades": trades,
    }
    return res


def candidate_rules(threshold: float) -> list[Rule]:
    """围绕「暴涨后做空 / 暴跌后反弹做多 / 暴涨延续做多」的多组规则。"""
    return [
        # 暴涨延续做多（强收盘 + 大盘顺风）
        Rule("暴涨延续·强收盘多", "surge", "long", hold_days=3, stop=0.06, tp=0.12,
             min_close_strength=0.8, min_vol_ratio=1.5, require_spy_bull=True),
        Rule("暴涨延续·创新高多", "surge", "long", hold_days=5, stop=0.07, tp=0.15,
             min_close_strength=0.7, require_high20=True, require_spy_bull=True),
        # 暴涨衰竭做空（弱收盘 / 已过热）
        Rule("暴涨衰竭·弱收盘空", "surge", "short", hold_days=3, stop=0.08, tp=0.10,
             max_close_strength=0.5, min_vol_ratio=2.0),
        Rule("暴涨过热·空", "surge", "short", hold_days=3, stop=0.08, tp=0.12,
             min_vol_ratio=3.0, pre20_max_abs=9.0),
        # 暴跌反弹做多
        Rule("暴跌反弹·未贴低多", "drop", "long", hold_days=3, stop=0.07, tp=0.12,
             min_close_strength=0.4, min_vol_ratio=2.0),
        Rule("暴跌反弹·恐慌多", "drop", "long", hold_days=2, stop=0.08, tp=0.10,
             min_vol_ratio=3.0),
        # 暴跌延续做空
        Rule("暴跌延续·破位空", "drop", "short", hold_days=3, stop=0.08, tp=0.12,
             max_close_strength=0.3, require_low20=True),
        # —— 10%阈值发现的高胜率反转子条件 ——
        Rule("下跌趋势·恐慌反弹多", "drop", "long", hold_days=3, stop=0.08, tp=0.15,
             max_pre20=-0.30),
        Rule("暴跌后·向上跳空多", "drop", "long", hold_days=3, stop=0.07, tp=0.12,
             min_gap=0.03),
        Rule("暴涨后·向下跳空反转多", "surge", "long", hold_days=3, stop=0.07, tp=0.12,
             max_gap=-0.03),
        # 暴跌破位空 · 调优版（h2/止损6%/止盈25%，卡尔玛最高）
        Rule("暴跌破位空·调优", "drop", "short", hold_days=2, stop=0.06, tp=0.25,
             max_close_strength=0.3, require_low20=True, max_per_day=2),
    ]


def optimize(
    data: dict[str, pd.DataFrame],
    ev: pd.DataFrame,
    *,
    fee_bps: float,
    slip_bps: float,
) -> pd.DataFrame:
    """围绕两条稳健边缘（暴跌破位空 / 暴跌反弹多 / 暴涨过热空）做参数网格，映射可达前沿。"""
    from itertools import product

    rows: list[dict] = []
    specs = [
        ("drop", "long", dict(max_pre20=-0.30)),                       # 恐慌反弹（深跌后再暴跌）
        ("drop", "long", dict(max_pre20=-0.20)),                       # 较温和下跌后反弹
        ("drop", "long", dict(max_pre20=-0.30, min_vol_ratio=2.0)),    # 恐慌反弹+放量
        ("drop", "short", dict(require_low20=True, max_close_strength=0.35)),
        ("surge", "short", dict(min_vol_ratio=3.0)),
    ]
    holds = [2, 3, 5, 10]
    stops = [0.06, 0.08, 0.10, 0.15]
    tps = [0.10, 0.15, 0.25, 0.40]
    caps = [1, 2, 3]
    for (d, side, extra), hold, stop, tp, cap in product(specs, holds, stops, tps, caps):
        r = Rule(
            name=f"{d}-{side}-h{hold}-s{stop}-t{tp}-c{cap}",
            direction=d, side=side, hold_days=hold, stop=stop, tp=tp, max_per_day=cap,
            **extra,
        )
        res = backtest_rule(data, ev, r, fee_bps=fee_bps, slip_bps=slip_bps)
        res.pop("_trades", None)
        full = res.get("全样本"); oos = res.get("样本外")
        if not isinstance(full, dict) or "error" in full:
            continue
        if full["交易次数"] < 30:
            continue
        h = hit_flags(full)
        rows.append({
            "事件": d, "方向": side, "持有": hold, "止损": stop, "止盈": tp, "每日": cap,
            **{k: extra[k] for k in extra},
            "笔数": full["交易次数"], "胜率": full["胜率"], "年化": full["年化"],
            "回撤": full["最大回撤"], "夏普": full["夏普"], "卡尔玛": full["卡尔玛"],
            "盈亏比": full["盈亏比"],
            "OOS年化": (oos or {}).get("年化"), "OOS胜率": (oos or {}).get("胜率"),
            "OOS回撤": (oos or {}).get("最大回撤"),
            "达标数": sum(h.values()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 排序：达标数 → 卡尔玛（年化/回撤综合）→ 夏普
    return df.sort_values(["达标数", "卡尔玛", "夏普"], ascending=False).reset_index(drop=True)


def hit_flags(p: dict) -> dict:
    if "error" in p:
        return {"年化>=100%": False, "回撤<10%": False, "胜率>=90%": False}
    return {
        "年化>=100%": bool(p["年化"] >= 1.0),
        "回撤<10%": bool(p["最大回撤"] >= -0.10),
        "胜率>=90%": bool(p["胜率"] >= 0.90),
    }


def run(
    *,
    pool: str,
    threshold: float,
    start: str,
    end: str,
    min_price: float,
    min_dvol_m: float,
    fee_bps: float,
    slip_bps: float,
    discover_only: bool,
    do_optimize: bool = False,
) -> dict:
    if pool == "liquid100":
        tickers = LIQUID100
    elif pool == "momentum":
        tickers = GAINER_MOMENTUM
    elif pool == "rgti":
        from quant.volatile_pool import load_pool
        tickers = load_pool()
    elif pool == "surge_drop":
        from quant.surge_drop_pool import load_pool as load_surge_drop_pool
        tickers = load_surge_drop_pool()
    else:
        cache = ROOT / "research" / "gainer_universe_cache.json"
        tickers = json.loads(cache.read_text()) if cache.exists() else GAINER_MOMENTUM
    tickers = sorted(dict.fromkeys([t for t in tickers if t and t != "SPY"]))
    print(f"股票池 {len(tickers)} 只 · {start}~{end} · 阈值±{threshold}% · 拉取行情…")

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    data = yahoo.fetch_batch(tickers, start, end)
    spy = yahoo.fetch_history("SPY", start, end)
    spy_close = spy["Close"]
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_close.index = pd.to_datetime(spy["Close"].index if hasattr(spy["Close"], "index") else spy.index)
    print(f"有效行情 {len(data)} 只，扫描事件…")

    ev = build_event_panel(
        data, spy_close.astype(float),
        threshold_pct=threshold, min_price=min_price, min_dvol_m=min_dvol_m,
    )
    if ev.empty:
        return {"error": "无事件"}
    ev.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    n_surge = int((ev["direction"] == "surge").sum())
    n_drop = int((ev["direction"] == "drop").sum())
    print(f"事件 {len(ev)} 条（暴涨 {n_surge} · 暴跌 {n_drop}）→ {EVENTS_CSV.name}")

    disc = discover(ev)
    doc: dict = {
        "updated": date.today().isoformat(),
        "pool": pool, "threshold_pct": threshold,
        "start": start, "end": end, "train_end": TRAIN_END,
        "min_price": min_price, "min_dvol_m": min_dvol_m,
        "events": {"total": int(len(ev)), "surge": n_surge, "drop": n_drop},
        "discover": disc,
    }

    if not discover_only:
        results = []
        all_trades = []
        for r in candidate_rules(threshold):
            res = backtest_rule(data, ev, r, fee_bps=fee_bps, slip_bps=slip_bps)
            tr = res.pop("_trades", None)
            if tr is not None:
                tr = tr.copy(); tr["rule"] = r.name
                all_trades.append(tr)
            if "全样本" in res:
                res["目标检验_全样本"] = hit_flags(res["全样本"])
                res["目标检验_样本外"] = hit_flags(res["样本外"])
            results.append(res)
        doc["strategies"] = results
        if all_trades:
            pd.concat(all_trades, ignore_index=True).to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")

    if do_optimize:
        print("网格寻优中（映射可达前沿）…")
        opt = optimize(data, ev, fee_bps=fee_bps, slip_bps=slip_bps)
        if not opt.empty:
            opt.to_csv(ROOT / "research" / "extreme15_optimize.csv", index=False, encoding="utf-8-sig")
            doc["optimize_top"] = opt.head(15).to_dict(orient="records")
            doc["optimize_best_winrate"] = opt.sort_values("胜率", ascending=False).head(5).to_dict(orient="records")
            doc["optimize_best_dd"] = opt.sort_values("回撤", ascending=False).head(5).to_dict(orient="records")

    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def _print_discover(disc: dict) -> None:
    print("\n" + "=" * 78)
    print("规律发现 · >阈值异动后的多空前向收益（次日开盘入场）")
    print("=" * 78)
    for label, blk in disc.items():
        print(f"\n【{label}】事件 {blk['事件数']} 条")
        for k in (1, 3, 5, 10):
            lo = blk[f"做多{k}日"]; sh = blk[f"做空{k}日"]
            if not lo.get("n"):
                continue
            print(f"  {k:>2}日: 做多 均{lo['mean%']:+.2f}% 胜{lo['win%']:.0f}%  | "
                  f"做空 均{sh['mean%']:+.2f}% 胜{sh['win%']:.0f}%  (n={lo['n']})")
        if blk.get("条件分档"):
            print("  — 条件分档（3日）—")
            for b in blk["条件分档"]:
                lo = b["做多3日"]; sh = b["做空3日"]
                print(f"    {b['条件']:<14} n={lo['n']:<4} 多 均{lo['mean%']:+.2f}%/胜{lo['win%']:.0f}%  "
                      f"空 均{sh['mean%']:+.2f}%/胜{sh['win%']:.0f}%")


def _print_strategies(doc: dict) -> None:
    res = doc.get("strategies") or []
    if not res:
        return
    print("\n" + "=" * 78)
    print("候选策略回测（次日开盘入场 + 止盈止损，等权日内组合，含成本）")
    print("=" * 78)
    print(f"{'策略':<18}{'方向':<6}{'笔数':>5}{'胜率':>7}{'年化':>9}{'回撤':>8}{'夏普':>7}{'卡尔玛':>7}")
    for r in res:
        if "全样本" not in r or "error" in r["全样本"]:
            print(f"{r['rule']:<18} 无足够交易")
            continue
        p = r["全样本"]
        side = "做多" if r["交易方向"] == "long" else "做空"
        print(f"{r['rule']:<18}{side:<6}{p['交易次数']:>5}{p['胜率']:>7.1%}"
              f"{p['年化']:>+9.1%}{p['最大回撤']:>+8.1%}{p['夏普']:>7.2f}{p['卡尔玛']:>7.2f}")
    print("\n  — 样本外 2024-07+ —")
    for r in res:
        if "样本外" not in r or "error" in r["样本外"]:
            continue
        p = r["样本外"]
        side = "做多" if r["交易方向"] == "long" else "做空"
        print(f"{r['rule']:<18}{side:<6}{p['交易次数']:>5}{p['胜率']:>7.1%}"
              f"{p['年化']:>+9.1%}{p['最大回撤']:>+8.1%}{p['夏普']:>7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="单日异动>阈值 双向规律+多空策略")
    ap.add_argument("--pool", choices=["liquid100", "momentum", "broad", "rgti", "surge_drop"], default="momentum")
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--start", default=(date.today() - timedelta(days=365 * 5 + 120)).isoformat())
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-dvol-m", type=float, default=100.0)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--slip-bps", type=float, default=15.0)
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--optimize", action="store_true", help="网格寻优，映射年化/回撤/胜率可达前沿")
    args = ap.parse_args()

    doc = run(
        pool=args.pool, threshold=args.threshold, start=args.start, end=args.end,
        min_price=args.min_price, min_dvol_m=args.min_dvol_m,
        fee_bps=args.fee_bps, slip_bps=args.slip_bps, discover_only=args.discover_only,
        do_optimize=args.optimize,
    )
    if doc.get("error"):
        print(doc["error"]); sys.exit(1)
    _print_discover(doc["discover"])
    if not args.discover_only:
        _print_strategies(doc)
    if args.optimize and doc.get("optimize_top"):
        print("\n" + "=" * 78)
        print("网格寻优 · 综合最优 Top10（按 达标数→卡尔玛→夏普）")
        print("=" * 78)
        for r in doc["optimize_top"][:10]:
            print(f"  {r['事件']}-{r['方向']:<5} h{r['持有']} 止损{r['止损']} 止盈{r['止盈']} /日{r['每日']} | "
                  f"n={r['笔数']:<4} 胜{r['胜率']:.0%} 年化{r['年化']:+.0%} 回撤{r['回撤']:+.0%} "
                  f"卡{r['卡尔玛']:.2f} | OOS年化{(r.get('OOS年化') or 0):+.0%} 达标{r['达标数']}/3")
        print("\n  最高胜率 Top5：")
        for r in doc["optimize_best_winrate"][:5]:
            print(f"    {r['事件']}-{r['方向']:<5} h{r['持有']} 止损{r['止损']} 止盈{r['止盈']} | "
                  f"胜{r['胜率']:.0%} 年化{r['年化']:+.0%} 回撤{r['回撤']:+.0%}")
        print("\n  最小回撤 Top5：")
        for r in doc["optimize_best_dd"][:5]:
            print(f"    {r['事件']}-{r['方向']:<5} h{r['持有']} 止损{r['止损']} 止盈{r['止盈']} | "
                  f"回撤{r['回撤']:+.0%} 年化{r['年化']:+.0%} 胜{r['胜率']:.0%}")
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
