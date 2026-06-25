#!/usr/bin/env python3
"""核心策略 5 年回测套件 · 汇总输出 research/strategy_suite_5y.json

用法:
    python research/strategy_suite_5y.py
    python research/strategy_suite_5y.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

OUT_JSON = ROOT / "research" / "strategy_suite_5y.json"


def _period(years: float = 5.0) -> dict[str, str]:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(years * 365.25) + 120)).strftime("%Y-%m-%d")
    return {"start": start, "end": end, "years": years}


def _row(
    sid: str,
    name: str,
    category: str,
    *,
    trades: int | None = None,
    win_rate: float | None = None,
    ann_return: float | None = None,
    total_return: float | None = None,
    max_dd: float | None = None,
    sharpe: float | None = None,
    detail: dict | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "id": sid,
        "name": name,
        "category": category,
        "trades": trades,
        "win_rate": win_rate,
        "ann_return": ann_return,
        "total_return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "detail": detail or {},
        "error": error,
    }


def bt_capital_flow(period: dict, *, quick: bool) -> dict:
    from research.flow_pattern_backtest import run_backtest

    doc = run_backtest(years=period["years"], quick=quick, min_dvol_m=30.0)
    patterns = doc.get("patterns") or {}
    core_ids = ("U_S2", "D_S2", "D_OFFERING")
    rows = [patterns[k] for k in core_ids if k in patterns]
    if not rows:
        return _row("capital_flow", "资金流向操盘痕迹", "量价", error="无有效规律样本")
    total_n = sum(r["sample_n"] for r in rows)
    wr = sum(r["win_rate_1d"] * r["sample_n"] for r in rows) / max(total_n, 1)
    mean_ret = sum(r["mean_ret_1d_pct"] * r["sample_n"] for r in rows) / max(total_n, 1)
    return _row(
        "capital_flow", "资金流向操盘痕迹", "量价",
        trades=total_n,
        win_rate=round(wr, 4),
        ann_return=None,
        detail={
            "method": "规律次日方向胜率（非组合净值）",
            "patterns": {k: patterns[k] for k in core_ids if k in patterns},
            "weighted_mean_ret_1d_pct": round(mean_ret, 3),
        },
    )


def bt_flow_strategy(period: dict, *, quick: bool) -> dict:
    from research.flow_strategy_backtest import run_backtest, _load_cfg

    cfg = _load_cfg()
    cfg.setdefault("backtest", {})["years"] = period["years"]
    cfg["backtest"]["quick"] = quick
    result = run_backtest(cfg)
    if "error" in result:
        return _row("flow_strategy", "资金流向组合", "量价", error=result["error"])
    s = result.get("summary_doc") or {}
    return _row(
        "flow_strategy", "资金流向组合", "量价",
        trades=int(s.get("总笔数") or s.get("交易天数") or 0),
        win_rate=float(s.get("笔胜率") or s.get("日胜率") or 0),
        ann_return=float(s.get("年化收益率") or 0),
        total_return=float(s.get("累计收益率") or 0),
        max_dd=float(s.get("最大回撤") or 0),
        sharpe=float(s.get("夏普比率") or 0),
        detail={"params": s.get("params"), "signal_rows": s.get("signal_rows")},
    )


def bt_meme_long(period: dict) -> dict:
    from research.ticker_pattern_backtest import run_ticker

    approved = json.loads((ROOT / "research" / "s8u_approved_tickers.json").read_text())
    tickers = approved.get("tickers") or []
    docs = [
        run_ticker(t, start=period["start"], end=period["end"], alloc_pct=0.10)
        for t in tickers
    ]
    s8u_rows = []
    for d in docs:
        for r in d.get("results") or []:
            if r.get("策略") == "S8U" and r.get("区间") == "全样本":
                s8u_rows.append(r)
    if not s8u_rows:
        return _row("meme_long", "Meme规律·Ultra80", "规律", error="S8U 无全样本结果")
    n = sum(int(r["笔数"]) for r in s8u_rows)
    wr = sum(float(r["胜率"]) * int(r["笔数"]) for r in s8u_rows) / max(n, 1)
    ann = float(np.mean([float(r["年化"]) for r in s8u_rows]))
    dd = float(np.min([float(r["最大回撤"]) for r in s8u_rows]))
    sh = float(np.mean([float(r["夏普"]) for r in s8u_rows if r.get("夏普")]))
    return _row(
        "meme_long", "Meme规律·Ultra80", "规律",
        trades=n, win_rate=round(wr, 4), ann_return=round(ann, 4),
        max_dd=round(dd, 4), sharpe=round(sh, 2),
        detail={"tickers": len(tickers), "per_ticker": s8u_rows},
    )


def bt_gain15(period: dict, *, quick: bool) -> dict:
    from quant.surge_drop_backtest import run_pool_backtest_suite
    from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

    pool = LIQUID100 if quick else GAINER_MOMENTUM
    suite = run_pool_backtest_suite(
        pool, years=int(period["years"]),
        threshold_pct=15.0, min_dvol_m=50.0,
        presets=["surge_chase", "surge_fade", "surge_weak"],
        end=period["end"],
    )
    strategies = {s["preset"]: s for s in suite.get("strategies") or []}
    chase = (strategies.get("surge_chase") or {}).get("全样本") or {}
    fade = (strategies.get("surge_fade") or strategies.get("surge_weak") or {}).get("全样本") or {}
    return _row(
        "gain15", "暴涨80%规则", "动量",
        trades=int(chase.get("交易次数") or 0),
        win_rate=float(chase.get("胜率") or 0),
        ann_return=float(chase.get("年化") or 0),
        max_dd=float(chase.get("最大回撤") or 0),
        sharpe=float(chase.get("夏普") or 0),
        detail={
            "method": "涨幅>15%事件 · surge_chase做多主腿",
            "events_total": suite.get("events", {}).get("total"),
            "surge_chase": chase,
            "surge_fade_short": fade,
        },
    )


def bt_extreme20(period: dict, *, quick: bool) -> dict:
    from research.surge20_optimize import run as opt_run

    doc = opt_run(pool="broad", quick=quick)
    combo = doc.get("combo_long_short") or {}
    if not combo or "error" in combo:
        ref = json.loads((ROOT / "research" / "surge20_refined_playbook.json").read_text())
        combo = (ref.get("combo_L1_S1") or {}).get("backtest") or {}
    return _row(
        "extreme20", "暴涨暴跌20%事件", "动量",
        trades=int(combo.get("交易次数") or 0),
        win_rate=float(combo.get("胜率") or 0),
        ann_return=float(combo.get("年化") or 0),
        max_dd=float(combo.get("回撤") or 0),
        sharpe=float(combo.get("夏普") or 0),
        detail={
            "combo": combo,
            "best_long": (doc.get("best_long") or [])[:2],
            "best_short": (doc.get("best_short") or [])[:2],
            "oos_win_rate": combo.get("OOS胜率"),
            "oos_ann": combo.get("OOS年化"),
        },
    )


def bt_whipsaw_short(period: dict, *, quick: bool) -> dict:
    from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100, fetch_gainer_data_yahoo
    from research.extreme15_pattern import Rule, backtest_rule, build_event_panel

    pool = LIQUID100 if quick else GAINER_MOMENTUM
    end = period["end"]
    start = period["start"]
    data, spy = fetch_gainer_data_yahoo(pool + ["SPY"], start, end)
    spy_close = spy["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy.index)
    events = build_event_panel(data, spy_close, threshold_pct=15.0, min_price=3.0, min_dvol_m=50.0)
    rule = Rule(
        "暴涨乏力做空", "surge", "short", hold_days=5, stop=0.10, tp=0.08,
        max_close_strength=0.75, min_vol_ratio=3.0, max_per_day=3,
    )
    from research.extreme15_pattern import backtest_rule

    direct = backtest_rule(data, events, rule, fee_bps=5, slip_bps=15)
    full = direct.get("全样本") or {}
    oos = direct.get("样本外") or {}
    return _row(
        "whipsaw_short", "做空涨幅榜·卖Call价差", "动量",
        trades=int(full.get("交易次数") or 0),
        win_rate=float(full.get("胜率") or 0),
        ann_return=float(full.get("年化") or 0),
        max_dd=float(full.get("最大回撤") or 0),
        sharpe=float(full.get("夏普") or 0),
        detail={"rule": rule.name, "full_sample": full, "oos_sample": oos},
    )


def bt_daily_pick_modules(period: dict, *, quick: bool) -> dict[str, dict]:
    from research.daily_pick_backtest import run_backtest

    res = run_backtest(
        start=period["start"], end=period["end"],
        quick=quick, profile="high_freq",
    )
    st = res.get("stats") or {}
    mods = st.get("分模块") or {}
    out: dict[str, dict] = {}
    mapping = {
        "收入·卖Call": ("bear_call", "卖Call价差·收租"),
        "5×舰队·CSP": ("fleet_csp", "5×CSP圣杯舰队"),
        "VRP波动率": ("vrp", "VRP波动率溢价"),
        "VRP波动率·CSP": ("vrp", "VRP波动率溢价"),
    }
    for mod_key, (sid, name) in mapping.items():
        m = mods.get(mod_key)
        if not m:
            continue
        out[sid] = _row(
            sid, name, "期权收入",
            trades=int(m.get("笔数") or 0),
            win_rate=float(m.get("胜率") or 0),
            ann_return=float(st.get("年化") or 0) if sid == "bear_call" else None,
            detail={"module": mod_key, "module_stats": m, "portfolio_ann": st.get("年化")},
        )
    out["_portfolio"] = _row(
        "daily_pick_portfolio", "每日选股组合", "聚合",
        trades=int(st.get("笔数") or 0),
        win_rate=float(st.get("胜率") or 0),
        ann_return=float(st.get("年化") or 0),
        total_return=float(st.get("累计收益%") or 0) / 100 if st.get("累计收益%") else None,
        detail={"stats": st, "modules": mods},
    )
    return out


def bt_fleet_csp(period: dict) -> dict:
    from research.screen_fleet_backtest import run

    doc = run(years=period["years"])
    accounts = doc.get("accounts") or []
    csp = [a for a in accounts if a.get("strategy_type") == "csp" or a.get("preset_id") == "csp"]
    if not csp:
        csp = accounts
    anns, wrs, dds, trades = [], [], [], 0
    per_acct = []
    for a in csp:
        s = a.get("stats") or {}
        if a.get("error"):
            continue
        anns.append(float(s.get("ann_return") or 0))
        wrs.append(float(s.get("trade_win_rate") or s.get("period_win_rate") or 0))
        dds.append(float(s.get("max_dd") or 0))
        trades += int(s.get("rebalance_count") or 0)
        per_acct.append({
            "label": a.get("label"), "ticker": a.get("description", "")[:20],
            "ann": s.get("ann_return"), "win_rate": s.get("trade_win_rate"),
            "max_dd": s.get("max_dd"), "trades": s.get("rebalance_count"),
        })
    return _row(
        "fleet_csp", "5×CSP圣杯舰队", "期权收入",
        trades=trades,
        win_rate=round(float(np.mean(wrs)), 4) if wrs else None,
        ann_return=round(float(np.mean(anns)), 4) if anns else None,
        max_dd=round(float(np.min(dds)), 4) if dds else None,
        detail={"accounts": per_acct, "fleet_name": doc.get("name")},
    )


def bt_sndk_iron(period: dict) -> dict:
    from research.sndk_iron_backtest import backtest_iron, summarize, fetch

    df = fetch("WDC", period["start"], period["end"])
    if df.empty:
        return _row("sndk_iron", "SNDK铁鹰收租", "期权收入", error="WDC 无数据")
    trades = backtest_iron(df, call_otm=0.15, put_otm=0.12, width_pct=0.02)
    s = summarize(trades, "WDC铁鹰")
    rors = [t.pnl_pct_margin for t in trades]
    eq = np.cumprod(1 + np.array(rors)) if rors else np.array([1.0])
    total = float(eq[-1] - 1)
    yrs = period["years"]
    ann = (1 + total) ** (1 / yrs) - 1 if yrs > 0 and total > -1 else total
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) > 1 else 0.0
    return _row(
        "sndk_iron", "SNDK铁鹰收租", "期权收入",
        trades=int(s.get("n") or 0),
        win_rate=float(s.get("胜率") or 0),
        ann_return=round(ann, 4),
        total_return=round(total, 4),
        max_dd=round(dd, 4),
        detail={"proxy_ticker": "WDC", "summary": s},
    )


def bt_gainer10() -> dict:
    rules_path = ROOT / "research" / "gainer10_sector_rules.json"
    rules = json.loads(rules_path.read_text()) if rules_path.exists() else {}
    hw = rules.get("portfolio_high_win") or {}
    return _row(
        "gainer10", "Gainer10+分板块高胜率", "动量",
        trades=int(hw.get("n") or 0),
        win_rate=float(hw.get("win%") or 0) / 100,
        ann_return=float(hw.get("CAGR%") or 0) / 100,
        max_dd=float(hw.get("最大回撤%") or 0) / 100,
        sharpe=float(hw.get("夏普") or 0),
        detail={
            "mode": "high_win",
            "filters": "L≥60%+avg≥3 · S≥80%",
            "portfolio_high_win": hw,
            "method": "5年事件+分板块网格优化",
        },
    )


def bt_amp_bear_call(period: dict) -> dict:
    from research.amp_options_backtest import load_universe, _collect_trades, settle, HOLD
    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from math import sqrt

    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    uni = load_universe()
    batch = y.fetch_batch(uni, period["start"], period["end"])
    batch = {t: d for t, d in batch.items() if d is not None and len(d) > 60}
    trades, _, _ = _collect_trades(batch, topn=50, min_amp=10)
    rets = [settle("callsp", S0, ST, sig, k_sd=1.0, w_pct=0.10, iv_mult=1.15) for S0, ST, sig in trades]
    rets = [r for r in rets if np.isfinite(r)]
    if not rets:
        return _row("bear_call_amp", "卖Call价差·amp池", "期权收入", error="无样本")
    arr = np.array(rets)
    win = float((arr > 0).mean())
    eq = np.cumprod(1 + arr / max(len(arr), 1))
    total = float(eq[-1] - 1) if len(eq) else 0
    years = period["years"]
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else total
    return _row(
        "bear_call", "卖Call价差·收租", "期权收入",
        trades=len(rets), win_rate=round(win, 4), ann_return=round(ann, 4),
        total_return=round(total, 4),
        detail={"method": "高成交额Top50+振幅>10% · BS卖Call价差", "avg_trade_ret": float(arr.mean())},
    )


def rank_rows(rows: list[dict]) -> list[dict]:
    scored = []
    for r in rows:
        if r.get("error"):
            scored.append({**r, "score": -999, "rank": 999})
            continue
        sh = float(r.get("sharpe") or 0)
        wr = float(r.get("win_rate") or 0)
        ann = float(r.get("ann_return") or 0)
        dd = abs(float(r.get("max_dd") or 0))
        dd_pen = max(0, 1 - dd)
        score = 0.35 * min(sh / 1.5, 1.0) + 0.30 * wr + 0.20 * dd_pen + 0.15 * min(ann / 0.5, 1.0)
        scored.append({**r, "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(scored, 1):
        r["rank"] = i
    return scored


def run_suite(*, quick: bool = False, years: float = 5.0) -> dict:
    period = _period(years)
    print(f"策略套件 5 年回测 · {period['start']} ~ {period['end']} · quick={quick}\n")

    rows: list[dict] = []
    steps = [
        ("capital_flow", lambda: bt_capital_flow(period, quick=quick)),
        ("flow_strategy", lambda: bt_flow_strategy(period, quick=quick)),
        ("meme_long", lambda: bt_meme_long(period)),
        ("gain15", lambda: bt_gain15(period, quick=quick)),
        ("extreme20", lambda: bt_extreme20(period, quick=quick)),
        ("whipsaw_short", lambda: bt_whipsaw_short(period, quick=quick)),
        ("fleet_csp", lambda: bt_fleet_csp(period)),
        ("sndk_iron", lambda: bt_sndk_iron(period)),
        ("gainer10", lambda: bt_gainer10()),
    ]

    for sid, fn in steps:
        print(f"▶ {sid} …")
        try:
            row = fn()
            rows.append(row)
            if row.get("error"):
                print(f"  ⚠ {row['error']}")
            else:
                print(
                    f"  ✓ 笔数={row.get('trades')} 胜率={row.get('win_rate')} "
                    f"年化={row.get('ann_return')} 夏普={row.get('sharpe')}"
                )
        except Exception as e:  # noqa: BLE001
            rows.append(_row(sid, sid, "?", error=str(e)))
            print(f"  ✗ {e}")

    print("▶ daily_pick 模块 …")
    try:
        dp_mods = bt_daily_pick_modules(period, quick=quick)
        for sid, row in dp_mods.items():
            if sid.startswith("_"):
                continue
            if sid not in {r["id"] for r in rows}:
                rows.append(row)
        if "_portfolio" in dp_mods:
            rows.append(dp_mods["_portfolio"])
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ daily_pick: {e}")

    if not any(r["id"] == "bear_call" and not r.get("error") for r in rows):
        print("▶ bear_call (amp池) …")
        try:
            rows.append(bt_amp_bear_call(period))
        except Exception as e:  # noqa: BLE001
            rows.append(_row("bear_call", "卖Call价差·收租", "期权收入", error=str(e)))

    ranked = rank_rows([r for r in rows if r["id"] != "daily_pick_portfolio"])
    doc = {
        "generated": date.today().isoformat(),
        "period": period,
        "quick_mode": quick,
        "strategies": ranked,
        "portfolio": next((r for r in rows if r["id"] == "daily_pick_portfolio"), None),
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 已写入 {OUT_JSON}")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="LIQUID100 子集加速")
    ap.add_argument("--years", type=float, default=5.0)
    args = ap.parse_args()
    doc = run_suite(quick=args.quick, years=args.years)
    print("\n【排名 Top10】")
    for r in doc["strategies"][:10]:
        if r.get("error"):
            print(f"  {r['rank']:2}. {r['name']} — 失败: {r['error']}")
        else:
            print(
                f"  {r['rank']:2}. {r['name']}  分={r['score']:.3f}  "
                f"胜率{r.get('win_rate', 0):.1%} 年化{r.get('ann_return') or 0:.1%}  "
                f"夏普{r.get('sharpe') or '—'}  笔数{r.get('trades')}"
            )


if __name__ == "__main__":
    main()
