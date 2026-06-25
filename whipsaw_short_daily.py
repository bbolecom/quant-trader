#!/usr/bin/env python3
"""涨幅榜暴涨乏力 → 卖Call信用价差 · 每日扫描（升级版「做空涨幅榜」）。

回测验证（research/gainer_top100_events.csv，2019-2026）：
  裸做空：胜率56% / 均-2.4%（被逼空尾部反噬）
  卖Call价差（定风险）：胜率76% / 均+1.34% · 逐年正收益
  大盘弱(SPY<MA20)时胜率79%、收益更高 → 自动加倍仓位

流程：
  1. 拉当日涨幅榜，筛 涨幅>阈值 + 成交额≥门槛
  2. 只留「乏力」票（收盘强度≤阈值，收在当日中下部）
  3. 看大盘 SPY vs MA20 → 仓位倍数（弱市 ×2）
  4. 对每只用真实期权链构建卖Call信用价差，给行权价/张数

用法：
    python whipsaw_short_daily.py
    python whipsaw_short_daily.py --dry-run
    python whipsaw_short_daily.py -c whipsaw_short_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant.option_chain import build_bear_call_spread, clear_chain_cache
from quant.screener import fetch_gainer_universe_live
from quant.data import fetch_history
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "whipsaw_short_config.json"
HISTORY_FILE = ROOT / "whipsaw_short_history.csv"


# 回测优选模式（research/gainer_top100_events.csv 2019-2026 组合净值）
MODE_PRESETS = {
    # 稳健：量比≥3 + 大盘弱 + Top3 → 83.7%胜率 / +20%年化 / -5%回撤
    "robust": {
        "min_vol_ratio": 3.0, "max_gain_rank": 3, "require_weak_market": True,
        "max_close_strength": 0.80, "risk_per_trade_pct": 0.02,
    },
    # 积极：量比≥3 + Top3，弱市加倍 → 76.8%胜率 / +71%年化 / -20%回撤
    "aggressive": {
        "min_vol_ratio": 3.0, "max_gain_rank": 3, "require_weak_market": False,
        "max_close_strength": 0.80, "risk_per_trade_pct": 0.02,
    },
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "account_size": 10000, "min_gain_pct": 15.0, "min_dvol_m": 50.0,
        "call_otm": 0.06, "width_pct": 0.12, "weak_market_multiplier": 2.0,
        "min_dte": 7, "max_dte": 21, "min_oi": 30, "max_candidates": 15,
        "notify": {"desktop": True},
    }
    # 模式预设补默认（显式配置项优先）
    preset = MODE_PRESETS.get(str(cfg.get("mode", "aggressive")), MODE_PRESETS["aggressive"])
    for k, v in preset.items():
        cfg.setdefault(k, v)
    return cfg


def _bar_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """返回 (收盘强度, 当日涨幅%, 成交额M, 量比) 取最后一根日线。"""
    if df is None or df.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    row = df.iloc[-1]
    hi, lo, cl = float(row["High"]), float(row["Low"]), float(row["Close"])
    vol = float(row.get("Volume", 0))
    cs = 0.5 if hi <= lo else (cl - lo) / (hi - lo)
    gain = (cl / float(df.iloc[-2]["Close"]) - 1.0) * 100 if len(df) >= 2 else float("nan")
    dvol_m = cl * vol / 1e6
    prior_vol = df["Volume"].iloc[:-1].tail(20).mean() if len(df) >= 6 else float("nan")
    vol_ratio = vol / prior_vol if prior_vol and prior_vol > 0 else float("nan")
    return cs, gain, dvol_m, vol_ratio


def market_regime() -> dict:
    """SPY vs MA20 判断牛熊，决定仓位倍数。"""
    try:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()
        spy = fetch_history("SPY", start=start, end=end)
        px = float(spy["Close"].iloc[-1])
        ma20 = float(spy["Close"].tail(20).mean())
        weak = px < ma20
        return {"SPY": round(px, 2), "MA20": round(ma20, 2), "弱市": weak}
    except Exception:  # noqa: BLE001
        return {"SPY": None, "MA20": None, "弱市": False}


def run_scan(cfg: dict) -> dict:
    clear_chain_cache()
    acct = float(cfg.get("account_size", 10000))
    min_gain = float(cfg.get("min_gain_pct", 15.0))
    min_dvol = float(cfg.get("min_dvol_m", 50.0))
    max_cs = float(cfg.get("max_close_strength", 0.80))
    min_vr = float(cfg.get("min_vol_ratio", 3.0))
    max_rank = int(cfg.get("max_gain_rank", 3))
    require_weak = bool(cfg.get("require_weak_market", False))
    mult = float(cfg.get("weak_market_multiplier", 2.0))
    risk_pct = float(cfg.get("risk_per_trade_pct", 0.02))

    reg = market_regime()
    weak = bool(reg.get("弱市"))
    pos_mult = mult if weak else 1.0

    # 稳健模式：大盘强时直接空仓（回测要求大盘弱才开）
    if require_weak and not weak:
        return {
            "date": date.today().isoformat(), "market": reg, "仓位倍数": 0.0,
            "mode": cfg.get("mode"),
            "config": {"min_gain_pct": min_gain, "min_vol_ratio": min_vr, "max_gain_rank": max_rank},
            "scan_stats": {"候选": 0, "可开仓": 0},
            "candidates": [],
            "note": "稳健模式·大盘强(SPY>MA20)→空仓等待",
        }

    snap = fetch_gainer_universe_live(count=int(cfg.get("gainer_count", 250)))
    cands: list[dict] = []
    if not snap.empty:
        snap = snap.copy()
        snap["涨幅%"] = pd.to_numeric(snap["涨幅%"], errors="coerce")
        snap = snap[snap["涨幅%"] >= min_gain].sort_values("涨幅%", ascending=False)
        snap = snap.head(max_rank)  # 只取涨幅榜 Top N（回测最强筛选）
        for rank_i, (_, r) in enumerate(snap.iterrows(), start=1):
            tk = str(r["代码"]).upper()
            try:
                start = (date.today() - timedelta(days=40)).isoformat()
                df = fetch_history(tk, start=start, end=date.today().isoformat())
            except Exception:  # noqa: BLE001
                continue
            cs, gain, dvol_m, vol_ratio = _bar_metrics(df)
            if not (gain == gain) or gain < min_gain:
                continue
            if dvol_m < min_dvol:
                continue
            if vol_ratio == vol_ratio and vol_ratio < min_vr:  # 天量过滤（核心筛选）
                continue
            if cs > max_cs:
                continue
            spot = float(df["Close"].iloc[-1])
            plan, why = build_bear_call_spread(
                tk, spot, acct,
                otm=float(cfg.get("call_otm", 0.06)),
                width_pct=float(cfg.get("width_pct", 0.12)),
                risk_per_trade=risk_pct * pos_mult,
                min_dte=int(cfg.get("min_dte", 7)),
                max_dte=int(cfg.get("max_dte", 21)),
                min_oi=int(cfg.get("min_oi", 30)),
            )
            item = {
                "代码": tk, "现价": round(spot, 2), "涨幅%": round(gain, 1),
                "榜单排名": rank_i,
                "量比": round(vol_ratio, 1) if vol_ratio == vol_ratio else None,
                "收盘强度": round(cs, 2), "成交额M": round(dvol_m, 0),
                "仓位倍数": pos_mult,
            }
            if plan:
                item.update({
                    "信号": "卖Call价差",
                    "到期": plan.expiry, "DTE": plan.dte,
                    "结构": plan.legs_label(),
                    "净权利金/share": round(plan.net_per_share, 2),
                    "收权利金$": round(plan.max_profit, 0),
                    "最大亏$": round(plan.max_loss, 0),
                    "建议张数": int(plan.contracts),
                })
            else:
                item.update({"信号": "无可行价差", "原因": why})
            cands.append(item)
            if len(cands) >= int(cfg.get("max_candidates", 15)):
                break

    actionable = [c for c in cands if c.get("信号") == "卖Call价差" and c.get("建议张数", 0) > 0]
    return {
        "date": date.today().isoformat(),
        "market": reg,
        "mode": cfg.get("mode"),
        "仓位倍数": pos_mult,
        "config": {"min_gain_pct": min_gain, "min_vol_ratio": min_vr,
                   "max_gain_rank": max_rank, "require_weak_market": require_weak},
        "scan_stats": {"候选": len(cands), "可开仓": len(actionable)},
        "candidates": cands,
    }


def format_lines(plan: dict) -> list[str]:
    reg = plan.get("market") or {}
    regime = "🔴 弱市(SPY<MA20)·仓位×2" if reg.get("弱市") else "🟢 强市·仓位×1"
    lines = [
        f"做空涨幅榜·卖Call价差 · {plan['date']} · 模式[{plan.get('mode','-')}]",
        f"  {regime}  SPY {reg.get('SPY')}/MA20 {reg.get('MA20')}",
        f"  候选 {plan['scan_stats']['候选']} · 可开仓 {plan['scan_stats']['可开仓']}",
        "",
    ]
    if plan.get("note"):
        lines.insert(1, f"  ⏸ {plan['note']}")
    act = [c for c in plan["candidates"] if c.get("信号") == "卖Call价差" and c.get("建议张数", 0) > 0]
    if not act:
        lines.append("  今日无「暴涨乏力 + 可行价差」标的（正常空仓）")
    for c in act:
        lines.append(
            f"  📉 {c['代码']} ${c['现价']} 涨{c['涨幅%']}% 量比{c.get('量比')} "
            f"Top{c.get('榜单排名')} → {c['结构']} {c['到期']}"
        )
        lines.append(
            f"     收${c['收权利金$']} 亏${c['最大亏$']} ×{c['建议张数']}张（倍数{c['仓位倍数']}）"
        )
    lines.append("")
    lines.append("法则：暴涨乏力卖Call价差·赚50%平·弱市加倍。回测76%胜率/年年正。")
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    today_json = ROOT / out.get("today_json", "research/whipsaw_short_today.json")
    today_csv = ROOT / out.get("today_csv", "research/whipsaw_short_today.csv")
    today_json.parent.mkdir(parents=True, exist_ok=True)
    today_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    if plan["candidates"]:
        pd.DataFrame(plan["candidates"]).to_csv(today_csv, index=False, encoding="utf-8-sig")
    hist_row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "日期": plan["date"], "弱市": plan["market"].get("弱市"),
        "候选": plan["scan_stats"]["候选"], "可开仓": plan["scan_stats"]["可开仓"],
    }
    if HISTORY_FILE.exists():
        pd.concat([pd.read_csv(HISTORY_FILE, encoding="utf-8-sig"), pd.DataFrame([hist_row])]).to_csv(
            HISTORY_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([hist_row]).to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="做空涨幅榜·卖Call价差每日扫描")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plan = run_scan(cfg)
    save_outputs(plan, cfg)
    text = "\n".join(format_lines(plan))
    print(text)

    if not args.dry_run and (cfg.get("notify") or {}).get("desktop", True):
        st = plan["scan_stats"]
        desktop_notify(f"做空涨幅榜 {plan['date']}", f"可开仓 {st['可开仓']} · 候选 {st['候选']}")


if __name__ == "__main__":
    main()
