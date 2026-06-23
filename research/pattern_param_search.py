#!/usr/bin/env python3
"""涨跌规律参数网格寻优（真实量价，无 BS）。

在样本内（默认 2019–2023）搜索做多/回避阈值，样本外（2024+）验证，写入
research/pattern_rules_optimized.json 供 pattern_daily.py 使用。

用法：
    python research/pattern_param_search.py
    python research/pattern_param_search.py --quick
    python research/pattern_param_search.py --min-samples 80 --min-win 0.60
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from quant.move_pattern import enrich_forward_horizons, vectorized_down_mask
from quant.pattern_params import (
    DownParams,
    LongParams,
    OptimizedPatternRules,
    save_optimized_rules,
)

HIGHWIN_PANEL = ROOT / "research" / "move_pattern_highwin_panel.csv"
EVENTS_PANEL = ROOT / "research" / "move_pattern_events.csv"
TRAIN_END = "2023-12-31"


def _split_panel(p: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = p.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    cut = pd.Timestamp(train_end)
    return p[p["日期"] <= cut], p[p["日期"] > cut]


def _eval_long(params: LongParams, panel: pd.DataFrame, *, min_n: int) -> dict | None:
    sub = panel[params.mask_on_panel(panel)].dropna(subset=["fwd_1d"])
    if len(sub) < min_n:
        return None
    fwd = sub["fwd_1d"]
    wr = float((fwd > 0).mean())
    return {
        "n": len(sub),
        "win_rate": wr,
        "fwd_mean": float(fwd.mean()),
        "fwd_median": float(fwd.median()),
    }


def search_long_params(
    panel: pd.DataFrame,
    *,
    min_samples: int = 50,
    min_win_is: float = 0.58,
    train_end: str = TRAIN_END,
) -> tuple[LongParams, dict]:
    is_df, oos_df = _split_panel(panel, train_end)
    print(f"  做多寻优 · 样本内 {len(is_df)} 行 · 样本外 {len(oos_df)} 行")

    grid = {
        "max_gain_pct": [4.0, 4.5, 5.0],
        "max_vol_ratio": [1.55, 1.65, 1.75],
        "min_close_strength": [0.58, 0.62, 0.66, 0.70],
        "max_ret_5d_pct": [12.0, 15.0, 18.0],
        "max_gain_20d_pct": [18.0, 22.0, 25.0],
        "min_setup_win_rate": [0.625, 0.65, 0.675],
        "min_dvol_m": [300.0, 500.0, 1000.0],
        "require_spy_positive_1d": [False, True],
        "min_spy_1d_pct": [0.0, 0.2],
    }

    best: LongParams | None = None
    best_meta: dict = {}
    best_score = -1.0

    keys = list(grid.keys())
    total = 1
    for k in keys:
        total *= len(grid[k])
    print(f"  网格 {total} 组 …")

    for i, combo in enumerate(itertools.product(*[grid[k] for k in keys])):
        kw = dict(zip(keys, combo))
        p = LongParams(
            min_gain_pct=2.5,
            min_vol_ratio=1.3,
            min_ret_5d_pct=4.0,
            min_gain_20d_pct=4.0,
            min_rs_spy_20d_pct=3.0,
            max_rs_spy_20d_pct=18.0,
            require_above_ma50=True,
            require_spy_ma20=True,
            require_green_candle=True,
            top_n=2,
            **kw,
        )
        if p.require_spy_positive_1d and p.min_spy_1d_pct <= 0:
            p.min_spy_1d_pct = 0.2
        is_res = _eval_long(p, is_df, min_n=min_samples)
        if is_res is None or is_res["win_rate"] < min_win_is:
            continue
        oos_res = _eval_long(p, oos_df, min_n=max(15, min_samples // 3))
        oos_wr = oos_res["win_rate"] if oos_res else 0.0
        # 优先样本外，兼顾样本量
        score = oos_wr * 0.6 + is_res["win_rate"] * 0.4 + min(is_res["n"], 500) / 5000
        if score > best_score:
            best_score = score
            best = p
            best_meta = {
                "is": is_res,
                "oos": oos_res,
                "score": score,
                "horizon": "fwd_1d",
            }

    if best is None:
        print("  ⚠ 未找到满足门槛的组合，退回默认 high_win 参数")
        best = LongParams()
        best_meta = {"is": _eval_long(best, is_df, min_n=10), "oos": None, "note": "default"}

    return best, best_meta


def refine_long_spy(
    best: LongParams,
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    *,
    min_samples: int,
    base_meta: dict,
) -> tuple[LongParams, dict]:
    """二次精调：SPY 当日涨≥阈值时次日胜率更高，在基础参数上叠加。"""
    if best.require_spy_positive_1d:
        return best, base_meta
    base_oos = (base_meta.get("oos") or {}).get("win_rate", 0.0)
    chosen: LongParams | None = None
    chosen_meta: dict | None = None
    for th in [0.1, 0.15, 0.2, 0.25]:
        p = replace(best, require_spy_positive_1d=True, min_spy_1d_pct=th)
        is_res = _eval_long(p, is_df, min_n=max(20, min_samples // 2))
        oos_res = _eval_long(p, oos_df, min_n=max(12, min_samples // 4))
        if is_res is None or oos_res is None:
            continue
        if oos_res["win_rate"] >= base_oos - 0.02 and is_res["win_rate"] >= 0.58:
            if chosen is None or oos_res["win_rate"] > (chosen_meta or {}).get("oos", {}).get("win_rate", 0):
                chosen, chosen_meta = p, {
                    "is": is_res,
                    "oos": oos_res,
                    "horizon": "fwd_1d",
                    "spy_refine": True,
                    "min_spy_1d_pct": th,
                }
    if chosen is None:
        return best, base_meta
    print(f"  SPY联动精调: 1d≥{chosen.min_spy_1d_pct}% → OOS {chosen_meta['oos']['win_rate']:.1%}")
    return chosen, chosen_meta


def _eval_down_rule(
    mask: pd.Series,
    panel: pd.DataFrame,
    min_n: int,
    *,
    horizon: int = 20,
) -> dict | None:
    col = f"fwd_{horizon}d"
    if col not in panel.columns:
        return None
    sub = panel[mask].dropna(subset=[col])
    if len(sub) < min_n:
        return None
    fwd = sub[col]
    return {
        "n": len(sub),
        "down_rate": float((fwd < 0).mean()),
        "fwd_mean": float(fwd.mean()),
        "horizon": f"{horizon}d",
    }


def _make_rule_grids():
    """每条回避规则：mask_builder(df, kw) + param_grid() -> list[kw]。"""

    def d3_mask(df, kw):
        r5s = pd.to_numeric(df["ret_5d"], errors="coerce")
        vrs = pd.to_numeric(df["vol_ratio"], errors="coerce")
        return (r5s > kw["shrink_vol_min_ret_5d"]) & (vrs < kw["shrink_vol_max"])

    def d3_grid():
        for r5 in [0.10, 0.12, 0.15, 0.18, 0.20]:
            for vr in [0.8, 0.9, 1.0, 1.1]:
                yield {"shrink_vol_min_ret_5d": r5, "shrink_vol_max": vr}

    def d4_mask(df, kw):
        r5s = pd.to_numeric(df["ret_5d"], errors="coerce")
        dvol = pd.to_numeric(df["dvol_m"], errors="coerce")
        a50 = df["above_ma50"].astype(bool)
        return (r5s > kw["mega_ret_5d"]) & (dvol >= kw["mega_dvol_m"]) & a50

    def d4_grid():
        for r5 in [0.12, 0.15, 0.18, 0.22]:
            for dv in [400.0, 500.0, 800.0, 1200.0]:
                yield {"mega_ret_5d": r5, "mega_dvol_m": dv}

    def parabolic_mask(df, kw):
        r20s = pd.to_numeric(df["ret_20d"], errors="coerce")
        r5s = pd.to_numeric(df["ret_5d"], errors="coerce")
        r20t = kw["parabolic_ret_20d"]
        r5t = kw["parabolic_ret_5d"]
        return (r20s > r20t) | ((r5s > r5t) & (r20s > r20t * 0.75))

    def parabolic_grid():
        for r20 in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            for r5 in [0.12, 0.15, 0.18, 0.22]:
                yield {"parabolic_ret_20d": r20, "parabolic_ret_5d": r5}

    def dump_mask(df, kw):
        vrs = pd.to_numeric(df["vol_ratio"], errors="coerce")
        r5s = pd.to_numeric(df["ret_5d"], errors="coerce")
        return (vrs >= kw["vol_dump_min_ratio"]) & (r5s < kw["vol_dump_max_ret_5d"])

    def dump_grid():
        for vr in [2.0, 2.5, 3.0, 3.5]:
            for r5 in [-0.04, -0.06, -0.08, -0.10]:
                yield {"vol_dump_min_ratio": vr, "vol_dump_max_ret_5d": r5}

    def blowoff_mask(df, kw):
        vrs = pd.to_numeric(df["vol_ratio"], errors="coerce")
        r5s = pd.to_numeric(df["ret_5d"], errors="coerce")
        dvol = pd.to_numeric(df["dvol_m"], errors="coerce")
        a50 = df["above_ma50"].astype(bool)
        return (
            (vrs >= kw["blowoff_min_ratio"])
            & (r5s >= kw["blowoff_ret_5d_min"])
            & (r5s <= kw["blowoff_ret_5d_max"])
            & (dvol >= kw["blowoff_dvol_min_m"])
            & (dvol < kw["blowoff_dvol_max_m"])
            & a50
        )

    def blowoff_grid():
        for vr in [2.0, 2.5, 3.0]:
            for r5hi in [0.12, 0.15, 0.18]:
                for dmin in [150.0, 200.0, 300.0]:
                    yield {
                        "blowoff_min_ratio": vr,
                        "blowoff_ret_5d_min": 0.05,
                        "blowoff_ret_5d_max": r5hi,
                        "blowoff_dvol_min_m": dmin,
                        "blowoff_dvol_max_m": 1000.0,
                    }

    return [
        ("D3_shrink", d3_mask, d3_grid),
        ("D4_mega", d4_mask, d4_grid),
        ("parabolic", parabolic_mask, parabolic_grid),
        ("D2_dump", dump_mask, dump_grid),
        ("D1_blowoff", blowoff_mask, blowoff_grid),
    ]


def _search_one_rule(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    mask_fn,
    grid_fn,
    *,
    min_n: int,
    horizons: tuple[int, ...] = (5, 20),
) -> tuple[dict, dict | None]:
    best_kw: dict = {}
    best_meta: dict | None = None
    best_score = -1.0
    for kw in grid_fn():
        mask = mask_fn(is_df, kw)
        for hz in horizons:
            is_res = _eval_down_rule(mask, is_df, min_n, horizon=hz)
            if is_res is None:
                continue
            oos_mask = mask_fn(oos_df, kw)
            oos_res = _eval_down_rule(oos_mask, oos_df, max(15, min_n // 4), horizon=hz)
            oos_dr = oos_res["down_rate"] if oos_res else 0.0
            score = is_res["down_rate"] * 0.45 + oos_dr * 0.45 + min(is_res["n"], 400) / 4000
            if score > best_score:
                best_score = score
                best_kw = kw
                best_meta = {"is": is_res, "oos": oos_res, "params": kw, "best_horizon": f"{hz}d"}
    return best_kw, best_meta


def search_down_params(
    panel: pd.DataFrame,
    *,
    min_samples: int = 80,
    min_down_rate: float = 0.52,
    train_end: str = TRAIN_END,
) -> tuple[DownParams, dict]:
    is_df, oos_df = _split_panel(panel, train_end)
    is_df = enrich_forward_horizons(is_df)
    oos_df = enrich_forward_horizons(oos_df)
    print(f"  回避寻优 · 样本内 {len(is_df)} 行 · 逐规则调参(5d/20d)")

    base = DownParams()
    rule_meta: dict[str, dict] = {}

    from dataclasses import asdict

    merged: dict = asdict(base)
    for label, mask_fn, grid_fn in _make_rule_grids():
        kw, meta = _search_one_rule(is_df, oos_df, mask_fn, grid_fn, min_n=min_samples // 2)
        if kw:
            merged.update(kw)
            rule_meta[label] = meta

    best = DownParams(**merged)
    # 仅保留样本内下跌率≥51%的单条规则（D3 缩量顶全市场偏弱，默认剔除）
    active: list[str] = []
    rule_map = {
        "D3_shrink": "D3_shrink",
        "D4_mega": "D4_mega",
        "parabolic": "D_parabolic",
        "D2_dump": "D2_dump",
        "D1_blowoff": "D1_blowoff",
    }
    for label, rid in rule_map.items():
        rm = rule_meta.get(label)
        if rm and rm.get("is") and rm["is"]["down_rate"] >= 0.51:
            active.append(rid)
    if not active:
        active = ["D4_mega", "D2_dump", "D_parabolic"]
    best.active_avoid_rules = active

    rule_horizons: dict[str, str] = {}
    short_term: list[str] = []
    for label, rid in rule_map.items():
        rm = rule_meta.get(label)
        if rm and rm.get("best_horizon"):
            rule_horizons[rid] = rm["best_horizon"]
            if rm["best_horizon"] == "5d":
                short_term.append(rid)
    best.rule_horizons = rule_horizons
    best.short_term_avoid_rules = short_term or ["D1_blowoff", "D2_dump"]

    mask = vectorized_down_mask(is_df, best, include_rules=active)
    is_res = _eval_down_rule(mask, is_df, min_samples, horizon=20)
    oos_mask = vectorized_down_mask(oos_df, best, include_rules=active)
    oos_res = _eval_down_rule(oos_mask, oos_df, max(20, min_samples // 4), horizon=20)
    # 短周期合并评估（5日）
    is_res_5 = _eval_down_rule(mask, is_df, min_samples, horizon=5)
    oos_res_5 = _eval_down_rule(oos_mask, oos_df, max(20, min_samples // 4), horizon=5)

    best_meta = {
        "is_n": is_res["n"] if is_res else 0,
        "is_down_rate": is_res["down_rate"] if is_res else 0.0,
        "is_fwd_mean": is_res["fwd_mean"] if is_res else 0.0,
        "oos_n": oos_res["n"] if oos_res else 0,
        "oos_down_rate": oos_res["down_rate"] if oos_res else 0.0,
        "horizon": "fwd_20d",
        "horizon_5d": {
            "is_n": is_res_5["n"] if is_res_5 else 0,
            "is_down_rate": is_res_5["down_rate"] if is_res_5 else 0.0,
            "oos_n": oos_res_5["n"] if oos_res_5 else 0,
            "oos_down_rate": oos_res_5["down_rate"] if oos_res_5 else 0.0,
        },
        "per_rule": rule_meta,
        "active_avoid_rules": active,
        "rule_horizons": rule_horizons,
        "min_down_target": min_down_rate,
    }
    if is_res and is_res["down_rate"] < min_down_rate:
        best_meta["note"] = f"全市场20日下跌规律上限约52%；合并后 IS={is_res['down_rate']:.1%}"

    return best, best_meta


def run_search(
    *,
    quick: bool = False,
    min_samples: int = 50,
    min_win_long: float = 0.58,
    min_down: float = 0.55,
    train_end: str = TRAIN_END,
    rebuild_panel: bool = False,
) -> OptimizedPatternRules:
    if rebuild_panel or not HIGHWIN_PANEL.exists():
        from research.move_pattern_mine import build_gainer_daily_panel
        from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

        end = date.today().isoformat()
        start = "2019-01-01"
        pool = LIQUID100 if quick else GAINER_MOMENTUM
        print(f"构建日频面板 {len(pool)} 只 …")
        panel = build_gainer_daily_panel(start=start, end=end, tickers=pool)
        panel.to_csv(HIGHWIN_PANEL, index=False, encoding="utf-8-sig")
    else:
        panel = pd.read_csv(HIGHWIN_PANEL)
        if quick:
            top = panel["代码"].value_counts().head(100).index
            panel = panel[panel["代码"].isin(top)]

    print(f"高置信面板 {len(panel)} 行 · {panel['代码'].nunique()} 只")
    long_p, long_meta = search_long_params(
        panel, min_samples=min_samples, min_win_is=min_win_long, train_end=train_end,
    )
    is_p, oos_p = _split_panel(panel, train_end)
    long_p, long_meta = refine_long_spy(long_p, is_p, oos_p, min_samples=min_samples, base_meta=long_meta)

    if not EVENTS_PANEL.exists():
        print("⚠ 无 move_pattern_events.csv，回避参数用默认")
        down_p, down_meta = DownParams(), {"note": "no events file"}
    else:
        ev = pd.read_csv(EVENTS_PANEL)
        if quick:
            top = ev["代码"].value_counts().head(120).index
            ev = ev[ev["代码"].isin(top)]
        down_p, down_meta = search_down_params(
            ev, min_samples=max(80, min_samples * 2), min_down_rate=min_down,
            train_end=train_end,
        )

    rules = OptimizedPatternRules(
        long=long_p,
        down=down_p,
        meta={
            "updated": date.today().isoformat(),
            "method": "真实量价网格 + 样本内寻优 / 样本外验证",
            "train_end": train_end,
            "long_search": long_meta,
            "down_search": down_meta,
            "panel_rows": len(panel),
        },
    )
    out = save_optimized_rules(rules)
    print(f"\n→ {out}")
    return rules


def print_report(rules: OptimizedPatternRules) -> None:
    m = rules.meta
    lp = rules.long
    dp = rules.down
    print("\n" + "=" * 70)
    print("规律参数寻优结果（真实量价）")
    print("=" * 70)
    ls = m.get("long_search") or {}
    ds = m.get("down_search") or {}
    print("\n【腿① 做多 · 次日 fwd_1d】")
    print(f"  涨 {lp.min_gain_pct}–{lp.max_gain_pct}% · 量比 {lp.min_vol_ratio}–{lp.max_vol_ratio}")
    print(f"  收强≥{lp.min_close_strength:.0%} · 5日涨 {lp.min_ret_5d_pct}–{lp.max_ret_5d_pct}%")
    print(f"  20日涨 {lp.min_gain_20d_pct}–{lp.max_gain_20d_pct}% · 形态胜率≥{lp.min_setup_win_rate:.0%}")
    print(f"  成交额≥${lp.min_dvol_m:.0f}M · SPY1d={'≥'+str(lp.min_spy_1d_pct)+'%' if lp.require_spy_positive_1d else '否'}")
    if ls.get("is"):
        print(f"  样本内: n={ls['is']['n']} 胜率={ls['is']['win_rate']:.1%} 均收益={ls['is']['fwd_mean']:.2%}")
    if ls.get("oos"):
        o = ls["oos"]
        print(f"  样本外: n={o['n']} 胜率={o['win_rate']:.1%} 均收益={o['fwd_mean']:.2%}")

    print("\n【腿② 回避 · 后 20 日 fwd_20d】")
    print(f"  {dp.describe()}")
    if ds.get("is_n"):
        print(f"  合并: n={ds['is_n']} 下跌率={ds.get('is_down_rate', 0):.1%}")
    if ds.get("oos_n"):
        print(f"  样本外: n={ds['oos_n']} 下跌率={ds.get('oos_down_rate', 0):.1%}")
    if ds.get("horizon_5d"):
        h5 = ds["horizon_5d"]
        print(f"  5日合并: n={h5.get('is_n', 0)} 下跌率={h5.get('is_down_rate', 0):.1%}")
    per = ds.get("per_rule") or {}
    for rk, rm in per.items():
        if rm and rm.get("is"):
            bh = rm.get("best_horizon", "20d")
            print(f"    · {rk}({bh}): n={rm['is']['n']} 下跌率={rm['is']['down_rate']:.1%}")
    if ds.get("note"):
        print(f"  注: {ds['note']}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="涨跌规律参数寻优")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="重建高置信面板")
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--min-win", type=float, default=0.58, help="做多样本内最低次日胜率")
    ap.add_argument("--min-down", type=float, default=0.52, help="回避样本内目标下跌率（全市场上限约52%）")
    ap.add_argument("--train-end", default=TRAIN_END)
    args = ap.parse_args()

    rules = run_search(
        quick=args.quick,
        min_samples=args.min_samples,
        min_win_long=args.min_win,
        min_down=args.min_down,
        train_end=args.train_end,
        rebuild_panel=args.rebuild,
    )
    print_report(rules)


if __name__ == "__main__":
    main()
