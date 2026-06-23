#!/usr/bin/env python3
"""5 日路径规律参数网格寻优（真实 OHLCV + 换手率）。

在 move_pattern_5d_events.csv 上搜索流动性 / 路径阈值 / 扩展过滤，
最大化 IS/OOS 命中率，写入 research/move_pattern_5d_optimized.json。

用法：
    python research/move_pattern_5d_param_search.py
    python research/move_pattern_5d_param_search.py --quick
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.pattern_5d_params import (
    ExtendedDownFilters,
    ExtendedUpFilters,
    LiquidityFilter,
    Optimized5dRules,
    PathThreshold,
    down_mask,
    save_optimized_5d,
    up_mask,
)

EVENTS_CSV = ROOT / "research" / "move_pattern_5d_events.csv"
TRAIN_END = "2023-12-31"


def _split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    cut = pd.Timestamp(TRAIN_END)
    return p[p["日期"] <= cut], p[p["日期"] > cut]


def _hit_rate(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    direction: str,
    up_th: float,
    dn_th: float,
    horizon: int = 5,
) -> dict | None:
    pu, pd_col = f"path_up_{horizon}d", f"path_down_{horizon}d"
    sub = df[mask].dropna(subset=[pu if direction == "up" else pd_col])
    if sub.empty:
        return None
    if direction == "up":
        hit = (sub[pu] >= up_th).mean()
        col = pu
    else:
        hit = (sub[pd_col] <= -dn_th).mean()
        col = pd_col
    return {
        "n": len(sub),
        "hit_rate": float(hit),
        "path_mean": float(sub[col].mean()),
    }


def search_up_params(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    *,
    up_th: float,
    min_samples: int,
    min_hit_is: float,
) -> tuple[ExtendedUpFilters, dict]:
    best: ExtendedUpFilters | None = None
    best_meta: dict = {}
    best_score = -1.0

    grid = {
        "min_vol_ratio": [1.5, 2.0, 2.5, 3.0],
        "min_ret_5d": [0.10, 0.15, 0.20],
        "min_close_strength": [0.55, 0.60, 0.65],
        "min_turnover_pct": [0.3, 0.5, 1.0],
        "min_dvol_m": [100.0, 200.0, 500.0],
        "require_above_ma20": [True, False],
        "min_up_vol_share": [0.0, 0.55],
    }
    keys = list(grid.keys())
    total = 1
    for k in keys:
        total *= len(grid[k])
    print(f"  做多网格 {total} 组 · 路径涨≥{up_th:.1%} …")

    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, combo))
        p = ExtendedUpFilters(require_above_ma50=True, **kw)
        m_is = up_mask(is_df, p)
        is_res = _hit_rate(is_df, m_is, direction="up", up_th=up_th, dn_th=up_th)
        if is_res is None or is_res["n"] < min_samples or is_res["hit_rate"] < min_hit_is:
            continue
        m_oos = up_mask(oos_df, p)
        oos_res = _hit_rate(oos_df, m_oos, direction="up", up_th=up_th, dn_th=up_th)
        oos_hit = oos_res["hit_rate"] if oos_res and oos_res["n"] >= max(15, min_samples // 4) else 0.0
        score = is_res["hit_rate"] * 0.4 + oos_hit * 0.5 + min(is_res["n"], 300) / 3000
        if score > best_score:
            best_score = score
            best = p
            best_meta = {"is": is_res, "oos": oos_res, "score": score, "params": kw}

    if best is None:
        best = ExtendedUpFilters()
        m = up_mask(is_df, best)
        best_meta = {"is": _hit_rate(is_df, m, direction="up", up_th=up_th, dn_th=up_th), "note": "default"}
    return best, best_meta


def search_down_params(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    *,
    dn_th: float,
    min_samples: int,
    min_hit_is: float,
) -> tuple[ExtendedDownFilters, dict]:
    best: ExtendedDownFilters | None = None
    best_meta: dict = {}
    best_score = -1.0

    grid = {
        "min_vol_ratio": [2.0, 2.5, 3.0],
        "max_ret_5d": [-0.04, -0.06, -0.08, -0.10],
        "max_close_strength": [0.35, 0.40, 0.45],
        "min_turnover_pct": [0.3, 0.5, 1.0],
        "min_dvol_m": [50.0, 100.0, 200.0],
        "require_below_ma50": [True, False],
    }
    keys = list(grid.keys())
    print(f"  做空网格 {len(list(itertools.product(*[grid[k] for k in keys])))} 组 · 路径跌≥{dn_th:.1%} …")

    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, combo))
        p = ExtendedDownFilters(**kw)
        m_is = down_mask(is_df, p)
        is_res = _hit_rate(is_df, m_is, direction="down", up_th=dn_th, dn_th=dn_th)
        if is_res is None or is_res["n"] < min_samples or is_res["hit_rate"] < min_hit_is:
            continue
        m_oos = down_mask(oos_df, p)
        oos_res = _hit_rate(oos_df, m_oos, direction="down", up_th=dn_th, dn_th=dn_th)
        oos_hit = oos_res["hit_rate"] if oos_res and oos_res["n"] >= max(15, min_samples // 4) else 0.0
        score = is_res["hit_rate"] * 0.4 + oos_hit * 0.5 + min(is_res["n"], 300) / 3000
        if score > best_score:
            best_score = score
            best = p
            best_meta = {"is": is_res, "oos": oos_res, "score": score, "params": kw}

    if best is None:
        best = ExtendedDownFilters()
        m = down_mask(is_df, best)
        best_meta = {"is": _hit_rate(is_df, m, direction="down", up_th=dn_th, dn_th=dn_th), "note": "default"}
    return best, best_meta


def search_liquidity_and_threshold(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    up_p: ExtendedUpFilters,
    down_p: ExtendedDownFilters,
    *,
    lock_path_pct: float | None = None,
) -> tuple[LiquidityFilter, PathThreshold, dict]:
    """在已选扩展过滤上，微调全局流动性与路径阈值（2%/2.5%/3%）。"""
    best_liq = LiquidityFilter()
    best_th = PathThreshold(up_pct=3.0, down_pct=3.0)
    best_meta: dict = {}
    best_score = -1.0

    pct_list = [lock_path_pct] if lock_path_pct else [2.0, 2.5, 3.0]
    for up_pct in pct_list:
        for dn_pct in pct_list:
            up_th, dn_th = up_pct / 100, dn_pct / 100
            for min_vr in [1.0, 1.3, 1.5]:
                for min_to in [0.3, 0.5]:
                    liq = LiquidityFilter(min_vol_ratio=min_vr, min_turnover_pct=min_to)
                    turn = pd.to_numeric(is_df["换手率%"], errors="coerce")
                    vr = pd.to_numeric(is_df["vol_ratio"], errors="coerce")
                    liq_m = (vr >= min_vr) & (turn >= min_to)
                    m_up = up_mask(is_df, up_p) & liq_m
                    m_dn = down_mask(is_df, down_p) & liq_m
                    iu = _hit_rate(is_df, m_up, direction="up", up_th=up_th, dn_th=dn_th)
                    id_ = _hit_rate(is_df, m_dn, direction="down", up_th=up_th, dn_th=dn_th)
                    if iu is None or id_ is None:
                        continue
                    ou = _hit_rate(oos_df, up_mask(oos_df, up_p) & (
                        pd.to_numeric(oos_df["vol_ratio"], errors="coerce") >= min_vr
                    ) & (pd.to_numeric(oos_df["换手率%"], errors="coerce") >= min_to),
                        direction="up", up_th=up_th, dn_th=dn_th)
                    od = _hit_rate(oos_df, down_mask(oos_df, down_p) & (
                        pd.to_numeric(oos_df["vol_ratio"], errors="coerce") >= min_vr
                    ) & (pd.to_numeric(oos_df["换手率%"], errors="coerce") >= min_to),
                        direction="down", up_th=up_th, dn_th=dn_th)
                    oos_u = ou["hit_rate"] if ou else 0
                    oos_d = od["hit_rate"] if od else 0
                    score = (iu["hit_rate"] + id_["hit_rate"]) * 0.2 + (oos_u + oos_d) * 0.25
                    if score > best_score:
                        best_score = score
                        best_liq = liq
                        best_th = PathThreshold(up_pct=up_pct, down_pct=dn_pct)
                        best_meta = {
                            "up_is": iu, "down_is": id_,
                            "up_oos": ou, "down_oos": od,
                            "score": score,
                        }
    return best_liq, best_th, best_meta


def run_search(
    *,
    min_samples: int = 60,
    min_hit_is: float = 0.68,
    quick: bool = False,
    lock_path_pct: float | None = None,
) -> Optimized5dRules:
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"请先运行 move_pattern_5d_mine.py 生成 {EVENTS_CSV}")

    panel = pd.read_csv(EVENTS_CSV)
    if quick:
        top = panel["代码"].value_counts().head(150).index
        panel = panel[panel["代码"].isin(top)]
    is_df, oos_df = _split(panel)
    print(f"5日寻优 · IS {len(is_df)} · OOS {len(oos_df)} 行")

    up_p, up_meta = search_up_params(
        is_df, oos_df, up_th=0.03, min_samples=min_samples, min_hit_is=min_hit_is,
    )
    down_p, down_meta = search_down_params(
        is_df, oos_df, dn_th=0.03, min_samples=min_samples, min_hit_is=min_hit_is - 0.03,
    )
    liq, th, liq_meta = search_liquidity_and_threshold(
        is_df, oos_df, up_p, down_p, lock_path_pct=lock_path_pct,
    )

    # 用最优阈值重新评估
    up_th, dn_th = th.up_pct / 100, th.down_pct / 100
    up_final = _hit_rate(is_df, up_mask(is_df, up_p), direction="up", up_th=up_th, dn_th=dn_th)
    up_oos = _hit_rate(oos_df, up_mask(oos_df, up_p), direction="up", up_th=up_th, dn_th=dn_th)
    dn_final = _hit_rate(is_df, down_mask(is_df, down_p), direction="down", up_th=up_th, dn_th=dn_th)
    dn_oos = _hit_rate(oos_df, down_mask(oos_df, down_p), direction="down", up_th=up_th, dn_th=dn_th)

    opt = Optimized5dRules(
        liquidity=liq,
        threshold=th,
        up=up_p,
        down=down_p,
        min_samples=min_samples,
        min_hit_is=min_hit_is,
        meta={
            "updated": date.today().isoformat(),
            "method": "真实OHLCV路径+换手率网格 · IS/OOS",
            "train_end": TRAIN_END,
            "up_search": up_meta,
            "down_search": down_meta,
            "liq_threshold_search": liq_meta,
            "final_up": {"is": up_final, "oos": up_oos},
            "final_down": {"is": dn_final, "oos": dn_oos},
            "panel_rows": len(panel),
        },
    )
    out = save_optimized_5d(opt)
    print(f"\n→ {out}")
    return opt


def print_report(opt: Optimized5dRules) -> None:
    m = opt.meta
    th = opt.threshold
    u, d = opt.up, opt.down
    print("\n" + "=" * 70)
    print("5 日路径参数寻优结果")
    print("=" * 70)
    print(f"路径阈值: 涨≥{th.up_pct}% / 跌≥{th.down_pct}%")
    print(
        f"流动性: 成交额≥${opt.liquidity.min_dvol_m}M 量比≥{opt.liquidity.min_vol_ratio} "
        f"换手≥{opt.liquidity.min_turnover_pct}%"
    )
    fu, fd = m.get("final_up") or {}, m.get("final_down") or {}
    if fu.get("is"):
        print(
            f"\n【做多】 量比≥{u.min_vol_ratio} 5日涨≥{u.min_ret_5d:.0%} 收强≥{u.min_close_strength:.0%} "
            f"换手≥{u.min_turnover_pct}% dvol≥${u.min_dvol_m}M MA50+MA20"
        )
        print(f"  IS: n={fu['is']['n']} 命中={fu['is']['hit_rate']:.1%}")
        if fu.get("oos"):
            print(f"  OOS: n={fu['oos']['n']} 命中={fu['oos']['hit_rate']:.1%}")
    if fd.get("is"):
        print(
            f"\n【做空/回避】 量比≥{d.min_vol_ratio} 5日跌≤{d.max_ret_5d:.0%} 收弱≤{d.max_close_strength:.0%} "
            f"换手≥{d.min_turnover_pct}% MA50下={d.require_below_ma50}"
        )
        print(f"  IS: n={fd['is']['n']} 命中={fd['is']['hit_rate']:.1%}")
        if fd.get("oos"):
            print(f"  OOS: n={fd['oos']['n']} 命中={fd['oos']['hit_rate']:.1%}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="5日路径参数寻优")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--min-samples", type=int, default=60)
    ap.add_argument("--min-hit", type=float, default=0.68, help="做多 IS 最低命中率")
    ap.add_argument("--min-path-pct", type=float, default=0.0,
                    help="锁定路径阈值(如3)，0=在2/2.5/3%间寻优")
    args = ap.parse_args()
    opt = run_search(
        min_samples=args.min_samples,
        min_hit_is=args.min_hit,
        quick=args.quick,
        lock_path_pct=args.min_path_pct or None,
    )
    print_report(opt)


if __name__ == "__main__":
    main()
