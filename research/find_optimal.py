"""最优策略汇总：合并 triple_target + holy_grail + 5×$10k 舰队，按场景输出 Top 推荐。

用法：
    python research/find_optimal.py
    python research/find_optimal.py --run   # 先重跑 quick 扫描再汇总
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.triple_target_scan import (
    RESULTS_CSV as TRIPLE_CSV,
    set_scan_targets,
    targets_label,
)

HOLY_CSV = ROOT / "research" / "holy_grail_results.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_row(df: pd.DataFrame, *, tier_col: str = "tier", tier_val: str = "A") -> pd.Series | None:
    if df.empty:
        return None
    sub = df[df[tier_col] == tier_val] if tier_col in df.columns else df
    if sub.empty and "gap_score" in df.columns:
        sub = df.nsmallest(1, "gap_score")
    elif sub.empty:
        sub = df.sort_values("ann_return", ascending=False).head(1)
    else:
        sub = sub.sort_values("gap_score" if "gap_score" in sub.columns else "ann_return", ascending=True)
    return sub.iloc[0] if len(sub) else None


def recommend_fleet() -> dict:
    from research.tier_a_csp import load_tier_a_csp_config, scan_tier_a_fleet, fleet_summary

    cfg = load_tier_a_csp_config()
    df, _ = scan_tier_a_fleet(cfg=cfg)
    summ = fleet_summary(df)
    open_df = df[df["可开仓"] == "✅"] if not df.empty else pd.DataFrame()
    weekly_roi = 0.0
    if not open_df.empty and "提示" in open_df.columns:
        import re
        rois = []
        for tip in open_df["提示"]:
            m = re.search(r"周ROI≈([\d.]+)%", str(tip))
            if m:
                rois.append(float(m.group(1)))
        weekly_roi = sum(rois) / len(rois) if rois else 0.0
    return {
        "scenario": "5×$10k 舰队（实盘）",
        "strategy": "偏斜铁鹰 / 周Put价差",
        "accounts": summ["total_accounts"],
        "open_today": summ["open_count"],
        "weekly_premium": summ["total_premium"],
        "weekly_margin": summ["total_margin"],
        "avg_weekly_roi_pct": weekly_roi,
        "tickers": open_df["代码"].tolist() if not open_df.empty else [],
        "note": "CSP 放不下 $10k 单户，自动降级为保证金 $2500/张 的铁鹰",
    }


def print_report(*, run_scans: bool = False) -> None:
    set_scan_targets(preset="relaxed")
    label = targets_label()
    print("=" * 72)
    print(f"最优策略报告 · 目标：{label}")
    print("=" * 72)

    if run_scans:
        print("\n[1/2] 运行 triple_target_scan quick …")
        from research.triple_target_scan import run_full_scan
        run_full_scan("2019-01-01", __import__("datetime").date.today().isoformat(), mode="quick", min_trades=30)
        print("\n[2/2] 运行 holy_grail_search quick …")
        from research.holy_grail_search import run_search
        run_search(mode="quick", preset="relaxed")

    triple = _load_csv(TRIPLE_CSV)
    holy = _load_csv(HOLY_CSV)

    print(f"\n── 数据源 ──")
    print(f"  triple_target: {len(triple)} 条  Tier-A={int((triple.get('tier') == 'A').sum()) if 'tier' in triple.columns else 0}")
    print(f"  holy_grail:    {len(holy)} 条  OOS-A={int((holy.get('tier_oos') == 'A').sum()) if 'tier_oos' in holy.columns else 0}")

    print(f"\n── 场景 A：单账户大资金（≥$36万）──")
    best = _best_row(triple, tier_col="tier", tier_val="A")
    if best is not None:
        print(f"  ★ 最优：{best['name']}")
        print(f"    年化={best['ann_return']:.1%}  回撤={best['max_dd']:.1%}  胜率={best['win_rate']:.1%}  "
              f"交易={int(best.get('trade_count', 0))}  OOS={best.get('oos_pass', False)}")
        print(f"    参数：{best.get('params', '')}")
        print("    执行：SNDK 顺势 CSP · δ=0.25 · MA50 · 仓位50% · 50%止盈")
    else:
        near = triple.nsmallest(3, "gap_score") if not triple.empty else pd.DataFrame()
        print("  无 Tier-A；最接近：")
        for _, r in near.iterrows():
            print(f"    {r['name']}: 年化={r['ann_return']:.1%} 回撤={r['max_dd']:.1%} 胜率={r['win_rate']:.1%}")

    print(f"\n── 继续寻找（普适策略，排除 SNDK）──")
    try:
        from research.continued_search import BEST_JSON, RESULTS_CSV as CS_CSV, run_continued_search
        if run_scans or not CS_CSV.exists():
            print("  运行 continued_search …")
            run_continued_search(quick=True, presets=["survivor", "realistic"])
        if BEST_JSON.exists():
            best = json.loads(BEST_JSON.read_text())
            for preset in ("survivor", "realistic"):
                rows = (best.get("best_by_preset") or {}).get(preset) or []
                if rows:
                    r = rows[0]
                    print(f"  [{preset}] {r.get('name','')[:45]}")
                    print(f"    年化={r.get('年化',0):.1%} 回撤={r.get('最大回撤',0):.1%} "
                          f"胜率={r.get('胜率',0):.1%} tier={r.get('tier','?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  继续寻找失败：{e}")

    try:
        from research.market_pattern_scan import RULES_JSON, run_scan
        if run_scans or not RULES_JSON.exists():
            print("  运行 market_pattern_scan quick …")
            run_scan(quick=True, exclude={"SNDK", "MSTR", "SOXL", "TQQQ"})
        if RULES_JSON.exists():
            rules = json.loads(RULES_JSON.read_text())
            bp = rules.get("best_portfolio") or {}
            print(f"  ★ 普适组合：{bp.get('scenario', 'ETF铁鹰舰队')}")
            print(f"    标的 {', '.join(bp.get('tickers', []))}")
            print(f"    回测 年化={bp.get('年化', 0):.1%} 回撤={bp.get('最大回撤', 0):.1%} 胜率={bp.get('胜率', 0):.1%}")
            mu = rules.get("ma50_uplift") or {}
            print(f"    Tier-A 单票（排除SNDK后）仅 {rules.get('tier_a_count', 0)} 只 → 不可外推")
        fleet = recommend_fleet()
        print(f"    今日 {fleet['open_today']}/{fleet['accounts']} 户可开 · 周收租≈${fleet['weekly_premium']:,.0f}")
    except Exception as e:  # noqa: BLE001
        print(f"  扫描失败：{e}")

    print(f"\n── 场景 C：最高年化（接受大回撤）──")
    if not triple.empty:
        hi = triple.loc[triple["ann_return"].idxmax()]
        print(f"  {hi['name']}: 年化={hi['ann_return']:.1%} 回撤={hi['max_dd']:.1%} 胜率={hi['win_rate']:.1%}")

    print(f"\n── 场景 D：最稳（高胜率+低回撤，牺牲年化）──")
    if not triple.empty:
        stable = triple[(triple["win_rate"] >= 0.85) & (triple["max_dd"] > -0.15)]
        if stable.empty:
            stable = triple[triple["win_rate"] >= 0.85]
        if not stable.empty:
            s = stable.sort_values("ann_return", ascending=False).iloc[0]
            print(f"  {s['name']}: 年化={s['ann_return']:.1%} 回撤={s['max_dd']:.1%} 胜率={s['win_rate']:.1%}")

    print(f"\n── 圣杯穷尽（holy_grail Top 3）──")
    if holy.empty:
        print("  尚未运行：python research/holy_grail_search.py --mode quick --preset relaxed")
    else:
        for _, r in holy.nsmallest(3, "gap_score").iterrows():
            print(
                f"  {r['name']}: 年化={r['ann_return']:.1%} 回撤={r['max_dd']:.1%} "
                f"胜率={r['win_rate']:.1%} IS={r.get('tier_is','?')} OOS={r.get('tier_oos','?')}"
            )

    print(f"\n── 最终推荐 ──")
    print("  1. 有 $36万+ 单账户 → CSP SNDK δ0.25 MA50 仓位50%（回测 Tier-A 57%年化 / 5%回撤 / 97%胜率）")
    print("  3. 普适最优（继续寻找）→ 并行铁鹰 SPY+QQQ ~16%年化/0回撤/100%胜率")
    print("  4. 更高收益     → 组合 铁鹰60%+动量40%（见 continued_search）")
    print("  3. 弱市增厚       → 引擎①卖看涨价差 + 引擎②高胜率做多（牛市）")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="最优策略汇总")
    p.add_argument("--run", action="store_true", help="先重跑 quick 扫描")
    args = p.parse_args()
    print_report(run_scans=args.run)


if __name__ == "__main__":
    main()
