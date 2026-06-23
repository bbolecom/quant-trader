#!/usr/bin/env python3
"""全市场资金轨迹规律挖掘：从成交额/量比/涨幅等前期特征归纳涨跌模式。

思路：大涨大跌需要资金推动 → 在发生前往往有「放量、动量、价位」等可观测轨迹。

用法：
    python research/move_pattern_mine.py
    python research/move_pattern_mine.py --quick
    python research/move_pattern_mine.py --min-dvol-m 30 --min-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.move_pattern import (
    enrich_buckets,
    extract_trajectory_features,
    live_matches,
    mine_high_win_rules,
    mine_rules_from_panel,
    MovePatternRule,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.liquid_tier_a_scan import _avg_dollar_vol, build_candidate_pool

EVENTS_CSV = ROOT / "research" / "move_pattern_events.csv"
BUCKETS_CSV = ROOT / "research" / "move_pattern_buckets.csv"
RULES_JSON = ROOT / "research" / "move_pattern_rules.json"
TODAY_CSV = ROOT / "research" / "move_pattern_today.csv"
HIGHWIN_PANEL_CSV = ROOT / "research" / "move_pattern_highwin_panel.csv"


def _spy_close_series(spy: pd.DataFrame) -> pd.Series:
    if spy is None or spy.empty:
        return pd.Series(dtype=float)
    if "Close" in spy.columns:
        c = spy["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return c.astype(float)
    return spy.iloc[:, 0].astype(float)


def build_gainer_daily_panel(
    *,
    start: str,
    end: str,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """构建日频全样本因子面板（含近8次形态胜率）。"""
    from research.gainer_daily_backtest import (
        GAINER_MOMENTUM,
        build_factor_panels,
        edge_setup_filters,
        fetch_gainer_data_yahoo,
        precompute_setup_edge,
    )

    pool = tickers or GAINER_MOMENTUM
    data, spy = fetch_gainer_data_yahoo(pool, start, end)
    spy_close = _spy_close_series(spy)
    panel = build_factor_panels(data, spy_close)
    if panel.empty:
        return panel
    fwd_ret: dict[str, pd.Series] = {}
    for tk, df in data.items():
        if df is None or df.empty:
            continue
        s = df["Close"].astype(float).pct_change(1).shift(-1)
        s.index = pd.to_datetime(df.index)
        fwd_ret[str(tk)] = s
    panel = precompute_setup_edge(panel, fwd_ret, edge_setup_filters())
    panel["fwd_1d"] = panel.apply(
        lambda r: float(fwd_ret.get(str(r["代码"]), pd.Series(dtype=float)).get(pd.Timestamp(r["日期"]), np.nan)),
        axis=1,
    )
    close_by_tk = {tk: df["Close"].astype(float) for tk, df in data.items() if df is not None}
    ret5, ret20 = [], []
    for _, r in panel.iterrows():
        tk, dt = str(r["代码"]), pd.Timestamp(r["日期"])
        c = close_by_tk.get(tk)
        if c is None or dt not in c.index:
            ret5.append(np.nan)
            ret20.append(np.nan)
            continue
        loc = c.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.start
        ret5.append(float(c.iloc[loc] / c.iloc[max(0, loc - 5)] - 1) if loc >= 5 else np.nan)
        ret20.append(float(c.iloc[loc] / c.iloc[max(0, loc - 20)] - 1) if loc >= 20 else np.nan)
    panel["ret_5d"] = ret5
    panel["ret_20d"] = ret20
    return panel


def run_highwin_mine(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_samples: int = 40,
    min_win_rate: float = 0.62,
    quick: bool = False,
) -> dict:
    """高置信模式：日频面板 + 严格模板 + 次日胜率。"""
    from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

    end = end or date.today().isoformat()
    pool = LIQUID100 if quick else GAINER_MOMENTUM
    print(f"高置信挖掘 · {len(pool)} 只 · 次日胜率门槛 ≥ {min_win_rate:.0%}")
    panel = build_gainer_daily_panel(start=start, end=end, tickers=pool)
    if panel.empty:
        return {"error": "面板为空"}
    panel.to_csv(HIGHWIN_PANEL_CSV, index=False, encoding="utf-8-sig")
    print(f"日频面板 {len(panel)} 行 → {HIGHWIN_PANEL_CSV}")

    rules = mine_high_win_rules(
        panel, min_samples=min_samples, min_win_rate=max(0.55, min_win_rate - 0.05),
    )
    rules.sort(key=lambda r: -r.win_rate)
    if min_win_rate > 0:
        preferred = [r for r in rules if r.win_rate >= min_win_rate]
        if preferred:
            rules = preferred
        elif rules:
            print(f"  ⚠ 无规则达 {min_win_rate:.0%}，展示最高胜率 Top{min(6, len(rules))} 条")

    # Income 锚点：期权收租类在全市场扫描中胜率 80%+（continued_search）
    cs_path = ROOT / "research" / "continued_search_best.json"
    if cs_path.exists():
        cs = json.loads(cs_path.read_text(encoding="utf-8"))
        for row in (cs.get("best_by_preset") or {}).get("realistic") or []:
            if row.get("tier") in ("A", "B") and float(row.get("胜率", 0) or 0) >= 0.75:
                rules.insert(0, MovePatternRule(
                    id="income_iron_condor",
                    direction="up",
                    description=row.get("name", "并行铁鹰收租"),
                    conditions={"source": "continued_search", "params": row.get("params", "")},
                    sample_n=int(row.get("交易数", 0) or 0),
                    fwd_mean=float(row.get("年化", 0) or 0),
                    win_rate=float(row.get("胜率", 0) or 0),
                    median_fwd=0.0,
                    action="高胜率收租腿：大盘 ETF 铁鹰，与轨迹选股互补",
                    win_horizon="1w",
                    tier="S",
                ))
                break

    # 锚点：参数寻优后的涨幅榜 TopN（组合级次日胜率）
    try:
        from research.gainer_daily_backtest import fetch_gainer_data_yahoo, search_win_rate_params
        data, spy = fetch_gainer_data_yahoo(pool, start, end)
        best_filt, best_res = search_win_rate_params(data, spy, start=start, end=end)
        if not best_res.get("error"):
            wr = float(best_res.get("日胜率", 0) or 0)
            rules.insert(0, MovePatternRule(
                id="gainer_optimized_top",
                direction="up",
                description=(
                    f"涨幅榜·寻优Top{best_filt.top_n}（温和涨{best_filt.min_gain_pct}-{best_filt.max_gain_pct}% "
                    f"+ 量比{best_filt.min_vol_ratio}-{best_filt.max_vol_ratio} + 形态胜率≥{best_filt.min_setup_win_rate:.0%}）"
                ),
                conditions={"source": "gainer_optimize", "top_n": best_filt.top_n},
                sample_n=int(best_res.get("交易天数", 0)),
                fwd_mean=float(best_res.get("年化收益率", 0) or 0),
                win_rate=wr,
                median_fwd=0.0,
                action=f"全市场每日扫描 Top{best_filt.top_n}，仅大盘顺风+历史形态验证日",
                win_horizon="1d",
                tier="S" if wr >= 0.65 else "A",
            ))
    except Exception as e:  # noqa: BLE001
        print(f"  寻优锚点跳过: {e}")

    up = [r for r in rules if r.direction == "up"]
    down = [r for r in rules if r.direction == "down"]
    all_rules = [r.to_dict() for r in rules]

    doc = {
        "updated": date.today().isoformat(),
        "mode": "highwin",
        "method": "日频面板 + 严格量价模板 + 次日胜率（≥62%）",
        "universe_note": "不绑定固定标的；高置信模板过滤后统计",
        "event_count": len(panel),
        "ticker_count": int(panel["代码"].nunique()),
        "min_win_rate": min_win_rate,
        "rules_up": [r.to_dict() for r in up],
        "rules_down": [r.to_dict() for r in down],
        "rules": all_rules,
    }
    RULES_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"高置信规则 {len(all_rules)} 条 → {RULES_JSON}")

    if all_rules:
        today = scan_today_highwin(all_rules, quick=quick)
        if not today.empty:
            today.to_csv(TODAY_CSV, index=False, encoding="utf-8-sig")
            print(f"今日高置信命中 {len(today)} 条 → {TODAY_CSV}")
    return doc


def scan_today_highwin(rules: list[dict], *, quick: bool = False) -> pd.DataFrame:
    """今日高置信扫描（涨幅榜因子 + 模板匹配）。"""
    from research.gainer_daily_backtest import (
        LIQUID100, GAINER_MOMENTUM, fetch_gainer_data_yahoo,
        high_win_filters, pick_top_gainers,
    )
    end = date.today().isoformat()
    start = (date.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    pool = LIQUID100 if quick else GAINER_MOMENTUM
    data, spy = fetch_gainer_data_yahoo(pool, start, end)
    if not data:
        return pd.DataFrame()
    spy_close = _spy_close_series(spy)
    picks = pick_top_gainers(data, pd.Timestamp(end), spy_close, high_win_filters())
    opt_rule = next((x for x in rules if x.get("id") == "gainer_optimized_top"), None)
    rows = []
    for _, r in picks.iterrows():
        rows.append({
            "代码": r["代码"],
            "日期": r.get("选股日期", end),
            "方向": "偏多",
            "规律": "涨幅榜·高置信Top2",
            "历史胜率": (opt_rule or {}).get("win_rate") or next((x.get("win_rate") for x in rules if "gainer" in str(x.get("id", ""))), None),
            "样本数": next((x.get("sample_n") for x in rules if x.get("id") == "gainer_highwin_top2"), None),
            "建议": r.get("选股理由", ""),
            "量比": round(float(r.get("量比", 0)), 2),
            "5日涨幅": round(float(r.get("涨幅20d%", 0)), 2),
            "成交额M": round(float(r.get("成交额USD", 0)) / 1e6, 1),
        })
    return pd.DataFrame(rows)


def build_event_panel(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 30.0,
    quick: bool = False,
    sample_stride: int = 3,
) -> pd.DataFrame:
    """拉全市场，逐标的提取轨迹事件。"""
    end = end or date.today().isoformat()
    pool = build_candidate_pool(use_broad=not quick, max_names=100 if quick else 0)
    print(f"候选 {len(pool)} 只 · 成交额门槛 ≥ ${min_dvol_m:.0f}M · {start}~{end}")

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)
    print(f"有效行情 {len(batch)} 只")

    rows: list[pd.DataFrame] = []
    n_liquid = 0
    for tk, df in batch.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m < min_dvol_m:
            continue
        n_liquid += 1
        feat = extract_trajectory_features(df, forward_days=20)
        if feat.empty:
            continue
        feat = enrich_buckets(feat)
        feat["代码"] = tk
        feat["avg_dvol_m"] = dvol_m
        # 稀疏采样加速（每 N 个交易日取 1 个事件，避免同股重叠过多）
        feat = feat.iloc[::sample_stride].copy()
        rows.append(feat)

    print(f"流动性通过 {n_liquid} 只")
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel["日期"] = pd.to_datetime(panel["日期"]).dt.strftime("%Y-%m-%d")
    return panel


def aggregate_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """分桶统计：各轨迹组合后的 forward 分布。"""
    if panel.empty:
        return pd.DataFrame()
    grp = panel.groupby(["vol_ratio桶", "ret_5d桶", "dvol桶", "above_ma50"], dropna=False)
    rows = []
    for keys, sub in grp:
        fwd = sub["fwd_20d"]
        rows.append({
            "量比桶": keys[0],
            "5日涨跌桶": keys[1],
            "成交额桶": keys[2],
            "站上MA50": keys[3],
            "样本数": len(sub),
            "后20日均值": float(fwd.mean()),
            "后20日中位": float(fwd.median()),
            "后20日上涨率": float((fwd > 0).mean()),
            "强涨率≥10%": float((fwd >= 0.10).mean()),
            "强跌率≤-10%": float((fwd <= -0.10).mean()),
        })
    bdf = pd.DataFrame(rows)
    return bdf.sort_values("样本数", ascending=False)


def scan_today(
    rules: list[dict],
    *,
    min_dvol_m: float = 30.0,
    quick: bool = False,
) -> pd.DataFrame:
    """用挖掘规则扫描今日全市场匹配。"""
    end = date.today().isoformat()
    start = (date.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    pool = build_candidate_pool(use_broad=not quick, max_names=80 if quick else 0)
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)

    hits: list[dict] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m < min_dvol_m:
            continue
        feat = extract_trajectory_features(df, forward_days=20)
        if feat.empty:
            continue
        feat = enrich_buckets(feat)
        last = feat.iloc[-1]
        as_of = pd.Timestamp(last["日期"]).strftime("%Y-%m-%d")
        for h in live_matches(tk, last, rules, as_of=as_of):
            hits.append(h)
    return pd.DataFrame(hits)


def run_mine(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    min_dvol_m: float = 30.0,
    min_samples: int = 40,
    min_win_rate: float = 0.58,
    quick: bool = False,
    from_cache: bool = False,
) -> dict:
    if from_cache and EVENTS_CSV.exists():
        print(f"从缓存加载 {EVENTS_CSV} …")
        panel = pd.read_csv(EVENTS_CSV)
    else:
        panel = build_event_panel(
            start=start, end=end, min_dvol_m=min_dvol_m, quick=quick,
        )
        if panel.empty:
            return {"error": "无事件样本"}

    panel.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    print(f"事件样本 {len(panel)} 条 → {EVENTS_CSV}")

    bdf = aggregate_buckets(panel)
    bdf.to_csv(BUCKETS_CSV, index=False, encoding="utf-8-sig")
    print(f"分桶统计 {len(bdf)} 组 → {BUCKETS_CSV}")

    rules = mine_rules_from_panel(
        panel, min_samples=min_samples, min_win_rate=min_win_rate,
    )
    up_rules = [r for r in rules if r.direction == "up"][:12]
    down_rules = [r for r in rules if r.direction == "down"][:12]
    all_rules = [r.to_dict() for r in up_rules + down_rules]

    doc = {
        "updated": date.today().isoformat(),
        "method": "全市场轨迹挖掘：量比×5日涨跌×成交额×MA50 → 后20日表现",
        "universe_note": "不绑定固定标的；任意满足流动性门槛的股票均纳入统计",
        "event_count": len(panel),
        "ticker_count": int(panel["代码"].nunique()),
        "rules_up": [r.to_dict() for r in up_rules],
        "rules_down": [r.to_dict() for r in down_rules],
        "rules": all_rules,
        "top_buckets": bdf.head(15).to_dict(orient="records") if not bdf.empty else [],
    }
    RULES_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"规则 {len(all_rules)} 条 → {RULES_JSON}")

    if all_rules:
        today = scan_today(all_rules, min_dvol_m=min_dvol_m, quick=quick)
        if not today.empty:
            today.to_csv(TODAY_CSV, index=False, encoding="utf-8-sig")
            print(f"今日命中 {len(today)} 条 → {TODAY_CSV}")

    return doc


def _print_report(doc: dict) -> None:
    print(f"\n{'=' * 64}")
    print("资金轨迹规律 · 挖掘报告")
    print(f"{'=' * 64}")
    print(f"样本 {doc.get('event_count', 0)} 条 · 覆盖 {doc.get('ticker_count', 0)} 只")

    for title, key in [("📈 做多规律（上涨前轨迹）", "rules_up"), ("📉 做空/回避规律（下跌前轨迹）", "rules_down")]:
        rules = doc.get(key) or []
        print(f"\n{title} ({len(rules)} 条)")
        for r in rules[:8]:
            wl = r.get("win_label", "胜率")
            fm = r.get("fwd_mean", r.get("fwd_20d_mean", 0))
            print(
                f"  · [{r.get('tier', '?')}] {r['pattern']}\n"
                f"    样本={r['sample_n']}  {wl}={r['win_rate']:.1%}  "
                f"均收益={fm:.2%}  → {r['action']}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="全市场资金轨迹规律挖掘")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--min-dvol-m", type=float, default=30.0)
    p.add_argument("--min-samples", type=int, default=40)
    p.add_argument("--mode", choices=["trajectory", "highwin", "5d"], default="highwin",
                   help="highwin=次日; trajectory=20日; 5d=5日路径+换手率")
    p.add_argument("--min-win-rate", type=float, default=None,
                   help="默认 highwin=0.62 trajectory=0.58")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--from-cache", action="store_true", help="从 move_pattern_events.csv 重新挖掘")
    p.add_argument("--today-only", action="store_true", help="仅扫描今日命中")
    args = p.parse_args()

    if args.today_only:
        path_5d = ROOT / "research" / "move_pattern_5d_rules.json"
        if args.mode == "5d" or (path_5d.exists() and not RULES_JSON.exists()):
            from research.move_pattern_5d_mine import LiquidityFilter, PathThreshold, scan_today_5d
            doc = json.loads(path_5d.read_text(encoding="utf-8"))
            liq = LiquidityFilter(**(doc.get("liquidity") or {}))
            th_d = doc.get("threshold") or {}
            th = PathThreshold(up_pct=float(th_d.get("up_pct", 3)), down_pct=float(th_d.get("down_pct", 3)))
            today = scan_today_5d(doc.get("rules") or [], liq=liq, th=th, quick=args.quick)
        elif RULES_JSON.exists():
            doc = json.loads(RULES_JSON.read_text(encoding="utf-8"))
            if doc.get("mode") == "highwin":
                today = scan_today_highwin(doc.get("rules") or [], quick=args.quick)
            else:
                today = scan_today(doc.get("rules") or [], min_dvol_m=args.min_dvol_m, quick=args.quick)
        else:
            print("请先运行完整挖掘生成 rules")
            return
        print(today.to_string() if not today.empty else "今日无命中")
        return

    min_wr = args.min_win_rate
    if min_wr is None:
        min_wr = 0.62 if args.mode == "highwin" else 0.58

    if args.mode == "5d":
        from research.move_pattern_5d_mine import LiquidityFilter, PathThreshold, run_5d_mine, print_report as pr5
        doc = run_5d_mine(
            start=args.start, end=args.end,
            liq=LiquidityFilter(min_dvol_m=max(args.min_dvol_m, 50)),
            th=PathThreshold(),
            min_samples=args.min_samples,
            min_hit_rate=min_wr,
            quick=args.quick,
            from_cache=args.from_cache,
        )
        if not doc.get("error"):
            pr5(doc)
        return

    if args.mode == "highwin":
        doc = run_highwin_mine(
            start=args.start, end=args.end,
            min_samples=args.min_samples, min_win_rate=min_wr, quick=args.quick,
        )
    else:
        doc = run_mine(
            start=args.start, end=args.end,
            min_dvol_m=args.min_dvol_m, min_samples=args.min_samples,
            min_win_rate=min_wr, quick=args.quick,
            from_cache=args.from_cache,
        )
    if "error" not in doc:
        _print_report(doc)


if __name__ == "__main__":
    main()
