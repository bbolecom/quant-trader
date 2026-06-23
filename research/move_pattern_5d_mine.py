#!/usr/bin/env python3
"""5 日路径涨跌规律挖掘（真实 OHLCV + 换手率 + 流动性）。

定义：
  · 5 日内「路径涨≥X%」：自信号日收盘起，未来 5 个交易日内最高价涨幅 ≥ X%
  · 5 日内「路径跌≥X%」：未来 5 日内最低价跌幅 ≥ X%
  · 换手率 = 当日成交量 / 流通股 × 100（yfinance fast_info）
  · 流动性：当日成交额 + 20 日均成交额 + 量比 + 换手率下限

用法：
    python research/move_pattern_5d_mine.py
    python research/move_pattern_5d_mine.py --quick --from-cache
    python research/move_pattern_5d_mine.py --today-only
    python research/move_pattern_5d_mine.py --up-pct 3 --down-pct 3 --min-turnover 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.move_pattern import (
    MovePatternRule,
    enrich_buckets,
    extract_trajectory_features_5d,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.liquid_tier_a_scan import _avg_dollar_vol, build_candidate_pool
from research.medallion_short import fetch_shares
from quant.pattern_5d_params import (
    ExtendedDownFilters,
    ExtendedUpFilters,
    LiquidityFilter,
    Optimized5dRules,
    PathThreshold,
    down_mask,
    load_optimized_5d,
    up_mask,
)

EVENTS_CSV = ROOT / "research" / "move_pattern_5d_events.csv"
BUCKETS_CSV = ROOT / "research" / "move_pattern_5d_buckets.csv"
RULES_JSON = ROOT / "research" / "move_pattern_5d_rules.json"
TODAY_CSV = ROOT / "research" / "move_pattern_5d_today.csv"
TRAIN_END = "2023-12-31"


def _liquidity_mask(panel: pd.DataFrame, liq: LiquidityFilter) -> pd.Series:
    dvol = pd.to_numeric(panel["dvol_m"], errors="coerce")
    avg = pd.to_numeric(panel.get("avg_dvol_m", dvol), errors="coerce")
    vr = pd.to_numeric(panel["vol_ratio"], errors="coerce")
    turn = pd.to_numeric(panel.get("换手率%", np.nan), errors="coerce")
    m = (dvol >= liq.min_dvol_m) & (avg >= liq.min_avg_dvol_m) & (vr >= liq.min_vol_ratio)
    if liq.min_turnover_pct > 0:
        if liq.require_turnover:
            m &= turn.notna() & (turn >= liq.min_turnover_pct)
        else:
            m &= turn.isna() | (turn >= liq.min_turnover_pct)
    return m


def build_5d_event_panel(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    liq: LiquidityFilter | None = None,
    th: PathThreshold | None = None,
    quick: bool = False,
    sample_stride: int = 3,
) -> pd.DataFrame:
    liq = liq or LiquidityFilter()
    th = th or PathThreshold()
    end = end or date.today().isoformat()
    pool = build_candidate_pool(use_broad=not quick, max_names=120 if quick else 0)
    print(f"候选 {len(pool)} 只 · 5日路径 ±{th.up_pct}% · 流动性 dvol≥${liq.min_dvol_m}M 换手≥{liq.min_turnover_pct}%")

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)
    print(f"有效行情 {len(batch)} 只 · 抓取流通股…")
    shares = fetch_shares(list(batch.keys()))

    rows: list[pd.DataFrame] = []
    n_pass = 0
    up_th = th.up_pct / 100.0
    dn_th = th.down_pct / 100.0
    for tk, df in batch.items():
        if df is None or df.empty or "Volume" not in df.columns:
            continue
        avg_dvol = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if avg_dvol < liq.min_avg_dvol_m:
            continue
        sh = shares.get(tk)
        feat = extract_trajectory_features_5d(
            df,
            shares_out=sh,
            horizon=th.horizon,
            up_threshold=up_th,
            down_threshold=dn_th,
        )
        if feat.empty:
            continue
        feat = enrich_buckets(feat)
        feat["代码"] = tk
        feat["avg_dvol_m"] = avg_dvol
        feat["shares_out"] = sh
        feat = feat.iloc[::sample_stride].copy()
        feat = feat[_liquidity_mask(feat, liq)]
        if feat.empty:
            continue
        n_pass += 1
        rows.append(feat)

    print(f"流动性+换手通过 {n_pass} 只")
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel["日期"] = pd.to_datetime(panel["日期"]).dt.strftime("%Y-%m-%d")
    return panel


def aggregate_5d_buckets(panel: pd.DataFrame, th: PathThreshold) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    hz = th.horizon
    pu, pd_col = f"path_up_{hz}d", f"path_down_{hz}d"
    grp = panel.groupby(["vol_ratio桶", "ret_5d桶", "dvol桶", "above_ma50"], dropna=False)
    rows = []
    up_th = th.up_pct / 100.0
    dn_th = th.down_pct / 100.0
    for keys, sub in grp:
        if len(sub) < 20:
            continue
        rows.append({
            "量比桶": keys[0],
            "5日涨跌桶": keys[1],
            "成交额桶": keys[2],
            "站上MA50": keys[3],
            "样本数": len(sub),
            f"5日涨≥阈值": float((sub[pu] >= up_th).mean()),
            f"5日跌≥阈值": float((sub[pd_col] <= -dn_th).mean()),
            f"fwd_{hz}d均值": float(sub[f"fwd_{hz}d"].mean()),
            "path_up均值": float(sub[pu].mean()),
            "path_down均值": float(sub[pd_col].mean()),
            "均换手率%": float(pd.to_numeric(sub["换手率%"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values("样本数", ascending=False)


def _split(panel: pd.DataFrame, train_end: str = TRAIN_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = panel.copy()
    p["日期"] = pd.to_datetime(p["日期"])
    cut = pd.Timestamp(train_end)
    return p[p["日期"] <= cut], p[p["日期"] > cut]


def mine_5d_rules(
    panel: pd.DataFrame,
    *,
    th: PathThreshold,
    liq: LiquidityFilter,
    min_samples: int = 80,
    min_hit_rate: float = 0.58,
    train_end: str = TRAIN_END,
) -> list[MovePatternRule]:
    """分桶挖掘 5 日路径命中率（IS 门槛 + OOS 验证）。"""
    is_df, oos_df = _split(panel, train_end)
    hz = th.horizon
    pu, pd_col = f"path_up_{hz}d", f"path_down_{hz}d"
    up_th, dn_th = th.up_pct / 100.0, th.down_pct / 100.0
    rules: list[MovePatternRule] = []
    rid = 0

    vol_buckets = ["1.0-1.5", "1.5-2.5", ">2.5"]
    ret_up = ["0~5%", "5~15%", ">15%"]
    ret_dn = ["-5~0%", "<-5%"]
    dvol_buckets = ["50-200M", "200M-1B", ">1B"]

    def _eval(sub_is: pd.DataFrame, sub_oos: pd.DataFrame, direction: str) -> tuple[float, float, int] | None:
        col = pu if direction == "up" else pd_col
        thv = up_th if direction == "up" else -dn_th
        if len(sub_is) < min_samples:
            return None
        if direction == "up":
            is_hit = float((sub_is[col] >= thv).mean())
            oos_hit = float((sub_oos[col] >= thv).mean()) if len(sub_oos) >= max(20, min_samples // 3) else 0.0
        else:
            is_hit = float((sub_is[col] <= thv).mean())
            oos_hit = float((sub_oos[col] <= thv).mean()) if len(sub_oos) >= max(20, min_samples // 3) else 0.0
        if is_hit < min_hit_rate:
            return None
        return is_hit, oos_hit, len(sub_is)

    for direction, ret_bs in [("up", ret_up), ("down", ret_dn)]:
        for vb in vol_buckets:
            for rb in ret_bs:
                for db in dvol_buckets:
                    for ma in [True, False]:
                        def _bm(df: pd.DataFrame) -> pd.Series:
                            return (
                                (df["vol_ratio桶"] == vb)
                                & (df["ret_5d桶"] == rb)
                                & (df["dvol桶"] == db)
                                & (df["above_ma50"] == ma)
                            )

                        sub_is = is_df[_bm(is_df)]
                        sub_oos = oos_df[_bm(oos_df)]
                        ev = _eval(sub_is, sub_oos, direction)
                        if ev is None:
                            continue
                        is_hit, oos_hit, n = ev
                        score = is_hit * 0.45 + oos_hit * 0.55
                        if score < min_hit_rate:
                            continue
                        rid += 1
                        tag = "涨幅" if direction == "up" else "跌幅"
                        ma_lbl = "MA50上" if ma else "MA50下"
                        desc = (
                            f"5日内{'涨' if direction == 'up' else '跌'}≥{th.up_pct if direction == 'up' else th.down_pct:.0f}% · "
                            f"量比{vb} · 5日{tag}{rb} · 成交额{db} · {ma_lbl}"
                        )
                        col = pu if direction == "up" else pd_col
                        fwd = sub_is[col]
                        rules.append(MovePatternRule(
                            id=f"5d_{direction}_{rid}",
                            direction=direction,
                            description=desc,
                            conditions={
                                "vol_ratio_bucket": vb,
                                "ret_5d_bucket": rb,
                                "dvol_bucket": db,
                                "above_ma50": ma,
                                "threshold_pct": th.up_pct if direction == "up" else th.down_pct,
                                "horizon": "5d_path",
                                "liquidity": asdict(liq),
                                "is_hit_rate": round(is_hit, 4),
                                "oos_hit_rate": round(oos_hit, 4),
                                "score": round(score, 4),
                            },
                            sample_n=n,
                            fwd_mean=float(fwd.mean()),
                            win_rate=round(is_hit, 4),
                            median_fwd=float(fwd.median()),
                            action="5日内路径达标 → 做多" if direction == "up" else "5日内路径达标 → 回避",
                            win_horizon="5d",
                            tier="A" if is_hit >= 0.65 and oos_hit >= 0.60 else "B",
                        ))

    # 单因子高置信规则
    for direction, mask_fn, desc, action in [
        ("up", lambda df: (df["vol_ratio"] >= 2.5) & (df["ret_5d"] > 0.05) & (df["dvol_m"] >= 200),
         f"5日内涨≥{th.up_pct:.0f}% · 量比≥2.5 + 5日涨>5% + 成交额>200M", "路径动量 → 做多"),
        ("down", lambda df: (df["vol_ratio"] >= 2.5) & (df["ret_5d"] < 0),
         f"5日内跌≥{th.down_pct:.0f}% · 量比≥2.5 + 5日已跌", "路径回避"),
        ("down", lambda df: (df["close_strength"] <= 0.35) & (df["vol_ratio"] >= 1.5) & (df["ret_5d"] < 0),
         f"5日内跌≥{th.down_pct:.0f}% · 收弱+放量+5日跌", "出货 → 回避"),
    ]:
        sub_is = is_df[mask_fn(is_df)]
        sub_oos = oos_df[mask_fn(oos_df)]
        ev = _eval(sub_is, sub_oos, direction)
        if ev is None:
            continue
        is_hit, oos_hit, n = ev
        rid += 1
        col = pu if direction == "up" else pd_col
        rules.append(MovePatternRule(
            id=f"5d_{direction}_{rid}",
            direction=direction,
            description=desc,
            conditions={"simple": desc, "threshold_pct": th.up_pct if direction == "up" else th.down_pct},
            sample_n=n,
            fwd_mean=float(sub_is[col].mean()),
            win_rate=round(is_hit, 4),
            median_fwd=float(sub_is[col].median()),
            action=action,
            win_horizon="5d",
            tier="B",
        ))

    rules.sort(key=lambda r: (-float(r.conditions.get("score", r.win_rate)), -r.sample_n))
    seen: set[str] = set()
    out: list[MovePatternRule] = []
    for r in rules:
        k = f"{r.direction}|{r.description}"
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out[:40]


def rules_from_optimized(opt: Optimized5dRules) -> list[MovePatternRule]:
    """将寻优参数转为可扫描规则。"""
    th = opt.threshold
    u, d = opt.up, opt.down
    meta = opt.meta or {}
    fu = meta.get("final_up") or {}
    fd = meta.get("final_down") or {}
    is_u = (fu.get("is") or {}).get("hit_rate", 0)
    oos_u = (fu.get("oos") or {}).get("hit_rate", 0)
    is_d = (fd.get("is") or {}).get("hit_rate", 0)
    oos_d = (fd.get("oos") or {}).get("hit_rate", 0)

    up_desc = (
        f"5日内涨≥{th.up_pct:.0f}% · 量比≥{u.min_vol_ratio} · 5日涨≥{u.min_ret_5d:.0%} · "
        f"收强≥{u.min_close_strength:.0%} · 换手≥{u.min_turnover_pct}% · "
        f"dvol≥${u.min_dvol_m:.0f}M · MA50{'+MA20' if u.require_above_ma20 else ''}"
    )
    dn_desc = (
        f"5日内跌≥{th.down_pct:.0f}% · 量比≥{d.min_vol_ratio} · 5日跌≤{d.max_ret_5d:.0%} · "
        f"收弱≤{d.max_close_strength:.0%} · 换手≥{d.min_turnover_pct}% · "
        f"{'MA50下' if d.require_below_ma50 else '全趋势'}"
    )
    from dataclasses import asdict as _asdict

    return [
        MovePatternRule(
            id="5d_opt_up",
            direction="up",
            description=up_desc,
            conditions={
                "rule_type": "optimized_up",
                "filters": _asdict(u),
                "threshold_pct": th.up_pct,
                "is_hit_rate": is_u,
                "oos_hit_rate": oos_u,
                "score": is_u * 0.4 + oos_u * 0.6,
            },
            sample_n=int((fu.get("is") or {}).get("n", 0)),
            fwd_mean=0.0,
            win_rate=float(is_u),
            median_fwd=0.0,
            action="5日内路径达标 → 做多（寻优）",
            win_horizon="5d",
            tier="S" if is_u >= 0.75 and oos_u >= 0.70 else "A",
        ),
        MovePatternRule(
            id="5d_opt_down",
            direction="down",
            description=dn_desc,
            conditions={
                "rule_type": "optimized_down",
                "filters": _asdict(d),
                "threshold_pct": th.down_pct,
                "is_hit_rate": is_d,
                "oos_hit_rate": oos_d,
                "score": is_d * 0.4 + oos_d * 0.6,
            },
            sample_n=int((fd.get("is") or {}).get("n", 0)),
            fwd_mean=0.0,
            win_rate=float(is_d),
            median_fwd=0.0,
            action="5日内路径达标 → 回避（寻优）",
            win_horizon="5d",
            tier="S" if is_d >= 0.70 and oos_d >= 0.65 else "A",
        ),
    ]


def match_5d_rule(row: pd.Series, rule: dict) -> bool:
    cond = rule.get("conditions") or {}
    rtype = cond.get("rule_type")
    if rtype == "optimized_up":
        p = ExtendedUpFilters(**(cond.get("filters") or {}))
        return bool(up_mask(pd.DataFrame([row.to_dict()]), p).iloc[0])
    if rtype == "optimized_down":
        p = ExtendedDownFilters(**(cond.get("filters") or {}))
        return bool(down_mask(pd.DataFrame([row.to_dict()]), p).iloc[0])
    if cond.get("simple"):
        return False
    vb = cond.get("vol_ratio_bucket")
    rb = cond.get("ret_5d_bucket")
    db = cond.get("dvol_bucket")
    ma = cond.get("above_ma50")
    if vb and str(row.get("vol_ratio桶", "")) != vb:
        return False
    if rb and str(row.get("ret_5d桶", "")) != rb:
        return False
    if db and str(row.get("dvol桶", "")) != db:
        return False
    if ma is not None and bool(row.get("above_ma50")) != bool(ma):
        return False
    return True


def scan_today_5d(
    rules: list[dict],
    *,
    liq: LiquidityFilter | None = None,
    th: PathThreshold | None = None,
    quick: bool = False,
    min_tier_hit: float = 0.60,
) -> pd.DataFrame:
    liq = liq or LiquidityFilter()
    th = th or PathThreshold()
    end = date.today().isoformat()
    start = (date.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    pool = build_candidate_pool(use_broad=not quick, max_names=100 if quick else 0)
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(pool, start, end)
    shares = fetch_shares(list(batch.keys()))

    tier_a = [r for r in rules if float(r.get("win_rate", 0) or 0) >= min_tier_hit]
    opt_rules = [r for r in rules if (r.get("conditions") or {}).get("rule_type", "").startswith("optimized")]
    if opt_rules:
        tier_a = opt_rules + tier_a
    elif not tier_a:
        tier_a = sorted(rules, key=lambda x: -float(x.get("win_rate", 0) or 0))[:20]

    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        avg_dvol = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if avg_dvol < liq.min_avg_dvol_m:
            continue
        sh = shares.get(tk)
        feat = extract_trajectory_features_5d(
            df, shares_out=sh, horizon=th.horizon,
            up_threshold=th.up_pct / 100, down_threshold=th.down_pct / 100,
        )
        if feat.empty:
            continue
        feat = enrich_buckets(feat)
        last = feat.iloc[-1]
        liq_row = pd.Series({**last.to_dict(), "avg_dvol_m": avg_dvol})
        if not bool(_liquidity_mask(pd.DataFrame([liq_row]), liq).iloc[0]):
            continue
        for rule in tier_a:
            if not match_5d_rule(last, rule):
                continue
            d = rule.get("direction", "up")
            key = (tk, d)
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "代码": tk,
                "方向": "偏多" if d == "up" else "偏空",
                "规律": rule.get("pattern", ""),
                "5日命中率": rule.get("win_rate"),
                "样本数": rule.get("sample_n"),
                "阈值%": (rule.get("conditions") or {}).get("threshold_pct", th.up_pct),
                "建议": rule.get("action", ""),
                "量比": round(float(last.get("vol_ratio", 0)), 2),
                "5日涨跌%": round(float(last.get("ret_5d", 0)) * 100, 1),
                "成交额M": round(float(last.get("dvol_m", 0)), 1),
                "换手率%": round(float(last.get("换手率%", 0) or 0), 2),
            })
    df_out = pd.DataFrame(hits)
    if not df_out.empty:
        df_out = df_out.sort_values(["方向", "5日命中率"], ascending=[True, False])
    return df_out


def run_5d_mine(
    *,
    start: str = "2019-01-01",
    end: str | None = None,
    liq: LiquidityFilter | None = None,
    th: PathThreshold | None = None,
    min_samples: int = 80,
    min_hit_rate: float = 0.58,
    quick: bool = False,
    from_cache: bool = False,
) -> dict:
    liq = liq or LiquidityFilter()
    th = th or PathThreshold()
    if from_cache and EVENTS_CSV.exists():
        print(f"从缓存加载 {EVENTS_CSV} …")
        panel = pd.read_csv(EVENTS_CSV)
    else:
        panel = build_5d_event_panel(
            start=start, end=end, liq=liq, th=th, quick=quick,
        )
        if panel.empty:
            return {"error": "无事件样本"}
        panel.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    print(f"事件 {len(panel)} 条 · {panel['代码'].nunique()} 只 → {EVENTS_CSV}")

    bdf = aggregate_5d_buckets(panel, th)
    bdf.to_csv(BUCKETS_CSV, index=False, encoding="utf-8-sig")
    print(f"分桶 {len(bdf)} 组 → {BUCKETS_CSV}")

    rules = mine_5d_rules(
        panel, th=th, liq=liq, min_samples=min_samples, min_hit_rate=min_hit_rate,
    )
    opt = load_optimized_5d()
    if opt.meta.get("final_up"):
        th = opt.threshold
        liq = opt.liquidity
        rules = rules_from_optimized(opt) + rules
    is_df, oos_df = _split(panel)
    liquid_n = int(_liquidity_mask(panel, liq).sum())
    up_rules = [r for r in rules if r.direction == "up"][:15]
    down_rules = [r for r in rules if r.direction == "down"][:15]
    all_rules = [r.to_dict() for r in up_rules + down_rules]

    top_meta = sorted(
        [
            {
                "direction": r.direction,
                "desc": r.description,
                "is_hit_rate": r.win_rate,
                "oos_hit_rate": (r.conditions or {}).get("oos_hit_rate", 0),
                "score": (r.conditions or {}).get("score", r.win_rate),
                "n": r.sample_n,
            }
            for r in rules
        ],
        key=lambda x: -float(x["score"]),
    )[:20]

    doc = {
        "updated": date.today().isoformat(),
        "mode": "5d_path",
        "method": (
            f"真实OHLCV · 5日内路径涨≥{th.up_pct}%/跌≥{th.down_pct}% · "
            f"成交额≥${liq.min_dvol_m}M · 量比≥{liq.min_vol_ratio} · "
            f"换手率≥{liq.min_turnover_pct}%"
        ),
        "universe_note": "全市场流动性标的，非 BS 期权回测",
        "event_count": len(panel),
        "ticker_count": int(panel["代码"].nunique()),
        "liquidity": asdict(liq),
        "threshold": {
            "up_pct": th.up_pct,
            "down_pct": th.down_pct,
            "use_path": True,
            "use_close": True,
        },
        "meta": {
            "liquidity": asdict(liq),
            "threshold": asdict(th),
            "is_rows": len(is_df),
            "oos_rows": len(oos_df),
            "liquid_rows": liquid_n,
            "top_buckets": top_meta,
            "optimized": opt.to_dict() if opt.meta.get("final_up") else None,
        },
        "rules_up": [r.to_dict() for r in up_rules],
        "rules_down": [r.to_dict() for r in down_rules],
        "rules": all_rules,
    }
    RULES_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"规则 {len(all_rules)} 条 → {RULES_JSON}")

    today = scan_today_5d(all_rules, liq=liq, th=th, quick=quick)
    if not today.empty:
        today.to_csv(TODAY_CSV, index=False, encoding="utf-8-sig")
        print(f"今日命中 {len(today)} 条 → {TODAY_CSV}")
    return doc


def print_report(doc: dict) -> None:
    print(f"\n{'=' * 70}")
    print("5 日路径规律 · 真实量价 + 换手率")
    print(f"{'=' * 70}")
    print(f"样本 {doc.get('event_count', 0)} · 标的 {doc.get('ticker_count', 0)} 只")
    print(doc.get("method", ""))
    for title, key in [("📈 5日路径做多", "rules_up"), ("📉 5日路径回避", "rules_down")]:
        rules = doc.get(key) or []
        print(f"\n{title} ({len(rules)} 条)")
        for r in rules[:8]:
            oos = (r.get("conditions") or {}).get("oos_hit_rate", 0)
            print(
                f"  · [{r.get('tier')}] {r['pattern']}\n"
                f"    样本={r['sample_n']}  IS={r['win_rate']:.1%}  OOS={oos:.1%}  → {r['action']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="5日路径涨跌规律挖掘")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--today-only", action="store_true")
    ap.add_argument("--min-samples", type=int, default=80)
    ap.add_argument("--optimize", action="store_true", help="先运行参数寻优")
    ap.add_argument("--min-hit-rate", type=float, default=0.58)
    ap.add_argument("--min-dvol-m", type=float, default=50.0)
    ap.add_argument("--min-avg-dvol-m", type=float, default=30.0)
    ap.add_argument("--min-vol-ratio", type=float, default=1.0)
    ap.add_argument("--min-turnover", type=float, default=0.3, help="换手率下限 %")
    ap.add_argument("--up-pct", type=float, default=3.0, help="5日内路径涨幅阈值 %")
    ap.add_argument("--down-pct", type=float, default=3.0, help="5日内路径跌幅阈值 %")
    args = ap.parse_args()

    liq = LiquidityFilter(
        min_dvol_m=args.min_dvol_m,
        min_avg_dvol_m=args.min_avg_dvol_m,
        min_vol_ratio=args.min_vol_ratio,
        min_turnover_pct=args.min_turnover,
    )
    th = PathThreshold(up_pct=args.up_pct, down_pct=args.down_pct)

    if args.optimize:
        from research.move_pattern_5d_param_search import print_report as pr_opt, run_search as run_opt

        opt = run_opt(min_samples=max(40, args.min_samples - 20), min_hit_is=max(0.65, args.min_hit_rate + 0.08))
        pr_opt(opt)

    if args.today_only:
        if not RULES_JSON.exists():
            print("请先运行完整挖掘")
            return
        doc = json.loads(RULES_JSON.read_text(encoding="utf-8-sig"))
        today = scan_today_5d(doc.get("rules") or [], liq=liq, th=th, quick=args.quick)
        print(today.to_string() if not today.empty else "今日无高置信 5 日路径命中")
        if not today.empty:
            today.to_csv(TODAY_CSV, index=False, encoding="utf-8-sig")
        return

    doc = run_5d_mine(
        start=args.start,
        end=args.end,
        liq=liq,
        th=th,
        min_samples=args.min_samples,
        min_hit_rate=args.min_hit_rate,
        quick=args.quick,
        from_cache=args.from_cache,
    )
    if doc.get("error"):
        print(doc["error"])
        return
    print_report(doc)


if __name__ == "__main__":
    main()
