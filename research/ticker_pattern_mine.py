#!/usr/bin/env python3
"""单标的涨跌规律挖掘（真实 OHLCV + 换手率）。

默认特斯拉 TSLA；也可指定其它 ticker。

用法：
    python research/ticker_pattern_mine.py
    python research/ticker_pattern_mine.py --ticker TSLA
    python research/ticker_pattern_mine.py --ticker TSLA --today
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.move_pattern import (
    assess_down_avoidance,
    assess_up_favor,
    compute_forward_path_labels,
    enrich_buckets,
    extract_trajectory_features_5d,
)
from quant.pattern_5d_params import ExtendedDownFilters, ExtendedUpFilters, down_mask, up_mask
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.liquid_tier_a_scan import _avg_dollar_vol
from research.medallion_short import fetch_shares

TRAIN_END = "2023-12-31"


@dataclass
class TickerRule:
    rule_id: str
    direction: str
    description: str
    horizon: str
    is_n: int
    is_rate: float
    oos_n: int
    oos_rate: float
    filters: dict
    action: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["日期"] = pd.to_datetime(d["日期"])
    cut = pd.Timestamp(TRAIN_END)
    return d[d["日期"] <= cut], d[d["日期"] > cut]


def build_ticker_panel(
    ticker: str,
    *,
    start: str = "2019-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    end = end or date.today().isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    df = yahoo.fetch_history(ticker.upper(), start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    sh = fetch_shares([ticker.upper()]).get(ticker.upper())
    feat = extract_trajectory_features_5d(
        df, shares_out=sh, horizon=5, up_threshold=0.02, down_threshold=0.02,
    )
    if feat.empty:
        return feat
    feat = enrich_buckets(feat)
    feat["代码"] = ticker.upper()
    feat["avg_dvol_m"] = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
    feat["shares_out"] = sh
    feat["日期"] = pd.to_datetime(feat["日期"]).dt.strftime("%Y-%m-%d")
    return feat


def _path_hit(sub: pd.DataFrame, direction: str, th: float = 0.02) -> float | None:
    if sub.empty:
        return None
    if direction == "up":
        return float((sub["path_up_5d"] >= th).mean())
    return float((sub["path_down_5d"] <= -th).mean())


def _fwd_hit(sub: pd.DataFrame, col: str, direction: str) -> float | None:
    if sub.empty or col not in sub.columns:
        return None
    s = sub.dropna(subset=[col])
    if s.empty:
        return None
    if direction == "up":
        return float((s[col] > 0).mean())
    return float((s[col] < 0).mean())


def _eval_mask(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    mask_is: pd.Series,
    mask_oos: pd.Series,
    *,
    direction: str,
    metric: str = "path_5d",
    th: float = 0.02,
    min_n: int = 15,
) -> dict | None:
    si, so = is_df[mask_is], oos_df[mask_oos]
    if len(si) < min_n:
        return None
    if metric == "path_5d":
        is_r = _path_hit(si, direction, th)
        oos_r = _path_hit(so, direction, th) if len(so) >= max(8, min_n // 3) else None
    elif metric == "fwd_1d":
        is_r = _fwd_hit(si, "ret_1d", direction)  # next day uses shift in panel - use fwd from paths
        # ret_1d in panel is same-day; use path_up as proxy or compute
        is_r = float((si["ret_1d"].shift(-1) > 0).mean()) if direction == "up" else float((si["ret_1d"].shift(-1) < 0).mean())
        oos_r = None
        if len(so) >= 8:
            oos_r = float((so["ret_1d"].shift(-1) > 0).mean()) if direction == "up" else float((so["ret_1d"].shift(-1) < 0).mean())
    else:
        col = "fwd_20d" if metric == "fwd_20d" else "fwd_5d"
        is_r = _fwd_hit(si, col, direction)
        oos_r = _fwd_hit(so, col, direction) if len(so) >= max(8, min_n // 3) else None
    if is_r is None:
        return None
    oos_v = oos_r if oos_r is not None else 0.0
    score = is_r * 0.4 + oos_v * 0.55 + min(len(si), 100) / 1000
    return {"is_n": len(si), "is_rate": is_r, "oos_n": len(so), "oos_rate": oos_r, "score": score}


def mine_ticker_rules(panel: pd.DataFrame, *, min_n: int = 15) -> list[TickerRule]:
    is_df, oos_df = _split(panel)
    rules: list[TickerRule] = []
    rid = 0

    def _add(direction, desc, meta, filters, action, horizon="5d_path"):
        nonlocal rid
        rid += 1
        rules.append(TickerRule(
            rule_id=f"{direction}_{rid}",
            direction=direction,
            description=desc,
            horizon=horizon,
            is_n=meta["is_n"],
            is_rate=meta["is_rate"],
            oos_n=meta["oos_n"],
            oos_rate=meta["oos_rate"] or 0.0,
            filters=filters,
            action=action,
        ))

    # --- 5日路径：网格 ---
    for direction, prefix in [("up", "涨"), ("down", "跌")]:
        for vr in [1.5, 2.0, 2.5, 3.0]:
            r5_list = [0.05, 0.10, 0.15, 0.20] if direction == "up" else [-0.05, -0.08, -0.10, -0.15]
            for r5 in r5_list:
                for ma in [True, False]:
                    for turn in [0.0, 2.0, 3.0]:
                        if direction == "up":
                            m = (
                                (panel["vol_ratio"] >= vr)
                                & (panel["ret_5d"] >= r5)
                                & (panel["above_ma50"] == ma)
                            )
                            ma_l = "MA50上" if ma else "MA50下"
                        else:
                            m = (panel["vol_ratio"] >= vr) & (panel["ret_5d"] <= r5)
                            if ma:
                                m &= ~panel["above_ma50"]
                                ma_l = "MA50下"
                            else:
                                ma_l = "不限"
                        if turn > 0:
                            m &= pd.to_numeric(panel["换手率%"], errors="coerce") >= turn
                        meta = _eval_mask(is_df, oos_df, m.loc[is_df.index], m.loc[oos_df.index], direction=direction)
                        if meta and meta["is_rate"] >= (0.65 if direction == "up" else 0.60):
                            desc = (
                                f"5日路径{prefix}≥2% · 量比≥{vr} · 5日{r5:+.0%} · {ma_l}"
                                + (f" · 换手≥{turn}%" if turn else "")
                            )
                            _add(
                                direction, desc, meta,
                                {"vol_ratio": vr, "ret_5d": r5, "ma50": ma, "turnover": turn},
                                f"{'做多' if direction == 'up' else '回避'}",
                            )

    # --- 固定高置信模板（TSLA 特征） ---
    templates = [
        ("up", "5日路径涨≥2% · 5日涨>10% + 换手>3%", lambda d: (d["ret_5d"] > 0.10) & (pd.to_numeric(d["换手率%"], errors="coerce") > 3), "高换手动量 → 5日内继续上冲"),
        ("up", "5日路径涨≥2% · 20日涨>30% 趋势延续", lambda d: d["ret_20d"] > 0.30, "强趋势中继 → 路径涨"),
        ("up", "5日路径涨≥2% · 爆量>3 + 5日涨5~15% + MA50", lambda d: (d["vol_ratio"] > 3) & d["ret_5d"].between(0.05, 0.15) & d["above_ma50"], "爆量突破段 → 路径涨"),
        ("down", "5日路径跌≥2% · 5日跌>10%", lambda d: d["ret_5d"] < -0.10, "深跌惯性 → 路径继续下探"),
        ("down", "5日路径跌≥2% · 20日涨>40% 过热", lambda d: d["ret_20d"] > 0.40, "抛物线过热 → 5日内回撤"),
        ("down", "5日路径跌≥2% · 5日涨>15% + 缩量<1", lambda d: (d["ret_5d"] > 0.15) & (d["vol_ratio"] < 1.0), "缩量顶 → 路径跌"),
        ("down", "5日路径跌≥2% · MA50下 + 5日跌>5%", lambda d: (~d["above_ma50"]) & (d["ret_5d"] < -0.05), "弱趋势放量跌 → 回避"),
        ("down", "5日路径跌≥2% · 收弱<35% + 放量>1.5", lambda d: (d["close_strength"] < 0.35) & (d["vol_ratio"] > 1.5), "出货形态 → 路径跌"),
    ]
    for direction, desc, fn, action in templates:
        m = fn(panel)
        meta = _eval_mask(is_df, oos_df, m.loc[is_df.index], m.loc[oos_df.index], direction=direction)
        if meta and meta["is_rate"] >= 0.55:
            _add(direction, desc, meta, {"template": desc}, action)

    # 20日方向
    for direction, desc, fn in [
        ("up", "后20日涨 · 20日涨>30%", lambda d: d["ret_20d"] > 0.30),
        ("down", "后20日跌 · 20日涨>40%", lambda d: d["ret_20d"] > 0.40),
        ("down", "后20日跌 · 5日涨>20%+缩量", lambda d: (d["ret_5d"] > 0.20) & (d["vol_ratio"] < 1)),
    ]:
        m = fn(panel)
        meta = _eval_mask(is_df, oos_df, m.loc[is_df.index], m.loc[oos_df.index], direction=direction, metric="fwd_20d", min_n=12)
        if meta and meta["is_rate"] >= 0.55:
            _add(direction, desc, meta, {"template": desc}, desc.split("·")[-1].strip(), horizon="20d")

    rules.sort(key=lambda r: -(r.is_rate * 0.4 + r.oos_rate * 0.55))
    seen: set[str] = set()
    out: list[TickerRule] = []
    for r in rules:
        if r.description in seen:
            continue
        seen.add(r.description)
        out.append(r)
    return out[:20]


def scan_today(ticker: str, rules: list[TickerRule], panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {}
    last = panel.iloc[-1]
    row = last.to_dict()
    hits_up: list[str] = []
    hits_down: list[str] = []

    templates_today = [
        ("up", lambda d: (d["ret_5d"] > 0.10) & (pd.to_numeric(d["换手率%"], errors="coerce") > 3), "5日涨>10% + 换手>3% → 5日路径涨"),
        ("up", lambda d: d["ret_20d"] > 0.30, "20日涨>30% → 趋势延续"),
        ("up", lambda d: (d["vol_ratio"] > 3) & d["ret_5d"].between(0.05, 0.15) & d["above_ma50"], "爆量>3 + 5日涨5~15% + MA50"),
        ("down", lambda d: d["ret_5d"] < -0.10, "5日跌>10% → 路径继续下探"),
        ("down", lambda d: d["ret_20d"] > 0.40, "20日涨>40% → 过热回撤"),
        ("down", lambda d: (d["ret_5d"] > 0.15) & (d["vol_ratio"] < 1.0), "5日涨>15% + 缩量 → 顶部"),
        ("down", lambda d: (~d["above_ma50"]) & (d["ret_5d"] < -0.05), "MA50下 + 5日跌>5%"),
        ("down", lambda d: (d["close_strength"] < 0.35) & (d["vol_ratio"] > 1.5), "收弱 + 放量 → 出货"),
    ]
    one = pd.DataFrame([row])
    for direction, fn, desc in templates_today:
        if fn(one).iloc[0]:
            (hits_up if direction == "up" else hits_down).append(desc)

    for r in rules:
        f = r.filters
        vr, r5 = f.get("vol_ratio"), f.get("ret_5d")
        if vr is None or r5 is None:
            continue
        ok = float(last["vol_ratio"]) >= float(vr)
        if r.direction == "up":
            ok &= float(last["ret_5d"]) >= float(r5)
            if f.get("ma50") is True:
                ok &= bool(last["above_ma50"])
            elif f.get("ma50") is False:
                ok &= not bool(last["above_ma50"])
        else:
            ok &= float(last["ret_5d"]) <= float(r5)
            if f.get("ma50"):
                ok &= not bool(last["above_ma50"])
        if f.get("turnover"):
            ok &= float(last.get("换手率%", 0) or 0) >= float(f["turnover"])
        if ok:
            (hits_up if r.direction == "up" else hits_down).append(r.description)

    avoid = assess_down_avoidance(last)
    favor = assess_up_favor(last)

    return {
        "日期": str(last.get("日期", "")),
        "收盘价特征": {
            "量比": round(float(last["vol_ratio"]), 2),
            "5日涨跌%": round(float(last["ret_5d"]) * 100, 1),
            "20日涨跌%": round(float(last.get("ret_20d", 0) or 0) * 100, 1),
            "换手率%": round(float(last.get("换手率%", 0) or 0), 2),
            "成交额M": round(float(last["dvol_m"]), 1),
            "MA50": "上" if bool(last["above_ma50"]) else "下",
            "收强": round(float(last["close_strength"]), 2),
        },
        "命中上涨规律": hits_up[:6],
        "命中下跌规律": hits_down[:6],
        "全市场回避标签": [h["rule_id"] + ": " + h["reason"] for h in avoid],
        "全市场做多加分": [t["note"] for t in favor],
    }


def run(ticker: str = "TSLA", *, start: str = "2019-01-01") -> dict:
    tk = ticker.upper()
    out_dir = ROOT / "research"
    panel = build_ticker_panel(tk, start=start)
    if panel.empty:
        return {"error": f"无法获取 {tk} 行情"}

    panel_path = out_dir / f"{tk.lower()}_pattern_events.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")

    rules = mine_ticker_rules(panel)
    is_df, oos_df = _split(panel)
    today = scan_today(tk, rules, panel)

    doc = {
        "updated": date.today().isoformat(),
        "ticker": tk,
        "method": "真实OHLCV + 换手率 + 5日路径High/Low + 20日收盘（非BS）",
        "event_count": len(panel),
        "is_rows": len(is_df),
        "oos_rows": len(oos_df),
        "avg_dvol_m": float(panel["dvol_m"].mean()),
        "avg_turnover_pct": float(pd.to_numeric(panel["换手率%"], errors="coerce").mean()),
        "rules_up": [r.to_dict() for r in rules if r.direction == "up"][:10],
        "rules_down": [r.to_dict() for r in rules if r.direction == "down"][:10],
        "today": today,
    }
    rules_path = out_dir / f"{tk.lower()}_pattern_rules.json"
    rules_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {rules_path}")
    print(f"→ {panel_path}")
    return doc


def print_report(doc: dict) -> None:
    tk = doc.get("ticker", "TSLA")
    print(f"\n{'=' * 72}")
    print(f"{tk} 涨跌规律 · 真实量价（IS 2019–2023 / OOS 2024+）")
    print(f"{'=' * 72}")
    print(f"样本 {doc.get('event_count')} 日 · 均成交额 ${doc.get('avg_dvol_m', 0):,.0f}M · 均换手 {doc.get('avg_turnover_pct', 0):.2f}%")

    for title, key in [("📈 上涨规律", "rules_up"), ("📉 下跌规律", "rules_down")]:
        print(f"\n{title}")
        for r in doc.get(key) or []:
            oos = f"{r['oos_rate']:.1%}" if r.get("oos_n", 0) >= 8 else "—"
            print(f"  · {r['description']}")
            print(f"    IS n={r['is_n']} {r['is_rate']:.1%}  OOS n={r.get('oos_n',0)} {oos}  → {r['action']}")

    t = doc.get("today") or {}
    if t:
        print(f"\n🎯 今日状态 ({t.get('日期')})")
        feat = t.get("收盘价特征") or {}
        for k, v in feat.items():
            print(f"  {k}: {v}")
        if t.get("命中上涨规律"):
            print("  上涨命中:", "；".join(t["命中上涨规律"][:3]))
        if t.get("命中下跌规律"):
            print("  下跌命中:", "；".join(t["命中下跌规律"][:3]))
        if t.get("全市场回避标签"):
            print("  全市场回避:", "；".join(t["全市场回避标签"][:2]))
    print(f"{'=' * 72}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="单标的涨跌规律挖掘")
    ap.add_argument("--ticker", default="TSLA")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--today", action="store_true", help="仅打印已保存结果中的今日")
    args = ap.parse_args()
    tk = args.ticker.upper()
    path = ROOT / "research" / f"{tk.lower()}_pattern_rules.json"
    if args.today and path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        print_report(doc)
        return
    doc = run(tk, start=args.start)
    if doc.get("error"):
        print(doc["error"])
        return
    print_report(doc)


if __name__ == "__main__":
    main()
