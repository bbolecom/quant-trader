#!/usr/bin/env python3
"""安飞士式做空 · 每日扫描 + 重点票破位触发盯盘。

两件事：
  1. 全市场跑「过热分」扫描（research.blowoff_relaxed_scan），给抛物线顶/派发候选。
  2. 对 watchlist（默认 CDNL/AMLX + 当日过热分 Top）逐只检测「破位大阴」触发：
       当日跌幅 ≤ trigger_drop_pct 且 收盘强度 ≤ trigger_clv 且 收盘 < MA10
       → 状态「✅破位可空」（安飞士式右侧确认点）；否则「观察」。

纪律：裸空尾部风险高，触发后优先卖Call价差。

用法：
    python blowoff_short_daily.py
    python blowoff_short_daily.py --dry-run
    python blowoff_short_daily.py -c blowoff_short_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "blowoff_short_config.json"
HISTORY_FILE = ROOT / "blowoff_short_history.csv"

DEFAULT_CFG = {
    "watchlist": ["CDNL", "AMLX"],
    "scan_count": 250,
    "scan_top": 25,
    "auto_watch_top": 8,          # 自动把当日过热分 Top N 也纳入触发盯盘
    "trigger_drop_pct": -4.0,     # 破位大阴：当日跌幅阈值
    "trigger_clv": 0.40,          # 收盘强度阈值（收在中下部）
    "max_mcap_b": 50.0,           # 市值上限（十亿美元）：>此值的大盘剔除（不会暴涨暴跌）
    "min_amp_ratio": 2.0,         # 跌宕起伏门槛：近一年 高/低 ≥ 此倍数
    "outputs": {
        "today_json": "research/blowoff_short_today.json",
        "today_csv": "research/blowoff_short_today.csv",
        "ios_bundle": "ios/Resources/blowoff_short_today.json",
    },
    "notify": {"desktop": True},
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged = dict(DEFAULT_CFG)
    merged.update(cfg)
    out = dict(DEFAULT_CFG["outputs"])
    out.update(cfg.get("outputs") or {})
    merged["outputs"] = out
    return merged


def _clv(o, h, l, c):
    rng = h - l
    return 0.5 if rng <= 0 else ((c - l) - (h - c)) / rng


def check_trigger(t: str, df: pd.DataFrame, cfg: dict) -> dict | None:
    """对单只票检测破位大阴触发。返回盯盘行（含状态）。"""
    if df is None or df.empty or len(df) < 12:
        return None
    o, h, l, c, v = (df[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    last = float(c.iloc[-1])
    chg = last / float(c.iloc[-2]) - 1
    clv_now = _clv(float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), last)
    ma10 = float(c.rolling(10).mean().iloc[-1])
    vma = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float("nan")
    vol_x = float(v.iloc[-1]) / vma if np.isfinite(vma) and vma > 0 else float("nan")
    prev_low = float(l.iloc[-2])
    ret_5d = last / float(c.iloc[-6]) - 1 if len(c) > 6 else float("nan")
    ret_20d = last / float(c.iloc[-21]) - 1 if len(c) > 21 else float("nan")

    drop_th = float(cfg.get("trigger_drop_pct", -4.0)) / 100.0
    clv_th = float(cfg.get("trigger_clv", 0.40))

    broke = chg <= drop_th and clv_now <= clv_th and last < ma10
    near_break = (last < ma10) or (last < prev_low) or (clv_now <= clv_th)

    if broke:
        status = "✅破位可空"
    elif near_break:
        status = "⚠️临界(盯)"
    else:
        status = "观察"

    return {
        "代码": t,
        "现价": round(last, 2),
        "当日%": round(chg * 100, 1),
        "5日%": round(ret_5d * 100, 1) if np.isfinite(ret_5d) else None,
        "20日%": round(ret_20d * 100, 1) if np.isfinite(ret_20d) else None,
        "收盘强度": round(clv_now, 2),
        "量倍": round(vol_x, 2) if np.isfinite(vol_x) else None,
        "MA10": round(ma10, 2),
        "破MA10": "是" if last < ma10 else "否",
        "前日低": round(prev_low, 2),
        "状态": status,
    }


MIN_DVOL_M = 20.0
MIN_PRICE = 2.0
MIN_RUNUP = 0.20


def fetch_universe_with_mcap(count: int) -> tuple[list[str], dict[str, float]]:
    """合并涨幅+跌幅+活跃榜，返回 (symbols, 市值map[十亿美元])。"""
    from quant.screener import fetch_gainer_universe_live, fetch_yahoo_screen
    syms: list[str] = []
    seen: set[str] = set()
    mcap_b: dict[str, float] = {}

    def _ingest(df):
        if df is None or df.empty:
            return
        has_mc = "市值USD" in df.columns
        for _, r in df.iterrows():
            s = str(r["代码"]).upper()
            if s not in seen:
                seen.add(s)
                syms.append(s)
            if has_mc:
                try:
                    mc = float(r["市值USD"])
                    if mc == mc and mc > 0:
                        mcap_b[s] = mc / 1e9
                except (TypeError, ValueError):
                    pass

    try:
        _ingest(fetch_gainer_universe_live(count=count))
    except Exception:  # noqa: BLE001
        pass
    for preset in ("day_losers", "most_actives"):
        try:
            _ingest(fetch_yahoo_screen(preset, count=count))
        except Exception:  # noqa: BLE001
            continue
    return syms, mcap_b


def analyze_cand(t: str, df: pd.DataFrame, mcap_b: float | None, cfg: dict) -> dict | None:
    """单只候选分析：市值过滤 + 抛物线/派发分类 + 振幅(跌宕起伏)评分。"""
    if df is None or df.empty or len(df) < 30:
        return None
    o, h, l, c, v = (df[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    last = float(c.iloc[-1])
    if last < MIN_PRICE:
        return None

    # 市值过滤：已知且超上限 → 剔除（大盘不会暴涨暴跌）
    max_mcap = float(cfg.get("max_mcap_b", 50.0))
    if mcap_b is not None and mcap_b > max_mcap:
        return None

    vma = float(v.rolling(20).mean().iloc[-1])
    if not np.isfinite(vma) or vma <= 0:
        return None
    dvol_m = last * float(v.iloc[-1]) / 1e6
    if dvol_m < MIN_DVOL_M:
        return None

    vol_x = float(v.iloc[-1]) / vma
    chg = last / float(c.iloc[-2]) - 1
    clv_now = _clv(float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), last)
    ret_5d = last / float(c.iloc[-6]) - 1 if len(c) > 6 else np.nan
    ret_10d = last / float(c.iloc[-11]) - 1 if len(c) > 11 else np.nan
    ret_20d = last / float(c.iloc[-21]) - 1 if len(c) > 21 else np.nan
    runup = np.nanmax([ret_5d, ret_10d, ret_20d])
    if not np.isfinite(runup) or runup < MIN_RUNUP:
        return None

    # 振幅（跌宕起伏）：整段窗口（≈1年）高/低倍数 + 年化波动率
    hi_all = float(h.max())
    lo_all = float(l.min())
    amp_ratio = hi_all / lo_all if lo_all > 0 else np.nan
    min_amp = float(cfg.get("min_amp_ratio", 2.0))
    if np.isfinite(amp_ratio) and amp_ratio < min_amp:
        return None  # 太温吞，非跌宕起伏票
    rv = float(c.pct_change().tail(60).std() * np.sqrt(252)) if len(c) > 61 else np.nan

    hi20 = float(h.iloc[-21:].max())
    off_hi20 = last / hi20 - 1
    ma10 = float(c.rolling(10).mean().iloc[-1])
    below_ma10 = last < ma10
    bb_mid = float(c.rolling(20).mean().iloc[-1])
    bb_std = float(c.rolling(20).std().iloc[-1])
    bb_up = bb_mid + 2 * bb_std
    above_bb = last / bb_up - 1 if np.isfinite(bb_up) and bb_up > 0 else 0.0

    score = 0.0
    score += min(runup, 2.0) * 40
    score += max(0.0, above_bb) * 180
    score += max(0.0, vol_x - 1.0) * 8
    score += (0.5 - clv_now) * 35
    score += min(amp_ratio, 6.0) * 12          # 振幅越大越优先（跌宕起伏）
    if np.isfinite(rv):
        score += min(rv, 3.0) * 10             # 高波动加分
    if off_hi20 <= -0.05:
        score += (-off_hi20) * 70
    if below_ma10:
        score += 12

    if off_hi20 >= -0.03 and clv_now <= 0.5 and vol_x >= 1.5:
        kind = "A贴顶天量收弱"
    elif below_ma10 and off_hi20 <= -0.08:
        kind = "B已破位下行"
    elif off_hi20 >= -0.05 and clv_now <= 0.4:
        kind = "C冲高回落派发"
    elif off_hi20 >= -0.03:
        kind = "D贴顶过热(强)"
    else:
        kind = "E回撤中"

    return {
        "代码": t,
        "现价": round(last, 2),
        "市值B": round(mcap_b, 2) if mcap_b is not None else None,
        "当日%": round(chg * 100, 1),
        "5日%": round(ret_5d * 100, 1) if np.isfinite(ret_5d) else None,
        "20日%": round(ret_20d * 100, 1) if np.isfinite(ret_20d) else None,
        "振幅倍": round(amp_ratio, 1) if np.isfinite(amp_ratio) else None,
        "年化波动": round(rv, 2) if np.isfinite(rv) else None,
        "量倍": round(vol_x, 2),
        "收强": round(clv_now, 2),
        "距20高%": round(off_hi20 * 100, 1),
        "破MA10": "是" if below_ma10 else "否",
        "额M": round(dvol_m, 0),
        "类型": kind,
        "过热分": round(score, 0),
    }


def run_scan(cfg: dict) -> dict:
    from quant.providers import DataConfig, get_provider, reset_provider_cache

    count = int(cfg.get("scan_count", 250))
    top = int(cfg.get("scan_top", 25))
    watch_cfg = [str(s).upper() for s in (cfg.get("watchlist") or [])]

    # 单次抓取：把 watchlist 并进全市场universe，避免二次抓取被限流
    print(f"① 拉涨幅+跌幅+活跃榜 (count={count}) + watchlist + 市值 …")
    universe, mcap_b = fetch_universe_with_mcap(count)
    seen = set(universe)
    for w in watch_cfg:
        if w not in seen:
            seen.add(w)
            universe.append(w)
    print(f"② 候选 {len(universe)} 只，单次抓 ≈1年日线（算振幅） …")
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    start = (date.today() - timedelta(days=370)).isoformat()
    end = date.today().isoformat()
    # 先单独小批量抓 watchlist（保证拿到，避免大批量末尾被限流丢弃），再抓大 universe
    watch_batch = y.fetch_batch(watch_cfg, start, end) if watch_cfg else {}
    big_batch = y.fetch_batch(universe, start, end)
    batch = {**big_batch, **{k: v for k, v in watch_batch.items() if v is not None}}

    # 1) 全市场过热分候选（市值过滤 + 振幅偏好）
    rows = []
    for t in universe:
        d = batch.get(t)
        try:
            r = analyze_cand(t, d, mcap_b.get(t), cfg) if d is not None else None
        except Exception:  # noqa: BLE001
            r = None
        if r:
            rows.append(r)
    candidates = sorted(rows, key=lambda r: -(r.get("过热分") or 0))[:top]

    # 2) watchlist = 配置 watchlist + 当日过热分 Top N
    watch = list(watch_cfg)
    auto_n = int(cfg.get("auto_watch_top", 8))
    for r in candidates[:auto_n]:
        code = str(r.get("代码", "")).upper()
        if code and code not in watch:
            watch.append(code)

    # 3) 从同一批数据做 watchlist 触发检测
    triggers: list[dict] = []
    for t in watch:
        d = batch.get(t)
        try:
            row = check_trigger(t, d, cfg) if d is not None else None
        except Exception:  # noqa: BLE001
            row = None
        if row:
            triggers.append(row)
    order = {"✅破位可空": 0, "⚠️临界(盯)": 1, "观察": 2}
    triggers.sort(key=lambda r: (order.get(r["状态"], 9), -(r.get("20日%") or 0)))

    actionable = [r for r in triggers if r["状态"] == "✅破位可空"]
    return {
        "date": date.today().isoformat(),
        "strategy": "安飞士式做空(抛物线顶+破位触发)",
        "config": {
            "trigger_drop_pct": cfg.get("trigger_drop_pct"),
            "trigger_clv": cfg.get("trigger_clv"),
            "watchlist": cfg.get("watchlist"),
        },
        "scan_stats": {
            "候选": len(candidates),
            "盯盘": len(triggers),
            "可空": len(actionable),
        },
        "candidates": candidates,
        "triggers": triggers,
    }


def format_lines(plan: dict) -> list[str]:
    s = plan["scan_stats"]
    lines = [
        f"安飞士式做空 · {plan['date']}",
        f"  全市场过热候选 {s['候选']} · 盯盘 {s['盯盘']} · ✅可空 {s['可空']}",
        "",
    ]
    act = [r for r in plan["triggers"] if r["状态"] == "✅破位可空"]
    if act:
        lines.append("  🔴 破位确认（可空 / 优先卖Call价差）:")
        for r in act:
            lines.append(
                f"    {r['代码']} ${r['现价']} {r['当日%']}% 收强{r['收盘强度']} "
                f"破MA10:{r['破MA10']} (20日{r.get('20日%')}%)"
            )
        lines.append("")
    watch = [r for r in plan["triggers"] if r["状态"] != "✅破位可空"]
    if watch:
        lines.append("  👀 盯盘中（未触发，等破位）:")
        for r in watch[:12]:
            lines.append(
                f"    {r['代码']} ${r['现价']} {r['当日%']}% 收强{r['收盘强度']} "
                f"[{r['状态']}]"
            )
        lines.append("")
    if plan["candidates"]:
        lines.append("  📈 跌宕起伏·过热 Top（市值≤上限，按过热分）:")
        for c in plan["candidates"][:10]:
            mc = c.get("市值B")
            mc_s = f"{mc}B" if mc is not None else "?"
            lines.append(
                f"    {c.get('代码')} ${c.get('现价')} 市值{mc_s} 振幅{c.get('振幅倍')}x "
                f"20日{c.get('20日%')}% {c.get('类型')} 分{c.get('过热分')}"
            )
    lines.append("")
    lines.append("纪律：仅✅破位才进；止损放新高上方；裸空风险高，优先卖Call价差。非投资建议。")
    return lines


def save_outputs(plan: dict, cfg: dict) -> None:
    out = cfg.get("outputs") or {}
    today_json = ROOT / out.get("today_json", "research/blowoff_short_today.json")
    today_json.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, ensure_ascii=False, indent=2)
    today_json.write_text(payload, encoding="utf-8")
    ios = out.get("ios_bundle")
    if ios:
        ios_path = ROOT / ios
        ios_path.parent.mkdir(parents=True, exist_ok=True)
        ios_path.write_text(payload, encoding="utf-8")
    today_csv = out.get("today_csv")
    if today_csv and plan["triggers"]:
        pd.DataFrame(plan["triggers"]).to_csv(
            ROOT / today_csv, index=False, encoding="utf-8-sig")
    hist_row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "日期": plan["date"],
        "候选": plan["scan_stats"]["候选"],
        "盯盘": plan["scan_stats"]["盯盘"],
        "可空": plan["scan_stats"]["可空"],
    }
    if HISTORY_FILE.exists():
        pd.concat([pd.read_csv(HISTORY_FILE, encoding="utf-8-sig"),
                   pd.DataFrame([hist_row])]).to_csv(
            HISTORY_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([hist_row]).to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="安飞士式做空·每日扫描+破位触发")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plan = run_scan(cfg)
    print("\n".join(format_lines(plan)))

    if args.dry_run:
        return
    save_outputs(plan, cfg)
    print(f"\n已写入 {cfg['outputs'].get('today_json')}")
    st = plan["scan_stats"]
    if (cfg.get("notify") or {}).get("desktop", True):
        try:
            from scan_daily import desktop_notify
            desktop_notify(f"安飞士式做空 {plan['date']}",
                           f"✅可空 {st['可空']} · 盯盘 {st['盯盘']} · 候选 {st['候选']}")
        except Exception:  # noqa: BLE001
            pass

    # 手机推送：仅当有 ✅破位可空 触发时才推（避免噪音）
    act = [r for r in plan["triggers"] if r["状态"] == "✅破位可空"]
    push_when = str(cfg.get("push_when", "actionable"))  # actionable=仅可空 / always=每次
    should_push = bool(act) if push_when == "actionable" else True
    if should_push:
        try:
            from quant.mobile_push import push_mobile
            if act:
                head = "、".join(
                    f"{r['代码']} ${r['现价']}({r['当日%']}%)" for r in act[:5])
                title = f"🔴做空触发 {len(act)}只 · {plan['date']}"
                body = f"破位可空：{head}\n优先卖Call价差，止损放新高上方。"
            else:
                title = f"做空扫描 · {plan['date']}"
                body = f"今日无破位触发 · 盯盘{st['盯盘']} · 候选{st['候选']}"
            push_mobile(cfg, title, body)
        except Exception as e:  # noqa: BLE001
            print(f"[手机推送] 异常: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
