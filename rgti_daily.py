#!/usr/bin/env python3
"""RGTI 每日盯盘 · 信号自动判别（基于 research/rgti_predictability.py 的统计）。

研究结论（2024-2026, 501 日）：
  - 暴涨 ≈ 不可提前感知（消息驱动跳空）→ 只顺势，别埋伏、别做空强势
  - 暴跌 ≈ 部分可感知：NR7 波动收敛（尤其高位）是下跌前兆
  - 恐慌跳空(-5%+) 次日反弹期望 +5.1%（大涨率31%）→ 崩盘找反弹别追空
  - 超买(RSI>75)/极度乖离(>MA20 40%) 次日仍 +3~4% → 不要猜顶做空

每日收盘后跑一次，对每个 ticker 输出「今日是否触发可操作信号 + 动作」。

用法：
    python rgti_daily.py                 # 跑 + 桌面提醒(仅有信号时)
    python rgti_daily.py --dry-run       # 只打印不提醒
    python rgti_daily.py -t RGTI,IONQ    # 临时换标的
    python rgti_daily.py -c rgti_config.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import fetch_history
from scan_daily import desktop_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "rgti_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "tickers": ["RGTI"],
        "panic_gap_pct": -5.0,
        "panic_chg_pct": -8.0,
        "nr_window": 7,
        "nr_high_pos": 0.7,
        "rsi_overbought": 75,
        "rsi_oversold": 35,
        "dist_ma20_extended": 0.40,
        "climax_vol_x": 2.5,
        "lookback_days": 80,
        "notify": {"desktop": True, "only_on_signal": True},
        "outputs": {
            "today_json": "research/rgti_today.json",
            "history_csv": "research/rgti_signal_history.csv",
        },
    }


def _rsi(c: pd.Series, n: int = 14) -> float:
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def analyze(tk: str, df: pd.DataFrame, cfg: dict) -> dict:
    o = df["Open"].astype(float); h = df["High"].astype(float)
    l = df["Low"].astype(float); c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    last_c = float(c.iloc[-1]); prev_c = float(c.iloc[-2])
    rng = float(h.iloc[-1] - l.iloc[-1])
    clv = 0.0 if rng <= 0 else ((last_c - l.iloc[-1]) - (h.iloc[-1] - last_c)) / rng
    chg = last_c / prev_c - 1
    gap = float(o.iloc[-1]) / prev_c - 1
    vma20 = float(v.rolling(20).mean().iloc[-1])
    vol_x = float(v.iloc[-1]) / vma20 if vma20 > 0 else float("nan")
    nr_w = int(cfg.get("nr_window", 7))
    today_rng = h.iloc[-1] - l.iloc[-1]
    nr = bool(today_rng == (h - l).iloc[-nr_w:].min())
    hi20 = float(h.iloc[-20:].max()); lo20 = float(l.iloc[-20:].min())
    pos20 = (last_c - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5
    ma20 = float(c.rolling(20).mean().iloc[-1])
    dist_ma20 = last_c / ma20 - 1
    rsi = _rsi(c)
    streak_up = bool(c.iloc[-1] > c.iloc[-2])

    snap = {
        "代码": tk, "现价": round(last_c, 2), "当日%": round(chg * 100, 1),
        "跳空%": round(gap * 100, 1), "量比": round(vol_x, 2), "收盘强度": round(clv, 2),
        "RSI": round(rsi, 0), "位置20": round(pos20 * 100, 0),
        "距MA20%": round(dist_ma20 * 100, 1), "NR7": nr,
    }

    signals: list[dict] = []
    # 1) 恐慌跳空/重挫 → 找反弹（BUY/反弹）
    if gap <= cfg.get("panic_gap_pct", -5.0) / 100 or chg <= cfg.get("panic_chg_pct", -8.0) / 100:
        signals.append({
            "类型": "恐慌重挫→反弹", "方向": "偏多/抢反弹", "强度": "强",
            "依据": f"跳空{snap['跳空%']}% / 当日{snap['当日%']}%（历史次日均+5.1%, 大涨率31%）",
            "动作": "别追空；可小仓博次日反弹或卖put价差收高IV权利金",
        })
    # 2) NR7 收敛 → 变盘前兆（偏空）
    if nr:
        high = pos20 >= cfg.get("nr_high_pos", 0.7)
        signals.append({
            "类型": "NR7波动收敛" + ("·高位" if high else ""),
            "方向": "偏空/防跌", "强度": "强" if high else "中",
            "依据": f"7日最窄, 位置{snap['位置20']}%（NR7次日均-1.6%, 高位NR7均-7%）",
            "动作": "提防向下变盘；持多减仓/做空首选此形态" + ("（高位更危险）" if high else ""),
        })
    # 3) 超买/极度乖离 → 别做空（动量延续）
    if rsi >= cfg.get("rsi_overbought", 75) or dist_ma20 >= cfg.get("dist_ma20_extended", 0.40):
        signals.append({
            "类型": "超买/极度乖离", "方向": "动量延续·勿空", "强度": "中",
            "依据": f"RSI{snap['RSI']} / 距MA20 {snap['距MA20%']}%（次日仍+3~4%）",
            "动作": "别猜顶做空；顺势持有或等NR7再考虑减",
        })
    # 4) 天量阳线收强 → 延续偏多
    if vol_x >= cfg.get("climax_vol_x", 2.5) and clv >= 0.5 and chg > 0:
        signals.append({
            "类型": "天量阳线收强", "方向": "偏多·延续", "强度": "中",
            "依据": f"量比{snap['量比']} 收强{snap['收盘强度']}（次日均+3.9%, 无大跌样本）",
            "动作": "顺势，别反向做空",
        })

    verdict = "🟡 无明确信号（弱势盘整/观望）"
    if signals:
        verdict = " ｜ ".join(f"{s['强度']}·{s['类型']}({s['方向']})" for s in signals)
    return {"snapshot": snap, "signals": signals, "verdict": verdict}


def run(cfg: dict, tickers: list[str]) -> dict:
    look = int(cfg.get("lookback_days", 80))
    start = (date.today() - timedelta(days=look)).isoformat()
    end = date.today().isoformat()
    results = []
    for tk in tickers:
        try:
            df = fetch_history(tk, start=start, end=end)
        except Exception as e:  # noqa: BLE001
            results.append({"代码": tk, "错误": str(e)})
            continue
        if df is None or len(df) < 25:
            results.append({"代码": tk, "错误": "数据不足"})
            continue
        results.append(analyze(tk, df, cfg))
    return {"date": date.today().isoformat(), "results": results}


def format_lines(out: dict) -> list[str]:
    lines = [f"RGTI 盯盘 · {out['date']}", ""]
    for r in out["results"]:
        if "错误" in r:
            lines.append(f"  ⚠️ {r['代码']}: {r['错误']}")
            continue
        s = r["snapshot"]
        lines.append(
            f"  {s['代码']} ${s['现价']} 当日{s['当日%']}% 跳空{s['跳空%']}% "
            f"量比{s['量比']} RSI{s['RSI']:.0f} 位置{s['位置20']:.0f}% 距MA20{s['距MA20%']}% NR7={s['NR7']}"
        )
        if not r["signals"]:
            lines.append("     🟡 无明确信号 → 观望（弱势盘整，无可操作边际）")
        for sig in r["signals"]:
            lines.append(f"     {'🟢' if '多' in sig['方向'] else '🔴' if '空' in sig['方向'] else '⚪'} "
                         f"[{sig['强度']}] {sig['类型']} — {sig['方向']}")
            lines.append(f"        依据: {sig['依据']}")
            lines.append(f"        动作: {sig['动作']}")
    lines += ["", "法则: 暴涨追不了别埋伏 | 超买别空 | NR7防跌找空 | 恐慌砸盘找反弹。"]
    return lines


def build_app_feed(out: dict) -> dict:
    """转成 iOS app json_generic 视图可渲染的 picks 形状（点击代码可开图）。"""
    picks: list[dict] = []
    actionable = 0
    for r in out["results"]:
        if "snapshot" not in r:
            continue
        s = r["snapshot"]
        sigs = r["signals"]
        if sigs:
            actionable += 1
        top = sigs[0] if sigs else None
        dir_emoji = "🟡 观望"
        if top:
            dir_emoji = ("🟢 " if "多" in top["方向"] else "🔴 " if "空" in top["方向"] else "⚪ ") + top["方向"]
        picks.append({
            "代码": s["代码"],
            "现价": s["现价"],
            "当日%": s["当日%"],
            "判定": dir_emoji,
            "信号": " / ".join(x["类型"] for x in sigs) if sigs else "无信号·观望",
            "动作": top["动作"] if top else "弱势盘整，无可操作边际",
            "RSI": s["RSI"],
            "位置20%": s["位置20"],
            "距MA20%": s["距MA20%"],
            "量比": s["量比"],
            "NR7": "是" if s["NR7"] else "否",
        })
    return {
        "date": out["date"],
        "title": "量子板块盯盘",
        "scan_stats": {"标的": len(picks), "有信号": actionable},
        "picks": picks,
        "法则": "暴涨追不了别埋伏 | 超买别空 | NR7防跌找空 | 恐慌砸盘找反弹",
    }


def save_outputs(out: dict, cfg: dict) -> None:
    o = cfg.get("outputs") or {}
    tj = ROOT / o.get("today_json", "research/rgti_today.json")
    hc = ROOT / o.get("history_csv", "research/rgti_signal_history.csv")
    tj.parent.mkdir(parents=True, exist_ok=True)
    tj.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # app feed（picks 形状）+ iOS bundle 同步
    feed = build_app_feed(out)
    feed_doc = json.dumps(feed, ensure_ascii=False, indent=2)
    for key in ("app_json", "ios_bundle"):
        rel = o.get(key)
        if not rel:
            continue
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(feed_doc, encoding="utf-8")
    rows = []
    for r in out["results"]:
        if "snapshot" not in r:
            continue
        s = r["snapshot"]
        rows.append({
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "日期": out["date"],
            **s, "信号数": len(r["signals"]),
            "信号": "; ".join(x["类型"] for x in r["signals"]) or "无",
        })
    if rows:
        df = pd.DataFrame(rows)
        if hc.exists():
            df = pd.concat([pd.read_csv(hc, encoding="utf-8-sig"), df], ignore_index=True)
        df.to_csv(hc, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="RGTI 每日盯盘信号")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("-t", "--tickers", type=str, default=None, help="逗号分隔，覆盖配置")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers \
        else [str(t).upper() for t in cfg.get("tickers", ["RGTI"])]

    out = run(cfg, tickers)
    save_outputs(out, cfg)
    print("\n".join(format_lines(out)))

    notify = cfg.get("notify") or {}
    has_sig = any(r.get("signals") for r in out["results"] if "signals" in r)
    if not args.dry_run and notify.get("desktop", True):
        if has_sig or not notify.get("only_on_signal", True):
            n = sum(len(r.get("signals", [])) for r in out["results"] if "signals" in r)
            tickers_with = [r["snapshot"]["代码"] for r in out["results"] if r.get("signals")]
            msg = f"{'/'.join(tickers_with) or '—'} 触发 {n} 个信号" if has_sig else "今日无信号"
            desktop_notify(f"RGTI盯盘 {out['date']}", msg)


if __name__ == "__main__":
    main()
