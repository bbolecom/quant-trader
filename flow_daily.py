#!/usr/bin/env python3
"""资金流向每日选股 · 操盘痕迹扫描。

识别量价轨迹中的「人为拉升 / 人为出货」规律，输出做多/做空/回避三池。

规律目录见 quant/capital_flow.py FLOW_CATALOG。

用法：
    python flow_daily.py
    python flow_daily.py --dry-run
    python flow_daily.py --catalog   # 仅打印规律目录
    python flow_daily.py -c flow_daily_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant.capital_flow import (
    build_daily_picks,
    format_catalog,
    load_flow_stats,
    scan_universe_flow,
    stats_hint_for_pattern,
)
from quant.flow_options import enrich_picks_with_live_chain
from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.screener import fetch_gainer_universe_live
from research.income_engine import get_regime
from research.liquid_tier_a_scan import build_candidate_pool

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "flow_daily_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _fetch_spy_series(yahoo, days: int = 120) -> pd.Series:
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    return spy


def _append_sec_only_alerts(
    scan_df: pd.DataFrame,
    alert_map: dict,
    batch: dict,
    *,
    spy: pd.Series,
    regime_bull: bool,
    spy_1d: float,
    min_dvol: float,
) -> pd.DataFrame:
    """仅有 SEC 融资公告、未进量价扫描的标的。"""
    from quant.capital_flow import enrich_flow_row, assess_flow_patterns

    if not alert_map:
        return scan_df
    existing = set(scan_df["代码"].astype(str).tolist()) if not scan_df.empty else set()
    rows = scan_df.to_dict("records") if not scan_df.empty else []
    for tk, alert in alert_map.items():
        if tk in existing:
            continue
        df = batch.get(tk)
        if df is None or df.empty:
            rows.append({
                "代码": tk,
                "现价": None,
                "信号": "做空",
                "策略动作": "买Put价差",
                "选股理由": f"SEC {alert.get('表格')} · {alert.get('关键词')} · {alert.get('公告日')}",
                "上涨规律": "—",
                "下跌规律": "D_OFFERING",
                "SEC融资": "是",
            })
            continue
        feat = enrich_flow_row(df, spy)
        if not feat or feat.get("dvol_m", 0) < min_dvol:
            continue
        feat["代码"] = tk
        res = assess_flow_patterns(feat, spy_bull=regime_bull, spy_1d_pct=spy_1d)
        down = "D_OFFERING"
        if res["下跌规律"]:
            down = "、".join(h["规律ID"] for h in res["下跌规律"])
            if "D_OFFERING" not in down:
                down += "、D_OFFERING"
        reason = f"SEC {alert.get('表格')} · {alert.get('关键词')} · {alert.get('公告日')}"
        rows.append({
            "代码": tk,
            "现价": round(float(feat["现价"]), 2),
            "信号": "做空",
            "策略动作": "买Put价差",
            "选股理由": reason,
            "上涨规律": "、".join(h["规律ID"] for h in res["上涨规律"]) or "—",
            "下跌规律": down,
            "涨幅%": feat.get("涨幅%"),
            "量比": round(float(feat.get("量比", 0)), 2),
            "收盘强度": round(float(feat.get("close_strength", 0)), 2),
            "成交额M": round(float(feat.get("dvol_m", 0)), 1),
            "5日涨%": round(float(feat.get("涨幅5d%", 0)), 1),
            "20日涨%": round(float(feat.get("涨幅20d%", 0)), 1),
            "MA50": "上" if feat.get("above_ma50") else "下",
            "SEC融资": "是",
        })
    return pd.DataFrame(rows) if rows else scan_df


def _attach_stats_hints(picks: list[dict], stats: dict) -> list[dict]:
    out: list[dict] = []
    for p in picks:
        row = dict(p)
        hints: list[str] = []
        for key in ("上涨规律", "下跌规律"):
            raw = str(row.get(key, ""))
            for pid in raw.replace("—", "").split("、"):
                pid = pid.strip()
                if not pid:
                    continue
                h = stats_hint_for_pattern(pid, stats)
                if h:
                    hints.append(f"{pid}:{h}")
        if hints:
            row["回测参考"] = "；".join(hints[:3])
        out.append(row)
    return out


def run_flow_scan(cfg: dict) -> dict:
    today = date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    min_dvol = float(cfg.get("min_dvol_m", 30))
    min_price = float(cfg.get("min_price", 3.0))
    long_n = int(cfg.get("long_top_n", 3))
    short_n = int(cfg.get("short_top_n", 3))
    avoid_n = int(cfg.get("avoid_top_n", 5))
    quick = bool(cfg.get("quick", False))
    gainer_count = int(cfg.get("gainer_count", 250))

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    regime = get_regime(yahoo)
    spy = _fetch_spy_series(yahoo)
    spy_1d = float(spy.iloc[-1] / spy.iloc[-2] - 1) * 100 if len(spy) >= 2 else 0.0

    # 股票池：涨幅榜 + 广谱流动性池
    snap = fetch_gainer_universe_live(count=gainer_count)
    tickers = set(snap["代码"].astype(str).tolist()) if not snap.empty else set()
    pool = build_candidate_pool(use_broad=not quick, max_names=80 if quick else 0)
    tickers.update(pool)
    tickers = sorted(tickers)

    start = (date.today() - timedelta(days=120)).isoformat()
    end = today
    batch = yahoo.fetch_batch(tickers, start, end)

    scan_df = scan_universe_flow(
        batch,
        spy_close=spy,
        spy_bull=regime.bull,
        spy_1d_pct=spy_1d,
        min_dvol_m=min_dvol,
        min_price=min_price,
    )

    sec_cfg = cfg.get("sec_filings") or {}
    filings_df = pd.DataFrame()
    alert_map: dict = {}
    if sec_cfg.get("enabled", True):
        try:
            filings_df = search_dilution_filings(days=int(sec_cfg.get("days", 5)))
            alert_map = dilution_alert_map(filings_df)
            scan_df = merge_offering_into_scan(scan_df, alert_map)
            scan_df = _append_sec_only_alerts(
                scan_df, alert_map, batch,
                spy=spy, regime_bull=regime.bull, spy_1d=spy_1d, min_dvol=min_dvol,
            )
        except Exception as exc:  # noqa: BLE001
            sec_cfg = {"error": str(exc)}

    flow_stats = load_flow_stats()
    pools = build_daily_picks(scan_df, long_top_n=long_n, short_top_n=short_n, avoid_top_n=avoid_n)

    picks: list[dict] = []
    for pool_name, label, mod in [
        ("long", "做多", "资金流入·拉升"),
        ("short", "做空", "资金流出·出货"),
        ("avoid", "回避", "陷阱·过热"),
    ]:
        df = pools.get(pool_name, pd.DataFrame())
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            sig = r["信号"]
            act = r["策略动作"]
            if sig == "做空" or "Put" in str(act) or "D_OFFERING" in str(r.get("下跌规律", "")):
                status = "待定价"
            elif pool_name in ("long", "short"):
                status = "可开仓"
            else:
                status = "观望"
            picks.append({
                "模块": mod,
                "账户": "资金流向",
                "代码": r["代码"],
                "状态": status,
                "方向": r["信号"],
                "策略动作": r["策略动作"],
                "选股理由": r["选股理由"],
                "上涨规律": r.get("上涨规律", "—"),
                "下跌规律": r.get("下跌规律", "—"),
                "涨幅%": r.get("涨幅%"),
                "量比": r.get("量比"),
                "收盘强度": r.get("收盘强度"),
                "成交额M": r.get("成交额M"),
                "5日涨%": r.get("5日涨%"),
                "20日涨%": r.get("20日涨%"),
                "SEC融资": r.get("SEC融资", ""),
            })

    picks = _attach_stats_hints(picks, flow_stats)
    account = float(cfg.get("account_size", 10_000))
    picks = enrich_picks_with_live_chain(picks, account, cfg.get("live_chain"))
    actionable = sum(1 for p in picks if p["状态"] == "可开仓")
    doc = {
        "选股日期": today,
        "选股时间": now,
        "philosophy": cfg.get(
            "philosophy",
            "量价轨迹识操盘：堆量拉升做多，冲顶出货回避/做空",
        ),
        "regime": {
            "bull": regime.bull,
            "label": regime.label,
            "spy": round(regime.spy, 2),
            "ma50": round(regime.ma50, 2),
            "spy_1d%": round(spy_1d, 2),
            "playbook": "牛市开做多池；弱市主做空/回避池",
        },
        "scan_stats": {
            "股票池数量": len(tickers),
            "命中数量": len(scan_df),
            "做多池": len(pools.get("long", [])),
            "做空池": len(pools.get("short", [])),
            "回避池": len(pools.get("avoid", [])),
            "SEC融资公告": len(alert_map),
        },
        "sec_filings": filings_df.to_dict(orient="records") if not filings_df.empty else [],
        "pattern_stats": flow_stats.get("patterns") or {},
        "summary": {
            "总条目": len(picks),
            "可开仓": actionable,
            "观望": len(picks) - actionable,
            "是否空仓日": actionable == 0,
            "大盘": regime.label,
        },
        "catalog": [p.id for p in __import__("quant.capital_flow", fromlist=["FLOW_CATALOG"]).FLOW_CATALOG],
        "picks": picks,
        "full_scan": scan_df.to_dict(orient="records") if not scan_df.empty else [],
    }
    return doc


def save_outputs(doc: dict, cfg: dict) -> None:
    outs = cfg.get("outputs") or {}
    jpath = ROOT / outs.get("today_json", "research/flow_daily_today.json")
    cpath = ROOT / outs.get("today_csv", "research/flow_daily_today.csv")
    hpath = ROOT / outs.get("history_csv", "flow_daily_history.csv")

    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(doc.get("picks") or [])
    if not df.empty:
        df.insert(0, "选股日期", doc["选股日期"])
        df.insert(1, "选股时间", doc["选股时间"])
        df.to_csv(cpath, index=False, encoding="utf-8-sig")
        header = not hpath.exists()
        df.to_csv(hpath, mode="a", header=header, index=False, encoding="utf-8-sig")


def print_report(doc: dict) -> None:
    s = doc["summary"]
    reg = doc.get("regime") or {}
    stats = doc.get("scan_stats") or {}
    print(f"\n{'=' * 58}")
    print(f"资金流向每日选股  {doc['选股日期']}  {doc['选股时间']}")
    print(f"{'=' * 58}")
    print(f"理念：{doc.get('philosophy', '')}")
    print(f"大盘：{reg.get('label', '')}  SPY {reg.get('spy')} / MA50 {reg.get('ma50')}  1日{reg.get('spy_1d%')}%")
    print(
        f"扫描 {stats.get('股票池数量', 0)} 只 → 命中 {stats.get('命中数量', 0)} 只 | "
        f"做多{stats.get('做多池', 0)} 做空{stats.get('做空池', 0)} 回避{stats.get('回避池', 0)} | "
        f"SEC融资{stats.get('SEC融资公告', 0)}"
    )
    print(f"可开仓 {s['可开仓']} ｜ 观望 {s['观望']} ｜ {'空仓日 ✓' if s['是否空仓日'] else '有信号'}")

    pstats = doc.get("pattern_stats") or {}
    if pstats:
        print("\n【规律回测胜率 · 次日】")
        for pid, row in sorted(pstats.items(), key=lambda x: -x[1].get("win_rate_1d", 0)):
            print(
                f"  {pid:12} {row.get('name', '')[:14]:14} "
                f"胜率{row.get('win_rate_1d', 0):.0%} n={row.get('sample_n')} "
                f"均{row.get('mean_ret_1d_pct', 0):+.2f}%"
            )

    sec_rows = doc.get("sec_filings") or []
    if sec_rows:
        print("\n【SEC 融资/稀释公告 · 近几日】")
        for r in sec_rows[:8]:
            print(f"  ⚠ {r.get('代码')} {r.get('表格')} {r.get('公告日')} · {r.get('关键词')}")

    print()

    sections = [
        ("【做多 · 资金流入/堆量拉升】", "做多"),
        ("【做空 · 出货/回吐】", "做空"),
        ("【回避 · 陷阱/过热】", "回避"),
    ]
    picks = doc.get("picks") or []
    for title, sig in sections:
        sub = [p for p in picks if p.get("方向") == sig]
        print(title)
        if not sub:
            print("  今日无命中")
        else:
            for p in sub:
                mark = "✅" if p.get("状态") == "可开仓" else "⏸"
                print(
                    f"  {mark} {p['代码']} {p['策略动作']} "
                    f"涨{p.get('涨幅%')}% 量比{p.get('量比')} 收强{p.get('收盘强度')}"
                )
                if p.get("期权结构"):
                    print(
                        f"      期权 {p.get('期权结构')} @{p.get('到期')} "
                        f"×{p.get('建议张数', 0)}张 最大亏${p.get('最大亏损$', '')}"
                    )
                print(f"      {p.get('选股理由', '')}")
                if p.get("回测参考"):
                    print(f"      回测: {p.get('回测参考')}")
                print(f"      ↑{p.get('上涨规律', '—')}  ↓{p.get('下跌规律', '—')}")
        print()

    print("规律目录：python flow_daily.py --catalog")
    print(f"→ {ROOT / 'research' / 'flow_daily_today.json'}")


def notify(doc: dict, cfg: dict) -> None:
    notify_cfg = cfg.get("notify") or {}
    if not notify_cfg.get("desktop", True):
        return
    sec_rows = doc.get("sec_filings") or []
    sec_alert = notify_cfg.get("sec_alert", True) and len(sec_rows) > 0
    if notify_cfg.get("only_when_action") and doc["summary"]["可开仓"] == 0 and not sec_alert:
        return
    try:
        from scan_daily import desktop_notify, email_notify
    except ImportError:
        return
    s = doc["summary"]
    reg = doc.get("regime") or {}
    parts: list[str] = []
    if s["可开仓"] > 0:
        title = f"资金流向 · {s['可开仓']}可开仓"
        for p in doc.get("picks") or []:
            if p.get("状态") != "可开仓":
                continue
            line = f"{p.get('代码')} {p.get('策略动作')}"
            if p.get("期权结构"):
                line += f" {p.get('期权结构')}×{p.get('建议张数', 0)}"
            parts.append(line)
        body = "；".join(parts[:5])
    else:
        title = "资金流向 · 今日观望"
        body = reg.get("label", "")
    if sec_alert:
        sec_line = "SEC融资:" + "、".join(
            f"{r.get('代码')}" for r in sec_rows[:5]
        )
        body = (body + " · " + sec_line) if body else sec_line
        if s["可开仓"] == 0:
            title = f"⚠ SEC融资公告 {len(sec_rows)}条"
    desktop_notify(title, body)
    if notify_cfg.get("email", {}).get("enabled"):
        email_notify(notify_cfg["email"], title, body)


def main() -> None:
    parser = argparse.ArgumentParser(description="资金流向每日选股")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CFG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--catalog", action="store_true", help="打印规律目录")
    parser.add_argument("--backtest", action="store_true", help="先跑规律回测再扫描")
    args = parser.parse_args()

    if args.catalog:
        for line in format_catalog():
            print(line)
        return

    cfg = load_config(Path(args.config))
    if args.backtest:
        from research.flow_pattern_backtest import run_backtest, print_summary

        print("运行规律回测…")
        doc_bt = run_backtest(
            years=float(cfg.get("backtest_years", 3)),
            quick=bool(cfg.get("quick", False)),
            min_dvol_m=float(cfg.get("min_dvol_m", 30)),
        )
        print_summary(doc_bt)
        print()

    cfg = load_config(Path(args.config))
    doc = run_flow_scan(cfg)
    print_report(doc)
    if not args.dry_run:
        save_outputs(doc, cfg)
        if not args.no_notify:
            notify(doc, cfg)


if __name__ == "__main__":
    main()
