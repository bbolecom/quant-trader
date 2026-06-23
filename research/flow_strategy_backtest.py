"""资金流向可操作策略 · 组合回测 + 今日选股。

用法：
    python research/flow_strategy_backtest.py
    python research/flow_strategy_backtest.py --years 5
    python research/flow_strategy_backtest.py --today
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from quant.flow_strategy import (
    FlowStrategyParams,
    build_signal_panel,
    load_strategy_config,
    run_portfolio_backtest,
    today_actionable_picks,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

CFG_PATH = ROOT / "flow_strategy_config.json"


def _load_cfg() -> dict:
    if CFG_PATH.exists():
        return json.loads(CFG_PATH.read_text(encoding="utf-8"))
    return {}


def run_backtest(cfg: dict | None = None) -> dict:
    cfg = cfg or _load_cfg()
    bt = cfg.get("backtest") or {}
    params = FlowStrategyParams.from_dict(cfg)
    years = float(bt.get("years", 3.0))
    quick = bool(bt.get("quick", False))
    capital = float(bt.get("initial_capital", 100_000))

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(years * 365) + 150)).isoformat()
    pool = LIQUID100 if quick else GAINER_MOMENTUM
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    batch = yahoo.fetch_batch(pool, start, end)

    panel = build_signal_panel(batch, spy, params)
    result = run_portfolio_backtest(panel, params, initial_capital=capital)
    if "error" in result:
        return result

    outs = cfg.get("outputs") or {}
    summary = {
        "generated": date.today().isoformat(),
        "strategy": params.name,
        "years": years,
        "universe": len(pool),
        "tickers_with_data": len(batch),
        "signal_rows": len(panel),
        "累计收益率": result["累计收益率"],
        "年化收益率": result["年化收益率"],
        "夏普比率": result["夏普比率"],
        "最大回撤": result["最大回撤"],
        "日胜率": result["日胜率"],
        "笔胜率": result["笔胜率"],
        "交易天数": result["交易天数"],
        "总笔数": result["总笔数"],
        "期末权益": result["final_equity"],
        "params": {
            "long_patterns": sorted(params.long_patterns),
            "short_patterns": sorted(params.short_patterns),
            "long_top_n": params.long_top_n,
            "short_top_n": params.short_top_n,
        },
    }

    spath = ROOT / outs.get("summary_json", "research/flow_strategy_backtest.json")
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    trades = result.get("交易明细")
    if trades is not None and not trades.empty:
        tpath = ROOT / outs.get("trades_csv", "research/flow_strategy_trades.csv")
        trades.to_csv(tpath, index=False, encoding="utf-8-sig")

    curve = result.get("权益曲线")
    if curve is not None and not curve.empty:
        cpath = ROOT / outs.get("equity_csv", "research/flow_strategy_equity.csv")
        curve.to_csv(cpath, index=False, encoding="utf-8-sig")

    result["summary_doc"] = summary
    return result


def run_today(cfg: dict | None = None) -> dict:
    cfg = cfg or _load_cfg()
    params = FlowStrategyParams.from_dict(cfg)
    outs = cfg.get("outputs") or {}

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=150)).isoformat()

    from quant.screener import fetch_gainer_universe_live
    from research.liquid_tier_a_scan import build_candidate_pool

    snap = fetch_gainer_universe_live(count=int(cfg.get("gainer_count", 250)))
    tickers = set(snap["代码"].astype(str).tolist()) if not snap.empty else set()
    tickers.update(build_candidate_pool(use_broad=True))
    batch = yahoo.fetch_batch(sorted(tickers), start, end)
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)

    picks = today_actionable_picks(batch, spy, params)
    doc = {
        "日期": end,
        "strategy": params.name,
        "picks": picks.to_dict(orient="records") if not picks.empty else [],
        "可交易": len(picks),
    }
    jpath = ROOT / outs.get("today_json", "research/flow_strategy_today.json")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def analyze_trades_by_pattern(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, r in trades_df.iterrows():
        for p in str(r.get("规律", "")).split("、"):
            p = p.strip()
            if not p:
                continue
            rows.append({
                "pattern": p,
                "方向": r.get("方向"),
                "策略收益%": float(r.get("策略收益%", 0)),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby("pattern")["策略收益%"]
    out = pd.DataFrame({
        "笔数": g.count(),
        "胜率": g.apply(lambda x: float((x > 0).mean())),
        "均收益%": g.mean(),
    })
    return out.sort_values("均收益%", ascending=False)


def print_backtest_report(result: dict) -> None:
    if "error" in result:
        print(f"回测失败: {result['error']}")
        return
    s = result.get("summary_doc") or {}
    print(f"\n{'=' * 60}")
    print(f"资金流向可操作策略回测 · {s.get('strategy')} · {s.get('years')}年")
    print(f"{'=' * 60}")
    print(f"股票池 {s.get('universe')} · 信号事件 {s.get('signal_rows')}")
    print(f"累计收益 {s.get('累计收益率', 0):.1%} · 年化 {s.get('年化收益率', 0):.1%}")
    print(f"夏普 {s.get('夏普比率', 0):.2f} · 最大回撤 {s.get('最大回撤', 0):.1%}")
    print(f"日胜率 {s.get('日胜率', 0):.1%} · 笔胜率 {s.get('笔胜率', 0):.1%}")
    print(f"交易天数 {s.get('交易天数')} · 总笔数 {s.get('总笔数')} · 期末 ${s.get('期末权益', 0):,.0f}")
    print(f"\n做多规律: {', '.join(s.get('params', {}).get('long_patterns', []))}")
    print(f"做空规律: {', '.join(s.get('params', {}).get('short_patterns', []))}")
    trades = result.get("交易明细")
    if trades is not None and not trades.empty:
        br = analyze_trades_by_pattern(trades)
        if not br.empty:
            print("\n【分规律表现】")
            for pid, row in br.iterrows():
                print(f"  {pid:12} 笔{int(row['笔数']):4d} 胜率{row['胜率']:.0%} 均{row['均收益%']:+.2f}%")
    print(f"\n→ {ROOT / 'research' / 'flow_strategy_backtest.json'}")


def print_today(doc: dict) -> None:
    print(f"\n可操作策略 · {doc.get('strategy')} · {doc.get('日期')}")
    picks = doc.get("picks") or []
    if not picks:
        print("  今日无符合规则的标的（空仓）")
        return
    for p in picks:
        print(
            f"  {'📈' if p['方向']=='做多' else '📉'} {p['代码']} {p['策略动作']} "
            f"涨{p.get('涨幅%')}% 量比{p.get('量比')} [{p.get('规律')}]"
        )
        print(f"      {p.get('选股理由', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="资金流向可操作策略回测")
    parser.add_argument("--years", type=float, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--today", action="store_true", help="仅输出今日可执行选股")
    args = parser.parse_args()

    cfg = _load_cfg()
    if args.years is not None:
        cfg.setdefault("backtest", {})["years"] = args.years
    if args.quick:
        cfg.setdefault("backtest", {})["quick"] = True

    if args.today:
        doc = run_today(cfg)
        print_today(doc)
        return

    result = run_backtest(cfg)
    print_backtest_report(result)


if __name__ == "__main__":
    main()
