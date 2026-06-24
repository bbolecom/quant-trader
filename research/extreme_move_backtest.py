"""Backtest liquid extreme +/-10% US stock move patterns.

Usage examples:
    python research/extreme_move_backtest.py --pool liquid100
    python research/extreme_move_backtest.py --pool momentum --start 2020-01-01 --end 2025-12-31
    python research/extreme_move_backtest.py --tickers NVDA,TSLA,COIN,MSTR
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.extreme_move_strategy import (  # noqa: E402
    ExtremeMoveConfig,
    scan_universe_events,
    simulate_event_trades,
    summarize_event_strategy,
)
from quant.providers import DataConfig, get_provider, reset_provider_cache  # noqa: E402
from quant.screener import fetch_broad_universe, fetch_sp500_tickers  # noqa: E402
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100  # noqa: E402


def _parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _resolve_universe(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return _parse_tickers(args.tickers)
    if args.pool == "liquid100":
        return LIQUID100
    if args.pool == "momentum":
        return GAINER_MOMENTUM
    if args.pool == "sp500":
        return fetch_sp500_tickers()
    if args.pool == "surge_drop":
        from quant.surge_drop_pool import load_pool as load_surge_drop_pool
        return load_surge_drop_pool()

    cache = ROOT / "research" / "extreme_move_universe_cache.json"
    if cache.exists() and args.max_tickers <= 0:
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass
    tickers = fetch_broad_universe(screen_count=args.screen_count, extra=LIQUID100)
    cache.write_text(json.dumps(tickers, ensure_ascii=False, indent=2))
    return tickers


def _print_summary(summary: dict) -> None:
    if summary.get("error"):
        print(f"错误：{summary['error']}")
        return
    print("\n" + "=" * 72)
    print("极端涨跌事件策略回测")
    print("=" * 72)
    print(f"交易次数     {summary['交易次数']}")
    print(f"交易日数     {summary['交易日数']}")
    print(f"累计收益率   {summary['累计收益率']:+.1%}")
    print(f"年化收益率   {summary['年化收益率']:+.1%}")
    print(f"最大回撤     {summary['最大回撤']:+.1%}")
    print(f"胜率         {summary['胜率']:.1%}")
    print(f"平均单笔     {summary['平均单笔收益']:+.2%}")
    print(f"盈亏比       {summary['盈亏比']:.2f}")
    print("\n目标检验（历史回测，不代表未来）：")
    print(f"年化 >= 100%  {'是' if summary['目标年化>=100%'] else '否'}")
    print(f"胜率 >= 90%   {'是' if summary['目标胜率>=90%'] else '否'}")
    print(f"回撤 > -10%   {'是' if summary['目标回撤>-10%'] else '否'}")


def _build_config(args: argparse.Namespace, **overrides) -> ExtremeMoveConfig:
    values = {
        "event_threshold_pct": args.threshold,
        "min_price": args.min_price,
        "min_dollar_vol_m": args.min_dollar_vol_m,
        "min_vol_ratio": args.min_vol_ratio,
        "hold_days": max(1, args.hold_days),
        "stop_loss_pct": max(0.0, args.stop_loss),
        "take_profit_pct": max(0.0, args.take_profit),
        "max_positions_per_day": max(1, args.top_per_day),
        "mode": args.mode,
    }
    values.update(overrides)
    return ExtremeMoveConfig(**values)


def _run_strategy(
    data: dict[str, pd.DataFrame],
    cfg: ExtremeMoveConfig,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = scan_universe_events(data, cfg, start=args.start, end=args.end)
    trades = simulate_event_trades(data, events, cfg, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    summary = summarize_event_strategy(trades, start=args.start, end=args.end)
    return events, trades, summary


def _score_summary(summary: dict) -> tuple:
    if summary.get("error"):
        return (-1, -999.0, -999.0, -999.0, 0)
    hit_count = int(summary["目标年化>=100%"]) + int(summary["目标胜率>=90%"]) + int(summary["目标回撤>-10%"])
    # Prefer candidates that satisfy more hard goals, then drawdown control,
    # then win rate, then CAGR, then enough samples.
    return (
        hit_count,
        float(summary["最大回撤"]),
        float(summary["胜率"]),
        float(summary["年化收益率"]),
        int(summary["交易次数"]),
    )


def _optimize_configs(
    data: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    candidates = []
    modes = [args.mode] if args.mode != "both" else ["surge", "drop", "both"]
    grid = product(
        modes,
        [args.min_dollar_vol_m, max(args.min_dollar_vol_m, 250.0), max(args.min_dollar_vol_m, 500.0)],
        [1.5, 2.0, 3.0],
        [0.72, 0.80, 0.88],
        [0.35, 0.45, 0.55],
        [1, 2, 3],
        [0.04, 0.06, 0.08],
        [0.08, 0.12, 0.16],
    )

    best_payload: tuple[pd.DataFrame, pd.DataFrame, dict] | None = None
    best_score: tuple | None = None
    seen: set[tuple] = set()
    for mode, dvol, vol_ratio, surge_strength, drop_strength, hold_days, stop_loss, take_profit in grid:
        key = (mode, dvol, vol_ratio, surge_strength, drop_strength, hold_days, stop_loss, take_profit)
        if key in seen:
            continue
        seen.add(key)
        cfg = _build_config(
            args,
            mode=mode,
            min_dollar_vol_m=dvol,
            min_vol_ratio=vol_ratio,
            surge_min_close_strength=surge_strength,
            drop_min_close_strength=drop_strength,
            hold_days=hold_days,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
        )
        events, trades, summary = _run_strategy(data, cfg, args)
        row = {
            "mode": mode,
            "min_dollar_vol_m": dvol,
            "min_vol_ratio": vol_ratio,
            "surge_min_close_strength": surge_strength,
            "drop_min_close_strength": drop_strength,
            "hold_days": hold_days,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "events": int(len(events)),
            **{k: v for k, v in summary.items() if k != "error"},
        }
        if summary.get("error"):
            row["error"] = summary["error"]
        candidates.append(row)

        score = _score_summary(summary)
        if best_score is None or score > best_score:
            best_score = score
            best_payload = (events, trades, summary)

    results = pd.DataFrame(candidates)
    if not results.empty:
        def _goal_col(name: str) -> pd.Series:
            if name not in results.columns:
                return pd.Series(False, index=results.index)
            return results[name].fillna(False).astype(bool)

        results["_达标数"] = (
            _goal_col("目标年化>=100%").astype(int)
            + _goal_col("目标胜率>=90%").astype(int)
            + _goal_col("目标回撤>-10%").astype(int)
        )
        results = results.sort_values(
            ["_达标数", "最大回撤", "胜率", "年化收益率", "交易次数"],
            ascending=[False, False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    if best_payload is None:
        return pd.DataFrame(), pd.DataFrame(), results, {"error": "没有可用候选。"}
    events, trades, summary = best_payload
    return events, trades, results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="美股暴涨/暴跌 10% 事件策略回测")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--pool", choices=["broad", "sp500", "liquid100", "momentum", "surge_drop"], default="momentum")
    parser.add_argument("--tickers", default="", help="逗号分隔代码；设置后忽略 --pool")
    parser.add_argument("--max-tickers", type=int, default=0, help="限制标的数量，便于快速试跑")
    parser.add_argument("--screen-count", type=int, default=250, help="broad 模式每个 Yahoo 榜单数量")
    parser.add_argument("--mode", choices=["both", "surge", "drop"], default="both")
    parser.add_argument("--threshold", type=float, default=10.0, help="暴涨/暴跌阈值，默认10%%")
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-vol-m", type=float, default=100.0)
    parser.add_argument("--min-vol-ratio", type=float, default=2.0)
    parser.add_argument("--hold-days", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=0.06)
    parser.add_argument("--take-profit", type=float, default=0.12)
    parser.add_argument("--top-per-day", type=int, default=3)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=15.0)
    parser.add_argument("--out-prefix", default=str(ROOT / "research" / "extreme_move"))
    parser.add_argument("--optimize", action="store_true", help="网格搜索更接近年化/胜率/回撤目标的参数")
    args = parser.parse_args()

    tickers = _resolve_universe(args)
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]
    tickers = sorted(dict.fromkeys(tickers))
    print(f"股票池 {len(tickers)} 只，区间 {args.start} ~ {args.end}，开始拉取 Yahoo 日线…")

    reset_provider_cache()
    provider = get_provider(DataConfig(provider="yahoo"))
    data = provider.fetch_batch(tickers, args.start, args.end)
    print(f"有效行情 {len(data)} 只，开始扫描 10% 极端涨跌事件…")

    if args.optimize:
        print("开始参数网格搜索（目标：年化>=100%、胜率>=90%、回撤>-10%）…")
        events, trades, optim, summary = _optimize_configs(data, args)
    else:
        cfg = _build_config(args)
        events, trades, summary = _run_strategy(data, cfg, args)
        optim = pd.DataFrame()

    _print_summary(summary)

    event_path = Path(f"{args.out_prefix}_events.csv")
    trade_path = Path(f"{args.out_prefix}_trades.csv")
    optim_path = Path(f"{args.out_prefix}_optimize.csv")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trade_path, index=False, encoding="utf-8-sig")
    if not optim.empty:
        optim.to_csv(optim_path, index=False, encoding="utf-8-sig")
    print(f"\n事件明细已存 {event_path}")
    print(f"交易明细已存 {trade_path}")
    if not optim.empty:
        print(f"参数寻优已存 {optim_path}")
        show_cols = [
            "mode", "min_dollar_vol_m", "min_vol_ratio", "hold_days", "stop_loss_pct",
            "take_profit_pct", "交易次数", "年化收益率", "胜率", "最大回撤", "_达标数",
        ]
        print("\nTop 10 参数候选：")
        print(optim.head(10)[show_cols].to_string(index=False))

    if not trades.empty:
        print("\n最近 10 笔交易：")
        cols = ["选股日期", "代码", "类型名", "净收益率", "退出原因", "说明"]
        print(trades.tail(10)[cols].to_string(index=False, formatters={"净收益率": "{:+.2%}".format}))


if __name__ == "__main__":
    main()
