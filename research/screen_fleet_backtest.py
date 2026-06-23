#!/usr/bin/env python3
"""批量回测 5×每日选股舰队，写入 research/screen_fleet_stats.json。

用法:
    python research/screen_fleet_backtest.py
    python research/screen_fleet_backtest.py --years 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from quant.daily_screen_fleet import (
    FLEET_CONFIG,
    STATS_JSON,
    account_strategy_label,
    backtest_account,
    fleet_accounts,
    is_csp_account,
    load_fleet_config,
    save_fleet_stats,
    tickers_for_preset,
)
from quant.daily_screen_fleet import preset_for_account
from quant.data import fetch_history_batch


def fetch_pool_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    return fetch_history_batch(tickers, start, end)


def run(years: float = 5.0, account_size: float = 10_000.0) -> dict:
    cfg = load_fleet_config()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(years * 365) + 400)).isoformat()

    # 合并各 screen 账户所需标的，一次拉取（CSP 账户用锚点，无需行情）
    all_tickers: list[str] = []
    for acct in fleet_accounts(cfg):
        if is_csp_account(acct):
            tk = str(acct.get("ticker", "")).upper()
            if tk and tk not in all_tickers:
                all_tickers.append(tk)
            continue
        preset = preset_for_account(acct)
        for t in tickers_for_preset(preset, acct):
            if t not in all_tickers:
                all_tickers.append(t)

    data: dict = {}
    if all_tickers:
        print(f"拉取 {len(all_tickers)} 只标的 ({start} ~ {end}) …")
        data = fetch_pool_data(all_tickers, start, end)
        print(f"有效 {len(data)} 只")
    else:
        print("全部为 CSP 锚点账户，跳过行情拉取")

    results: list[dict] = []
    for acct in fleet_accounts(cfg):
        label = acct.get("label", acct["id"])
        strat_name = account_strategy_label(acct)
        if is_csp_account(acct):
            sub = {str(acct.get("ticker", "")).upper(): data.get(str(acct.get("ticker", "")).upper())}
            sub = {k: v for k, v in sub.items() if v is not None}
            print(f"  锚点 {label} · {strat_name} …")
        else:
            preset = preset_for_account(acct)
            pool = tickers_for_preset(preset, acct)
            sub = {t: data[t] for t in pool if t in data}
            print(f"  回测 {label} · {preset.name} ({len(sub)} 只) …")
        res = backtest_account(
            acct, sub, years=years,
            initial_capital=account_size,
        )
        if "error" in res:
            print(f"    ⚠ {res['error']}")
            results.append({
                "account_id": acct["id"],
                "label": label,
                "role": acct.get("role", ""),
                "description": acct.get("description", ""),
                "preset_id": acct.get("preset_id", "csp"),
                "preset_name": strat_name,
                "strategy_type": acct.get("strategy_type", "screen"),
                "error": res["error"],
            })
        else:
            s = res["stats"]
            print(
                f"    年化={s['ann_return']:.1%} 回撤={s['max_dd']:.1%} "
                f"胜率={s.get('trade_win_rate', float('nan')):.0%}"
            )
            bt = res.pop("backtest")
            _ = bt
            results.append(res)

    doc = {
        "name": cfg.get("name", "5×圣杯舰队"),
        "account_size": float(cfg.get("account_size", account_size)),
        "years": years,
        "generated": date.today().isoformat(),
        "data_end": end,
        "target_profile": cfg.get("target_profile"),
        "note": "CSP 账户指标来自 triple_target_scan 锚点" if any(is_csp_account(a) for a in fleet_accounts(cfg)) else "",
        "accounts": results,
    }
    save_fleet_stats(doc)
    print(f"\n✓ 已写入 {STATS_JSON}")
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description="5×每日选股舰队批量回测")
    p.add_argument("--years", type=float, default=5.0)
    p.add_argument("--account-size", type=float, default=10_000.0)
    args = p.parse_args()
    doc = run(years=args.years, account_size=args.account_size)
    print(json.dumps({
        "accounts": [
            {
                "label": a.get("label"),
                "preset": a.get("preset_name"),
                "stats": a.get("stats"),
                "error": a.get("error"),
            }
            for a in doc["accounts"]
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
