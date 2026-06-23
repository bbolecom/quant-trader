"""日历价差(calendar spread)候选扫描器。

日历价差 = 卖近月 + 买远月（同行权价），赌标的到近月到期时**钉在行权价附近**、
靠近月时间价值衰减更快赚钱。它**怕大幅移动、爱横盘**，且远月腿越贵越占资金。
所以理想标的 = 低价(小账户买得起) + 横盘震荡(钉得住) + 适中偏高 IV(权利金厚) + 流动性好。

筛选：
  价格 ∈ [min_price, max_price]（默认 $20–60）
  日均成交额 ≥ min_dvol_m（流动性，默认 $20M）
  年化 RV ∈ [min_rv, max_rv]（默认 40%–120%：有权利金但别太疯）
  效率比 ER ≤ max_er（默认 0.35；ER 低=来回磨=适合日历）

对每只算 ATM 日历（卖 7 天 / 买 37 天）的净付权利金、占账户比例、盈利区间，
按"横盘度 × 波动溢价"打分排序。

用法：
    python research/calendar_scan.py
    python research/calendar_scan.py --account 10000 --max-price 80
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant import decline_income as di
from quant.vol_decay import DEFAULT_VRP
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100


def load_universe() -> list[str]:
    uni = set(GAINER_MOMENTUM) | set(LIQUID100)
    cache = ROOT / "research" / "gainer_universe_cache.json"
    if cache.exists():
        try:
            uni |= set(json.loads(cache.read_text()))
        except Exception:  # noqa: BLE001
            pass
    return sorted(t for t in uni if t and t not in {"SPY", "QQQ", "XLE", "XLF"})


def efficiency_ratio(close: pd.Series, n: int = 30) -> float:
    """Kaufman 效率比：|净移动| / Σ|日变动|。低=来回磨(适合日历)，高=单边趋势。"""
    c = close.iloc[-(n + 1):].astype(float)
    if len(c) < n + 1:
        return 1.0
    net = abs(float(c.iloc[-1] - c.iloc[0]))
    path = float(c.diff().abs().sum())
    return net / path if path > 0 else 1.0


def calendar_cost(S: float, iv: float, dn: int = 7, df: int = 37) -> tuple[float, float, float]:
    """ATM 日历净付（看涨腿）/股 + 近月1σ移动 + 远月单腿成本/股。"""
    K = round(S)
    Tn, Tf = dn / 365, df / 365
    debit = di.bs_call_price(S, K, Tf, iv) - di.bs_call_price(S, K, Tn, iv)
    far_leg = di.bs_call_price(S, K, Tf, iv)
    sigW = iv * sqrt(dn / 365)
    return debit, sigW, far_leg


def run(account: float, min_price: float, max_price: float, min_dvol_m: float,
        min_rv: float, max_rv: float, max_er: float, topn: int) -> None:
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=200)).isoformat()
    uni = load_universe()
    print(f"票池 {len(uni)} 只｜抓取 {start} ~ {end} …")
    batch = y.fetch_batch(uni, start, end)
    batch = {t: d for t, d in batch.items() if d is not None and len(d) > 60}
    print(f"有效 {len(batch)} 只\n")

    rows = []
    for t, df_ in batch.items():
        c = df_["Close"].astype(float).dropna()
        v = df_["Volume"].astype(float)
        if len(c) < 65:
            continue
        S = float(c.iloc[-1])
        if not (min_price <= S <= max_price):
            continue
        dvol = float((c * v).iloc[-20:].mean())
        if dvol < min_dvol_m * 1e6:
            continue
        rv = float(c.pct_change(fill_method=None).rolling(20).std().iloc[-1] * sqrt(252))
        if not np.isfinite(rv) or not (min_rv <= rv * 100 <= max_rv):
            continue
        er = efficiency_ratio(c, 30)
        if er > max_er:
            continue
        ma50 = float(c.rolling(50).mean().iloc[-1])
        iv = rv * (1 + DEFAULT_VRP)
        debit, sigW, far_leg = calendar_cost(S, iv)
        debit_c = debit * 100
        score = rv * (1 - er)  # 高波动溢价 × 横盘度
        rows.append({
            "代码": t, "现价": round(S, 2), "RV%": round(rv * 100, 0),
            "效率比ER": round(er, 2), "偏离MA50%": round((S / ma50 - 1) * 100, 1),
            "成交额M": round(dvol / 1e6, 0),
            "日历净付$": round(debit_c, 0), "占账户%": round(debit_c / account * 100, 1),
            "盈利区±%": round(sigW * 100, 0), "评分": round(score * 100, 1),
        })

    if not rows:
        print("没有符合条件的标的。可放宽 --max-price 或 --max-er。")
        return
    df = pd.DataFrame(rows).sort_values("评分", ascending=False).head(topn)
    print("=" * 92)
    print(f"日历价差候选（横盘+高IV+低价）｜账户 ${account:,.0f}｜价 [{min_price:.0f},{max_price:.0f}] "
          f"RV [{min_rv:.0f},{max_rv:.0f}]% ER≤{max_er}")
    print("=" * 92)
    print(df.to_string(index=False))
    print("\n读法：")
    print("  · 效率比ER 越低=越横盘(钉得住)，最适合日历；偏离MA50 越接近0 越居中。")
    print("  · 日历净付$ = 卖7天/买37天 ATM 看涨日历的成本(1张)；占账户% 越小越玩得动。")
    print("  · 盈利区±% = 近月 1σ 周波动，股价在此范围内波动最有利；超出太多日历会亏。")
    print("  · 评分 = RV ×(1−ER)：兼顾权利金厚度与横盘度，越高越优先。")
    print("\n注：BS 近似估值(IV=RV×1.3)；实盘以券商期权链与真实期限结构为准；财报周不做。")


def main() -> None:
    p = argparse.ArgumentParser(description="日历价差候选扫描")
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--min-price", type=float, default=20.0)
    p.add_argument("--max-price", type=float, default=60.0)
    p.add_argument("--min-dvol-m", type=float, default=20.0)
    p.add_argument("--min-rv", type=float, default=40.0)
    p.add_argument("--max-rv", type=float, default=120.0)
    p.add_argument("--max-er", type=float, default=0.35)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()
    run(args.account, args.min_price, args.max_price, args.min_dvol_m,
        args.min_rv, args.max_rv, args.max_er, args.top)


if __name__ == "__main__":
    main()
