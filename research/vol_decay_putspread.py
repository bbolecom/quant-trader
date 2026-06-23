"""用期权吃波动率衰减：看跌价差 / 保护性做空 回测（UVIX / UVXY / VXX）。

背景：做多波动率 ETF 长期 contango 衰减 → 看空它有结构性优势，但裸空会被
vol spike 爆仓（如 2024/8/5 UVIX 单日 +84%）。解决办法是用**定义风险的期权结构**：

  - long_put     ：买 ATM 看跌（有限风险做空，最多亏权利金）
  - put_spread   ：买 ATM 看跌 + 卖深虚看跌（更便宜、历年更稳）
  - prot_call    ：空 1x + 买虚值看涨当尾部保险（保留较多上行）

期权用「已实现波动 × IV 倍数」近似定价（真实 IV 通常更高 → 买保护更贵，
脚本用 iv_mult 默认 1.3 做保守修正）。月度滚动，输出总收益/CAGR/逐年/最差月。

用法：
    python research/vol_decay_putspread.py
    python research/vol_decay_putspread.py --tickers UVIX UVXY VXX --hold 21
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache

R = 0.045
TRADING_DAYS = 252

INVERSE_VOL_TARGETS = {
    "UVIX": "2x 做多 VIX 短期期货",
    "UVXY": "1.5x 做多 VIX 短期期货",
    "VXX": "1x 做多 VIX 短期期货",
}

STRUCTURES = {
    "long_put": "买 ATM 看跌（有限风险）",
    "put_spread": "买 ATM 看跌 / 卖深虚看跌（价差）",
    "prot_call": "空1x + 买虚值看涨保护",
}


def _ncdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(S: float, K: float, T: float, r: float, sig: float, typ: str) -> float:
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if typ == "c" else (K - S))
    d1 = (log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    if typ == "c":
        return S * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)
    return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


@dataclass
class PutSpreadConfig:
    structure: str = "put_spread"
    hold: int = 21          # 滚动周期（交易日）
    lower_otm: float = 0.40  # 看跌价差下方腿在 -40%
    call_otm: float = 0.30   # 保护性看涨在 +30%
    iv_mult: float = 1.3     # IV ≈ 已实现波动 × 倍数（保守）
    rv_window: int = 20


def backtest_vol_decay(
    px: pd.Series,
    cfg: PutSpreadConfig | None = None,
) -> dict:
    """单标的、单结构回测。返回总收益/CAGR/最大回撤/逐年/最差月等。"""
    cfg = cfg or PutSpreadConfig()
    px = px.astype(float).dropna()
    if len(px) < cfg.rv_window + cfg.hold + 5:
        return {"error": "数据不足"}
    rv = px.pct_change().rolling(cfg.rv_window).std() * sqrt(TRADING_DAYS)
    idx = px.index
    T = cfg.hold / TRADING_DAYS

    cur = 1.0
    eq_pts: list[tuple] = []
    monthly_rets: list[float] = []
    yearly: dict[int, float] = {}
    i = cfg.rv_window
    while i < len(idx) - 1:
        j = min(i + cfg.hold, len(idx) - 1)
        S0 = float(px.iloc[i]); S1 = float(px.iloc[j])
        sig = float(rv.iloc[i]) if np.isfinite(rv.iloc[i]) else 1.5
        sig = min(max(sig * cfg.iv_mult, 0.8), 4.0)
        m = S1 / S0 - 1
        if cfg.structure == "long_put":
            prem = bs(S0, S0, T, R, sig, "p") / S0
            pnl = max(S0 - S1, 0) / S0 - prem
        elif cfg.structure == "prot_call":
            K = S0 * (1 + cfg.call_otm)
            prem = bs(S0, K, T, R, sig, "c") / S0
            pnl = -m + max(S1 - K, 0) / S0 - prem
        else:  # put_spread
            Kl = S0 * (1 - cfg.lower_otm)
            prem = (bs(S0, S0, T, R, sig, "p") - bs(S0, Kl, T, R, sig, "p")) / S0
            payoff = (max(S0 - S1, 0) - max(Kl - S1, 0)) / S0
            pnl = payoff - prem
        cur *= (1 + pnl)
        eq_pts.append((idx[j], cur))
        monthly_rets.append(pnl)
        yr = idx[j].year
        yearly[yr] = (yearly.get(yr, 1.0)) * (1 + pnl)
        i = j

    if not monthly_rets:
        return {"error": "无有效周期"}
    eq = pd.Series([p for _, p in eq_pts], index=[d for d, _ in eq_pts])
    rets = pd.Series(monthly_rets)
    years = max((eq.index[-1] - idx[cfg.rv_window]).days / 365.25, 0.1)
    total = cur - 1
    cagr = cur ** (1 / years) - 1
    dd = float((eq / eq.cummax() - 1).min())
    periods_per_year = TRADING_DAYS / cfg.hold
    sharpe = float(rets.mean() / rets.std() * sqrt(periods_per_year)) if rets.std() > 0 else 0.0
    return {
        "总收益": total,
        "CAGR": cagr,
        "最大回撤": dd,
        "夏普": sharpe,
        "胜率": float((rets > 0).mean()),
        "最差月": float(rets.min()),
        "周期数": len(rets),
        "逐年": {k: v - 1 for k, v in sorted(yearly.items())},
        "权益": eq,
    }


def compare_structures(
    tickers: list[str],
    *,
    start: str,
    end: str,
    cfg: PutSpreadConfig | None = None,
    structures: list[str] | None = None,
) -> pd.DataFrame:
    """跨标的 × 结构对比，返回汇总表（含逐年列）。"""
    cfg = cfg or PutSpreadConfig()
    structures = structures or list(STRUCTURES.keys())
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    rows: list[dict] = []
    all_years: set[int] = set()
    cache: dict[str, dict] = {}
    for t in tickers:
        try:
            df = y.fetch_history(t, start, end)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        px = df["Close"]
        for s in structures:
            c = PutSpreadConfig(**{**cfg.__dict__, "structure": s})
            res = backtest_vol_decay(px, c)
            if res.get("error"):
                continue
            cache[(t, s)] = res
            all_years.update(res["逐年"].keys())
    years_sorted = sorted(all_years)
    for (t, s), res in cache.items():
        row = {
            "标的": t,
            "结构": STRUCTURES[s],
            "总收益": res["总收益"],
            "CAGR": res["CAGR"],
            "最大回撤": res["最大回撤"],
            "夏普": res["夏普"],
            "胜率": res["胜率"],
            "最差月": res["最差月"],
        }
        for yr in years_sorted:
            row[str(yr)] = res["逐年"].get(yr, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["UVIX", "UVXY", "VXX"])
    p.add_argument("--start", default="2022-03-30")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--hold", type=int, default=21)
    p.add_argument("--lower-otm", type=float, default=0.40)
    p.add_argument("--iv-mult", type=float, default=1.3)
    args = p.parse_args()

    cfg = PutSpreadConfig(hold=args.hold, lower_otm=args.lower_otm, iv_mult=args.iv_mult)
    print(f"期权吃 vol 衰减回测  标的={args.tickers}  {args.start}~{args.end}")
    print(f"参数：滚动{args.hold}日 · 价差下腿-{args.lower_otm:.0%} · IV≈已实现×{args.iv_mult}\n")
    df = compare_structures(args.tickers, start=args.start, end=args.end, cfg=cfg)
    if df.empty:
        print("无结果（数据拉取失败？）")
        return
    pd.set_option("display.width", 220, "display.max_columns", 30)
    disp = df.copy()
    pct_cols = ["总收益", "CAGR", "最大回撤", "胜率", "最差月"] + [c for c in disp.columns if c.isdigit()]
    for c in pct_cols:
        disp[c] = disp[c].map(lambda x: f"{x*100:+.0f}%" if pd.notna(x) else "-")
    disp["夏普"] = disp["夏普"].map(lambda x: f"{x:.2f}")
    print(disp.to_string(index=False))

    out = ROOT / "research" / "vol_decay_putspread_results.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n结果已存 {out}")
    print("\n注：期权按已实现波动近似定价（×IV倍数保守修正）；真实 IV 更高、点差更大。")
    print("定义风险结构永不爆仓，但保护成本会吃掉部分衰减利润。仅供研究。")


if __name__ == "__main__":
    main()
