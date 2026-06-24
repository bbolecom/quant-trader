#!/usr/bin/env python3
"""RGTI 类似股票池（超高波动 · 题材/动量小盘）。

RGTI（Rigetti，量子计算）画像：低~中价、年化波动 ≥0.8、年均「单日异动>10%」≥12 天、
成交额 ≥$50M（流动性合格）。大盘蓝筹（NVDA/AAPL 波动 0.3~0.5）自动被排除。

提供两种用法：
  1. 精选静态名单 RGTI_LIKE_SEED（量子/太空/核能/AI/矿企等题材高波票）。
  2. 数据驱动 build_rgti_like_pool()：按特征从全市场候选里筛选，可随时重建并缓存。

用法：
    python3 quant/volatile_pool.py                 # 用种子+全市场重建并缓存
    python3 quant/volatile_pool.py --from-seed-only # 仅校验种子名单
    python3 quant/volatile_pool.py --min-vol 0.9 --min-big-days 15
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POOL_JSON = ROOT / "research" / "rgti_like_pool.json"

# 精选种子：与 RGTI 同类的超高波动题材/动量票（量子/太空/核能/AI/矿企/EV/生科/meme）
RGTI_LIKE_SEED: list[str] = [
    # 量子计算
    "RGTI", "QBTS", "IONQ", "QUBT", "ARQQ", "LAES", "QMCO", "QSI",
    # 太空 / eVTOL / 无人机
    "ASTS", "LUNR", "RKLB", "ACHR", "JOBY", "SPCE", "RDW", "RCAT", "KULR",
    # 核能 / 新能源动量
    "OKLO", "SMR", "NNE", "OUST", "EOSE", "PLUG", "FCEL", "BLNK", "CHPT", "RUN",
    # AI / 数据小盘
    "SOUN", "BBAI", "AI", "TEM", "RXRX", "SERV", "INOD", "POET", "PRCT",
    # 加密矿企 / 高 beta
    "MARA", "RIOT", "CLSK", "BTBT", "BITF", "HUT", "IREN", "WULF", "CIFR", "CORZ", "HIVE", "APLD",
    # 生科高波 / meme
    "SAVA", "VKTX", "NVAX", "ATAI", "AVXL", "IBRX", "VERA",
    # EV / 中概高波
    "NIO", "LCID", "RIVN", "XPEV", "LI", "NVTS",
    # meme / 散户动量
    "GME", "AMC", "KOSS", "OPEN", "CVNA", "UPST", "AFRM", "DJT", "SOFI",
]


@dataclass
class RgtiLikeFilter:
    """RGTI 相似度筛选阈值（基于 ~2 年日线）。"""

    min_realized_vol: float = 0.80      # 年化波动率下限
    min_big_move_days: float = 12.0     # 年均「单日 |涨跌|≥10%」天数下限
    big_move_pct: float = 0.10
    min_dvol_m: float = 50.0            # 60日均成交额下限（百万美元）
    min_price: float = 3.0
    max_price: float = 250.0
    min_history_days: int = 120


def profile_ticker(df: pd.DataFrame, filt: RgtiLikeFilter | None = None) -> dict | None:
    """计算单只标的的 RGTI 画像指标，返回 dict（不判定）或 None（数据不足）。"""
    filt = filt or RgtiLikeFilter()
    if df is None or len(df) < filt.min_history_days:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    ret = close.pct_change()
    rv = float(ret.std() * np.sqrt(252))
    big = float((ret.abs() >= filt.big_move_pct).mean() * 252)
    dvol_m = float((close * vol).tail(60).mean() / 1e6)
    price = float(close.iloc[-1])
    return {
        "代码": str(df.attrs.get("ticker", "")),
        "价": round(price, 2),
        "年化波动": round(rv, 2),
        "年均大异动天": round(big, 1),
        "成交额M": round(dvol_m, 1),
    }


def is_rgti_like(prof: dict, filt: RgtiLikeFilter | None = None) -> bool:
    filt = filt or RgtiLikeFilter()
    return (
        prof["年化波动"] >= filt.min_realized_vol
        and prof["年均大异动天"] >= filt.min_big_move_days
        and prof["成交额M"] >= filt.min_dvol_m
        and filt.min_price <= prof["价"] <= filt.max_price
    )


def build_rgti_like_pool(
    candidates: list[str] | None = None,
    *,
    filt: RgtiLikeFilter | None = None,
    include_seed: bool = True,
    lookback_days: int = 730,
) -> pd.DataFrame:
    """从候选池拉历史 → 按 RGTI 画像筛选，返回命中明细（按波动降序）。"""
    from quant.providers import DataConfig, get_provider, reset_provider_cache

    filt = filt or RgtiLikeFilter()
    pool: list[str] = []
    if include_seed:
        pool.extend(RGTI_LIKE_SEED)
    if candidates:
        pool.extend(candidates)
    else:
        cache = ROOT / "research" / "gainer_universe_cache.json"
        if cache.exists():
            pool.extend(json.loads(cache.read_text()))
    tickers = sorted(dict.fromkeys([t.strip().upper() for t in pool if t and t.strip() and t != "SPY"]))

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(tickers, start, end)

    rows: list[dict] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        df.attrs["ticker"] = tk.upper()
        prof = profile_ticker(df, filt)
        if prof and is_rgti_like(prof, filt):
            prof["代码"] = tk.upper()
            prof["种子"] = tk.upper() in set(RGTI_LIKE_SEED)
            rows.append(prof)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("年化波动", ascending=False).reset_index(drop=True)


def load_pool() -> list[str]:
    """读取已缓存的 RGTI 类似池；无缓存则回退种子名单。"""
    if POOL_JSON.exists():
        try:
            doc = json.loads(POOL_JSON.read_text())
            return doc.get("tickers") or list(RGTI_LIKE_SEED)
        except json.JSONDecodeError:
            pass
    return list(RGTI_LIKE_SEED)


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 RGTI 类似股票池")
    ap.add_argument("--from-seed-only", action="store_true", help="只校验种子名单，不扫全市场")
    ap.add_argument("--min-vol", type=float, default=0.80)
    ap.add_argument("--min-big-days", type=float, default=12.0)
    ap.add_argument("--min-dvol-m", type=float, default=50.0)
    args = ap.parse_args()

    filt = RgtiLikeFilter(
        min_realized_vol=args.min_vol,
        min_big_move_days=args.min_big_days,
        min_dvol_m=args.min_dvol_m,
    )
    cands = list(RGTI_LIKE_SEED) if args.from_seed_only else None
    print("拉取行情并按 RGTI 画像筛选…")
    df = build_rgti_like_pool(cands, filt=filt, include_seed=True)
    if df.empty:
        print("无命中标的。")
        return

    print("\n" + "=" * 70)
    print(f"RGTI 类似池 · {len(df)} 只（年化波动≥{filt.min_realized_vol} · "
          f"年均>10%异动≥{filt.min_big_move_days:.0f}天 · 成交额≥${filt.min_dvol_m:.0f}M）")
    print("=" * 70)
    print(f"{'代码':<7}{'价':>9}{'年化波动':>9}{'大异动天':>9}{'成交额M':>10}{'  种子'}")
    for _, r in df.iterrows():
        print(f"{r['代码']:<7}{r['价']:>9.2f}{r['年化波动']:>9.2f}{r['年均大异动天']:>9.1f}"
              f"{r['成交额M']:>10.0f}{'   ✓' if r['种子'] else ''}")

    tickers = df["代码"].tolist()
    POOL_JSON.write_text(json.dumps({
        "updated": date.today().isoformat(),
        "filter": {
            "min_realized_vol": filt.min_realized_vol,
            "min_big_move_days": filt.min_big_move_days,
            "min_dvol_m": filt.min_dvol_m,
            "price_range": [filt.min_price, filt.max_price],
        },
        "count": len(tickers),
        "tickers": tickers,
        "detail": df.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {POOL_JSON}（{len(tickers)} 只）")


if __name__ == "__main__":
    main()
