"""5 年「爱暴涨也爱暴跌」股票池 · 画像、构建与加载。

筛选逻辑（数据驱动，非主观名单）：
  - 流动性：60 日均成交额 ≥ min_dvol_m
  - 暴涨偏好：年均单日涨幅 ≥ surge_pct 的天数
  - 暴跌偏好：年均单日跌幅 ≤ -drop_pct 的天数
  - 综合分：暴涨天 + 暴跌天 + 双向均衡度 + 年化波动

用法：
    python3 quant/surge_drop_pool.py
    python3 quant/surge_drop_pool.py --years 5 --top 60
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POOL_JSON = ROOT / "research" / "surge_drop_pool.json"
POOL_CSV = ROOT / "research" / "surge_drop_pool.csv"

# 已知高波动种子（优先纳入候选，最终仍过数据门槛）
SURGE_DROP_SEED: list[str] = [
    "SMCI", "MSTR", "COIN", "PLTR", "HOOD", "SOFI", "RIVN", "LCID", "NIO", "XPEV",
    "MARA", "RIOT", "CLSK", "IREN", "WULF", "CORZ", "GME", "AMC", "CVNA", "UPST",
    "RGTI", "QBTS", "IONQ", "QUBT", "SOUN", "AI", "ASTS", "RKLB", "ACHR", "JOBY",
    "VKTX", "NVAX", "MRNA", "SOXL", "TQQQ", "SNDK", "ARM", "MU", "WDC", "STX",
    "DKNG", "SNAP", "PINS", "ROKU", "APP", "DUOL", "HIMS", "OPEN", "DJT", "BBBY",
]


@dataclass
class SurgeDropFilter:
    """5 年暴涨/暴跌池筛选门槛。"""

    years: int = 5
    surge_pct: float = 7.0          # 暴涨日阈值（%）
    drop_pct: float = 7.0           # 暴跌日阈值（%）
    min_surge_days_yr: float = 3.0  # 年均暴涨天数
    min_drop_days_yr: float = 3.0   # 年均暴跌天数
    min_extreme_days_yr: float = 8.0  # 年均 |涨跌|≥阈值 合计
    min_dvol_m: float = 50.0
    min_price: float = 3.0
    max_price: float = 500.0
    min_history_days: int = 400
    min_realized_vol: float = 0.45
    top_n: int = 80


def profile_ticker(
    df: pd.DataFrame,
    *,
    ticker: str = "",
    filt: SurgeDropFilter | None = None,
) -> dict | None:
    """计算单票 5 年暴涨/暴跌画像。"""
    filt = filt or SurgeDropFilter()
    if df is None or len(df) < filt.min_history_days:
        return None

    df = df[~df.index.duplicated(keep="last")].sort_index()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    ret = close.pct_change()
    valid = ret.dropna()
    if len(valid) < filt.min_history_days:
        return None

    years = max(len(valid) / 252.0, 0.5)
    surge_th = filt.surge_pct / 100.0
    drop_th = -filt.drop_pct / 100.0

    surge_n = int((valid >= surge_th).sum())
    drop_n = int((valid <= drop_th).sum())
    extreme_n = surge_n + drop_n
    surge_yr = surge_n / years
    drop_yr = drop_n / years
    extreme_yr = extreme_n / years

    rv = float(valid.std() * np.sqrt(252))
    dvol_m = float((close * vol).tail(60).mean() / 1e6)
    price = float(close.iloc[-1])

    # 15% 级别极端日（研究用）
    big_surge_n = int((valid >= 0.15).sum())
    big_drop_n = int((valid <= -0.15).sum())

    # 双向均衡：暴涨与暴跌都活跃
    if surge_yr > 0 and drop_yr > 0:
        balance = min(surge_yr, drop_yr) / max(surge_yr, drop_yr)
    else:
        balance = 0.0

    # 综合分：极端频率 + 波动 + 双向性
    score = extreme_yr * 1.0 + rv * 8.0 + balance * 5.0

    return {
        "代码": ticker.upper() if ticker else "",
        "价": round(price, 2),
        "年化波动": round(rv, 3),
        "年均暴涨天": round(surge_yr, 2),
        "年均暴跌天": round(drop_yr, 2),
        "年均极端天": round(extreme_yr, 2),
        "暴涨总天数": surge_n,
        "暴跌总天数": drop_n,
        "15%暴涨天": big_surge_n,
        "15%暴跌天": big_drop_n,
        "双向均衡": round(balance, 3),
        "成交额M": round(dvol_m, 1),
        "综合分": round(score, 2),
        "样本年数": round(years, 2),
    }


def passes_filter(prof: dict, filt: SurgeDropFilter | None = None) -> bool:
    filt = filt or SurgeDropFilter()
    return (
        prof["年均暴涨天"] >= filt.min_surge_days_yr
        and prof["年均暴跌天"] >= filt.min_drop_days_yr
        and prof["年均极端天"] >= filt.min_extreme_days_yr
        and prof["成交额M"] >= filt.min_dvol_m
        and filt.min_price <= prof["价"] <= filt.max_price
        and prof["年化波动"] >= filt.min_realized_vol
    )


def build_surge_drop_pool(
    candidates: list[str] | None = None,
    *,
    filt: SurgeDropFilter | None = None,
    include_seed: bool = True,
) -> pd.DataFrame:
    """拉取历史并筛选暴涨/暴跌偏好池。"""
    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from quant.screener import fetch_broad_universe
    from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100
    from research.liquid_tier_a_scan import build_candidate_pool

    filt = filt or SurgeDropFilter()
    pool: list[str] = []
    if include_seed:
        pool.extend(SURGE_DROP_SEED)
    pool.extend(LIQUID100)
    pool.extend(GAINER_MOMENTUM)
    pool.extend(build_candidate_pool(use_broad=True, max_names=0))
    if candidates:
        pool.extend(candidates)
    else:
        try:
            pool.extend(fetch_broad_universe(screen_count=300, extra=LIQUID100))
        except Exception:  # noqa: BLE001
            cache = ROOT / "research" / "gainer_universe_cache.json"
            if cache.exists():
                pool.extend(json.loads(cache.read_text()))

    tickers = sorted(dict.fromkeys(t.strip().upper() for t in pool if t and t.strip() and t != "SPY"))

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=filt.years * 365 + 120)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(tickers, start, end)

    rows: list[dict] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        prof = profile_ticker(df, ticker=tk, filt=filt)
        if prof is None:
            continue
        prof["种子"] = tk.upper() in set(SURGE_DROP_SEED)
        prof["入选"] = passes_filter(prof, filt)
        rows.append(prof)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    selected = out[out["入选"]].sort_values("综合分", ascending=False)
    if filt.top_n > 0:
        selected = selected.head(filt.top_n)
    # 保留全量明细供研究，标记 top
    out["Top池"] = out["代码"].isin(selected["代码"].tolist())
    return out.sort_values(["Top池", "综合分"], ascending=[False, False]).reset_index(drop=True)


def save_pool(df: pd.DataFrame, filt: SurgeDropFilter, *, meta: dict | None = None) -> dict:
    """写入 JSON + CSV，返回文档。"""
    selected = df[df["Top池"]].copy() if "Top池" in df.columns else df[df.get("入选", False)]
    tickers = selected["代码"].tolist()
    doc = {
        "updated": date.today().isoformat(),
        "description": "5年爱暴涨也爱暴跌 · 高流动性极端波动池",
        "filter": asdict(filt),
        "count": len(tickers),
        "tickers": tickers,
        "summary": {
            "candidates_scanned": int(len(df)),
            "avg_extreme_days_yr": round(float(selected["年均极端天"].mean()), 2) if not selected.empty else 0,
            "avg_surge_days_yr": round(float(selected["年均暴涨天"].mean()), 2) if not selected.empty else 0,
            "avg_drop_days_yr": round(float(selected["年均暴跌天"].mean()), 2) if not selected.empty else 0,
            "avg_realized_vol": round(float(selected["年化波动"].mean()), 3) if not selected.empty else 0,
        },
        "detail": selected.to_dict(orient="records"),
        "all_profiles": df.to_dict(orient="records"),
    }
    if meta:
        doc["research"] = meta

    POOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    POOL_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")
    return doc


def load_pool() -> list[str]:
    """读取已缓存池；无缓存则回退种子。"""
    if POOL_JSON.exists():
        try:
            doc = json.loads(POOL_JSON.read_text(encoding="utf-8"))
            if doc.get("tickers"):
                return list(doc["tickers"])
        except (json.JSONDecodeError, OSError):
            pass
    return list(SURGE_DROP_SEED)


def load_pool_doc() -> dict:
    if POOL_JSON.exists():
        try:
            return json.loads(POOL_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"tickers": list(SURGE_DROP_SEED), "count": len(SURGE_DROP_SEED)}


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 5 年暴涨/暴跌偏好股票池")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--surge-pct", type=float, default=7.0)
    ap.add_argument("--drop-pct", type=float, default=7.0)
    ap.add_argument("--min-surge-yr", type=float, default=3.0)
    ap.add_argument("--min-drop-yr", type=float, default=3.0)
    ap.add_argument("--min-extreme-yr", type=float, default=8.0)
    ap.add_argument("--min-dvol-m", type=float, default=50.0)
    ap.add_argument("--min-vol", type=float, default=0.45)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--seed-only", action="store_true")
    args = ap.parse_args()

    filt = SurgeDropFilter(
        years=args.years,
        surge_pct=args.surge_pct,
        drop_pct=args.drop_pct,
        min_surge_days_yr=args.min_surge_yr,
        min_drop_days_yr=args.min_drop_yr,
        min_extreme_days_yr=args.min_extreme_yr,
        min_dvol_m=args.min_dvol_m,
        min_realized_vol=args.min_vol,
        top_n=args.top,
    )
    cands = list(SURGE_DROP_SEED) if args.seed_only else None
    print(f"扫描候选 · {filt.years} 年 · ±{filt.surge_pct}% 极端日…")
    df = build_surge_drop_pool(cands, filt=filt, include_seed=True)
    if df.empty:
        print("无数据。")
        return

    selected = df[df["Top池"]] if "Top池" in df.columns else df[df["入选"]]
    doc = save_pool(df, filt)
    n = doc["count"]

    print("\n" + "=" * 78)
    print(f"暴涨/暴跌池 · Top {n} · 年均±{filt.surge_pct}%极端 ≥{filt.min_extreme_yr}天 · "
          f"成交额≥${filt.min_dvol_m:.0f}M")
    print("=" * 78)
    print(f"{'代码':<7}{'价':>8}{'波动':>7}{'暴涨/年':>8}{'暴跌/年':>8}{'极端/年':>8}{'均衡':>6}{'成交额M':>9}")
    for _, r in selected.iterrows():
        print(
            f"{r['代码']:<7}{r['价']:>8.2f}{r['年化波动']:>7.2f}"
            f"{r['年均暴涨天']:>8.1f}{r['年均暴跌天']:>8.1f}{r['年均极端天']:>8.1f}"
            f"{r['双向均衡']:>6.2f}{r['成交额M']:>9.0f}"
        )
    print(f"\n→ {POOL_JSON.name} · {POOL_CSV.name}")


if __name__ == "__main__":
    main()
