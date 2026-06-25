"""多空双杀池：专挑「暴涨杀空、暴跌杀多」来回扫的高反转票（SPCE/RGTI 类）。

与 surge_drop_pool（爱暴涨也爱暴跌）的区别：
  surge_drop 看「极端频率」；本池更看 **反转率（whipsaw）**——
  大涨/大跌后第二天反向的概率越高，越是「多空双杀」。

核心指标：
  · 反转率 = 在 |当日涨跌|≥阈值 的日子里，次日反向的比例
  · 方向切换率 = 日收益正负号翻转的频率（越高越「锯齿」）
  · 趋势持续度（efficiency ratio 反向）= 越低越震荡
  · 年化波动 / 双向大异动 / 流动性
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

POOL_JSON = ROOT / "research" / "whipsaw_pool.json"
POOL_CSV = ROOT / "research" / "whipsaw_pool.csv"

# 多空双杀种子：低价高波 meme / 题材 / 杠杆 ETF（暴涨暴跌来回扫）
WHIPSAW_SEED: list[str] = [
    "SPCE", "RGTI", "QBTS", "IONQ", "QUBT", "SOUN", "BBAI", "LUNR", "ASTS", "RCAT",
    "ACHR", "JOBY", "NVTS", "OUST", "EOSE", "ONDS", "KULR", "AAOI", "OPEN", "WULF",
    "MARA", "RIOT", "CLSK", "BITF", "HUT", "IREN", "CIFR", "BTBT", "GME", "AMC",
    "NVAX", "SAVA", "VKTX", "DJT", "LCID", "NIO", "TSLL", "MSTX", "CONL", "SOXL",
    "NNE", "SMR", "OKLO", "PLUG", "FCEL", "DNA", "CHPT", "AFRM", "UPST", "HOOD",
]


@dataclass
class WhipsawFilter:
    years: int = 3
    move_th_pct: float = 5.0        # 「大异动」阈值（%）：触发反转判定
    min_reversal_rate: float = 0.42  # 反转率下限（大异动后次日反向概率）
    min_flip_rate: float = 0.45     # 日收益正负翻转频率下限
    min_big_up_yr: float = 6.0      # 年均大涨天（≥move_th）
    min_big_down_yr: float = 6.0    # 年均大跌天（≤-move_th）
    min_realized_vol: float = 0.60
    min_dvol_m: float = 50.0
    min_price: float = 1.5
    max_price: float = 300.0
    min_history_days: int = 250
    top_n: int = 60


def profile_ticker(
    df: pd.DataFrame, *, ticker: str = "", filt: WhipsawFilter | None = None,
) -> dict | None:
    """计算单票「多空双杀」画像。"""
    filt = filt or WhipsawFilter()
    if df is None or len(df) < filt.min_history_days:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    ret = close.pct_change().dropna()
    if len(ret) < filt.min_history_days:
        return None

    years = max(len(ret) / 252.0, 0.5)
    th = filt.move_th_pct / 100.0

    big_up = ret >= th
    big_down = ret <= -th
    big = big_up | big_down
    big_up_yr = int(big_up.sum()) / years
    big_down_yr = int(big_down.sum()) / years

    # 反转率：大异动日，次日符号与当日相反的比例
    sign = np.sign(ret)
    next_sign = sign.shift(-1)
    big_idx = big & next_sign.notna()
    if big_idx.sum() >= 10:
        reversal_rate = float((sign[big_idx] * next_sign[big_idx] < 0).mean())
    else:
        reversal_rate = np.nan

    # 方向切换率：相邻两日收益符号翻转的比例（锯齿度）
    flip_rate = float((sign * sign.shift(1) < 0).mean())

    # 趋势持续度（efficiency ratio）：净位移/路程，越低越震荡
    win = min(20, len(close) - 1)
    disp = (close.diff(win).abs())
    path = close.diff().abs().rolling(win).sum()
    er = float((disp / path.replace(0, np.nan)).dropna().mean()) if win > 0 else np.nan
    chop = 1.0 - er if np.isfinite(er) else np.nan  # 震荡度

    rv = float(ret.std() * np.sqrt(252))
    dvol_m = float((close * vol).tail(60).mean() / 1e6)
    price = float(close.iloc[-1])

    balance = (min(big_up_yr, big_down_yr) / max(big_up_yr, big_down_yr)) if big_up_yr > 0 and big_down_yr > 0 else 0.0

    # 双杀分：反转率 + 锯齿度 + 波动 + 双向均衡（这才是「多空双杀」的核心）
    rev = reversal_rate if np.isfinite(reversal_rate) else 0.0
    chop_s = chop if np.isfinite(chop) else 0.0
    score = rev * 40 + flip_rate * 20 + chop_s * 15 + rv * 8 + balance * 5

    return {
        "代码": ticker.upper() if ticker else "",
        "价": round(price, 2),
        "年化波动": round(rv, 3),
        "反转率": round(rev, 3),
        "切换率": round(flip_rate, 3),
        "震荡度": round(chop_s, 3) if np.isfinite(chop_s) else None,
        "年均大涨天": round(big_up_yr, 1),
        "年均大跌天": round(big_down_yr, 1),
        "双向均衡": round(balance, 3),
        "成交额M": round(dvol_m, 1),
        "双杀分": round(score, 2),
        "样本年数": round(years, 2),
    }


def passes_filter(prof: dict, filt: WhipsawFilter | None = None) -> bool:
    filt = filt or WhipsawFilter()
    rev = prof.get("反转率") or 0
    return (
        rev >= filt.min_reversal_rate
        and (prof.get("切换率") or 0) >= filt.min_flip_rate
        and (prof.get("年均大涨天") or 0) >= filt.min_big_up_yr
        and (prof.get("年均大跌天") or 0) >= filt.min_big_down_yr
        and (prof.get("年化波动") or 0) >= filt.min_realized_vol
        and (prof.get("成交额M") or 0) >= filt.min_dvol_m
        and filt.min_price <= (prof.get("价") or 0) <= filt.max_price
    )


def build_whipsaw_pool(
    candidates: list[str] | None = None,
    *,
    filt: WhipsawFilter | None = None,
    include_seed: bool = True,
    use_broad: bool = True,
) -> pd.DataFrame:
    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

    filt = filt or WhipsawFilter()
    pool: list[str] = []
    if include_seed:
        pool.extend(WHIPSAW_SEED)
    pool.extend(GAINER_MOMENTUM)
    pool.extend(LIQUID100)
    if candidates:
        pool.extend(candidates)
    cache = ROOT / "research" / "gainer_universe_cache.json"
    if cache.exists():
        try:
            pool.extend(json.loads(cache.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    if use_broad:
        try:
            from quant.screener import fetch_broad_universe
            pool.extend(fetch_broad_universe(screen_count=400, extra=LIQUID100))
        except Exception:  # noqa: BLE001
            pass
    tickers = sorted(dict.fromkeys(t.strip().upper() for t in pool if t and t.strip() and t != "SPY"))

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=filt.years * 365 + 60)).isoformat()
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
        prof["种子"] = tk.upper() in set(WHIPSAW_SEED)
        prof["入选"] = passes_filter(prof, filt)
        rows.append(prof)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    selected = out[out["入选"]].sort_values("双杀分", ascending=False)
    if filt.top_n > 0:
        selected = selected.head(filt.top_n)
    out["Top池"] = out["代码"].isin(selected["代码"].tolist())
    return out.sort_values(["Top池", "双杀分"], ascending=[False, False]).reset_index(drop=True)


def save_pool(df: pd.DataFrame, filt: WhipsawFilter) -> dict:
    selected = df[df["Top池"]].copy() if "Top池" in df.columns else df[df.get("入选", False)]
    tickers = selected["代码"].tolist()
    doc = {
        "updated": date.today().isoformat(),
        "description": "多空双杀池 · 高反转率/锯齿型暴涨暴跌票",
        "filter": asdict(filt),
        "count": len(tickers),
        "tickers": tickers,
        "summary": {
            "candidates_scanned": int(len(df)),
            "avg_reversal_rate": round(float(selected["反转率"].mean()), 3) if not selected.empty else 0,
            "avg_flip_rate": round(float(selected["切换率"].mean()), 3) if not selected.empty else 0,
            "avg_realized_vol": round(float(selected["年化波动"].mean()), 3) if not selected.empty else 0,
        },
        "detail": selected.to_dict(orient="records"),
        "all_profiles": df.to_dict(orient="records"),
    }
    POOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    POOL_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")
    return doc


def load_pool() -> list[str]:
    if POOL_JSON.exists():
        try:
            doc = json.loads(POOL_JSON.read_text(encoding="utf-8"))
            if doc.get("tickers"):
                return list(doc["tickers"])
        except (json.JSONDecodeError, OSError):
            pass
    return list(WHIPSAW_SEED)


def load_pool_doc() -> dict:
    if POOL_JSON.exists():
        try:
            return json.loads(POOL_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"tickers": list(WHIPSAW_SEED), "count": len(WHIPSAW_SEED)}


def main() -> None:
    ap = argparse.ArgumentParser(description="构建多空双杀池")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--move-th", type=float, default=5.0)
    ap.add_argument("--min-reversal", type=float, default=0.42)
    ap.add_argument("--min-dvol-m", type=float, default=50.0)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--seed-only", action="store_true")
    args = ap.parse_args()

    filt = WhipsawFilter(
        years=args.years, move_th_pct=args.move_th,
        min_reversal_rate=args.min_reversal, min_dvol_m=args.min_dvol_m, top_n=args.top,
    )
    cands = list(WHIPSAW_SEED) if args.seed_only else None
    print(f"扫描多空双杀候选 · {filt.years}年 · 阈值±{filt.move_th_pct}%…")
    df = build_whipsaw_pool(cands, filt=filt, include_seed=True)
    if df.empty:
        print("无数据。")
        return
    selected = df[df["Top池"]] if "Top池" in df.columns else df[df["入选"]]
    doc = save_pool(df, filt)

    print("\n" + "=" * 84)
    print(f"多空双杀池 · Top {doc['count']} · 反转率≥{filt.min_reversal_rate} · 切换率≥{filt.min_flip_rate}")
    print("=" * 84)
    print(f"{'代码':<7}{'价':>8}{'波动':>7}{'反转率':>8}{'切换率':>8}{'震荡度':>8}{'大涨/年':>8}{'大跌/年':>8}{'成交额M':>9}")
    for _, r in selected.iterrows():
        print(
            f"{r['代码']:<7}{r['价']:>8.2f}{r['年化波动']:>7.2f}{r['反转率']:>8.2f}"
            f"{r['切换率']:>8.2f}{(r['震荡度'] or 0):>8.2f}{r['年均大涨天']:>8.1f}"
            f"{r['年均大跌天']:>8.1f}{r['成交额M']:>9.0f}"
        )
    print(f"\n→ {POOL_JSON.name} · {POOL_CSV.name}")


if __name__ == "__main__":
    main()
