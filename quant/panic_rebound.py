"""恐慌反弹做多策略（全市场验证最优的方向性事件策略）。

规律（research/extreme15_pattern.py 全市场 400 只 / 5 年回测得出）：
  某票过去 20 日已深跌（≥30%），当日再单日暴跌（≥10%）放出恐慌盘，
  次日开盘做多，持有数日反弹 → 样本外年化 +54%~+77%、胜率 54%~63%、回撤 -18%。

本模块只负责「信号判定 + 当日实盘扫描」，数据源走 quant.providers（Yahoo）。
回测逻辑在 research/extreme15_pattern.py，二者参数保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PanicReboundConfig:
    """恐慌反弹参数（默认=全市场验证的稳健头档：持3日/止损8%/止盈15%）。"""

    drop_pct: float = 10.0          # 当日跌幅阈值（≥该值视为恐慌盘）
    pre20_drop_pct: float = 30.0    # 前20日累计跌幅阈值（已深跌才抄）
    min_price: float = 5.0
    min_dvol_m: float = 100.0       # 当日成交额下限（百万美元，保流动性）
    hold_days: int = 3
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.15
    max_positions: int = 3


def detect_signal(df: pd.DataFrame, cfg: PanicReboundConfig | None = None) -> dict[str, Any] | None:
    """判定一只标的的「最后一根 K 线」是否触发恐慌反弹做多信号。

    返回信号字典（含参考价位）或 None。入场为次日开盘，故价位以最新收盘价为参考。
    """
    cfg = cfg or PanicReboundConfig()
    if df is None or len(df) < 25:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    if close.iloc[-1] < cfg.min_price:
        return None

    ret1 = float(close.iloc[-1] / close.iloc[-2] - 1.0)
    pre20 = float(close.iloc[-2] / close.iloc[-22] - 1.0) if len(close) >= 22 else np.nan
    dvol_m = float(close.iloc[-1] * vol.iloc[-1]) / 1e6
    vr = float(vol.iloc[-1] / vol.iloc[-21:-1].mean()) if len(vol) >= 21 else np.nan

    if ret1 > -cfg.drop_pct / 100.0:
        return None
    if not np.isfinite(pre20) or pre20 > -cfg.pre20_drop_pct / 100.0:
        return None
    if dvol_m < cfg.min_dvol_m:
        return None

    ref = float(close.iloc[-1])
    return {
        "代码": str(df.attrs.get("ticker", "")),
        "最新价": round(ref, 2),
        "当日跌%": round(ret1 * 100, 1),
        "前20日跌%": round(pre20 * 100, 1),
        "量比": round(vr, 2) if np.isfinite(vr) else None,
        "成交额M": round(dvol_m, 1),
        "入场": "次日开盘市价做多",
        "持有": f"{cfg.hold_days}日",
        "止损价≈": round(ref * (1 - cfg.stop_loss_pct), 2),
        "止盈价≈": round(ref * (1 + cfg.take_profit_pct), 2),
        "止损%": round(cfg.stop_loss_pct * 100, 1),
        "止盈%": round(cfg.take_profit_pct * 100, 1),
    }


def scan_live(
    cfg: PanicReboundConfig | None = None,
    *,
    screen_count: int = 250,
    pool: str = "market",
) -> pd.DataFrame:
    """当日实盘扫描：候选 → 拉历史 → 命中恐慌反弹规律的候选。

    pool:
      "market"  全市场今日跌幅榜/活跃榜（默认，覆盖广）
      "rgti"    仅 RGTI 类似高波池（research/rgti_like_pool.json，信号更纯）
      "both"    两者并集
    """
    cfg = cfg or PanicReboundConfig()
    from quant.providers import DataConfig, get_provider, reset_provider_cache

    cands: list[str] = []
    if pool in ("rgti", "both"):
        from quant.volatile_pool import load_pool
        cands.extend(load_pool())
    if pool in ("market", "both"):
        from quant.screener import fetch_yahoo_screen
        for preset in ("day_losers", "most_actives", "small_cap_gainers"):
            try:
                df = fetch_yahoo_screen(preset, count=screen_count)
                if not df.empty:
                    cands.extend(df["代码"].astype(str).str.upper().tolist())
            except Exception:  # noqa: BLE001
                continue
    tickers = sorted(dict.fromkeys([t.upper() for t in cands if t and t != "SPY"]))
    if not tickers:
        return pd.DataFrame()

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=80)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(tickers, start, end)

    rows: list[dict[str, Any]] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        df.attrs["ticker"] = tk.upper()
        sig = detect_signal(df, cfg)
        if sig:
            sig["代码"] = tk.upper()
            rows.append(sig)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 已跌越深、成交额越大优先
    return out.sort_values(["前20日跌%", "成交额M"], ascending=[True, False]).reset_index(drop=True)
