"""暴涨/暴跌 ≥20% 事件策略 · L1/S1/L2/S2（research/surge20_refined_playbook.json）。

回测（400只全市场 / 5年 / $2·$50M 流动性）：
  L1 弱收盘+大盘多：胜率67% · OOS 71% · 年化+37%（OOS +59%）
  S1 深跌后暴涨空：  胜率55% · OOS 77% · 年化+20%（OOS +52%）
  L2 暴跌恐慌反弹：  109笔 · OOS 年化+60%
  S2 缩量利好出尽空：5日空胜率62% · 回撤-22%
  组合 L1+S1：       胜率59% · OOS 70% · 年化+60%
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

SignalId = Literal["L1", "S1", "L2", "S2"]
Side = Literal["long", "short"]


@dataclass
class Extreme20Config:
    """默认参数 = surge20_optimize 网格最优档。"""

    threshold_pct: float = 20.0
    min_price: float = 2.0
    min_dvol_m: float = 50.0

    # L1 弱收盘续涨·顺风
    l1_max_close_strength: float = 0.30
    l1_hold_days: int = 5
    l1_stop_pct: float = 0.10
    l1_tp_pct: float = 0.15

    # S1 深跌反弹衰竭·空
    s1_pre20_drop_pct: float = 30.0
    s1_hold_days: int = 3
    s1_stop_pct: float = 0.10
    s1_tp_pct: float = 0.12

    # L2 暴跌恐慌反弹
    l2_min_gap_pct: float = -3.0
    l2_hold_days: int = 5
    l2_stop_pct: float = 0.08
    l2_tp_pct: float = 0.15

    # S2 缩量利好出尽·空
    s2_max_vol_ratio: float = 1.5
    s2_hold_days: int = 5
    s2_stop_pct: float = 0.08
    s2_tp_pct: float = 0.10

    priority: tuple[str, ...] = ("L1", "S1", "L2", "S2")
    enabled: tuple[str, ...] = ("L1", "S1", "L2", "S2")
    bear_enabled: tuple[str, ...] = ("L2", "S1")
    max_signals_per_day: int = 5
    # 多空组合：同日允许多头+空头各取 TopN（回测 L1+S1 胜率59% 年化+57%）
    combo_mode: bool = True
    max_long_per_day: int = 1
    max_short_per_day: int = 3


SIGNAL_META: dict[str, dict[str, str]] = {
    "L1": {
        "name": "弱收盘续涨·顺风",
        "side": "long",
        "event": "surge",
        "backtest": "全39笔 胜67% 年化+37% · OOS 71%/+59%",
    },
    "S1": {
        "name": "深跌反弹衰竭·空",
        "side": "short",
        "event": "surge",
        "backtest": "全62笔 胜55% 年化+20% · OOS 77%/+52%",
    },
    "L2": {
        "name": "暴跌恐慌反弹",
        "side": "long",
        "event": "drop",
        "backtest": "全109笔 · OOS 年化+60%",
    },
    "S2": {
        "name": "缩量利好出尽·空",
        "side": "short",
        "event": "surge",
        "backtest": "5日空胜率62% · 回撤-22%",
    },
}


def _bar_features(df: pd.DataFrame) -> dict[str, float] | None:
    if df is None or len(df) < 25:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    cl = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    if cl < 1 or prev < 1:
        return None

    ret1 = cl / prev - 1.0
    pre20 = float(close.iloc[-2] / close.iloc[-22] - 1.0) if len(close) >= 22 else float("nan")
    dvol_m = cl * float(vol.iloc[-1]) / 1e6
    vma20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float("nan")
    vol_ratio = float(vol.iloc[-1] / vma20) if vma20 and vma20 > 0 else float("nan")
    hl = float(high.iloc[-1] - low.iloc[-1])
    close_strength = 0.5 if hl <= 0 else float((cl - low.iloc[-1]) / hl)
    gap = float(open_.iloc[-1] / prev - 1.0)

    return {
        "close": cl,
        "ret1": ret1,
        "ret1_pct": ret1 * 100.0,
        "pre20": pre20,
        "pre20_pct": pre20 * 100.0 if np.isfinite(pre20) else float("nan"),
        "dvol_m": dvol_m,
        "vol_ratio": vol_ratio,
        "close_strength": close_strength,
        "gap": gap,
        "gap_pct": gap * 100.0,
    }


def _liquidity_ok(f: dict[str, float], cfg: Extreme20Config) -> bool:
    return f["close"] >= cfg.min_price and f["dvol_m"] >= cfg.min_dvol_m


def _price_levels(ref: float, side: Side, stop_pct: float, tp_pct: float) -> dict[str, float]:
    if side == "long":
        return {
            "止损价≈": round(ref * (1 - stop_pct), 2),
            "止盈价≈": round(ref * (1 + tp_pct), 2),
        }
    return {
        "止损价≈": round(ref * (1 + stop_pct), 2),
        "止盈价≈": round(ref * (1 - tp_pct), 2),
    }


def _try_l1(f: dict[str, float], cfg: Extreme20Config, *, spy_bull: bool) -> dict[str, Any] | None:
    th = cfg.threshold_pct / 100.0
    if f["ret1"] < th:
        return None
    if f["close_strength"] > cfg.l1_max_close_strength:
        return None
    if not spy_bull:
        return None
    ref = f["close"]
    px = _price_levels(ref, "long", cfg.l1_stop_pct, cfg.l1_tp_pct)
    return {
        "策略ID": "L1",
        "策略": SIGNAL_META["L1"]["name"],
        "方向": "做多",
        "side": "long",
        "事件": "暴涨",
        "信号": "可开仓",
        "最新价": round(ref, 2),
        "涨幅%": round(f["ret1_pct"], 1),
        "收盘强度": round(f["close_strength"], 2),
        "成交额M": round(f["dvol_m"], 1),
        "量比": round(f["vol_ratio"], 2) if np.isfinite(f["vol_ratio"]) else None,
        "入场": "次日开盘市价做多",
        "持有": f"{cfg.l1_hold_days}日",
        "止损%": round(cfg.l1_stop_pct * 100, 1),
        "止盈%": round(cfg.l1_tp_pct * 100, 1),
        **px,
        "依据": f"涨{f['ret1_pct']:+.1f}% 弱收{f['close_strength']:.2f} SPY>MA20 · {SIGNAL_META['L1']['backtest']}",
    }


def _try_s1(f: dict[str, float], cfg: Extreme20Config) -> dict[str, Any] | None:
    th = cfg.threshold_pct / 100.0
    if f["ret1"] < th:
        return None
    if not np.isfinite(f["pre20"]) or f["pre20"] > -cfg.s1_pre20_drop_pct / 100.0:
        return None
    ref = f["close"]
    px = _price_levels(ref, "short", cfg.s1_stop_pct, cfg.s1_tp_pct)
    return {
        "策略ID": "S1",
        "策略": SIGNAL_META["S1"]["name"],
        "方向": "做空",
        "side": "short",
        "事件": "暴涨",
        "信号": "可开仓",
        "最新价": round(ref, 2),
        "涨幅%": round(f["ret1_pct"], 1),
        "前20日%": round(f["pre20_pct"], 1),
        "成交额M": round(f["dvol_m"], 1),
        "量比": round(f["vol_ratio"], 2) if np.isfinite(f["vol_ratio"]) else None,
        "入场": "次日开盘市价做空",
        "持有": f"{cfg.s1_hold_days}日",
        "止损%": round(cfg.s1_stop_pct * 100, 1),
        "止盈%": round(cfg.s1_tp_pct * 100, 1),
        **px,
        "依据": f"深跌后死猫跳 前20{f['pre20_pct']:+.0f}% 今涨{f['ret1_pct']:+.1f}% · {SIGNAL_META['S1']['backtest']}",
    }


def _try_l2(f: dict[str, float], cfg: Extreme20Config, *, spy_bull: bool) -> dict[str, Any] | None:
    th = cfg.threshold_pct / 100.0
    if f["ret1"] > -th:
        return None
    if spy_bull:
        return None
    if f["gap"] > cfg.l2_min_gap_pct / 100.0:
        return None
    ref = f["close"]
    px = _price_levels(ref, "long", cfg.l2_stop_pct, cfg.l2_tp_pct)
    return {
        "策略ID": "L2",
        "策略": SIGNAL_META["L2"]["name"],
        "方向": "做多",
        "side": "long",
        "事件": "暴跌",
        "信号": "可开仓",
        "最新价": round(ref, 2),
        "跌幅%": round(f["ret1_pct"], 1),
        "跳空%": round(f["gap_pct"], 1),
        "成交额M": round(f["dvol_m"], 1),
        "入场": "次日开盘市价做多",
        "持有": f"{cfg.l2_hold_days}日",
        "止损%": round(cfg.l2_stop_pct * 100, 1),
        "止盈%": round(cfg.l2_tp_pct * 100, 1),
        **px,
        "依据": f"跌{f['ret1_pct']:+.1f}% 跳空{f['gap_pct']:+.1f}% SPY<MA20 · {SIGNAL_META['L2']['backtest']}",
    }


def _try_s2(f: dict[str, float], cfg: Extreme20Config) -> dict[str, Any] | None:
    th = cfg.threshold_pct / 100.0
    if f["ret1"] < th:
        return None
    if not np.isfinite(f["vol_ratio"]) or f["vol_ratio"] >= cfg.s2_max_vol_ratio:
        return None
    ref = f["close"]
    px = _price_levels(ref, "short", cfg.s2_stop_pct, cfg.s2_tp_pct)
    return {
        "策略ID": "S2",
        "策略": SIGNAL_META["S2"]["name"],
        "方向": "做空",
        "side": "short",
        "事件": "暴涨",
        "信号": "可开仓",
        "最新价": round(ref, 2),
        "涨幅%": round(f["ret1_pct"], 1),
        "量比": round(f["vol_ratio"], 2),
        "成交额M": round(f["dvol_m"], 1),
        "入场": "次日开盘市价做空",
        "持有": f"{cfg.s2_hold_days}日",
        "止损%": round(cfg.s2_stop_pct * 100, 1),
        "止盈%": round(cfg.s2_tp_pct * 100, 1),
        **px,
        "依据": f"涨{f['ret1_pct']:+.1f}% 缩量{f['vol_ratio']:.1f}x 利好出尽 · {SIGNAL_META['S2']['backtest']}",
    }


_DETECTORS = {
    "L1": lambda f, cfg, reg: _try_l1(f, cfg, spy_bull=reg["spy_bull"]),
    "S1": lambda f, cfg, reg: _try_s1(f, cfg),
    "L2": lambda f, cfg, reg: _try_l2(f, cfg, spy_bull=reg["spy_bull"]),
    "S2": lambda f, cfg, reg: _try_s2(f, cfg),
}


def detect_signals(
    df: pd.DataFrame,
    cfg: Extreme20Config | None = None,
    *,
    spy_bull: bool = True,
    ticker: str = "",
) -> list[dict[str, Any]]:
    """对单票最后一根 K 线检测全部命中策略（按 priority 排序）。"""
    cfg = cfg or Extreme20Config()
    f = _bar_features(df)
    if f is None or not _liquidity_ok(f, cfg):
        return []
    reg = {"spy_bull": spy_bull}
    hits: list[dict[str, Any]] = []
    for sid in cfg.priority:
        if sid not in cfg.enabled or sid not in _DETECTORS:
            continue
        sig = _DETECTORS[sid](f, cfg, reg)
        if sig:
            sig["代码"] = ticker.upper() if ticker else str(df.attrs.get("ticker", "")).upper()
            hits.append(sig)
    return hits


def pick_best_signal(signals: list[dict[str, Any]], cfg: Extreme20Config | None = None) -> dict[str, Any] | None:
    """同一标的只保留最高优先级信号。"""
    if not signals:
        return None
    cfg = cfg or Extreme20Config()
    order = {s: i for i, s in enumerate(cfg.priority)}
    return min(signals, key=lambda x: order.get(str(x.get("策略ID")), 99))


def _enabled_for_regime(cfg: Extreme20Config, *, spy_bull: bool) -> tuple[str, ...]:
    return cfg.enabled if spy_bull else cfg.bear_enabled


def _sort_hits(hits: list[dict[str, Any]], cfg: Extreme20Config) -> list[dict[str, Any]]:
    pri = {s: i for i, s in enumerate(cfg.priority)}
    return sorted(
        hits,
        key=lambda x: (pri.get(str(x.get("策略ID")), 99), -float(x.get("成交额M") or 0)),
    )


def select_combo_day(
    hits: list[dict[str, Any]],
    cfg: Extreme20Config,
    *,
    spy_bull: bool,
) -> list[dict[str, Any]]:
    """多空组合：按 regime 启用腿，多头/空头各取 TopN（不同标的可同日并存）。"""
    allowed = set(_enabled_for_regime(cfg, spy_bull=spy_bull))
    filtered = [h for h in hits if str(h.get("策略ID")) in allowed]
    if not filtered:
        return []

    # 同一标的只保留最高优先级
    by_ticker: dict[str, dict[str, Any]] = {}
    for h in _sort_hits(filtered, cfg):
        tk = str(h.get("代码", "")).upper()
        if tk and tk not in by_ticker:
            by_ticker[tk] = h
    uniq = list(by_ticker.values())

    longs = _sort_hits([h for h in uniq if h.get("side") == "long"], cfg)
    shorts = _sort_hits([h for h in uniq if h.get("side") == "short"], cfg)
    out = longs[: max(cfg.max_long_per_day, 0)] + shorts[: max(cfg.max_short_per_day, 0)]
    return out[: cfg.max_signals_per_day]


def market_regime() -> dict[str, Any]:
    """SPY vs MA20。"""
    from quant.data import fetch_history

    try:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()
        spy = fetch_history("SPY", start=start, end=end)
        px = float(spy["Close"].iloc[-1])
        ma20 = float(spy["Close"].tail(20).mean())
        bull = px > ma20
        return {
            "SPY": round(px, 2),
            "MA20": round(ma20, 2),
            "spy_bull": bull,
            "站上MA20": bull,
            "regime": "顺风(MA20上)" if bull else "逆风(MA20下)",
        }
    except Exception:  # noqa: BLE001
        return {"SPY": None, "MA20": None, "spy_bull": True, "站上MA20": True, "regime": "未知"}


def scan_live(
    cfg: Extreme20Config | None = None,
    *,
    screen_count: int = 300,
) -> pd.DataFrame:
    """扫描涨幅榜 + 跌幅榜，输出今日可开仓信号。"""
    cfg = cfg or Extreme20Config()
    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from quant.screener import fetch_yahoo_screen

    cands: list[str] = []
    for preset in ("day_gainers", "day_losers", "most_actives"):
        try:
            df = fetch_yahoo_screen(preset, count=screen_count)
            if not df.empty:
                cands.extend(df["代码"].astype(str).str.upper().tolist())
        except Exception:  # noqa: BLE001
            continue

    tickers = sorted(dict.fromkeys(t for t in cands if t and t != "SPY"))
    if not tickers:
        return pd.DataFrame()

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=80)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(tickers, start, end)
    reg = market_regime()
    spy_bull = bool(reg.get("spy_bull", True))

    all_hits: list[dict[str, Any]] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        hits = detect_signals(df, cfg, spy_bull=spy_bull, ticker=tk.upper())
        if cfg.combo_mode:
            all_hits.extend(hits)
        else:
            best = pick_best_signal(hits, cfg)
            if best:
                all_hits.append(best)

    if not all_hits:
        return pd.DataFrame()

    if cfg.combo_mode:
        picked = select_combo_day(all_hits, cfg, spy_bull=spy_bull)
    else:
        pri = {s: i for i, s in enumerate(cfg.priority)}
        out = pd.DataFrame(all_hits)
        out["_pri"] = out["策略ID"].map(lambda x: pri.get(str(x), 99))
        picked = (
            out.sort_values(["_pri", "成交额M"], ascending=[True, False])
            .head(cfg.max_signals_per_day)
            .drop(columns=["_pri"])
            .to_dict(orient="records")
        )

    if not picked:
        return pd.DataFrame()
    return pd.DataFrame(picked).reset_index(drop=True)


def config_from_dict(d: dict[str, Any]) -> Extreme20Config:
    """从 JSON 配置构建 Extreme20Config。"""
    base = asdict(Extreme20Config())
    for k, v in d.items():
        if k in base and not isinstance(v, dict):
            base[k] = v
    if "enabled_strategies" in d:
        base["enabled"] = tuple(d["enabled_strategies"])
    if "bear_enabled_strategies" in d:
        base["bear_enabled"] = tuple(d["bear_enabled_strategies"])
    return Extreme20Config(**base)


def run_backtest_summary(*, pool: str = "broad") -> dict[str, Any]:
    """调用 research 寻优模块，返回组合回测摘要（需网络）。"""
    from datetime import timedelta as td

    from research.surge20_optimize import run as opt_run

    return opt_run(pool=pool, quick=True)
