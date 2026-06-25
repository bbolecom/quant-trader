"""多空组合策略 · Extreme20 L1/S1 + Flow U_S2/D_S2 · 质量分排序 · 高胜率过滤。

5年回测锚点（高胜率模式 max_short=1）：
  Extreme20 腿：91笔 胜60.4% 年化+56%
  Flow 腿：430笔 胜53.3% 年化+12%
  日收益等权组合：日胜率54% 年化+38%
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant.extreme20_strategy import (
    Extreme20Config,
    config_from_dict as e20_config_from_dict,
    detect_signals,
    market_regime,
    select_combo_day,
)
from quant.flow_strategy import FlowStrategyParams, today_actionable_picks

Leg = Literal["e20", "flow"]

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "research" / "longshort_combo_rules.json"


@dataclass
class LongShortComboConfig:
    """统一多空组合配置。"""

    name: str = "longshort_combo_v1_highwin"
    # Extreme20 子配置（嵌套 dict 或路径）
    extreme20: dict[str, Any] = field(default_factory=dict)
    flow: dict[str, Any] = field(default_factory=dict)
    screen_count: int = 300
    gainer_count: int = 250

    # 组合限额
    max_long_per_day: int = 1
    max_short_per_day: int = 1
    max_signals_per_day: int = 2

    # 高胜率质量过滤
    min_quality_score: float = 0.55
    use_quality_rank: bool = True
    min_dvol_m_boost: float = 80.0
    l1_max_close_strength: float = 0.25
    s1_pre20_drop_pct: float = 35.0
    s1_max_surge_pct: float = 35.0
    s1_min_vol_ratio: float = 1.2
    require_spy_bull_for_l1: bool = True
    flow_min_setup_win: float = 0.55

    long_weight: float = 0.5
    short_weight: float = 0.5


def config_from_dict(d: dict[str, Any]) -> LongShortComboConfig:
    base = asdict(LongShortComboConfig())
    for k, v in d.items():
        if k in base and not isinstance(v, dict):
            base[k] = v
    if "extreme20" in d:
        base["extreme20"] = d["extreme20"]
    if "flow" in d:
        base["flow"] = d["flow"]
    return LongShortComboConfig(**base)


def load_rules(path: Path | None = None) -> dict[str, Any]:
    p = path or RULES_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _e20_cfg(combo: LongShortComboConfig) -> Extreme20Config:
    raw = dict(combo.extreme20)
    raw.setdefault("combo_mode", True)
    raw.setdefault("enabled_strategies", ["L1", "S1"])
    raw.setdefault("bear_enabled_strategies", ["L2", "S1"])
    raw["max_long_per_day"] = combo.max_long_per_day
    raw["max_short_per_day"] = combo.max_short_per_day
    raw["max_signals_per_day"] = combo.max_signals_per_day
    raw["l1_max_close_strength"] = combo.l1_max_close_strength
    raw["s1_pre20_drop_pct"] = combo.s1_pre20_drop_pct
    raw["min_dvol_m"] = max(float(raw.get("min_dvol_m", 50)), combo.min_dvol_m_boost)
    return e20_config_from_dict(raw)


def _flow_params(combo: LongShortComboConfig) -> FlowStrategyParams:
    raw = dict(combo.flow)
    raw.setdefault("long_top_n", combo.max_long_per_day)
    raw.setdefault("short_top_n", combo.max_short_per_day)
    raw["min_recent_setup_win_rate"] = max(
        float(raw.get("min_recent_setup_win_rate", 0)),
        combo.flow_min_setup_win,
    )
    raw["use_recent_setup_win"] = True
    return FlowStrategyParams.from_dict(raw)


def quality_score(sig: dict[str, Any], cfg: LongShortComboConfig) -> float:
    """0~1 质量分，越高越优先。"""
    sid = str(sig.get("策略ID", sig.get("leg_id", "")))
    leg = str(sig.get("leg", "e20"))
    score = 0.35

    dvol = float(sig.get("成交额M") or sig.get("dvol_m") or 0)
    if dvol >= cfg.min_dvol_m_boost:
        score += 0.15 * min(dvol / 200.0, 1.0)

    if leg == "e20" or sid in ("L1", "S1", "L2", "S2"):
        cs = float(sig.get("收盘强度", sig.get("close_strength", 0.5)) or 0.5)
        vr = float(sig.get("量比", sig.get("vol_ratio", 1)) or 1)
        pre20_pct = float(sig.get("前20日%", sig.get("pre20_pct", 0)) or 0)
        ret1 = abs(float(sig.get("涨幅%", sig.get("跌幅%", 0)) or 0))

        if sid == "L1":
            score += 0.25 * max(0, (cfg.l1_max_close_strength - cs) / max(cfg.l1_max_close_strength, 0.01))
            score += 0.1 * min(vr / 3.0, 1.0)
        elif sid == "S1":
            score += 0.2 * min(abs(pre20_pct) / 50.0, 1.0) if pre20_pct < 0 else 0
            score += 0.15 * max(0, (cfg.s1_max_surge_pct - ret1) / cfg.s1_max_surge_pct)
            score += 0.1 * min(vr / 2.5, 1.0)
        elif sid == "L2":
            score += 0.2
    else:
        wr = float(sig.get("setup_win_rate", sig.get("近期胜率", 0)) or 0)
        if wr > 0:
            score += 0.35 * min(wr, 1.0)
        vr = float(sig.get("量比", 1) or 1)
        score += 0.1 * min(vr / 3.0, 1.0)

    bt_wr = float(sig.get("backtest_win", 0) or 0)
    if bt_wr > 0:
        score += 0.1 * bt_wr

    return round(min(max(score, 0.0), 1.0), 4)


def passes_quality_filter(sig: dict[str, Any], cfg: LongShortComboConfig, *, spy_bull: bool) -> bool:
    sid = str(sig.get("策略ID", ""))
    leg = str(sig.get("leg", "e20"))
    dvol = float(sig.get("成交额M") or sig.get("dvol_m") or 0)
    if dvol < cfg.min_dvol_m_boost and leg == "e20":
        return False

    if sid == "L1":
        if cfg.require_spy_bull_for_l1 and not spy_bull:
            return False
        cs = float(sig.get("收盘强度", 1) or 1)
        if cs > cfg.l1_max_close_strength:
            return False
    elif sid == "S1":
        pre20_pct = float(sig.get("前20日%", 0) or 0)
        ret1 = abs(float(sig.get("涨幅%", 0) or 0))
        vr = float(sig.get("量比", 0) or 0)
        if pre20_pct > -cfg.s1_pre20_drop_pct:
            return False
        if ret1 > cfg.s1_max_surge_pct:
            return False
        if vr < cfg.s1_min_vol_ratio:
            return False

    if leg == "flow":
        wr = float(sig.get("setup_win_rate", sig.get("近期胜率", 1)) or 1)
        if wr < cfg.flow_min_setup_win:
            return False

    return quality_score(sig, cfg) >= cfg.min_quality_score


def _tag(sig: dict[str, Any], *, leg: Leg, cfg: LongShortComboConfig) -> dict[str, Any]:
    out = dict(sig)
    out["leg"] = leg
    out["模块"] = "多空组合"
    if leg == "flow":
        out.setdefault("side", "long" if "多" in str(out.get("方向", "")) else "short")
        out.setdefault("策略ID", "FLOW-L" if out.get("side") == "long" else "FLOW-S")
        out["setup_win_rate"] = float(out.get("近期胜率", out.get("setup_win_rate", 0.6)) or 0.6)
    out["质量分"] = quality_score(out, cfg)
    return out


def merge_and_select(
    e20_hits: list[dict[str, Any]],
    flow_picks: list[dict[str, Any]],
    cfg: LongShortComboConfig,
    *,
    spy_bull: bool,
) -> list[dict[str, Any]]:
    """合并两腿、去重、质量过滤、多空各取 TopN。"""
    ecfg = _e20_cfg(cfg)
    e20_sel = select_combo_day(e20_hits, ecfg, spy_bull=spy_bull)
    pool: list[dict[str, Any]] = [_tag(h, leg="e20", cfg=cfg) for h in e20_sel]
    pool.extend(_tag(p, leg="flow", cfg=cfg) for p in flow_picks)

    filtered = [s for s in pool if passes_quality_filter(s, cfg, spy_bull=spy_bull)]
    if not filtered:
        return []

    by_ticker: dict[str, dict[str, Any]] = {}
    for s in filtered:
        tk = str(s.get("代码", "")).upper()
        if not tk:
            continue
        prev = by_ticker.get(tk)
        if prev is None or float(s.get("质量分", 0)) > float(prev.get("质量分", 0)):
            by_ticker[tk] = s

    uniq = list(by_ticker.values())
    if cfg.use_quality_rank:
        key_fn = lambda x: (-float(x.get("质量分", 0)), -float(x.get("成交额M") or 0))
    else:
        key_fn = lambda x: (-float(x.get("成交额M") or 0),)

    longs = sorted([s for s in uniq if s.get("side") == "long"], key=key_fn)
    shorts = sorted([s for s in uniq if s.get("side") == "short"], key=key_fn)
    out = longs[: cfg.max_long_per_day] + shorts[: cfg.max_short_per_day]
    for s in out:
        s["质量分"] = quality_score(s, cfg)
        s["信号"] = "可开仓"
    return out[: cfg.max_signals_per_day]


def scan_live(cfg: LongShortComboConfig | None = None) -> pd.DataFrame:
    """扫描今日多空组合信号。"""
    cfg = cfg or LongShortComboConfig()
    ecfg = _e20_cfg(cfg)
    fparams = _flow_params(cfg)
    reg = market_regime()
    spy_bull = bool(reg.get("spy_bull", True))

    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from quant.screener import fetch_yahoo_screen

    cands: list[str] = []
    for preset in ("day_gainers", "day_losers", "most_actives"):
        try:
            df = fetch_yahoo_screen(preset, count=cfg.screen_count)
            if not df.empty:
                cands.extend(df["代码"].astype(str).str.upper().tolist())
        except Exception:  # noqa: BLE001
            continue

    tickers = sorted(dict.fromkeys(t for t in cands if t and t != "SPY"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=80)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    batch = yahoo.fetch_batch(tickers, start, end) if tickers else {}
    spy_df = yahoo.fetch_history("SPY", start, end)
    spy_close = spy_df["Close"].astype(float)

    e20_hits: list[dict[str, Any]] = []
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        e20_hits.extend(detect_signals(df, ecfg, spy_bull=spy_bull, ticker=tk.upper()))

    flow_df = today_actionable_picks(batch, spy_close, fparams)
    flow_picks = flow_df.to_dict(orient="records") if not flow_df.empty else []

    picked = merge_and_select(e20_hits, flow_picks, cfg, spy_bull=spy_bull)
    if not picked:
        return pd.DataFrame()
    return pd.DataFrame(picked).reset_index(drop=True)


def backtest_summary(cfg: LongShortComboConfig | None = None, *, quick: bool = False) -> dict[str, Any]:
    """5年组合回测摘要（需网络）。"""
    from research.longshort_combo_optimize import run_backtest_combo

    return run_backtest_combo(config_from_dict(asdict(cfg or LongShortComboConfig())), quick=quick)
