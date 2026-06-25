"""Gainer10+ 策略单元测试（离线）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.gainer10_strategy import Gainer10Config, _features, classify


def _df(closes: list[float], *, gap_pct: float = 0.0, vol: float = 5e7) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx, dtype=float)
    o = c.shift(1).fillna(c.iloc[0])
    if gap_pct:
        o.iloc[-1] = c.iloc[-2] * (1 + gap_pct)
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.02
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.98
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol}, index=idx)


def test_features_on_spike_day() -> None:
    rise = list(np.linspace(50, 120, 180))
    spike = rise + [132.0]
    f = _features(_df(spike, gap_pct=0.06))
    assert f is not None
    assert f["chg"] > 0.08
    assert f["gap"] >= 0.05


def test_classify_rule_a_tech_momentum() -> None:
    cfg = Gainer10Config(sector_mode=False)
    f = {"close": 100, "chg": 0.12, "clv": 0.6, "gap": 0.06, "vol_x": 3.0,
         "ext20": 0.45, "rsi": 78, "dvol_m": 150}
    sig = classify(f, "Technology", bull=True, cfg=cfg)
    assert sig is not None
    assert sig.信号 == "续涨A"


def test_classify_rule_b_balanced() -> None:
    cfg = Gainer10Config(sector_mode=False)
    f = {"close": 80, "chg": 0.11, "clv": 0.5, "gap": 0.07, "vol_x": 2.5,
         "ext20": 0.25, "rsi": 65, "dvol_m": 120}
    sig = classify(f, "Financial Services", bull=True, cfg=cfg)
    assert sig is not None
    assert sig.信号 == "续涨B"
    assert sig.限价入场 == round(80 * 0.95, 2)


def test_classify_rule_s_short_weak() -> None:
    cfg = Gainer10Config(sector_mode=False)
    f = {"close": 20, "chg": 0.15, "clv": 0.2, "gap": -0.02, "vol_x": 2.0,
         "ext20": -0.05, "rsi": 45, "dvol_m": 110}
    sig = classify(f, "Healthcare", bull=False, cfg=cfg)
    assert sig is not None
    assert sig.信号 == "做空S"
    assert sig.方向 == "short"
    assert sig.持有天 == 10


def test_classify_long_skips_when_bear_market() -> None:
    cfg = Gainer10Config(require_bull=True, sector_mode=False)
    f = {"close": 100, "chg": 0.12, "clv": 0.6, "gap": 0.06, "vol_x": 3.0,
         "ext20": 0.45, "rsi": 78, "dvol_m": 150}
    assert classify(f, "Technology", bull=False, cfg=cfg) is None


def test_classify_short_works_in_bear_market() -> None:
    cfg = Gainer10Config(require_bull=True, sector_mode=False)
    f = {"close": 20, "chg": 0.15, "clv": 0.2, "gap": -0.02, "vol_x": 2.0,
         "ext20": -0.05, "rsi": 45, "dvol_m": 110}
    sig = classify(f, "Healthcare", bull=False, cfg=cfg)
    assert sig is not None
    assert sig.信号 == "做空S"


def test_classify_sector_short_high_win() -> None:
    """可选消费空头规则 win 91.9% ≥ 80% 阈值。"""
    cfg = Gainer10Config(sector_mode=True, sector_short_min_win=80.0)
    f = {"close": 30, "chg": 0.12, "clv": 0.3, "gap": -0.01, "vol_x": 1.5,
         "ext20": -0.02, "rsi": 50, "dvol_m": 120}
    sig = classify(f, "Consumer Cyclical", bull=False, cfg=cfg)
    assert sig is not None
    assert sig.信号 == "做空·可选消费"
    assert sig.方向 == "short"


def test_sector_only_skips_legacy() -> None:
    cfg = Gainer10Config(sector_mode=True, sector_only=True, sector_long_min_win=99.0)
    f = {"close": 100, "chg": 0.12, "clv": 0.6, "gap": 0.06, "vol_x": 3.0,
         "ext20": 0.45, "rsi": 78, "dvol_m": 150}
    assert classify(f, "Technology", bull=True, cfg=cfg) is None


def test_classify_sector_long_unknown() -> None:
    """未知板块多头 win 65.6% ≥ 60% 阈值（高胜率模式）。"""
    cfg = Gainer10Config(sector_mode=True, sector_only=True)
    f = {"close": 50, "chg": 0.11, "clv": 0.55, "gap": 0.06, "vol_x": 2.0,
         "ext20": 0.42, "rsi": 65, "dvol_m": 150}
    sig = classify(f, "Unknown", bull=True, cfg=cfg)
    assert sig is not None
    assert sig.信号 == "续涨·未知"
    assert sig.方向 == "long"
