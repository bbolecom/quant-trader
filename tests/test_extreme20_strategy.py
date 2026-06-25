"""extreme20_strategy unit tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.extreme20_strategy import (
    Extreme20Config,
    config_from_dict,
    detect_signals,
    pick_best_signal,
    select_combo_day,
)


def _ohlcv(n: int = 30, price: float = 10.0, vol: float = 2_000_000.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = np.full(n, price, dtype=float)
    o = c * 0.99
    h = c * 1.02
    l = c * 0.98
    v = np.full(n, vol, dtype=float)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def test_l1_weak_close_bull_surge():
    df = _ohlcv()
    prev = float(df["Close"].iloc[-2])
    # +25% but close near low (weak)
    df.iloc[-1, df.columns.get_loc("Open")] = prev * 1.05
    df.iloc[-1, df.columns.get_loc("High")] = prev * 1.30
    df.iloc[-1, df.columns.get_loc("Low")] = prev * 1.18
    df.iloc[-1, df.columns.get_loc("Close")] = prev * 1.21  # 弱收盘 ~0.27
    df.iloc[-1, df.columns.get_loc("Volume")] = 5_000_000
    hits = detect_signals(df, Extreme20Config(min_dvol_m=1), spy_bull=True, ticker="TEST")
    ids = [h["策略ID"] for h in hits]
    assert "L1" in ids


def test_s1_deep_drop_then_surge():
    df = _ohlcv(n=30, price=8.0)
    # crash pre20: set older prices high
    for i in range(-22, -2):
        df.iloc[i, df.columns.get_loc("Close")] = 15.0
        df.iloc[i, df.columns.get_loc("High")] = 15.5
        df.iloc[i, df.columns.get_loc("Low")] = 14.5
    prev = 6.0
    df.iloc[-2, df.columns.get_loc("Close")] = prev
    df.iloc[-1, df.columns.get_loc("Open")] = prev * 1.02
    df.iloc[-1, df.columns.get_loc("High")] = prev * 1.28
    df.iloc[-1, df.columns.get_loc("Low")] = prev * 1.01
    df.iloc[-1, df.columns.get_loc("Close")] = prev * 1.25
    df.iloc[-1, df.columns.get_loc("Volume")] = 8_000_000
    hits = detect_signals(df, Extreme20Config(min_dvol_m=1), spy_bull=False, ticker="DEAD")
    assert any(h["策略ID"] == "S1" for h in hits)


def test_l2_panic_drop():
    df = _ohlcv()
    prev = float(df["Close"].iloc[-2])
    df.iloc[-1, df.columns.get_loc("Open")] = prev * 0.92
    df.iloc[-1, df.columns.get_loc("High")] = prev * 0.94
    df.iloc[-1, df.columns.get_loc("Low")] = prev * 0.75
    df.iloc[-1, df.columns.get_loc("Close")] = prev * 0.78
    df.iloc[-1, df.columns.get_loc("Volume")] = 6_000_000
    hits = detect_signals(df, Extreme20Config(min_dvol_m=1), spy_bull=False, ticker="PANIC")
    assert any(h["策略ID"] == "L2" for h in hits)


def test_s2_low_volume_surge():
    df = _ohlcv(vol=1_000_000)
    prev = float(df["Close"].iloc[-2])
    df.iloc[-1, df.columns.get_loc("Open")] = prev * 1.18
    df.iloc[-1, df.columns.get_loc("High")] = prev * 1.26
    df.iloc[-1, df.columns.get_loc("Low")] = prev * 1.17
    df.iloc[-1, df.columns.get_loc("Close")] = prev * 1.24
    df.iloc[-1, df.columns.get_loc("Volume")] = 800_000  # vol_ratio < 1.5 vs 1M avg
    hits = detect_signals(df, Extreme20Config(min_dvol_m=1), spy_bull=True, ticker="GAP")
    assert any(h["策略ID"] == "S2" for h in hits)


def test_pick_best_prefers_l1_over_s2():
    hits = [
        {"策略ID": "S2", "side": "short"},
        {"策略ID": "L1", "side": "long"},
    ]
    best = pick_best_signal(hits)
    assert best["策略ID"] == "L1"


def test_l1_skipped_when_bear():
    df = _ohlcv()
    prev = float(df["Close"].iloc[-2])
    df.iloc[-1, df.columns.get_loc("High")] = prev * 1.30
    df.iloc[-1, df.columns.get_loc("Low")] = prev * 1.18
    df.iloc[-1, df.columns.get_loc("Close")] = prev * 1.21
    df.iloc[-1, df.columns.get_loc("Volume")] = 5_000_000
    hits = detect_signals(df, Extreme20Config(min_dvol_m=1), spy_bull=False, ticker="X")
    assert not any(h["策略ID"] == "L1" for h in hits)


def test_select_combo_bull_long_and_short():
    cfg = Extreme20Config(
        enabled=("L1", "S1"),
        max_long_per_day=1,
        max_short_per_day=2,
        max_signals_per_day=3,
    )
    hits = [
        {"策略ID": "L1", "side": "long", "代码": "AAA", "成交额M": 100},
        {"策略ID": "S1", "side": "short", "代码": "BBB", "成交额M": 200},
        {"策略ID": "S1", "side": "short", "代码": "CCC", "成交额M": 150},
        {"策略ID": "L1", "side": "long", "代码": "DDD", "成交额M": 50},
    ]
    picked = select_combo_day(hits, cfg, spy_bull=True)
    ids = [p["代码"] for p in picked]
    assert ids == ["AAA", "BBB", "CCC"]


def test_select_combo_bear_uses_l2_not_l1():
    cfg = Extreme20Config(
        enabled=("L1", "S1"),
        bear_enabled=("L2", "S1"),
        max_long_per_day=1,
        max_short_per_day=1,
    )
    hits = [
        {"策略ID": "L1", "side": "long", "代码": "AAA", "成交额M": 100},
        {"策略ID": "L2", "side": "long", "代码": "BBB", "成交额M": 80},
        {"策略ID": "S1", "side": "short", "代码": "CCC", "成交额M": 90},
    ]
    picked = select_combo_day(hits, cfg, spy_bull=False)
    codes = {p["代码"] for p in picked}
    assert "AAA" not in codes
    assert codes == {"BBB", "CCC"}


def test_config_from_dict_bear_enabled():
    cfg = config_from_dict(
        {
            "enabled_strategies": ["L1", "S1"],
            "bear_enabled_strategies": ["L2", "S1"],
            "combo_mode": True,
        }
    )
    assert cfg.enabled == ("L1", "S1")
    assert cfg.bear_enabled == ("L2", "S1")
    assert cfg.combo_mode is True
