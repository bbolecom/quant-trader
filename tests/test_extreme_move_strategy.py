"""extreme_move_strategy unit tests without network access."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.extreme_move_strategy import (
    ExtremeMoveConfig,
    classify_extreme_row,
    compute_extreme_features,
    scan_ticker_events,
    scan_universe_events,
    simulate_event_trades,
    summarize_event_strategy,
)


def _base_ohlcv(n: int = 90, *, price: float = 30.0, volume: float = 5_000_000.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.full(n, price, dtype=float)
    open_ = close * 0.995
    high = close * 1.01
    low = close * 0.99
    vol = np.full(n, volume, dtype=float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def _surge_df() -> pd.DataFrame:
    df = _base_ohlcv()
    i = -6
    prev = float(df["Close"].iloc[i - 1])
    df.iloc[i, df.columns.get_loc("Open")] = prev * 1.03
    df.iloc[i, df.columns.get_loc("Close")] = prev * 1.12
    df.iloc[i, df.columns.get_loc("High")] = prev * 1.13
    df.iloc[i, df.columns.get_loc("Low")] = prev * 1.02
    df.iloc[i, df.columns.get_loc("Volume")] = 18_000_000
    # Give the simulated trade a next-session profit path.
    df.iloc[i + 1, df.columns.get_loc("Open")] = prev * 1.125
    df.iloc[i + 1, df.columns.get_loc("High")] = prev * 1.20
    df.iloc[i + 1, df.columns.get_loc("Low")] = prev * 1.11
    df.iloc[i + 1, df.columns.get_loc("Close")] = prev * 1.18
    return df


def _drop_df() -> pd.DataFrame:
    df = _base_ohlcv()
    i = -6
    prev = float(df["Close"].iloc[i - 1])
    df.iloc[i, df.columns.get_loc("Open")] = prev * 0.94
    df.iloc[i, df.columns.get_loc("Close")] = prev * 0.89
    df.iloc[i, df.columns.get_loc("High")] = prev * 0.96
    df.iloc[i, df.columns.get_loc("Low")] = prev * 0.84
    df.iloc[i, df.columns.get_loc("Volume")] = 16_000_000
    df.iloc[i + 1, df.columns.get_loc("Open")] = prev * 0.90
    df.iloc[i + 1, df.columns.get_loc("High")] = prev * 0.98
    df.iloc[i + 1, df.columns.get_loc("Low")] = prev * 0.89
    df.iloc[i + 1, df.columns.get_loc("Close")] = prev * 0.96
    return df


def test_compute_extreme_features_has_required_columns():
    feats = compute_extreme_features(_surge_df())
    for col in ["涨跌幅_pct", "成交额USD", "量比", "收盘强度", "后3日_pct"]:
        assert col in feats.columns


def test_classifies_liquid_surge():
    df = _surge_df()
    feats = compute_extreme_features(df)
    row = feats.iloc[-6]
    kind, score, note = classify_extreme_row(row, ExtremeMoveConfig(min_dollar_vol_m=1))
    assert kind == "surge_continuation"
    assert score > 0
    assert "涨" in note


def test_classifies_drop_rebound_when_not_closed_on_low():
    df = _drop_df()
    events = scan_ticker_events("DROP", df, ExtremeMoveConfig(min_dollar_vol_m=1, mode="drop"))
    assert any(e.类型 == "drop_rebound" for e in events)


def test_liquidity_filter_rejects_low_dollar_volume():
    df = _surge_df()
    events = scan_ticker_events("THIN", df, ExtremeMoveConfig(min_dollar_vol_m=1_000))
    assert events == []


def test_simulate_trades_and_summary():
    cfg = ExtremeMoveConfig(min_dollar_vol_m=1, max_positions_per_day=1, hold_days=3)
    data = {"AAA": _surge_df(), "BBB": _drop_df()}
    events = scan_universe_events(data, cfg)
    trades = simulate_event_trades(data, events, cfg, fee_bps=0, slippage_bps=0)
    summary = summarize_event_strategy(trades)
    assert not events.empty
    assert not trades.empty
    assert summary["交易次数"] == len(trades)
    assert 0 <= summary["胜率"] <= 1
