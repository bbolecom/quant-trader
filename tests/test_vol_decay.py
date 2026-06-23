"""波动率衰减策略模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import vol_decay as vd


def _ohlcv(n: int = 300, drift: float = 0.0003, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    ret = rng.normal(drift, 0.015, n)
    close = start * np.cumprod(1 + ret)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": 5e6},
        index=idx,
    )


def test_realized_vol_positive():
    df = _ohlcv()
    rv = vd.realized_vol(df["Close"])
    assert rv.iloc[-1] > 0


def test_inverse_etf_signal_above_ma():
    # 强上行 → 应在均线上方
    n = 120
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.linspace(50, 120, n)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6}, index=idx)
    sig = vd.inverse_etf_signal(df, "SVIX", ma_window=50)
    assert "持有" in sig.action or "建仓" in sig.action
    assert sig.close > sig.ma


def test_inverse_etf_signal_below_ma():
    n = 120
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.linspace(120, 50, n)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6}, index=idx)
    sig = vd.inverse_etf_signal(df, "SVIX", ma_window=50)
    assert "清仓" in sig.action


def test_estimate_csp_returns_strike_and_premium():
    K, prem, yld = vd.estimate_csp(100.0, 0.40)
    assert K < 100
    assert prem > 0
    assert yld > 0


def test_analyze_csp_ticker_filters_low_volume(monkeypatch):
    df = _ohlcv(250)
    f = vd.CspFilters(min_dollar_vol_m=999999)  # 极高门槛
    assert vd.analyze_csp_ticker("TEST", df, f) is None


def test_analyze_csp_ticker_passes_good_candidate():
    df = _ohlcv(260, drift=0.001)
    f = vd.CspFilters(min_dollar_vol_m=0, min_rv_pct=0, max_rv_pct=200, require_above_ma200=False)
    cand = vd.analyze_csp_ticker("TEST", df, f)
    assert cand is not None
    assert cand.put_strike > 0
    assert cand.est_premium > 0


def test_ma_timing_backtest():
    df = _ohlcv(300)
    stats = vd.ma_timing_backtest(df["Close"], ma_window=50)
    assert "买入持有" in stats
    assert "均线择时" in stats


def test_daily_playbook_includes_steps():
    sig = vd.InverseEtfSignal("SVIX", "x", "2024-01-01", 10, 9, 50, 0.1, "🟢 持有", "ok")
    vix = vd.VixAlert(18, 17, 0.02, "🟢 正常", "fine")
    table = pd.DataFrame({"代码": ["NVDA", "AMD"]})
    steps = vd.daily_playbook(sig, vix, table)
    assert len(steps) >= 3
    assert any("SVIX" in s for s in steps)
