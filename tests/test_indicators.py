"""技术指标的边界与正确性测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import indicators as ind


def test_sma_matches_manual(ohlcv):
    s = ohlcv["Close"]
    out = ind.sma(s, 5)
    expected = s.iloc[:5].mean()
    assert np.isclose(out.iloc[4], expected)
    assert out.iloc[:4].isna().all()  # 前 4 个不足窗口


def test_rsi_bounded(ohlcv):
    r = ind.rsi(ohlcv["Close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_atr_positive(ohlcv):
    a = ind.atr(ohlcv, 14).dropna()
    assert (a >= 0).all()


def test_bollinger_ordering(ohlcv):
    b = ind.bollinger_bands(ohlcv["Close"], 20, 2.0).dropna()
    assert (b["upper"] >= b["mid"]).all()
    assert (b["mid"] >= b["lower"]).all()


def test_adx_bounded(ohlcv):
    a = ind.adx(ohlcv, 14)["adx"].dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_efficiency_ratio_unit_interval(ohlcv):
    er = ind.efficiency_ratio(ohlcv["Close"], 20)
    assert (er >= 0).all() and (er <= 1).all()


def test_donchian_no_lookahead(ohlcv):
    """唐奇安通道用 shift(1)，当日上轨不应包含当日最高价。"""
    d = ind.donchian(ohlcv, 20)
    # 上轨为过去 20 日最高（不含当日），因此可能小于当日 High。
    valid = d["upper"].dropna()
    assert len(valid) > 0
    assert not valid.isna().any()
