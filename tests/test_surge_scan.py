"""暴涨扫描模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import indicators as ind
from quant.surge_scan import (
    SurgeScanConfig,
    classify_surge_row,
    compute_surge_features,
    scan_ticker_history,
)


def _make_breakout_df() -> pd.DataFrame:
    """合成：60 日横盘收口 → 1 日放量突破。"""
    n = 90
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.full(n, 30.0)
    close[60:80] = 30.0 + np.sin(np.linspace(0, 4, 20)) * 0.3
    close[80:] = 30.0
    # 突破日
    close[-1] = 33.5
    open_ = close * 0.99
    open_[-1] = 31.0
    high = np.maximum(open_, close) * 1.01
    high[-1] = 34.0
    low = np.minimum(open_, close) * 0.99
    low[-1] = 30.8
    vol = np.full(n, 2_000_000.0)
    vol[-1] = 8_000_000.0
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _make_continuation_df() -> pd.DataFrame:
    """合成：强趋势后沿上轨加速。"""
    n = 90
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(30, 38, n - 5)
    close = np.concatenate([close, np.linspace(38, 51, 5)])
    open_ = close * 0.995
    high = close * 1.025
    low = close * 0.975
    vol = np.full(n, 3_000_000.0)
    vol[-1] = 10_000_000.0
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_williams_r_range():
    df = _make_continuation_df()
    wr = ind.williams_r(df, 6)
    assert wr.iloc[-1] >= -20


def test_compute_surge_features_columns():
    df = _make_breakout_df()
    feats = compute_surge_features(df)
    for col in ["涨幅_pct", "量比", "WR", "boll_width_pctile", "创20日高"]:
        assert col in feats.columns


def test_detects_breakout_surge():
    df = _make_breakout_df()
    cfg = SurgeScanConfig(min_dvol_m=0.0, boll_squeeze_pctile=0.35)
    hits = scan_ticker_history("TEST", df, cfg)
    kinds = {h.类型 for h in hits}
    assert "breakout" in kinds


def test_detects_continuation_surge():
    df = _make_continuation_df()
    cfg = SurgeScanConfig(min_dvol_m=0.0)
    hits = scan_ticker_history("TEST", df, cfg)
    kinds = {h.类型 for h in hits}
    assert "continuation" in kinds


def test_classify_returns_none_on_quiet_day():
    df = _make_breakout_df()
    feats = compute_surge_features(df)
    row = feats.iloc[50].copy()
    row["Close"] = 30.0
    row["成交额USD"] = 100e6
    kind, _, _ = classify_surge_row(row, SurgeScanConfig(min_dvol_m=0.0))
    assert kind is None
