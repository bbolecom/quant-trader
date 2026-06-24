"""暴涨/暴跌池模块测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.surge_drop_pool import SurgeDropFilter, passes_filter, profile_ticker


def _make_extreme_df() -> pd.DataFrame:
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    ret = rng.normal(0, 0.04, n)
    ret[rng.random(n) < 0.04] = 0.12
    ret[rng.random(n) < 0.04] = -0.12
    close = 20 * np.cumprod(1 + ret)
    vol = np.full(n, 5_000_000.0)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": vol,
        },
        index=idx,
    )


def test_profile_ticker_metrics():
    prof = profile_ticker(_make_extreme_df(), ticker="TEST")
    assert prof is not None
    assert prof["年均暴涨天"] > 0
    assert prof["年均暴跌天"] > 0
    assert prof["综合分"] > 0


def test_passes_filter():
    prof = {
        "年均暴涨天": 5.0,
        "年均暴跌天": 4.0,
        "年均极端天": 9.0,
        "成交额M": 100.0,
        "价": 25.0,
        "年化波动": 0.8,
    }
    assert passes_filter(prof, SurgeDropFilter())
