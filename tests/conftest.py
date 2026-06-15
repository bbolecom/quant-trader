"""pytest 公共夹具：构造可复现的合成行情数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(seed: int = 0, n: int = 600, drift: float = 0.0005, vol: float = 0.018) -> pd.DataFrame:
    """生成一段带高低价的合成日线行情。"""
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv(seed=42)


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    """近似线性强上行趋势。"""
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(7)
    # 低噪声的近线性上行，确保是无歧义的强趋势（ADX 明显 > 25）。
    close = np.linspace(100, 240, len(idx)) * (1 + rng.normal(0, 0.0012, len(idx)))
    close = np.abs(close)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )
    return df


@pytest.fixture
def multi_data() -> dict[str, pd.DataFrame]:
    return {f"T{i}": _make_ohlcv(seed=i, n=500) for i in range(4)}
