"""磁盘行情缓存测试。"""

from __future__ import annotations

import pandas as pd

from quant.market_cache import read_cached, write_cached, clear_cache


def test_market_cache_roundtrip():
    clear_cache()
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({
        "Open": [1, 2, 3, 4, 5],
        "High": [1.1, 2.1, 3.1, 4.1, 5.1],
        "Low": [0.9, 1.9, 2.9, 3.9, 4.9],
        "Close": [1, 2, 3, 4, 5],
        "Volume": [100, 100, 100, 100, 100],
    }, index=idx)
    write_cached("yahoo", "TEST", "2024-01-01", "2024-01-10", df)
    hit = read_cached("yahoo", "TEST", "2024-01-01", "2024-01-10")
    assert hit is not None
    assert len(hit) == 5
    assert hit["Close"].iloc[-1] == 5
