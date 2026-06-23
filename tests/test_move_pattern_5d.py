"""5 日路径标签单元测试。"""

import pandas as pd
import numpy as np

from quant.move_pattern import compute_forward_path_labels


def test_path_up_within_5d():
    # 5 天内最高价比首日收盘涨 5%
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.Series([100.0] * 10, index=idx)
    high = close.copy()
    high.iloc[2] = 105.0
    low = close.copy()
    df = pd.DataFrame({"Close": close, "High": high, "Low": low})
    lab = compute_forward_path_labels(df, horizon=5, up_threshold=0.03, down_threshold=0.03)
    assert lab.loc[idx[0], "hit_up_5d"]
    assert lab.loc[idx[0], "path_up_5d"] >= 0.05 - 1e-9


def test_path_down_within_5d():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.Series([100.0] * 10, index=idx)
    low = close.copy()
    low.iloc[3] = 96.0
    high = close.copy()
    df = pd.DataFrame({"Close": close, "High": high, "Low": low})
    lab = compute_forward_path_labels(df, horizon=5, up_threshold=0.03, down_threshold=0.03)
    assert lab.loc[idx[0], "hit_down_5d"]
    assert lab.loc[idx[0], "path_down_5d"] <= -0.04 + 1e-9
