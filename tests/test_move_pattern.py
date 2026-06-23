"""资金轨迹规律单元测试。"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant.move_pattern import (
    enrich_buckets,
    extract_trajectory_features,
    mine_rules_from_panel,
    match_rule,
)


def _fake_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    ret = rng.normal(0.001, 0.02, n)
    close = 100 * (1 + ret).cumprod()
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "Close": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Volume": vol,
    }, index=idx)


def test_extract_trajectory_features():
    df = _fake_ohlcv()
    feat = extract_trajectory_features(df, forward_days=20)
    assert not feat.empty
    assert "vol_ratio" in feat.columns
    assert "fwd_20d" in feat.columns


def test_mine_rules_smoke():
    df = _fake_ohlcv(300)
    feat = enrich_buckets(extract_trajectory_features(df, forward_days=20))
    feat["代码"] = "TEST"
    # 复制多份模拟面板
    panel = pd.concat([feat] * 50, ignore_index=True)
    rules = mine_rules_from_panel(panel, min_samples=30, min_win_rate=0.50)
    assert isinstance(rules, list)


def test_match_rule():
    row = pd.Series({
        "vol_ratio桶": "1.5-2.5",
        "ret_5d桶": "0~5%",
        "dvol桶": "200M-1B",
        "above_ma50": True,
    })
    rule = {
        "conditions": {
            "vol_ratio_bucket": "1.5-2.5",
            "ret_5d_bucket": "0~5%",
            "dvol_bucket": "200M-1B",
            "above_ma50": True,
        }
    }
    assert match_rule(row, rule)
