"""pattern_daily 下跌回避规则单元测试。"""

import pandas as pd

from quant.move_pattern import assess_down_avoidance, assess_up_favor, vectorized_down_mask
from quant.pattern_params import DownParams


def test_shrink_vol_top():
    row = pd.Series({"vol_ratio": 0.37, "ret_5d": 0.50, "ret_20d": 0.45, "dvol_m": 19000, "above_ma50": True})
    p = DownParams(active_avoid_rules=["D3_shrink"])
    hits = assess_down_avoidance(row, p)
    assert any(h["rule_id"] == "D3_shrink_vol_top" for h in hits)


def test_vol_dump():
    row = pd.Series({"vol_ratio": 2.8, "ret_5d": -0.08, "ret_20d": -0.05, "dvol_m": 500, "above_ma50": True})
    hits = assess_down_avoidance(row)
    assert any(h["rule_id"] == "D2_vol_dump" for h in hits)


def test_vectorized_down_mask():
    panel = pd.DataFrame([
        {"vol_ratio": 0.37, "ret_5d": 0.50, "ret_20d": 0.45, "dvol_m": 19000, "above_ma50": True},
        {"vol_ratio": 1.1, "ret_5d": 0.02, "ret_20d": 0.05, "dvol_m": 80, "above_ma50": True},
    ])
    mask = vectorized_down_mask(panel, DownParams(active_avoid_rules=["D3_shrink", "D_parabolic"]))
    assert mask.iloc[0] and not mask.iloc[1]


def test_up_favor_u1():
    row = pd.Series({"vol_ratio": 1.2, "ret_5d": 0.03, "dvol_m": 1200, "above_ma50": True})
    tags = assess_up_favor(row)
    assert any(t["rule_id"] == "U1" for t in tags)


def test_clean_row_no_hits():
    row = pd.Series({"vol_ratio": 1.1, "ret_5d": 0.02, "ret_20d": 0.05, "dvol_m": 80, "above_ma50": True})
    assert assess_down_avoidance(row) == []
