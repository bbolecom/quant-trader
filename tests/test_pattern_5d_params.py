"""5日路径寻优掩码测试。"""

import pandas as pd

from quant.pattern_5d_params import ExtendedUpFilters, up_mask


def test_up_mask_momentum():
    row = pd.DataFrame([{
        "vol_ratio": 2.6,
        "ret_5d": 0.22,
        "close_strength": 0.62,
        "dvol_m": 500,
        "换手率%": 1.2,
        "above_ma50": True,
        "above_ma20": True,
        "up_vol_share": 0.5,
    }])
    p = ExtendedUpFilters(min_vol_ratio=2.5, min_ret_5d=0.20, min_turnover_pct=1.0)
    assert bool(up_mask(row, p).iloc[0])
