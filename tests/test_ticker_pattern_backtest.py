"""规律策略回测 smoke test。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.ticker_pattern_strategy import long_momentum as _long_momentum, short_shrink_top as _short_shrink_top
from research.ticker_pattern_backtest import STRATEGIES, backtest_strategy, summarize_trades


def test_signal_helpers():
    row = pd.Series({
        "vol_ratio": 2.5,
        "ret_5d": 0.08,
        "ret_20d": 0.35,
        "above_ma50": True,
        "换手率%": 3.0,
        "close_strength": 0.6,
        "ret_1d": 0.02,
        "代码": "SMCI",
    })
    assert _long_momentum(row, spy_bull=True)
    top = pd.Series({"vol_ratio": 0.8, "ret_5d": 0.18, "ret_20d": 0.2, "above_ma50": True})
    assert _short_shrink_top(top)


def test_summarize_empty():
    s = summarize_trades(pd.DataFrame())
    assert s["笔数"] == 0
    assert s["胜率"] == 0.0


def test_strategies_defined():
    ids = {s.id for s in STRATEGIES}
    assert "S1" in ids and "S7" in ids and "S8" not in ids  # S8 is combo variant
