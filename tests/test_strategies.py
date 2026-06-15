"""策略层测试：所有策略产出合法仓位，且元数据齐全。"""

from __future__ import annotations

import pandas as pd
import pytest

from quant import strategies


@pytest.mark.parametrize("name", strategies.list_strategies())
def test_strategy_positions_valid(name, ohlcv):
    strat = strategies.get_strategy(name)
    pos = strat.generate(ohlcv, allow_short=True)
    assert isinstance(pos, pd.Series)
    assert pos.index.equals(ohlcv.index)
    assert not pos.isna().any()
    assert pos.between(-1, 1).all()


@pytest.mark.parametrize("name", strategies.list_strategies())
def test_long_only_clipping(name, ohlcv):
    strat = strategies.get_strategy(name)
    pos = strat.generate(ohlcv, allow_short=False)
    assert (pos >= 0).all()


@pytest.mark.parametrize("name", strategies.list_strategies())
def test_metadata_present(name):
    strat = strategies.get_strategy(name)
    assert strat.category
    assert strat.best_market
    assert strat.applicability


def test_registry_has_expected_count():
    assert len(strategies.list_strategies()) >= 11


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        strategies.get_strategy("不存在的策略")
