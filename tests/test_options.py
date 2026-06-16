"""期权损益计算模块测试。"""

from __future__ import annotations

import numpy as np
import pytest

from quant import options as opt


def test_long_call_payoff_at_strike_and_above():
    legs = opt.long_call(strike=100.0, premium=5.0, qty=1.0)
    assert legs[0].payoff(100.0) == pytest.approx(-500.0)
    assert legs[0].payoff(110.0) == pytest.approx(500.0)


def test_long_put_payoff():
    legs = opt.long_put(strike=100.0, premium=4.0, qty=1.0)
    assert legs[0].payoff(100.0) == pytest.approx(-400.0)
    assert legs[0].payoff(90.0) == pytest.approx(600.0)


def test_bull_call_spread_max_loss_and_profit():
    legs = opt.bull_call_spread(100.0, 6.0, 110.0, 2.0, qty=1.0)
    res = opt.analyze(legs, spot=100.0, width=0.3, n=301)
    assert res.max_loss == pytest.approx(-400.0)
    assert res.max_profit == pytest.approx(600.0)
    assert len(res.breakevens) == 1
    assert res.breakevens[0] == pytest.approx(104.0)


def test_bear_put_spread_limited_risk():
    legs = opt.bear_put_spread(100.0, 5.0, 90.0, 1.0, qty=1.0)
    res = opt.analyze(legs, spot=100.0, width=0.3, n=301)
    assert res.max_loss == pytest.approx(-400.0)
    assert res.max_profit == pytest.approx(600.0)


def test_collar_limited_profit_and_loss():
    legs = opt.collar(spot=100.0, put_strike=90.0, put_premium=4.0,
                      call_strike=110.0, call_premium=2.0, qty=1.0)
    res = opt.analyze(legs, spot=100.0, width=0.4, n=401)
    assert res.max_profit == pytest.approx(800.0)
    assert res.max_loss == pytest.approx(-1200.0)
    assert len(res.breakevens) >= 1


def test_iron_condor_max_profit_is_net_credit():
    legs = opt.iron_condor(
        80, 1.0, 90, 3.0, 110, 3.0, 120, 1.0, qty=1.0,
    )
    res = opt.analyze(legs, spot=100.0, width=0.5, n=401)
    assert res.max_profit == pytest.approx(400.0)
    assert res.net_cost == pytest.approx(400.0)


def test_list_strategies_not_empty():
    assert len(opt.list_strategies()) >= 8
    assert "买入认沽 (Long Put)" in opt.STRATEGY_INFO
