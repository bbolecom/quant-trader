"""meme_route 单元测试。"""

from __future__ import annotations

import pandas as pd

from quant.meme_route import (
    estimate_bear_put_debit_spread,
    parse_meme_route,
    pnl_put_spread_hold,
    route_action,
)


def _row(**kw) -> pd.Series:
    base = {
        "代码": "NVDA",
        "涨幅%": 8.0,
        "振幅%": 10.0,
        "量比": 2.0,
        "收盘强度": 0.4,
        "涨幅20d%": 15.0,
        "相对SPY20d%": 5.0,
        "SPY1d涨%": -0.5,
    }
    base.update(kw)
    return pd.Series(base)


def test_parse_defaults_spy_ma50_and_gain_12():
    mrc = parse_meme_route({})
    assert mrc.short_fade.regime_filter == "SPY_MA50"
    assert mrc.exclude_gain_pct == 12.0
    assert mrc.short_fade.structure == "put_spread"


def test_blocklist_skips_when_no_short_setup():
    mrc = parse_meme_route({"meme_route": {"enabled": True}})
    assert route_action(_row(代码="GME", **{"收盘强度": 0.8}), mrc, spy_bear=True) == "skip"


def test_extreme_gain_skips():
    mrc = parse_meme_route({"meme_route": {"enabled": True, "exclude_gain_pct": 12.0}})
    assert route_action(_row(**{"涨幅%": 13.0, "振幅%": 18.0, "收盘强度": 0.8}), mrc) == "skip"


def test_short_fade_requires_spy_bear():
    mrc = parse_meme_route({"meme_route": {"enabled": True}})
    assert route_action(_row(代码="GME"), mrc, spy_bear=False) == "skip"
    assert route_action(_row(代码="GME"), mrc, spy_bear=True) == "short_fade"


def test_normal_goes_bear_call():
    mrc = parse_meme_route({"meme_route": {"enabled": True}})
    assert route_action(_row(代码="NVDA", **{"涨幅%": 5.0, "收盘强度": 0.7}), mrc) == "bear_call"


def test_put_spread_pricing():
    kl, ks, debit, max_l, max_p = estimate_bear_put_debit_spread(100.0, 0.5)
    assert kl > ks
    assert debit > 0
    assert max_l >= debit


def test_put_spread_profits_on_drop():
    _, _, _, _, pct = pnl_put_spread_hold(100.0, 92.0, 0.6, hold_days=1)
    assert pct > 0
