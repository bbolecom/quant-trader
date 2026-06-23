"""flow_options 单元测试（无网络）。"""

from __future__ import annotations

from quant.flow_options import wants_bear_call, wants_put_spread


def test_wants_put_spread_offering():
    assert wants_put_spread({
        "信号": "回避",
        "策略动作": "回避追涨",
        "下跌规律": "D_OFFERING",
    })
    assert wants_put_spread({"信号": "做空", "策略动作": "买Put价差"})
    assert not wants_bear_call({"策略动作": "买Put价差"})


def test_wants_bear_call():
    assert wants_bear_call({"策略动作": "卖Call价差", "下跌规律": "D_A1"})
    assert not wants_bear_call({"策略动作": "买Put价差"})
