"""option_chain 选择逻辑单测（注入链 DataFrame，无网络）。"""

from __future__ import annotations

import pandas as pd

from quant.option_chain import (
    pick_bear_call,
    pick_bear_put_debit,
    pick_csp,
    pick_put_credit,
)


def _calls() -> pd.DataFrame:
    return pd.DataFrame([
        {"strike": 100, "bid": 4.0, "ask": 4.4, "openInterest": 500, "volume": 80, "impliedVolatility": 0.5},
        {"strike": 105, "bid": 2.2, "ask": 2.5, "openInterest": 300, "volume": 60, "impliedVolatility": 0.5},
        {"strike": 110, "bid": 1.1, "ask": 1.3, "openInterest": 200, "volume": 40, "impliedVolatility": 0.5},
        {"strike": 120, "bid": 0.4, "ask": 0.5, "openInterest": 150, "volume": 20, "impliedVolatility": 0.5},
    ])


def _puts() -> pd.DataFrame:
    return pd.DataFrame([
        {"strike": 80, "bid": 0.4, "ask": 0.5, "openInterest": 150, "volume": 20, "impliedVolatility": 0.5},
        {"strike": 90, "bid": 1.1, "ask": 1.3, "openInterest": 200, "volume": 40, "impliedVolatility": 0.5},
        {"strike": 95, "bid": 2.2, "ask": 2.5, "openInterest": 300, "volume": 60, "impliedVolatility": 0.5},
        {"strike": 100, "bid": 4.0, "ask": 4.4, "openInterest": 500, "volume": 80, "impliedVolatility": 0.5},
    ])


def test_bear_call_picks_real_strikes():
    short, long, why = pick_bear_call(_calls(), spot=100.0, otm=0.05, width_pct=0.04)
    assert why == ""
    assert short.strike == 105      # nearest >= 105
    assert long.strike == 110       # nearest >= 105*1.04 = 109.2
    assert short.bid - long.ask > 0  # 有正净权利金


def test_bear_call_skips_illiquid_strike():
    # 最近 OTM 行权价(105)无持仓 → 跳过稀薄档、改选下一档流动行权价(110)，而非选中 105。
    calls = _calls().copy()
    calls.loc[calls.strike == 105, "openInterest"] = 0
    short, long, why = pick_bear_call(calls, spot=100.0, otm=0.05, width_pct=0.05, min_oi=25)
    assert why == ""
    assert short is not None and short.strike == 110   # 已跳过稀薄的 105
    assert short.oi >= 25                                # 选中的卖腿本身必须流动


def test_bear_call_bails_when_all_illiquid():
    # 所有候选行权价都无持仓 → 无可成交结构，宁可观望：返回 None + 原因。
    calls = _calls().copy()
    calls["openInterest"] = 0
    short, long, why = pick_bear_call(calls, spot=100.0, otm=0.05, width_pct=0.05, min_oi=25)
    assert short is None and long is None
    assert "流动" in why                                 # “无足够 OTM 流动卖腿”


def test_bear_call_skips_wide_spread_strike():
    # 最近 OTM(105) 买卖价差过大 → 跳过它、改选下一档(110)。
    calls = _calls().copy()
    calls.loc[calls.strike == 105, "ask"] = 9.0
    short, long, why = pick_bear_call(calls, spot=100.0, otm=0.05, width_pct=0.05)
    assert why == ""
    assert short is not None and short.strike == 110


def test_put_credit_picks_below_spot():
    short, long, why = pick_put_credit(_puts(), spot=100.0, otm=0.05, width_pct=0.05)
    assert why == ""
    assert short.strike == 95
    assert long.strike < short.strike


def test_bear_put_debit_structure():
    long, short, why = pick_bear_put_debit(_puts(), spot=100.0, otm=0.0, width_pct=0.10)
    assert why == ""
    assert long.strike > short.strike   # 买高卖低
    assert long.right == "P" and short.right == "P"


def test_csp_picks_otm_put():
    leg, why = pick_csp(_puts(), spot=100.0, otm=0.05)
    assert why == ""
    assert leg.strike == 95
    assert leg.action == "sell"


def test_empty_chain_returns_reason():
    short, long, why = pick_bear_call(pd.DataFrame(), spot=100.0)
    assert short is None and "无 call 链" in why
