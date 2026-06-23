"""单测：quant.options_history 的选腿与结算纯逻辑（注入 DataFrame，不联网）。"""

import pandas as pd
import pytest

from quant.options_history import (
    nearest_expiry,
    pick_bear_call_eod,
    pick_csp_eod,
    pick_bear_put_debit_eod,
    settle_bear_call_at_expiry,
    settle_csp_at_expiry,
    settle_bear_put_debit_at_expiry,
)


def _chain(day="2024-06-14", exp="2024-06-28", spot=210.0):
    rows = []
    for strike in [180, 190, 200, 210, 220, 230, 240, 250]:
        # 真实型：内在值 + 随价外距离衰减的时间价值
        tv = max(0.05, 10.0 - 0.12 * abs(strike - spot))
        c_mid = max(0.0, spot - strike) + tv
        rows.append({"date": day, "expiration": exp, "strike": float(strike),
                     "call_put": "Call", "bid": round(c_mid - 0.1, 2), "ask": round(c_mid + 0.1, 2),
                     "vol": 0.30, "delta": 0.5})
        p_mid = max(0.0, strike - spot) + tv
        rows.append({"date": day, "expiration": exp, "strike": float(strike),
                     "call_put": "Put", "bid": round(p_mid - 0.1, 2), "ask": round(p_mid + 0.1, 2),
                     "vol": 0.32, "delta": -0.5})
    df = pd.DataFrame(rows)
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


def test_nearest_expiry_window():
    df = _chain()
    exp = nearest_expiry(df, "2024-06-14", min_dte=2, max_dte=45)
    assert exp == pd.Timestamp("2024-06-28")


def test_nearest_expiry_empty():
    assert nearest_expiry(pd.DataFrame(), "2024-06-14") is None


def test_bear_call_picks_real_otm_strikes():
    df = _chain()
    plan, why = pick_bear_call_eod(df, spot=210.0, day="2024-06-14", otm=0.05, width_pct=0.04)
    assert plan is not None, why
    # 卖腿 >= 210*1.05=220.5 → 230；买腿 >= 230*1.04=239.2 → 240
    assert plan["short_strike"] == 230.0
    assert plan["long_strike"] == 240.0
    assert plan["credit"] > 0
    assert plan["width"] == 10.0


def test_bear_call_no_strike_far_otm():
    df = _chain()
    plan, why = pick_bear_call_eod(df, spot=210.0, day="2024-06-14", otm=0.30)
    assert plan is None
    assert "OTM" in why or "保护" in why


def test_csp_picks_otm_put():
    df = _chain()
    plan, why = pick_csp_eod(df, spot=210.0, day="2024-06-14", otm=0.05)
    assert plan is not None, why
    # <= 210*0.95 = 199.5 → 取最高的 190
    assert plan["short_strike"] == 190.0
    assert plan["credit"] > 0
    assert plan["collateral"] == 190.0 * 100


def test_bear_put_debit_picks():
    df = _chain()
    plan, why = pick_bear_put_debit_eod(df, spot=210.0, day="2024-06-14", otm=0.0, width_pct=0.05)
    assert plan is not None, why
    assert plan["long_strike"] > plan["short_strike"]
    assert plan["debit"] > 0


def test_settle_bear_call_win_when_below_short():
    plan = {"short_strike": 225.0, "long_strike": 240.0, "credit": 1.0, "width": 15.0}
    # 到期收盘 200 < 卖腿 → 价差归零，赚满 credit
    assert settle_bear_call_at_expiry(plan, 200.0) == pytest.approx(100.0)
    # 到期 250 > 买腿 → 最大亏 = (width-credit)*100
    assert settle_bear_call_at_expiry(plan, 250.0) == pytest.approx((1.0 - 15.0) * 100)


def test_settle_csp():
    plan = {"short_strike": 200.0, "credit": 2.0}
    assert settle_csp_at_expiry(plan, 210.0) == pytest.approx(200.0)  # OTM 赚满
    assert settle_csp_at_expiry(plan, 190.0) == pytest.approx((2.0 - 10.0) * 100)


def test_settle_bear_put_debit():
    plan = {"long_strike": 200.0, "short_strike": 190.0, "debit": 3.0, "width": 10.0}
    # 跌到 185 → 价差满值 10，赚 (10-3)*100
    assert settle_bear_put_debit_at_expiry(plan, 185.0) == pytest.approx(700.0)
    # 涨到 210 → 价差归零，亏掉 debit
    assert settle_bear_put_debit_at_expiry(plan, 210.0) == pytest.approx(-300.0)
