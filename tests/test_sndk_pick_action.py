"""SNDK 舰队每日选股 · 策略动作文案。"""

from types import SimpleNamespace


def _leg(strike: float):
    return SimpleNamespace(strike=strike, oi=100, iv=0.25)


def _iron_plan():
    cs, cl, ps, pl = _leg(520), _leg(525), _leg(480), _leg(475)
    return SimpleNamespace(
        structure="iron_condor",
        legs=(cs, cl, ps, pl),
        spot=500.0,
        expiry="2026-07-17",
        dte=23,
        contracts=1,
        net_per_contract=753.0,
        collateral=500.0,
        note="",
    )


def test_format_pick_action_iron_condor():
    import sndk_iron_daily as sid

    text = sid.format_pick_action(_iron_plan())
    assert "铁鹰" in text
    assert "卖C$520/买C$525" in text
    assert "卖P$480/买P$475" in text
    assert "盈利区" in text


def test_format_pick_reason_includes_spot_and_credit():
    import sndk_iron_daily as sid

    text = sid.format_pick_reason(_iron_plan())
    assert "现价$500.00" in text
    assert "DTE23" in text
    assert "收$753" in text
