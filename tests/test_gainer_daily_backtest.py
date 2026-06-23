"""gainer_daily_backtest 单元测试（无网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.gainer_daily_backtest import (
    GainerProFilters,
    apply_pro_filters,
    filters_for_mode,
    pick_top_gainers,
    pro_snapshot_at_date,
    score_gainers,
    ultra_high_win_filters,
)


def _mock_data(n: int = 120, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    tickers = tickers or ["AAA", "BBB", "CCC"]
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    out = {}
    for j, t in enumerate(tickers):
        close = 100 * (1 + 0.002 * (np.arange(n) + j * 0.1))
        close[-1] *= 1.05 + j * 0.01  # 最后一日不同涨幅
        vol = np.full(n, 5_000_000.0)
        vol[-1] = 8_000_000.0 + j * 1_000_000
        out[t] = pd.DataFrame({"Close": close, "Volume": vol}, index=idx)
    spy = pd.DataFrame({"Close": 100 * (1 + 0.001 * np.arange(n))}, index=idx)
    out["_SPY"] = spy
    return out


def test_pro_snapshot_has_factors():
    data = _mock_data()
    spy = data["_SPY"]["Close"]
    snap = pro_snapshot_at_date({k: v for k, v in data.items() if k != "_SPY"}, data["_SPY"].index[-1], spy)
    assert not snap.empty
    assert "量比" in snap.columns
    assert "站上MA20" in snap.columns


def test_score_gainers_orders():
    df = pd.DataFrame([
        {"代码": "A", "涨幅%": 5, "量比": 2, "相对SPY20d%": 3, "涨幅20d%": 10, "站上MA20": True},
        {"代码": "B", "涨幅%": 3, "量比": 1.5, "相对SPY20d%": 1, "涨幅20d%": 5, "站上MA20": True},
    ])
    ranked = score_gainers(df)
    assert ranked.iloc[0]["代码"] == "A"
    assert "综合分" in ranked.columns


def test_pick_top_gainers():
    data = _mock_data(tickers=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    spy = data["_SPY"]["Close"]
    as_of = data["AAA"].index[-1]
    filt = GainerProFilters(
        min_dollar_vol_m=0, min_mcap_b=0, min_gain_pct=-100, max_gain_pct=100,
        min_vol_ratio=0, max_vol_ratio=100, min_rs_20d_pct=-999, max_rs_20d_pct=999,
        require_above_ma20=False, require_above_ma50=False, require_rs_vs_spy=False,
        require_spy_above_ma20=False, require_spy_positive_5d=False,
        require_spy_positive_1d=False, require_green_candle=False, min_close_strength=0,
        min_gain_20d_pct=-999, max_gain_20d_pct=999,
        min_setup_win_rate=0, min_setup_samples=0,
        top_n=3, min_candidates=1,
    )
    top = pick_top_gainers(
        {k: v for k, v in data.items() if k != "_SPY"}, as_of, spy, filt,
    )
    assert len(top) <= 3
    assert "选股理由" in top.columns


def test_filters_for_mode():
    ultra = filters_for_mode("ultra")
    assert ultra.max_gain_pct == 3.5
    assert ultra.require_spy_positive_1d is True
    assert ultra.min_spy_1d_pct == 0.4
    weekly = filters_for_mode("weekly")
    assert weekly.min_setup_win_rate == 0.55
    assert filters_for_mode("legacy").top_n == 5


def test_ultra_high_win_stricter_than_highwin():
    ultra = ultra_high_win_filters()
    high = filters_for_mode("highwin")
    assert ultra.max_gain_pct <= high.max_gain_pct
    assert ultra.min_close_strength >= high.min_close_strength
