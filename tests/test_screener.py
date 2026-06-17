"""策略选股模块测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.screener import (
    SECTORS,
    ScreenFilters,
    add_rationale_to_merged,
    apply_filters,
    forward_backward_metrics,
    merge_snapshot_backtest,
    pick_rationale,
    quotes_to_dataframe,
    run_historical_daily_screen,
    screen_at_date,
    sector_cn,
    snapshot_at_date,
    stamp_selection_date,
    summarize_backtest,
)


def test_quotes_to_dataframe_parses_fields():
    resp = {
        "quotes": [
            {
                "symbol": "AAA",
                "shortName": "Alpha",
                "regularMarketPrice": 10.0,
                "regularMarketVolume": 1_000_000,
                "sharesOutstanding": 100_000_000,
                "marketCap": 5_000_000_000,
                "regularMarketChangePercent": 5.5,
                "sector": "Technology",
            }
        ]
    }
    df = quotes_to_dataframe(resp)
    assert len(df) == 1
    assert df.iloc[0]["代码"] == "AAA"
    assert df.iloc[0]["涨幅%"] == pytest.approx(5.5)
    assert df.iloc[0]["换手率%"] == pytest.approx(1.0)
    assert df.iloc[0]["成交额USD"] == pytest.approx(10_000_000.0)


def test_apply_filters_gain_and_mcap():
    snapshot = pd.DataFrame(
        [
            {"代码": "A", "涨幅%": 12.0, "成交额USD": 50e6, "换手率%": 2.0, "市值USD": 20e9},
            {"代码": "B", "涨幅%": -5.0, "成交额USD": 50e6, "换手率%": 2.0, "市值USD": 20e9},
            {"代码": "C", "涨幅%": 8.0, "成交额USD": 1e6, "换手率%": 2.0, "市值USD": 20e9},
        ]
    )
    f = ScreenFilters(min_gain_pct=0, max_gain_pct=100, min_dollar_vol_m=10, min_mcap_b=1, max_mcap_b=100)
    out = apply_filters(snapshot, f)
    assert list(out["代码"]) == ["A"]


def test_sector_cn_maps_known_and_unknown():
    assert sector_cn("Technology") == "科技"
    assert sector_cn("") == ""
    assert sector_cn("Unknown Sector") == "Unknown Sector"


def test_quotes_to_dataframe_keeps_sector_columns():
    resp = {"quotes": [{"symbol": "AAA", "regularMarketPrice": 10.0,
                        "regularMarketVolume": 100, "sector": "Technology"}]}
    df = quotes_to_dataframe(resp)
    assert df.iloc[0]["_行业EN"] == "Technology"
    assert df.iloc[0]["行业"] == "科技"


def test_apply_filters_sector_filter():
    snapshot = pd.DataFrame(
        [
            {"代码": "TECH", "涨幅%": 5.0, "成交额USD": 50e6, "换手率%": 2.0,
             "市值USD": 20e9, "_行业EN": "Technology", "行业": "科技"},
            {"代码": "BANK", "涨幅%": 6.0, "成交额USD": 50e6, "换手率%": 2.0,
             "市值USD": 20e9, "_行业EN": "Financial Services", "行业": "金融"},
            {"代码": "NODATA", "涨幅%": 7.0, "成交额USD": 50e6, "换手率%": 2.0,
             "市值USD": 20e9, "_行业EN": "", "行业": ""},
        ]
    )
    f = ScreenFilters(min_dollar_vol_m=10, min_mcap_b=1, max_mcap_b=100,
                      sectors=["Technology"])
    out = apply_filters(snapshot, f)
    # 科技保留，金融剔除，行业缺失的不被硬排除
    assert "TECH" in set(out["代码"])
    assert "BANK" not in set(out["代码"])
    assert "NODATA" in set(out["代码"])


def test_all_sectors_have_chinese_labels():
    assert "Technology" in SECTORS
    assert all(isinstance(v, str) and v for v in SECTORS.values())


def test_merge_and_summarize():
    snapshot = pd.DataFrame([{"代码": "A", "名称": "Alpha", "涨幅%": 10.0}])
    bt = pd.DataFrame(
        [
            {
                "代码": "A",
                "策略累计收益": 0.2,
                "策略年化收益": 0.1,
                "基准收益": 0.05,
                "超额收益": 0.15,
                "夏普比率": 1.2,
                "最大回撤": -0.08,
                "胜率": 0.6,
                "交易次数": 5,
                "期末资金": 120_000,
                "当前信号": "🟢 做多",
            }
        ]
    )
    merged = merge_snapshot_backtest(snapshot, bt)
    assert len(merged) == 1
    summ = summarize_backtest(bt)
    assert summ["入选数量"] == 1.0
    assert summ["盈利标的占比"] == 1.0


def test_pick_rationale_includes_gain_and_rank():
    row = pd.Series({"代码": "AAPL", "涨幅%": 12.5, "成交额USD": 50e6, "行业": "科技"})
    f = ScreenFilters(min_gain_pct=0, max_gain_pct=100, min_dollar_vol_m=10, lookback_days=20)
    text = pick_rationale(row, f, rank=1)
    assert "近20日涨幅" in text
    assert "排名第 1" in text


def test_forward_backward_metrics(ohlcv):
    as_of = ohlcv.index[200]
    m = forward_backward_metrics(ohlcv, as_of, forward_days=10, backward_days=10)
    assert "入选价" in m
    assert "后10日收益" in m
    assert "前10日最大回撤" in m


def test_screen_at_date_and_historical_replay(multi_data):
    f = ScreenFilters(min_gain_pct=-100, max_gain_pct=1000, min_dollar_vol_m=0, lookback_days=5)
    best = max(multi_data.keys(), key=lambda t: len(multi_data[t]))
    as_of = multi_data[best].index[120]
    picks = screen_at_date(multi_data, f, as_of, top_n=2)
    assert not picks.empty
    assert "选股日期" in picks.columns
    assert "选股理由" in picks.columns

    start = multi_data[best].index[80].strftime("%Y-%m-%d")
    end = multi_data[best].index[200].strftime("%Y-%m-%d")
    hres = run_historical_daily_screen(
        multi_data, f, start=start, end=end,
        rebalance_days=20, top_picks=2, forward_days=10, backward_days=10,
    )
    daily = hres["daily_picks"]
    assert not daily.empty
    assert "选股理由" in daily.columns


def test_add_rationale_to_merged():
    merged = pd.DataFrame([{"代码": "A", "涨幅%": 5.0}])
    f = ScreenFilters(lookback_days=10)
    out = add_rationale_to_merged(merged, f, "2024-06-15")
    assert "选股理由" in out.columns
    assert out["选股日期"].iloc[0] == "2024-06-15"
    assert "选股日 2024-06-15" in out["选股理由"].iloc[0]


def test_stamp_selection_date_first_column():
    df = pd.DataFrame([{"代码": "A", "涨幅%": 1.0}])
    out = stamp_selection_date(df, "2023-01-05")
    assert out.columns[0] == "选股日期"
    assert out["选股日期"].iloc[0] == "2023-01-05"
