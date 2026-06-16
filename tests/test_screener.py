"""策略选股模块测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from quant.screener import (
    SECTORS,
    ScreenFilters,
    apply_filters,
    merge_snapshot_backtest,
    quotes_to_dataframe,
    sector_cn,
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
