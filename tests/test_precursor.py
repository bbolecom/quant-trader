"""异动前兆扫描模块测试。"""

from __future__ import annotations

from quant import precursor


def test_scan_precursors_does_not_crash(ohlcv):
    hits = precursor.scan_precursors(ohlcv)
    assert isinstance(hits, list)


def test_scan_universe_returns_dataframe(multi_data):
    table = precursor.scan_universe(multi_data, min_score=0.0)
    assert not table.empty
    assert "代码" in table.columns
    assert "前兆得分" in table.columns
