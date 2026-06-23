"""SEC 融资公告 + merge 单元测试（无网络）。"""

from __future__ import annotations

import pandas as pd

from quant.sec_filings import dilution_alert_map, merge_offering_into_scan


def test_merge_offering_downgrades_long():
    scan = pd.DataFrame([
        {
            "代码": "NXTS",
            "信号": "做多",
            "策略动作": "次日做多",
            "选股理由": "温和堆量",
            "上涨规律": "U_S1",
            "下跌规律": "—",
            "涨幅%": 5.0,
        },
    ])
    alerts = {
        "NXTS": {
            "表格": "8-K",
            "公告日": "2026-06-22",
            "关键词": "registered direct offering",
            "公司": "Nexentis",
            "链接": "https://sec.gov",
        },
    }
    out = merge_offering_into_scan(scan, alerts)
    row = out.iloc[0]
    assert row["信号"] == "做空"
    assert row["策略动作"] == "买Put价差"
    assert "D_OFFERING" in str(row["下跌规律"])
    assert row["SEC融资"] == "是"
    assert "SEC" in row["选股理由"]


def test_dilution_alert_map_latest():
    df = pd.DataFrame([
        {"代码": "ABC", "公告日": "2026-06-20", "表格": "8-K", "关键词": "atm", "公司": "A"},
        {"代码": "ABC", "公告日": "2026-06-22", "表格": "8-K", "关键词": "direct", "公司": "A"},
    ])
    m = dilution_alert_map(df)
    assert "ABC" in m
    assert "direct" in m["ABC"]["关键词"]
