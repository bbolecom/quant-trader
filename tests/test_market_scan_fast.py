"""全市场快扫单元测试（无网络）。"""

from __future__ import annotations

import pandas as pd

from quant.market_scan_fast import ScanConfig, _base_filter, _match_rules, _row_to_signal, run_market_scan


def test_match_rules_gainer10():
    cfg = ScanConfig()
    row = pd.Series({
        "代码": "TEST", "名称": "Test", "最新价": 10.0, "涨幅%": 12.0,
        "成交额USD": 150e6, "_gain": 12.0, "_dvol_m": 150.0, "_price": 10.0,
    })
    tags = _match_rules(row, cfg)
    assert "Gainer10+" in tags


def test_base_filter_min_liquidity():
    cfg = ScanConfig(min_dvol_m=10.0, min_price=2.0)
    df = pd.DataFrame([
        {"代码": "A", "最新价": 5.0, "涨幅%": 8.0, "成交额USD": 20e6},
        {"代码": "B", "最新价": 1.0, "涨幅%": 20.0, "成交额USD": 200e6},
        {"代码": "C", "最新价": 8.0, "涨幅%": 3.0, "成交额USD": 1e6},
    ])
    out = _base_filter(df, cfg)
    assert set(out["代码"]) == {"A"}


def test_row_to_signal_real_quotes():
    row = pd.Series({
        "代码": "AAPL", "名称": "Apple", "_gain": 11.0, "_dvol_m": 120.0,
        "_price": 200.0, "_来源": "当日涨幅榜", "行业": "科技",
    })
    sig = _row_to_signal(row, ["Gainer10+"], {"AAPL": {"现价": 200.0, "振幅%": 5.0, "RV%": 40.0}})
    assert sig["数据源"] == "真实行情"
    assert sig["数据有效"] is True
    assert sig["代码"] == "AAPL"


def test_run_market_scan_mock(monkeypatch):
    snap = pd.DataFrame([
        {
            "代码": "NVDA", "名称": "NVIDIA", "最新价": 900.0, "涨幅%": 12.5,
            "成交额USD": 5e9, "成交量": 1e7, "换手率%": 2.0, "市值USD": 2e12,
            "_行业EN": "Technology", "行业": "科技", "_来源": "当日涨幅榜",
        },
        {
            "代码": "PENNY", "名称": "Penny", "最新价": 1.0, "涨幅%": 25.0,
            "成交额USD": 1e6, "成交量": 1e6, "换手率%": 1.0, "市值USD": 1e8,
            "_行业EN": "", "行业": "", "_来源": "小盘活跃",
        },
    ])

    def fake_parallel(*_a, **_k):
        return snap

    def fake_rv(tickers, **_k):
        return {t: {"现价": 900.0, "振幅%": 4.0, "RV%": 35.0} for t in tickers}

    monkeypatch.setattr("quant.market_scan_fast.fetch_yahoo_screens_parallel", fake_parallel)
    monkeypatch.setattr("quant.market_scan_fast._enrich_rv_batch", fake_rv)

    doc = run_market_scan(ScanConfig(enrich_rv=True, enrich_top_n=10))
    assert doc["within_budget"] is True
    assert doc["scan_stats"]["universe"] == 2
    codes = [s["代码"] for s in doc["signals"]]
    assert "NVDA" in codes
    assert "PENNY" not in codes
