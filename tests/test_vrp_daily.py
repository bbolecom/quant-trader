"""vrp_daily.py 单元测试（无网络）。"""

from __future__ import annotations

import pandas as pd

from quant import vol_decay
from vrp_daily import build_history_row, format_notification, resolve_csp_tickers


def test_resolve_csp_tickers_default():
    cfg = {"csp": {"pool": "default"}}
    tickers = resolve_csp_tickers(cfg)
    assert "NVDA" in tickers
    assert len(tickers) >= 10


def test_resolve_csp_tickers_custom():
    cfg = {"csp": {"pool": "custom", "custom_tickers": ["aapl", "MSFT"]}}
    assert resolve_csp_tickers(cfg) == ["AAPL", "MSFT"]


def test_format_notification_with_etf_and_csp():
    sig = vol_decay.InverseEtfSignal(
        "SVIX", "x", "2024-06-01", 25.0, 24.0, 50, 0.04,
        "🟢 持有 / 可建仓", "detail",
    )
    vix = vol_decay.VixAlert(18.0, 17.0, 0.02, "🟢 正常", "ok")
    csp = pd.DataFrame({"代码": ["NVDA", "AMD"], "月化收益%": [2.1, 1.8]})
    result = {"etf_sig": sig, "vix": vix, "csp_table": csp, "errors": []}
    title, body = format_notification(result)
    assert "SVIX" in title
    assert "NVDA" in body
    assert "VIX" in body


def test_build_history_row():
    sig = vol_decay.InverseEtfSignal(
        "SVIX", "x", "2024-06-01", 25.0, 24.0, 50, 0.04,
        "🟢 持有", "detail",
    )
    csp = pd.DataFrame({"代码": ["NVDA"]})
    row = build_history_row({
        "etf": "SVIX",
        "etf_sig": sig,
        "vix": None,
        "csp_table": csp,
        "playbook": ["step1"],
        "errors": [],
    })
    assert row["反向ETF"] == "SVIX"
    assert row["CSP首选"] == "NVDA"
    assert "step1" in row["执行清单"]
