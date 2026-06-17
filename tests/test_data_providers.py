"""行情数据源单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant.data import fetch_history, get_data_source_info
from quant.providers import get_provider, load_data_config, reset_provider_cache
from quant.providers.base import normalize_ohlcv
from quant.providers.config import DataConfig
from quant.providers.polygon import PolygonProvider
from quant.providers.yahoo import YahooProvider


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    reset_provider_cache()
    yield
    reset_provider_cache()


def test_normalize_ohlcv_flattens_multiindex():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw = pd.DataFrame(
        {
            ("Close", "AAPL"): [100.0, 101.0, 102.0],
            ("Open", "AAPL"): [99.0, 100.0, 101.0],
            ("High", "AAPL"): [101.0, 102.0, 103.0],
            ("Low", "AAPL"): [98.0, 99.0, 100.0],
            ("Volume", "AAPL"): [1e6, 1.1e6, 1.2e6],
        },
        index=idx,
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    out = normalize_ohlcv(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 3


def test_load_data_config_falls_back_without_polygon_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("DATA_PROVIDER", "polygon")
    # 隔离本地 secrets.toml，确保测试只看环境变量。
    monkeypatch.setattr("quant.providers.config._load_secrets_file", lambda: {})
    cfg = load_data_config()
    assert cfg.provider == "yahoo"


def test_polygon_provider_parses_response():
    provider = PolygonProvider("test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"t": 1704067200000, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 1000},
            {"t": 1704153600000, "o": 10.5, "h": 12, "l": 10, "c": 11.5, "v": 1200},
        ]
    }
    with patch("quant.providers.polygon.requests.get", return_value=mock_resp):
        df = provider.fetch_history("AAPL", "2024-01-01", "2024-01-05")
    assert len(df) == 2
    assert "Close" in df.columns
    assert float(df["Close"].iloc[-1]) == pytest.approx(11.5)


def test_get_provider_respects_config():
    reset_provider_cache()
    p = get_provider(DataConfig(provider="yahoo"))
    assert isinstance(p, YahooProvider)


def test_get_data_source_info():
    info = get_data_source_info()
    assert "provider" in info
    assert "label" in info
