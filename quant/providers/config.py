"""行情数据源配置：从 Streamlit secrets 或环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    provider: str = "yahoo"          # yahoo | polygon | alpaca
    polygon_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_data_url: str = "https://data.alpaca.markets"


def _read_secret(section: str, key: str) -> str:
    try:
        import streamlit as st

        block = st.secrets.get(section, {})
        if isinstance(block, dict):
            val = block.get(key, "")
            return str(val).strip() if val else ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def load_data_config() -> DataConfig:
    """加载数据源配置（secrets 优先，其次环境变量）。"""
    provider = (
        _read_secret("data", "provider")
        or os.environ.get("DATA_PROVIDER", "")
        or "yahoo"
    ).strip().lower()

    polygon_key = (
        _read_secret("data", "polygon_api_key")
        or _read_secret("api", "polygon_key")
        or os.environ.get("POLYGON_API_KEY", "")
    ).strip()

    alpaca_key = (
        _read_secret("data", "alpaca_api_key")
        or os.environ.get("ALPACA_API_KEY", "")
    ).strip()
    alpaca_secret = (
        _read_secret("data", "alpaca_api_secret")
        or os.environ.get("ALPACA_API_SECRET", "")
    ).strip()
    alpaca_url = (
        _read_secret("data", "alpaca_data_url")
        or os.environ.get("ALPACA_DATA_URL", "")
        or "https://data.alpaca.markets"
    ).strip()

    # 未配置密钥时自动回退 Yahoo，避免无声失败。
    if provider == "polygon" and not polygon_key:
        provider = "yahoo"
    if provider == "alpaca" and (not alpaca_key or not alpaca_secret):
        provider = "yahoo"

    return DataConfig(
        provider=provider,
        polygon_api_key=polygon_key,
        alpaca_api_key=alpaca_key,
        alpaca_api_secret=alpaca_secret,
        alpaca_data_url=alpaca_url.rstrip("/"),
    )


def provider_label(cfg: DataConfig) -> str:
    labels = {
        "yahoo": "Yahoo Finance（免费 · 备用）",
        "polygon": "Polygon.io（专业 · 交易所级）",
        "alpaca": "Alpaca Markets（专业 · 券商级）",
    }
    return labels.get(cfg.provider, cfg.provider)
