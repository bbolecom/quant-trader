"""行情数据源配置：从 Streamlit secrets 或环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    provider: str = "yahoo"          # yahoo | polygon | alpaca
    polygon_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_data_url: str = "https://data.alpaca.markets"


@lru_cache(maxsize=1)
def _load_secrets_file() -> dict:
    """直接解析 .streamlit/secrets.toml（供非 Streamlit 运行时，如 CLI 脚本/测试）。"""
    # 项目根：config.py -> providers -> quant -> 项目根
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            try:
                import tomllib  # Python 3.11+
                with open(path, "rb") as f:
                    return tomllib.load(f)
            except ModuleNotFoundError:
                try:
                    import tomli
                    with open(path, "rb") as f:
                        return tomli.load(f)
                except ModuleNotFoundError:
                    import toml
                    with open(path, "r", encoding="utf-8") as f:
                        return toml.load(f)
        except Exception:  # noqa: BLE001
            continue
    return {}


def _extract(block: object, key: str) -> str:
    """从 dict / Streamlit AttrDict 等映射对象里取值。"""
    if block is None:
        return ""
    getter = getattr(block, "get", None)
    if callable(getter):
        try:
            val = getter(key, "")
        except TypeError:
            val = getter(key)
    elif isinstance(block, dict):
        val = block.get(key, "")
    else:
        return ""
    return str(val).strip() if val else ""


def _read_secret(section: str, key: str) -> str:
    # 1) Streamlit 运行时（st.secrets 的子段是 AttrDict，并非 dict 子类）
    try:
        import streamlit as st

        val = _extract(st.secrets.get(section, {}), key)
        if val:
            return val
    except Exception:  # noqa: BLE001
        pass
    # 2) 直接读 secrets.toml（CLI / 测试）
    val = _extract(_load_secrets_file().get(section, {}), key)
    if val:
        return val
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
