"""行情数据源注册与工厂。"""

from __future__ import annotations

from functools import lru_cache

from .alpaca import AlpacaProvider
from .base import DataError, MarketDataProvider
from .config import DataConfig, load_data_config, provider_label
from .polygon import PolygonProvider
from .yahoo import YahooProvider

__all__ = [
    "DataConfig",
    "MarketDataProvider",
    "get_provider",
    "load_data_config",
    "provider_label",
    "reset_provider_cache",
]


def _build_provider(cfg: DataConfig) -> MarketDataProvider:
    if cfg.provider == "polygon":
        return PolygonProvider(cfg.polygon_api_key)
    if cfg.provider == "alpaca":
        return AlpacaProvider(cfg.alpaca_api_key, cfg.alpaca_api_secret, cfg.alpaca_data_url)
    if cfg.provider == "yahoo":
        return YahooProvider()
    raise DataError(f"未知行情数据源：{cfg.provider}（可选 yahoo / polygon / alpaca）")


@lru_cache(maxsize=4)
def _cached_provider(provider: str, polygon_key: str, alpaca_key: str, alpaca_secret: str, alpaca_url: str):
    cfg = DataConfig(
        provider=provider,
        polygon_api_key=polygon_key,
        alpaca_api_key=alpaca_key,
        alpaca_api_secret=alpaca_secret,
        alpaca_data_url=alpaca_url,
    )
    return _build_provider(cfg)


def get_provider(cfg: DataConfig | None = None) -> MarketDataProvider:
    """获取当前配置的行情数据源实例。"""
    c = cfg or load_data_config()
    return _cached_provider(
        c.provider, c.polygon_api_key, c.alpaca_api_key, c.alpaca_api_secret, c.alpaca_data_url,
    )


def reset_provider_cache() -> None:
    """测试或切换配置后清缓存。"""
    _cached_provider.cache_clear()
