"""Tests for the public chart API hardening (ticker validation + rate limit).

纯单元测试，不依赖 httpx/TestClient，也不联网。
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cloud.chart_api.main import _SlidingWindowLimiter, normalize_ticker


@pytest.mark.parametrize("raw,expected", [
    ("spy", "SPY"),
    ("  aapl ", "AAPL"),
    ("brk.b", "BRK.B"),
    ("brk-b", "BRK-B"),
])
def test_normalize_ticker_valid(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "—",
    "1ABC",                    # 不能数字开头
    "TOOOOOLONGSYM",           # 超过 10 位
    "AAPL; DROP TABLE",        # 注入式脏输入
    "../etc/passwd",
    "A B",
])
def test_normalize_ticker_invalid(raw: str) -> None:
    with pytest.raises(HTTPException) as exc:
        normalize_ticker(raw)
    assert exc.value.status_code == 400


def test_rate_limiter_allows_then_blocks() -> None:
    limiter = _SlidingWindowLimiter(limit=3, window=60.0)
    assert limiter.allow("1.1.1.1") is True
    assert limiter.allow("1.1.1.1") is True
    assert limiter.allow("1.1.1.1") is True
    # 第 4 次超过窗口内额度 → 拒绝
    assert limiter.allow("1.1.1.1") is False
    # 不同 key 独立计数
    assert limiter.allow("2.2.2.2") is True


def test_rate_limiter_window_slides() -> None:
    limiter = _SlidingWindowLimiter(limit=2, window=0.05)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False
    import time
    time.sleep(0.06)  # 窗口滑过，旧命中过期
    assert limiter.allow("k") is True
