"""K-line live API — deploy to Render/Railway or run locally on :8503.

防御性加固（v1.1）：
- ticker 白名单正则校验，绝不把脏输入直接下传给数据源；
- 单实例内存滑动窗口限流，防止被当作免费 yfinance 代理刷爆；
- CORS 允许来源可由环境变量收紧（默认 "*" 保持对外行为不变）。
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.chart_live import chart_period_args, fetch_live_chart  # noqa: E402

# --- 配置（环境变量可覆盖；默认值保证对外行为与旧版一致） ---
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("CHART_API_ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
] or ["*"]
_RATE_LIMIT_PER_MIN = max(1, int(os.environ.get("CHART_API_RATE_LIMIT_PER_MIN", "60")))
_RATE_WINDOW_SEC = 60.0

# 合法美股代码：字母开头，允许字母/数字/点/连字符，长度 1~10（覆盖 BRK.B、SPY 等）。
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_ticker(raw: str) -> str:
    """规范化并校验股票代码；非法直接 400，杜绝脏输入下传数据源。"""
    sym = (raw or "").strip().upper()
    if not _TICKER_RE.match(sym):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")
    return sym


class _SlidingWindowLimiter:
    """单实例内存滑动窗口限流（Render free 单实例，足够）。线程安全。"""

    def __init__(self, limit: int, window: float) -> None:
        self._limit = max(1, limit)
        self._window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._limit:
                return False
            q.append(now)
            # 机会式清理空队列，防止 IP 字典无界增长。
            if len(self._hits) > 4096:
                for stale in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale]
            return True


_limiter = _SlidingWindowLimiter(_RATE_LIMIT_PER_MIN, _RATE_WINDOW_SEC)


def _client_key(request: Request) -> str:
    """取真实客户端 IP：Render 等反代把真实 IP 放在 X-Forwarded-For。"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 健康检查不计入限流，避免 Render 探活被拒。
        if request.url.path != "/health" and not _limiter.allow(_client_key(request)):
            return JSONResponse(status_code=429, content={"detail": "Too many requests"})
        return await call_next(request)


app = FastAPI(title="Quant Trader Chart API", version="1.1")
# 注意中间件顺序：先加限流（内层），后加 CORS（外层），
# 这样 429 响应也能带上 CORS 头，浏览器侧可正常读取。
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/v1/chart/{ticker}")
def get_chart(ticker: str, period: str = "daily") -> dict:
    sym = normalize_ticker(ticker)
    yf_period, interval = chart_period_args(period)
    doc = fetch_live_chart(sym, period=yf_period, interval=interval)
    if not doc:
        raise HTTPException(status_code=404, detail=f"No chart data for {sym}")
    return doc
