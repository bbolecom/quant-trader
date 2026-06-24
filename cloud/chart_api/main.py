"""K-line live API — deploy to Render/Railway or run locally on :8503."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.chart_live import chart_period_args, fetch_live_chart  # noqa: E402

app = FastAPI(title="Quant Trader Chart API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/v1/chart/{ticker}")
def get_chart(ticker: str, period: str = "daily") -> dict:
    yf_period, interval = chart_period_args(period)
    doc = fetch_live_chart(ticker, period=yf_period, interval=interval)
    if not doc:
        raise HTTPException(status_code=404, detail=f"No chart data for {ticker.upper()}")
    return doc
