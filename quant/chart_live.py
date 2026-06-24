"""Live OHLCV chart fetch (yfinance) — shared by API, export, daily_pick."""

from __future__ import annotations

from datetime import datetime
from typing import Any

PERIOD_MAP: dict[str, tuple[str, str]] = {
    "daily": ("6mo", "1d"),
    "weekly": ("2y", "1wk"),
    "monthly": ("5y", "1mo"),
    "6mo": ("6mo", "1d"),
    "2y": ("2y", "1wk"),
    "5y": ("5y", "1mo"),
}


def chart_period_args(period: str) -> tuple[str, str]:
    key = (period or "daily").strip().lower()
    return PERIOD_MAP.get(key, ("6mo", "1d"))


def fetch_live_chart(
    ticker: str,
    *,
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, Any] | None:
    import yfinance as yf

    sym = (ticker or "").strip().upper()
    if not sym or sym in {"—", "-"}:
        return None
    try:
        hist = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
    except Exception:  # noqa: BLE001
        return None
    if hist is None or hist.empty:
        return None

    bars: list[dict[str, Any]] = []
    for idx, row in hist.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        bars.append(
            {
                "date": d.isoformat(),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": round(float(row.get("Volume") or 0)),
            }
        )
    return {
        "ticker": sym,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": period,
        "interval": interval,
        "source": "yfinance-live",
        "bars": bars,
    }
