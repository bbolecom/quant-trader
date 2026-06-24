"""本地磁盘行情缓存 · 减少重复联网、避免 App 长时间阻塞断连。"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import date
from pathlib import Path

import pandas as pd

from .providers.base import REQUIRED_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "research" / ".cache" / "market"
META_FILE = CACHE_DIR / "_meta.json"

# 结束日为今天：6 小时刷新；历史固定区间：7 天
TTL_TODAY_SEC = 6 * 3600
TTL_HIST_SEC = 7 * 86400


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(provider: str, ticker: str, start: str, end: str, interval: str) -> str:
    raw = f"{provider}|{ticker.upper()}|{start}|{end}|{interval}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _ttl_for_end(end: str) -> int:
    try:
        end_d = pd.Timestamp(end).date()
    except (TypeError, ValueError):
        return TTL_TODAY_SEC
    if end_d >= date.today():
        return TTL_TODAY_SEC
    return TTL_HIST_SEC


def _meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.meta.json"


def _data_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"


def read_cached(
    provider: str,
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame | None:
    """读取有效缓存；过期或损坏返回 None。"""
    _ensure_dir()
    key = _key(provider, ticker, start, end, interval)
    meta_p = _meta_path(key)
    data_p = _data_path(key)
    if not meta_p.exists() or not data_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        age = time.time() - float(meta.get("ts", 0))
        if age > float(meta.get("ttl", TTL_TODAY_SEC)):
            return None
        df = pickle.loads(data_p.read_bytes())
        if df is None or df.empty:
            return None
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return None
        return df
    except (OSError, pickle.PickleError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_cached(
    provider: str,
    ticker: str,
    start: str,
    end: str,
    df: pd.DataFrame,
    interval: str = "1d",
) -> None:
    if df is None or df.empty:
        return
    _ensure_dir()
    key = _key(provider, ticker, start, end, interval)
    _data_path(key).write_bytes(pickle.dumps(df))
    _meta_path(key).write_text(
        json.dumps({
            "provider": provider,
            "ticker": ticker.upper(),
            "start": str(start),
            "end": str(end),
            "interval": interval,
            "ts": time.time(),
            "ttl": _ttl_for_end(str(end)),
            "rows": int(len(df)),
        }, ensure_ascii=False),
        encoding="utf-8",
    )


def cache_stats() -> dict:
    _ensure_dir()
    files = list(CACHE_DIR.glob("*.pkl"))
    return {"entries": len(files), "dir": str(CACHE_DIR)}


def clear_cache() -> int:
    _ensure_dir()
    n = 0
    for p in CACHE_DIR.glob("*"):
        if p.name.startswith("_"):
            continue
        p.unlink(missing_ok=True)
        n += 1
    return n
