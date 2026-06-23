"""SEC EDGAR 8-K 融资/稀释公告扫描（NXTS 类「先拉后融」检测）。

数据源：SEC full-text search (efts.sec.gov)。
合规：须设置 User-Agent（见 SEC_FILING_UA 环境变量）。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_UA = os.environ.get(
    "SEC_FILING_UA",
    "QuantResearch flow_daily/1.0 (contact: research@localhost)",
)

_CACHE_PATH = Path(__file__).resolve().parents[1] / "research" / "sec_filings_cache.json"

DILUTION_KEYWORDS = (
    "registered direct offering",
    "securities purchase agreement",
    "at-the-market",
    "atm offering",
    "equity offering",
    "private placement",
    "warrants to purchase",
    "dilution",
    "shelf registration",
)

FORM_TYPES = ("8-K", "S-3", "424B5", "424B2")


def _http_get(url: str, *, timeout: float = 25.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, *, timeout: float = 25.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_tickers_from_hit(hit: dict) -> list[str]:
    tickers: list[str] = []
    for key in ("tickers", "ticker"):
        val = hit.get(key)
        if isinstance(val, list):
            tickers.extend(str(t).upper() for t in val if t)
        elif val:
            tickers.append(str(val).upper())
    names = hit.get("display_names") or []
    if isinstance(names, str):
        names = [names]
    for name in names:
        text = str(name)
        for m in re.findall(r"\(([A-Z]{1,5})\)", text):
            if m not in {"CIK", "NYSE", "NASDAQ"} and len(m) <= 5:
                tickers.append(m)
    entity = str(hit.get("entityName", ""))
    for m in re.findall(r"\(([A-Z]{1,5})\)", entity):
        if m not in {"CIK"}:
            tickers.append(m)
    return list(dict.fromkeys(tickers))


def _load_cache(days: int) -> pd.DataFrame | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        doc = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if doc.get("date") != date.today().isoformat() or int(doc.get("days", 0)) != days:
            return None
        rows = doc.get("rows") or []
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:  # noqa: BLE001
        return None


def _save_cache(df: pd.DataFrame, days: int) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps({
            "date": date.today().isoformat(),
            "days": days,
            "rows": df.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def search_dilution_filings(
    *,
    start: date | None = None,
    end: date | None = None,
    days: int = 5,
    max_hits: int = 80,
    keywords: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """搜索近期含稀释关键词的 SEC 公告。"""
    cached = _load_cache(days)
    if cached is not None:
        return cached

    end_d = end or date.today()
    start_d = start or (end_d - timedelta(days=days))
    rows: list[dict] = []
    seen: set[str] = set()
    kws = keywords or DILUTION_KEYWORDS[:4]  # 默认前4个关键词，加速日扫
    for kw in kws:
        params = urllib.parse.urlencode({
            "q": kw,
            "forms": ",".join(FORM_TYPES),
            "dateRange": "custom",
            "startdt": start_d.isoformat(),
            "enddt": end_d.isoformat(),
        })
        url = f"https://efts.sec.gov/LATEST/search-index?{params}"
        try:
            data = _http_get_json(url)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
            time.sleep(0.2)
            continue

        hits = data.get("hits") or {}
        for hit in (hits.get("hits") or [])[:max_hits]:
            src = hit.get("_source") or hit
            adsh = str(src.get("adsh", ""))
            if adsh and adsh in seen:
                continue
            if adsh:
                seen.add(adsh)
            tickers = _parse_tickers_from_hit(src)
            filed = src.get("file_date") or src.get("period_ending") or ""
            form = str(src.get("form_type") or src.get("form") or "8-K")
            entity = str(src.get("display_names", src.get("entityName", "")))
            ciks = src.get("ciks") or []
            cik = str(ciks[0]) if ciks else ""
            link_cik = cik.lstrip("0") if cik else ""
            rows.append({
                "代码": tickers[0] if tickers else "",
                "全部代码": ",".join(tickers),
                "表格": form,
                "公告日": filed,
                "关键词": kw,
                "公司": entity[:80],
                "链接": f"https://www.sec.gov/Archives/edgar/data/{link_cik}/{adsh.replace('-', '')}/",
            })
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["代码"].astype(str).str.len() > 0]
    out = df.drop_duplicates(subset=["代码", "公告日", "关键词"], keep="first")
    _save_cache(out, days)
    return out


def dilution_alert_map(filings_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """ticker → 最新融资公告摘要。"""
    out: dict[str, dict[str, Any]] = {}
    if filings_df is None or filings_df.empty:
        return out
    df = filings_df.copy()
    df["公告日"] = pd.to_datetime(df["公告日"], errors="coerce")
    df = df.sort_values("公告日", ascending=False)
    for _, r in df.iterrows():
        tk = str(r["代码"]).upper()
        if not tk or tk in out:
            continue
        out[tk] = {
            "表格": r.get("表格", "8-K"),
            "公告日": str(r.get("公告日", "")),
            "关键词": r.get("关键词", ""),
            "公司": r.get("公司", ""),
            "链接": r.get("链接", ""),
        }
    return out


def merge_offering_into_scan(
    scan_df: pd.DataFrame,
    alert_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """命中 8-K 融资公告的标的 → 追加 D_OFFERING 规律、降级为回避/做空。"""
    if scan_df.empty or not alert_map:
        return scan_df
    out = scan_df.copy()
    for i, row in out.iterrows():
        tk = str(row.get("代码", "")).upper()
        alert = alert_map.get(tk)
        if not alert:
            continue
        down = str(row.get("下跌规律", "—"))
        if down == "—":
            down = "D_OFFERING"
        elif "D_OFFERING" not in down:
            down = down + "、D_OFFERING"
        kw = alert.get("关键词", "融资")
        filed = alert.get("公告日", "")
        reason = f"SEC {alert.get('表格', '8-K')} · {kw} · {filed}"
        out.at[i, "下跌规律"] = down
        prev_sig = row.get("信号", "回避")
        out.at[i, "信号"] = "做空" if prev_sig == "做多" else prev_sig
        if prev_sig == "做多" or "D_OFFERING" in down:
            out.at[i, "策略动作"] = "买Put价差"
        out.at[i, "信号"] = "做空" if out.at[i, "策略动作"] == "买Put价差" else out.at[i, "信号"]
        prev_reason = str(row.get("选股理由", ""))
        out.at[i, "选股理由"] = f"{reason} | {prev_reason}" if prev_reason else reason
        out.at[i, "SEC融资"] = "是"
    return out
