"""真实历史期权链（DoltHub: post-no-preference/options，免费 EOD 数据）。

与 option_chain.py（当前真实链，yfinance）互补，这里取**历史每日 EOD 期权链**，
用于真实回测——真实行权价/到期/bid/ask/IV，全程不靠 Black-Scholes 估值。

数据：act_symbol, date, expiration, strike, call_put(Call/Put), bid, ask, vol(IV), delta...
覆盖：约 2019 年起，但**不是每个交易日都全**（免费集有缺口）；无数据的日期回测里如实跳过。

查询走 DoltHub SQL API（较慢），本地 parquet 缓存避免重复拉取。
大规模回测建议 `dolt clone post-no-preference/options` 到本地跑（快几个数量级）。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DOLT_BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
CACHE_DIR = Path(__file__).resolve().parents[1] / "research" / "options_cache"
_NUM_COLS = ["strike", "bid", "ask", "vol", "delta"]


def _dolt_query(sql: str, *, timeout: int = 90, retries: int = 2) -> pd.DataFrame:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            r = requests.get(DOLT_BASE, params={"q": sql}, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            status = j.get("query_execution_status")
            if status and status != "Success":
                last_err = j.get("query_execution_message", "查询失败")
                return pd.DataFrame()
            return pd.DataFrame(j.get("rows", []))
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    if last_err:
        # 静默返回空，调用方按"无数据"处理
        return pd.DataFrame()
    return pd.DataFrame()


def fetch_eod_chain(sym: str, day: str, *, use_cache: bool = True) -> pd.DataFrame:
    """取 sym 在 day(YYYY-MM-DD) 的真实 EOD 期权链。无数据返回空 DataFrame。"""
    sym = str(sym).upper()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fpath = CACHE_DIR / f"{sym}_{day}.parquet"
    if use_cache and fpath.exists():
        return pd.read_parquet(fpath)
    sql = (
        "SELECT `date`,expiration,strike,call_put,bid,ask,vol,delta "
        f"FROM option_chain WHERE act_symbol='{sym}' AND `date`='{day}'"
    )
    df = _dolt_query(sql)
    if not df.empty:
        for c in _NUM_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    # 缓存（含空结果，避免反复打无数据的日期）
    try:
        df.to_parquet(fpath)
    except Exception:  # noqa: BLE001
        pass
    return df


def has_data(sym: str, day: str) -> bool:
    return not fetch_eod_chain(sym, day).empty


# ---------- 从真实 EOD 链选合约（纯逻辑，可单测） ----------

def _calls(chain: pd.DataFrame) -> pd.DataFrame:
    return chain[chain["call_put"].str.lower().eq("call")].copy()


def _puts(chain: pd.DataFrame) -> pd.DataFrame:
    return chain[chain["call_put"].str.lower().eq("put")].copy()


def nearest_expiry(chain: pd.DataFrame, day: str, *, min_dte: int = 2, max_dte: int = 45) -> pd.Timestamp | None:
    if chain.empty:
        return None
    d0 = pd.Timestamp(day)
    exps = pd.to_datetime(chain["expiration"]).dropna().unique()
    cand = sorted(pd.Timestamp(e) for e in exps if min_dte <= (pd.Timestamp(e) - d0).days <= max_dte)
    if cand:
        return cand[0]
    rest = sorted(pd.Timestamp(e) for e in exps if (pd.Timestamp(e) - d0).days >= min_dte)
    return rest[0] if rest else None


def pick_bear_call_eod(
    chain: pd.DataFrame, spot: float, day: str, *,
    otm: float = 0.08, width_pct: float = 0.10, min_dte: int = 2, max_dte: int = 45,
) -> tuple[dict | None, str]:
    """真实 EOD 卖看涨价差：卖近端OTM call(收bid)，买更高call(付ask)。"""
    if chain.empty or spot <= 0:
        return None, "无链数据"
    exp = nearest_expiry(chain, day, min_dte=min_dte, max_dte=max_dte)
    if exp is None:
        return None, "无合适到期"
    calls = _calls(chain)
    calls = calls[pd.to_datetime(calls["expiration"]) == exp].sort_values("strike")
    if calls.empty:
        return None, "该到期无call"
    short = calls[calls["strike"] >= spot * (1 + otm)].head(1)
    if short.empty:
        return None, "无足够OTM行权价"
    s = short.iloc[0]
    long = calls[calls["strike"] >= float(s["strike"]) * (1 + width_pct) + 1e-9].head(1)
    if long.empty:
        return None, "无更高保护腿"
    l = long.iloc[0]
    if not (float(s["bid"]) > 0 and float(l["ask"]) >= 0):
        return None, "卖腿无买价"
    credit = float(s["bid"]) - float(l["ask"])
    if credit <= 0:
        return None, "真实净权利金≤0"
    width = float(l["strike"]) - float(s["strike"])
    return {
        "expiration": exp, "dte": (exp - pd.Timestamp(day)).days,
        "short_strike": float(s["strike"]), "long_strike": float(l["strike"]),
        "credit": credit, "width": width,
        "max_loss": (width - credit) * 100, "max_profit": credit * 100,
        "short_iv": float(s.get("vol", 0) or 0),
    }, ""


def pick_csp_eod(
    chain: pd.DataFrame, spot: float, day: str, *,
    otm: float = 0.10, min_dte: int = 2, max_dte: int = 45,
) -> tuple[dict | None, str]:
    if chain.empty or spot <= 0:
        return None, "无链数据"
    exp = nearest_expiry(chain, day, min_dte=min_dte, max_dte=max_dte)
    if exp is None:
        return None, "无合适到期"
    puts = _puts(chain)
    puts = puts[pd.to_datetime(puts["expiration"]) == exp].sort_values("strike")
    cand = puts[puts["strike"] <= spot * (1 - otm)]
    if cand.empty:
        return None, "无足够OTM行权价"
    p = cand.iloc[-1]
    if not float(p["bid"]) > 0:
        return None, "无买价"
    return {
        "expiration": exp, "dte": (exp - pd.Timestamp(day)).days,
        "short_strike": float(p["strike"]), "credit": float(p["bid"]),
        "collateral": float(p["strike"]) * 100, "short_iv": float(p.get("vol", 0) or 0),
    }, ""


def pick_bear_put_debit_eod(
    chain: pd.DataFrame, spot: float, day: str, *,
    otm: float = 0.0, width_pct: float = 0.10, min_dte: int = 2, max_dte: int = 45,
) -> tuple[dict | None, str]:
    if chain.empty or spot <= 0:
        return None, "无链数据"
    exp = nearest_expiry(chain, day, min_dte=min_dte, max_dte=max_dte)
    if exp is None:
        return None, "无合适到期"
    puts = _puts(chain)
    puts = puts[pd.to_datetime(puts["expiration"]) == exp].sort_values("strike")
    longc = puts[puts["strike"] <= spot * (1 - otm)]
    if longc.empty:
        return None, "无合适行权价"
    lg = longc.iloc[-1]
    shortc = puts[puts["strike"] <= float(lg["strike"]) * (1 - width_pct) - 1e-9]
    if shortc.empty:
        return None, "无更低腿"
    sh = shortc.iloc[-1]
    debit = float(lg["ask"]) - float(sh["bid"])
    if debit <= 0:
        return None, "真实净成本≤0"
    width = float(lg["strike"]) - float(sh["strike"])
    return {
        "expiration": exp, "dte": (exp - pd.Timestamp(day)).days,
        "long_strike": float(lg["strike"]), "short_strike": float(sh["strike"]),
        "debit": debit, "width": width,
        "max_loss": debit * 100, "max_profit": (width - debit) * 100,
    }, ""


# ---------- 到期真实结算（用真实标的收盘价的内在价值） ----------

def settle_bear_call_at_expiry(plan: dict, underlying_close_at_exp: float) -> float:
    """卖看涨价差到期真实盈亏($/张)：已收 credit，减去价差内在值。"""
    st = float(underlying_close_at_exp)
    spread_intrinsic = min(max(0.0, st - plan["short_strike"]), plan["width"])
    pnl = plan["credit"] - spread_intrinsic
    return pnl * 100


def settle_csp_at_expiry(plan: dict, underlying_close_at_exp: float) -> float:
    """卖看跌到期真实盈亏($/张)：收 credit，减去 put 内在值。"""
    st = float(underlying_close_at_exp)
    put_intrinsic = max(0.0, plan["short_strike"] - st)
    return (plan["credit"] - put_intrinsic) * 100


def settle_bear_put_debit_at_expiry(plan: dict, underlying_close_at_exp: float) -> float:
    """买看跌价差到期真实盈亏($/张)：价差内在值 - 已付 debit。"""
    st = float(underlying_close_at_exp)
    spread_intrinsic = min(max(0.0, plan["long_strike"] - st), plan["width"])
    return (spread_intrinsic - plan["debit"]) * 100
