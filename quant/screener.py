"""策略选股：按行情指标筛选标的，并对入选标的批量回测。

数据来源：Yahoo Finance 预置选股器 + 历史行情计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from . import backtest, strategies
from .data import DataError, fetch_history


# 预置股票池（yfinance 内置选股器名称 → 中文说明）
UNIVERSE_PRESETS: dict[str, str] = {
    "day_gainers": "当日涨幅榜",
    "day_losers": "当日跌幅榜",
    "most_actives": "成交活跃榜",
    "small_cap_gainers": "小盘活跃",
    "aggressive_small_caps": "激进小盘",
    "growth_technology_stocks": "科技成长股",
    "undervalued_large_caps": "低估大盘股",
}

# Wikipedia 拉取失败时的备用列表（流动性较好的大盘股）
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "KO", "PEP", "COST", "AVGO", "MCD", "CSCO", "TMO", "ACN", "ABT",
    "CRM", "NFLX", "AMD", "LIN", "DHR", "TXN", "INTC", "QCOM", "ORCL", "IBM",
]

# 行业板块（Yahoo 英文 → 中文）。Yahoo 的 sector 字段沿用 GICS 11 大板块。
SECTORS: dict[str, str] = {
    "Technology": "科技",
    "Healthcare": "医疗健康",
    "Financial Services": "金融",
    "Consumer Cyclical": "可选消费",
    "Consumer Defensive": "必需消费",
    "Communication Services": "通信服务",
    "Industrials": "工业",
    "Energy": "能源",
    "Basic Materials": "原材料",
    "Real Estate": "房地产",
    "Utilities": "公用事业",
}


def sector_cn(name: str) -> str:
    """把 Yahoo 英文行业名转成中文，未知则原样返回。"""
    if not name:
        return ""
    return SECTORS.get(str(name).strip(), str(name).strip())


@dataclass
class ScreenFilters:
    """选股筛选条件。"""

    min_gain_pct: float = -100.0       # 涨幅下限（%）
    max_gain_pct: float = 1000.0       # 涨幅上限（%）
    min_dollar_vol_m: float = 0.0      # 成交额下限（百万 USD，取近均）
    min_turnover_pct: float = 0.0      # 换手率下限（%）
    max_turnover_pct: float = 100.0    # 换手率上限（%）
    min_mcap_b: float = 0.0            # 市值下限（十亿美元）
    max_mcap_b: float = 10_000.0       # 市值上限（十亿美元）
    lookback_days: int = 1             # 涨幅统计周期（交易日）
    sectors: list[str] | None = None   # 仅保留这些行业（英文名）；None/空 = 不限


def _ensure_yfinance() -> None:
    if yf is None:
        raise DataError("未安装 yfinance，请先运行: pip install yfinance")


def fetch_sp500_tickers() -> list[str]:
    """拉取标普 500 成分股代码。"""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Symbol",
        )
        syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return [s.strip().upper() for s in syms if s.strip()]
    except Exception:  # noqa: BLE001
        return list(_FALLBACK_TICKERS)


def quotes_to_dataframe(response: dict[str, Any]) -> pd.DataFrame:
    """把 yfinance screen 返回的 quotes 转为标准 DataFrame。"""
    quotes = response.get("quotes") or []
    rows: list[dict[str, Any]] = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue
        price = q.get("regularMarketPrice") or q.get("intradayprice")
        vol = q.get("regularMarketVolume") or q.get("dayvolume") or 0
        shares = q.get("sharesOutstanding") or q.get("impliedSharesOutstanding")
        mcap = q.get("marketCap") or q.get("intradaymarketcap")
        chg_pct = q.get("regularMarketChangePercent")
        if chg_pct is None:
            chg_pct = q.get("percentchange")
        turnover = (float(vol) / float(shares) * 100.0) if shares and float(shares) > 0 else np.nan
        dollar_vol = float(price) * float(vol) if price is not None and vol else np.nan
        sector_en = str(q.get("sector") or "").strip()
        rows.append(
            {
                "代码": str(sym).upper(),
                "名称": q.get("shortName") or q.get("longName") or sym,
                "最新价": price,
                "涨幅%": chg_pct,
                "成交量": vol,
                "成交额USD": dollar_vol,
                "换手率%": turnover,
                "市值USD": mcap,
                "_行业EN": sector_en,
                "行业": sector_cn(sector_en),
            }
        )
    return pd.DataFrame(rows)


def fetch_yahoo_screen(preset: str, count: int = 50) -> pd.DataFrame:
    """调用 Yahoo 预置选股器。"""
    _ensure_yfinance()
    if preset not in UNIVERSE_PRESETS:
        raise DataError(f"未知选股池：{preset}")
    resp = yf.screen(preset, count=min(max(count, 1), 250))
    if not isinstance(resp, dict) or not resp.get("quotes"):
        return pd.DataFrame()
    return quotes_to_dataframe(resp)


def fetch_sector(ticker: str) -> str:
    """尽力获取单只标的的行业（英文 GICS 板块名）；失败返回空串。"""
    _ensure_yfinance()
    try:
        info = yf.Ticker(ticker).get_info()
        return str(info.get("sector") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def build_snapshot_from_history(
    tickers: list[str],
    start: str | date,
    end: str | date,
    lookback_days: int = 20,
    with_sector: bool = False,
) -> pd.DataFrame:
    """基于历史行情批量计算涨幅、成交额、换手率、市值（估算）。

    with_sector=True 时额外逐只拉取行业（较慢，会发起额外网络请求）。
    """
    _ensure_yfinance()
    if not tickers:
        return pd.DataFrame()

    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers,
        start=start_str,
        end=end_str,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    multi = isinstance(raw.columns, pd.MultiIndex)

    for t in tickers:
        try:
            if multi:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].dropna(how="all")
            else:
                sub = raw.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            close = sub["Close"].astype(float)
            vol = sub["Volume"].astype(float)
            lb = min(lookback_days, len(close) - 1)
            if lb < 1:
                continue
            gain = (close.iloc[-1] / close.iloc[-1 - lb] - 1.0) * 100.0
            dollar_vol = (close * vol).tail(lb).mean()
            turnover = np.nan
            mcap = np.nan
            try:
                info = yf.Ticker(t).fast_info
                shares = getattr(info, "shares", None)
                mcap_val = getattr(info, "market_cap", None)
                if shares and float(shares) > 0:
                    turnover = float(vol.iloc[-1]) / float(shares) * 100.0
                if mcap_val:
                    mcap = float(mcap_val)
            except Exception:  # noqa: BLE001
                pass
            sector_en = fetch_sector(t) if with_sector else ""
            rows.append(
                {
                    "代码": t,
                    "名称": t,
                    "最新价": float(close.iloc[-1]),
                    "涨幅%": float(gain),
                    "成交量": float(vol.iloc[-1]),
                    "成交额USD": float(dollar_vol),
                    "换手率%": turnover,
                    "市值USD": mcap,
                    "_行业EN": sector_en,
                    "行业": sector_cn(sector_en),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def apply_filters(snapshot: pd.DataFrame, filters: ScreenFilters) -> pd.DataFrame:
    """按涨幅 / 成交额 / 换手率 / 市值 / 行业筛选。"""
    if snapshot.empty:
        return snapshot.copy()

    df = snapshot.copy()
    df["涨幅%"] = pd.to_numeric(df["涨幅%"], errors="coerce")
    df["成交额USD"] = pd.to_numeric(df["成交额USD"], errors="coerce")
    df["换手率%"] = pd.to_numeric(df["换手率%"], errors="coerce")
    df["市值USD"] = pd.to_numeric(df["市值USD"], errors="coerce")

    min_dollar = filters.min_dollar_vol_m * 1_000_000
    min_mcap = filters.min_mcap_b * 1_000_000_000
    max_mcap = filters.max_mcap_b * 1_000_000_000

    mask = (
        df["涨幅%"].between(filters.min_gain_pct, filters.max_gain_pct, inclusive="both")
        & (df["成交额USD"].fillna(0) >= min_dollar)
        & (df["换手率%"].fillna(0).between(filters.min_turnover_pct, filters.max_turnover_pct, inclusive="both")
           | df["换手率%"].isna())  # 无换手率数据时不硬排除
        & ((df["市值USD"].fillna(min_mcap).between(min_mcap, max_mcap, inclusive="both"))
           | df["市值USD"].isna())
    )

    if filters.sectors:
        wanted = {str(s).strip() for s in filters.sectors if str(s).strip()}
        if wanted:
            sector_col = df["_行业EN"] if "_行业EN" in df.columns else df.get("行业", pd.Series("", index=df.index))
            sector_col = sector_col.fillna("").astype(str)
            # 行业未知（空串）的标的不因行业条件被排除，避免误杀缺数据标的。
            mask = mask & (sector_col.isin(wanted) | (sector_col == ""))

    out = df.loc[mask].copy()
    out = out.sort_values("涨幅%", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def backtest_universe(
    tickers: list[str],
    start: str | date,
    end: str | date,
    strategy_name: str,
    params: dict[str, float] | None = None,
    *,
    allow_short: bool = False,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """对一组标的运行同一策略回测，返回汇总表。"""
    params = params or {}
    strat = strategies.get_strategy(strategy_name)
    rows: list[dict[str, Any]] = []

    for t in tickers:
        try:
            df = fetch_history(t, start=start, end=end)
            if len(df) < 60:
                continue
            pos = strat.generate(df, allow_short=allow_short, **params)
            res = backtest.run_backtest(
                df, pos,
                initial_capital=initial_capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            s = res.stats
            last_pos = float(pos.iloc[-1]) if len(pos) else 0.0
            signal = "🟢 做多" if last_pos > 0 else ("🔴 做空" if last_pos < 0 else "⚪ 空仓")
            rows.append(
                {
                    "代码": t,
                    "策略累计收益": s["累计收益率"],
                    "策略年化收益": s["年化收益率"],
                    "基准收益": s["基准收益率"],
                    "超额收益": s["累计收益率"] - s["基准收益率"],
                    "夏普比率": s["夏普比率"],
                    "最大回撤": s["最大回撤"],
                    "胜率": s["胜率"],
                    "交易次数": int(s["交易次数"]),
                    "期末资金": s["期末资金"],
                    "当前信号": signal,
                }
            )
        except Exception:  # noqa: BLE001
            continue

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("策略累计收益", ascending=False).reset_index(drop=True)


def merge_snapshot_backtest(snapshot: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """合并选股快照与回测结果。"""
    if snapshot.empty or bt.empty:
        return pd.DataFrame()
    cols = ["代码", "名称", "最新价", "涨幅%", "成交额USD", "换手率%", "市值USD", "行业"]
    base = snapshot[[c for c in cols if c in snapshot.columns]].copy()
    merged = base.merge(bt, on="代码", how="inner")
    return merged.sort_values("策略累计收益", ascending=False).reset_index(drop=True)


def summarize_backtest(bt: pd.DataFrame) -> dict[str, float]:
    """组合层面汇总：等权平均收益等。"""
    if bt.empty:
        return {}
    return {
        "入选数量": float(len(bt)),
        "平均累计收益": float(bt["策略累计收益"].mean()),
        "平均年化收益": float(bt["策略年化收益"].mean()),
        "平均夏普": float(bt["夏普比率"].mean()),
        "平均最大回撤": float(bt["最大回撤"].mean()),
        "盈利标的占比": float((bt["策略累计收益"] > 0).mean()),
        "平均超额收益": float(bt["超额收益"].mean()),
    }


def build_universe_snapshot(
    pool: str,
    start: str | date,
    end: str | date,
    *,
    lookback_days: int = 20,
    pool_size: int = 50,
    custom_tickers: list[str] | None = None,
    with_sector: bool = False,
) -> pd.DataFrame:
    """根据股票池来源构造初选快照。

    pool 取值：UNIVERSE_PRESETS 的键、"sp500" 或 "custom"。
    """
    if pool == "custom":
        tickers = [t.strip().upper() for t in (custom_tickers or []) if t.strip()]
        return build_snapshot_from_history(
            tickers, start, end, lookback_days=lookback_days, with_sector=with_sector,
        )
    if pool == "sp500":
        tickers = fetch_sp500_tickers()[: int(pool_size)]
        return build_snapshot_from_history(
            tickers, start, end, lookback_days=lookback_days, with_sector=with_sector,
        )
    snapshot = fetch_yahoo_screen(pool, count=int(pool_size))
    if snapshot.empty:
        return snapshot
    # Yahoo 预置选股已带行业；若需要更长的涨幅周期，则回退到历史快照重算。
    if lookback_days > 1:
        hist = build_snapshot_from_history(
            snapshot["代码"].tolist(), start, end,
            lookback_days=lookback_days, with_sector=False,
        )
        if not hist.empty:
            # 用历史快照的指标，但补回 Yahoo 已有的行业信息。
            sector_map = dict(zip(snapshot["代码"], snapshot.get("_行业EN", "")))
            name_map = dict(zip(snapshot["代码"], snapshot.get("名称", "")))
            hist["_行业EN"] = hist["代码"].map(sector_map).fillna("")
            hist["行业"] = hist["_行业EN"].map(sector_cn)
            hist["名称"] = hist["代码"].map(name_map).fillna(hist["名称"])
            return hist
    return snapshot


def run_screen(
    filters: ScreenFilters,
    start: str | date,
    end: str | date,
    *,
    pool: str = "day_gainers",
    pool_size: int = 50,
    custom_tickers: list[str] | None = None,
    strategy_name: str | None = None,
    params: dict[str, float] | None = None,
    max_backtest: int = 20,
    allow_short: bool = False,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """端到端选股流水线：建池 → 筛选 → （可选）批量回测。

    返回 dict：snapshot / filtered / backtest / merged / summary。
    """
    with_sector = bool(filters.sectors)
    snapshot = build_universe_snapshot(
        pool, start, end,
        lookback_days=filters.lookback_days,
        pool_size=pool_size,
        custom_tickers=custom_tickers,
        with_sector=with_sector,
    )
    result: dict[str, Any] = {
        "snapshot": snapshot,
        "filtered": pd.DataFrame(),
        "backtest": pd.DataFrame(),
        "merged": pd.DataFrame(),
        "summary": {},
    }
    if snapshot.empty:
        return result

    filtered = apply_filters(snapshot, filters)
    result["filtered"] = filtered
    if filtered.empty or not strategy_name:
        return result

    targets = filtered["代码"].head(int(max_backtest)).tolist()
    bt = backtest_universe(
        targets, start, end, strategy_name,
        params=params,
        allow_short=allow_short,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    result["backtest"] = bt
    result["merged"] = merge_snapshot_backtest(filtered, bt)
    result["summary"] = summarize_backtest(bt)
    return result


def load_screen_history(path: str | Path = "screen_history.csv") -> pd.DataFrame:
    """读取每日选股历史 CSV（不存在则返回空表）。"""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def summarize_screen_history(df: pd.DataFrame) -> pd.DataFrame:
    """按选股批次汇总：入选数量、平均策略收益、盈利占比。"""
    if df.empty or "选股时间" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("选股时间", sort=False)
    rows = []
    for ts, grp in g:
        ret = pd.to_numeric(grp.get("策略累计收益"), errors="coerce")
        rows.append({
            "选股时间": ts,
            "股票池": grp["股票池"].iloc[0] if "股票池" in grp.columns else "",
            "策略": grp["策略"].iloc[0] if "策略" in grp.columns else "",
            "入选数": len(grp),
            "平均策略收益": ret.mean(),
            "盈利占比": (ret > 0).mean() if len(ret) else 0.0,
        })
    return pd.DataFrame(rows)
