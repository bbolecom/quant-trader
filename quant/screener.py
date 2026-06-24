"""策略选股：按行情指标筛选标的，并对入选标的批量回测。

数据来源：Yahoo Finance 预置选股器 + 历史行情计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from . import backtest, strategies
from .data import DataError, fetch_history, fetch_history_batch


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


def _read_html_tables(url: str, **kwargs) -> list[pd.DataFrame]:
    """带浏览器 User-Agent 拉取并解析 HTML 表格。

    Wikipedia 等站点已对 pandas/urllib 默认 UA 返回 403，必须显式带 UA 抓取后再 read_html。
    """
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; quant-screener/1.0)"}
    )
    html = urllib.request.urlopen(req, timeout=20).read()  # noqa: S310
    return pd.read_html(html, **kwargs)


def fetch_sp500_tickers() -> list[str]:
    """拉取标普 500 成分股代码。"""
    try:
        tables = _read_html_tables(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            match="Symbol",
        )
        syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        cleaned = [s.strip().upper() for s in syms if s.strip()]
        return cleaned or list(_FALLBACK_TICKERS)
    except Exception:  # noqa: BLE001
        return list(_FALLBACK_TICKERS)


def fetch_nasdaq100_tickers() -> list[str]:
    """拉取纳斯达克 100 成分股。"""
    try:
        tables = _read_html_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        for table in tables:
            if "Ticker" in table.columns:
                syms = table["Ticker"].astype(str).str.replace(".", "-", regex=False).tolist()
                return [s.strip().upper() for s in syms if s.strip()]
    except Exception:  # noqa: BLE001
        pass
    return []


# 用于「全市场」候选池：Yahoo 多榜 + 指数成分（不限于标普500）
BROAD_SCREEN_PRESETS: tuple[str, ...] = (
    "day_gainers",
    "most_actives",
    "small_cap_gainers",
    "aggressive_small_caps",
    "growth_technology_stocks",
)


def fetch_broad_universe(
    *,
    screen_count: int = 250,
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    """合并 Yahoo 涨幅/活跃/小盘等多榜 + 指数成分，去重。符合流动性条件的票均可入选。"""
    seen: set[str] = set()
    out: list[str] = []

    def _add(symbols: list[str]) -> None:
        for raw in symbols:
            t = str(raw).strip().upper()
            if t and t not in seen:
                seen.add(t)
                out.append(t)

    for preset in BROAD_SCREEN_PRESETS:
        try:
            df = fetch_yahoo_screen(preset, count=screen_count)
            if not df.empty:
                _add(df["代码"].tolist())
        except Exception:  # noqa: BLE001
            continue
    if include_sp500:
        _add(fetch_sp500_tickers())
    if include_nasdaq100:
        _add(fetch_nasdaq100_tickers())
    if extra:
        _add(extra)
    return out


def fetch_gainer_universe_live(count: int = 250) -> pd.DataFrame:
    """当日全市场涨幅相关候选：合并多个 Yahoo 榜，保留市值/成交额字段。"""
    frames: list[pd.DataFrame] = []
    for preset in ("day_gainers", "most_actives", "small_cap_gainers", "aggressive_small_caps"):
        try:
            df = fetch_yahoo_screen(preset, count=count)
            if not df.empty:
                df = df.copy()
                df["_来源"] = UNIVERSE_PRESETS.get(preset, preset)
                frames.append(df)
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("涨幅%", ascending=False, na_position="last")
    merged = merged.drop_duplicates(subset=["代码"], keep="first")
    return merged.reset_index(drop=True)


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
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    batch = fetch_history_batch(tickers, start_str, end_str)
    if not batch:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for t in tickers:
        sub = batch.get(t)
        if sub is None or sub.empty or "Close" not in sub.columns:
            continue
        try:
            close = sub["Close"].astype(float)
            vol = sub["Volume"].astype(float)
            lb = min(lookback_days, len(close) - 1)
            if lb < 1:
                continue
            gain = (close.iloc[-1] / close.iloc[-1 - lb] - 1.0) * 100.0
            dollar_vol = (close * vol).tail(lb).mean()
            turnover = np.nan
            mcap = np.nan
            if with_sector:
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


def merge_snapshot_backtest(
    snapshot: pd.DataFrame,
    bt: pd.DataFrame,
    selection_date: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """合并选股快照与回测结果；可选写入选股日期。"""
    if snapshot.empty or bt.empty:
        return pd.DataFrame()
    cols = ["代码", "名称", "最新价", "涨幅%", "成交额USD", "换手率%", "市值USD", "行业"]
    base = snapshot[[c for c in cols if c in snapshot.columns]].copy()
    merged = base.merge(bt, on="代码", how="inner")
    merged = merged.sort_values("策略累计收益", ascending=False).reset_index(drop=True)
    if selection_date is not None:
        merged = stamp_selection_date(merged, selection_date)
    return merged


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


# ---------------------------------------------------------------------------
# 按日选股 · 选股理由 · 前后收益/回撤 · 选股日回测
# ---------------------------------------------------------------------------

def normalize_selection_date(value: str | date | pd.Timestamp | None = None) -> str:
    """统一选股日期格式为 YYYY-MM-DD。"""
    if value is None:
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def stamp_selection_date(df: pd.DataFrame, selection_date: str | date | pd.Timestamp) -> pd.DataFrame:
    """为选股结果表写入/覆盖「选股日期」，并置于首列。"""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["选股日期"] = normalize_selection_date(selection_date)
    cols = ["选股日期"] + [c for c in out.columns if c != "选股日期"]
    return out[cols]


def _window_return(close: pd.Series) -> float:
    if len(close) < 2:
        return np.nan
    base = float(close.iloc[0])
    if base == 0:
        return np.nan
    return float(close.iloc[-1] / base - 1.0)


def _window_max_drawdown(close: pd.Series) -> float:
    if len(close) < 2:
        return np.nan
    eq = close / float(close.iloc[0])
    return float((eq / eq.cummax() - 1.0).min())


def snapshot_at_date(
    data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """用截至 as_of 的历史数据构造快照（无未来函数）。"""
    rows: list[dict[str, Any]] = []
    as_of = pd.Timestamp(as_of)
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        hist = df.loc[df.index <= as_of]
        if len(hist) < lookback + 2:
            continue
        close = hist["Close"].astype(float)
        vol = hist["Volume"].astype(float)
        lb = lookback
        if lb <= 1:
            gain = (close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0 if len(close) >= 2 else 0.0
        else:
            gain = (close.iloc[-1] / close.iloc[-lb - 1] - 1.0) * 100.0
        avg_dollar = float((close.tail(lb) * vol.tail(lb)).mean()) if lb > 0 else float(close.iloc[-1] * vol.iloc[-1])
        rows.append({
            "代码": ticker.upper(),
            "名称": ticker.upper(),
            "涨幅%": float(gain),
            "成交额USD": avg_dollar,
            "换手率%": np.nan,
            "市值USD": np.nan,
            "最新价": float(close.iloc[-1]),
        })
    return pd.DataFrame(rows)


def pick_rationale(
    row: pd.Series,
    filters: ScreenFilters,
    rank: int = 0,
    selection_date: str | date | pd.Timestamp | None = None,
) -> str:
    """生成单只标的的选股理由（人话）。"""
    parts: list[str] = []
    if selection_date is not None:
        parts.append(f"选股日 {normalize_selection_date(selection_date)}")
    lb = filters.lookback_days
    gain = row.get("涨幅%")
    if pd.notna(gain):
        parts.append(f"近{lb}日涨幅 {float(gain):+.1f}%，落在设定区间 [{filters.min_gain_pct:.1f}%, {filters.max_gain_pct:.1f}%]")
    dvol = row.get("成交额USD")
    if pd.notna(dvol) and filters.min_dollar_vol_m > 0:
        parts.append(f"日均成交额约 ${float(dvol)/1e6:.1f}M（下限 {filters.min_dollar_vol_m:.0f}M）")
    turnover = row.get("换手率%")
    if pd.notna(turnover):
        parts.append(f"换手率 {float(turnover):.2f}%")
    mcap = row.get("市值USD")
    if pd.notna(mcap):
        parts.append(f"市值约 ${float(mcap)/1e9:.1f}B")
    sector = row.get("行业") or row.get("_行业EN")
    if sector and str(sector).strip():
        parts.append(f"行业：{sector}")
    if rank > 0:
        parts.append(f"筛选池内涨幅排名第 {rank}")
    return "；".join(parts) if parts else "满足当前全部筛选条件"


def forward_backward_metrics(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    forward_days: int = 20,
    backward_days: int = 20,
) -> dict[str, float]:
    """计算选股日前后的持有期收益与最大回撤（买入持有视角，选股日收盘价为基准）。"""
    close = df["Close"].astype(float)
    as_of = pd.Timestamp(as_of)
    hist = close.loc[close.index <= as_of]
    if hist.empty:
        return {}
    pick_price = float(hist.iloc[-1])

    back_slice = hist.tail(backward_days + 1)
    back_ret = _window_return(back_slice)
    back_dd = _window_max_drawdown(back_slice)

    future = close.loc[close.index > as_of].head(forward_days + 1)
    if len(future) >= 2:
        fwd_ret = float(future.iloc[-1] / pick_price - 1.0)
        fwd_eq = future / pick_price
        fwd_dd = float((fwd_eq / fwd_eq.cummax() - 1.0).min())
    else:
        fwd_ret = fwd_dd = np.nan

    return {
        "入选价": pick_price,
        "前向天数": float(forward_days),
        "后向天数": float(backward_days),
        f"前{backward_days}日收益": back_ret,
        f"前{backward_days}日最大回撤": back_dd,
        f"后{forward_days}日收益": fwd_ret,
        f"后{forward_days}日最大回撤": fwd_dd,
    }


def backtest_pick_forward(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    strategy_name: str,
    params: dict[str, float] | None = None,
    *,
    forward_days: int = 20,
    warmup_bars: int = 120,
    allow_short: bool = False,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, float]:
    """从选股日次日开始，在 forward_days 窗口内用指定策略回测（指标预热用选股日前数据）。"""
    as_of = pd.Timestamp(as_of)
    params = params or {}
    hist = df.loc[df.index <= as_of].tail(warmup_bars)
    future_idx = df.index[df.index > as_of]
    if len(hist) < 60 or len(future_idx) == 0:
        return {"策略后向收益": np.nan, "策略后向最大回撤": np.nan}

    end_i = min(forward_days, len(future_idx))
    forward = df.loc[future_idx[:end_i]]
    combined = pd.concat([hist, forward])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    strat = strategies.get_strategy(strategy_name)
    pos = strat.generate(combined, allow_short=allow_short, **params)
    res = backtest.run_backtest(
        combined, pos,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    fwd_ret = res.returns.loc[res.returns.index > as_of]
    if fwd_ret.empty:
        return {"策略后向收益": np.nan, "策略后向最大回撤": np.nan}
    eq = (1.0 + fwd_ret).cumprod()
    return {
        "策略后向收益": float(eq.iloc[-1] - 1.0),
        "策略后向最大回撤": float((eq / eq.cummax() - 1.0).min()),
    }


def signal_direction_at(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    strategy_name: str,
    params: dict[str, float] | None = None,
    *,
    allow_short: bool = False,
    warmup_bars: int = 120,
) -> float:
    """返回选股日策略给出的目标仓位信号：>0 做多，<0 做空，0 观望（无未来数据）。"""
    as_of = pd.Timestamp(as_of)
    hist = df.loc[df.index <= as_of].tail(warmup_bars)
    if len(hist) < 30:
        return 0.0
    strat = strategies.get_strategy(strategy_name)
    pos = strat.generate(hist, allow_short=allow_short, **(params or {}))
    if pos is None or len(pos) == 0:
        return 0.0
    val = pos.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def direction_label(signal: float) -> str:
    if signal > 0:
        return "做多"
    if signal < 0:
        return "做空"
    return "观望"


def build_trade_plan(
    picks: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    strategy_name: str,
    params: dict[str, float] | None = None,
    *,
    forward_days: int = 20,
    capital: float = 100_000.0,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """为每只入选股生成交易计划：方向/仓位/金额/选股理由 + 选股后 N 日盈亏与回撤。

    - 方向：选股日策略信号（做多/做空/观望，无未来数据）
    - 仓位：入选股等权分配，权重 = |信号|（0~1）
    - 金额：capital × 权重 ÷ 入选数（建议投入）
    - 后 N 日盈亏：方向调整后的买入持有收益 × 投入金额（赚为正、亏为负）
    - 策略后 N 日收益：策略自行择时（含中途离场/反手）的收益口径
    """
    if picks.empty:
        return picks
    as_of = pd.Timestamp(as_of)
    n = max(len(picks), 1)
    per_slot = float(capital) / n
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(picks.iterrows()):
        ticker = str(row["代码"]).upper()
        df = data.get(ticker)
        rec: dict[str, Any] = {
            "选股日期": as_of.strftime("%Y-%m-%d"),
            "代码": ticker,
            "选股理由": row.get("选股理由", ""),
        }
        if df is None or df.empty:
            rows.append(rec)
            continue

        signal = signal_direction_at(
            df, as_of, strategy_name, params,
            allow_short=allow_short,
        )
        weight = min(abs(signal), 1.0)
        dollar = per_slot * weight
        rec["方向"] = direction_label(signal)
        rec["建议仓位%"] = round(weight * 100.0, 1)
        rec["建议金额USD"] = round(dollar, 2)

        fb = forward_backward_metrics(df, as_of, forward_days=forward_days, backward_days=1)
        raw_fwd = fb.get(f"后{forward_days}日收益", np.nan)
        fwd_dd = fb.get(f"后{forward_days}日最大回撤", np.nan)
        dir_sign = 1.0 if signal > 0 else (-1.0 if signal < 0 else 0.0)
        directional_ret = dir_sign * raw_fwd if pd.notna(raw_fwd) else np.nan
        pnl = directional_ret * dollar if pd.notna(directional_ret) else np.nan

        rec[f"后{forward_days}日收益%"] = round(directional_ret * 100.0, 2) if pd.notna(directional_ret) else np.nan
        rec["盈亏"] = ("观望" if dir_sign == 0 else ("赚" if pd.notna(pnl) and pnl > 0 else ("亏" if pd.notna(pnl) and pnl < 0 else "持平")))
        rec["盈亏金额USD"] = round(pnl, 2) if pd.notna(pnl) else np.nan
        rec[f"后{forward_days}日最大回撤%"] = round(fwd_dd * 100.0, 2) if pd.notna(fwd_dd) else np.nan

        strat_perf = backtest_pick_forward(
            df, as_of, strategy_name, params=params,
            forward_days=forward_days, allow_short=allow_short,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        sret = strat_perf.get("策略后向收益", np.nan)
        rec["策略后向收益%"] = round(sret * 100.0, 2) if pd.notna(sret) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def screen_at_date(
    data: dict[str, pd.DataFrame],
    filters: ScreenFilters,
    as_of: pd.Timestamp,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """在指定交易日按历史数据选股（无未来函数），并附选股理由。"""
    snap = snapshot_at_date(data, as_of, filters.lookback_days)
    if snap.empty:
        return snap
    filtered = apply_filters(snap, filters)
    if filtered.empty:
        return filtered
    out = filtered.sort_values("涨幅%", ascending=False).head(int(top_n)).copy()
    out["选股日期"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    out["选股理由"] = [
        pick_rationale(row, filters, rank=i + 1, selection_date=as_of)
        for i, (_, row) in enumerate(out.iterrows())
    ]
    return out.reset_index(drop=True)


def enrich_picks_with_performance(
    picks: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    *,
    forward_days: int = 20,
    backward_days: int = 20,
    strategy_name: str | None = None,
    params: dict[str, float] | None = None,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """为入选标的补充前后收益/回撤及（可选）策略后向回测。"""
    if picks.empty:
        return picks
    rows: list[dict[str, Any]] = []
    as_of = pd.Timestamp(as_of)
    for _, row in picks.iterrows():
        ticker = str(row["代码"]).upper()
        df = data.get(ticker)
        rec = row.to_dict()
        if df is None or df.empty:
            rows.append(rec)
            continue
        perf = forward_backward_metrics(df, as_of, forward_days=forward_days, backward_days=backward_days)
        rec.update(perf)
        if strategy_name:
            strat_perf = backtest_pick_forward(
                df, as_of, strategy_name, params=params,
                forward_days=forward_days, allow_short=allow_short,
                fee_bps=fee_bps, slippage_bps=slippage_bps,
            )
            rec.update(strat_perf)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_historical_daily_screen(
    data: dict[str, pd.DataFrame],
    filters: ScreenFilters,
    *,
    start: str | date,
    end: str | date,
    rebalance_days: int = 5,
    top_picks: int = 5,
    forward_days: int = 20,
    backward_days: int = 20,
    strategy_name: str | None = None,
    params: dict[str, float] | None = None,
    allow_short: bool = False,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    """历史按日（每 rebalance_days 个交易日）回放选股，并计算前后收益/回撤与策略后向回测。

    返回 daily_picks（明细表）、by_date（按日汇总）、summary（整体统计）。
    """
    if not data:
        return {"daily_picks": pd.DataFrame(), "by_date": pd.DataFrame(), "summary": {}}

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    best = max(data.keys(), key=lambda t: len(data[t]))
    calendar = data[best].index
    calendar = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
    if len(calendar) < rebalance_days + forward_days + 30:
        return {"error": "数据不足，请扩大日期范围"}

    all_picks: list[pd.DataFrame] = []
    step = max(int(rebalance_days), 1)
    warmup = max(filters.lookback_days + 30, 60)

    for i in range(warmup, len(calendar) - forward_days, step):
        as_of = calendar[i]
        picks = screen_at_date(data, filters, as_of, top_n=top_picks)
        if picks.empty:
            continue
        enriched = enrich_picks_with_performance(
            picks, data, as_of,
            forward_days=forward_days,
            backward_days=backward_days,
            strategy_name=strategy_name,
            params=params,
            allow_short=allow_short,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        all_picks.append(enriched)

    if not all_picks:
        return {"daily_picks": pd.DataFrame(), "by_date": pd.DataFrame(), "summary": {}}

    daily = pd.concat(all_picks, ignore_index=True)
    fwd_col = f"后{forward_days}日收益"
    back_col = f"前{backward_days}日收益"
    strat_col = "策略后向收益"

    by_date_rows: list[dict[str, Any]] = []
    for dt, grp in daily.groupby("选股日期", sort=False):
        fwd = pd.to_numeric(grp.get(fwd_col), errors="coerce")
        strat = pd.to_numeric(grp.get(strat_col), errors="coerce") if strat_col in grp.columns else pd.Series(dtype=float)
        by_date_rows.append({
            "选股日期": dt,
            "入选数": len(grp),
            "入选代码": ", ".join(grp["代码"].astype(str).tolist()),
            "平均后向收益": float(fwd.mean()) if fwd.notna().any() else np.nan,
            "后向盈利占比": float((fwd > 0).mean()) if fwd.notna().any() else np.nan,
            "平均策略后向收益": float(strat.mean()) if len(strat) and strat.notna().any() else np.nan,
        })
    by_date = pd.DataFrame(by_date_rows)

    fwd_all = pd.to_numeric(daily.get(fwd_col), errors="coerce")
    back_all = pd.to_numeric(daily.get(back_col), errors="coerce")
    summary = {
        "选股批次数": float(len(by_date)),
        "入选总人次": float(len(daily)),
        "平均后向收益": float(fwd_all.mean()) if fwd_all.notna().any() else np.nan,
        "后向盈利占比": float((fwd_all > 0).mean()) if fwd_all.notna().any() else np.nan,
        "平均前向收益": float(back_all.mean()) if back_all.notna().any() else np.nan,
    }
    if strat_col in daily.columns:
        s = pd.to_numeric(daily[strat_col], errors="coerce")
        if s.notna().any():
            summary["平均策略后向收益"] = float(s.mean())
            summary["策略后向盈利占比"] = float((s > 0).mean())

    return {"daily_picks": daily, "by_date": by_date, "summary": summary}


def add_rationale_to_merged(
    merged: pd.DataFrame,
    filters: ScreenFilters,
    selection_date: str | date | pd.Timestamp,
) -> pd.DataFrame:
    """给选股合并表补充选股日期与选股理由（日期必填）。"""
    if merged.empty:
        return merged
    sel = normalize_selection_date(selection_date)
    out = stamp_selection_date(merged, sel)
    ranks = out.sort_values("涨幅%", ascending=False).reset_index(drop=True)
    rank_map = {str(r["代码"]): i + 1 for i, r in ranks.iterrows()}
    out["选股理由"] = [
        pick_rationale(row, filters, rank=rank_map.get(str(row["代码"]), 0), selection_date=sel)
        for _, row in out.iterrows()
    ]
    # 选股日期 | 代码 | 选股理由 优先展示
    prefer = [c for c in ["选股日期", "代码", "名称", "选股理由"] if c in out.columns]
    rest = [c for c in out.columns if c not in prefer]
    return out[prefer + rest]


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
    selection_date: str | date | pd.Timestamp | None = None,
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

    selection_date：本次选股的确定日期（某年某月某日）；默认等于 end。
    返回 dict：selection_date / snapshot / filtered / backtest / merged / summary。
    """
    sel = normalize_selection_date(selection_date or end)
    data_end = sel  # 行情与指标截至选股日，避免未来数据
    with_sector = bool(filters.sectors)
    snapshot = build_universe_snapshot(
        pool, start, data_end,
        lookback_days=filters.lookback_days,
        pool_size=pool_size,
        custom_tickers=custom_tickers,
        with_sector=with_sector,
    )
    result: dict[str, Any] = {
        "selection_date": sel,
        "snapshot": snapshot,
        "filtered": pd.DataFrame(),
        "backtest": pd.DataFrame(),
        "merged": pd.DataFrame(),
        "summary": {},
    }
    if snapshot.empty:
        return result

    filtered = stamp_selection_date(apply_filters(snapshot, filters), sel)
    result["filtered"] = filtered
    if filtered.empty or not strategy_name:
        if not filtered.empty:
            result["filtered"] = add_rationale_to_merged(
                filtered.drop(columns=["选股理由"], errors="ignore"),
                filters, sel,
            )
        return result

    targets = filtered["代码"].head(int(max_backtest)).tolist()
    bt = backtest_universe(
        targets, start, data_end, strategy_name,
        params=params,
        allow_short=allow_short,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    result["backtest"] = bt
    result["merged"] = add_rationale_to_merged(
        merge_snapshot_backtest(filtered, bt, selection_date=sel), filters, sel,
    )
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
    date_col = "选股日期" if "选股日期" in df.columns else "选股时间"
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(date_col, sort=False)
    rows = []
    for ts, grp in g:
        ret = pd.to_numeric(grp.get("策略累计收益"), errors="coerce")
        rows.append({
            "选股日期": ts,
            "股票池": grp["股票池"].iloc[0] if "股票池" in grp.columns else "",
            "策略": grp["策略"].iloc[0] if "策略" in grp.columns else "",
            "入选数": len(grp),
            "平均策略收益": ret.mean(),
            "盈利占比": (ret > 0).mean() if len(ret) else 0.0,
        })
    return pd.DataFrame(rows)
