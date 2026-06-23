"""波动率衰减策略：反向波动率 ETF 择时 + 卖认沽(CSP) 候选扫描。

基于历史研究结论：
    - 反向波动率 ETF（SVIX/SVXY）+ 均线择时，表达「做空波动率衰减」且风险有限。
    - 高成交额、中等偏高已实现波动率个股更适合卖认沽吃 theta（勿卖裸 call / 宽跨式）。

CSP 权利金为 Black-Scholes + 波动率风险溢价(VRP) 估算，非真实期权链报价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .data import fetch_history, fetch_history_batch

TRADING_DAYS = 252
RFR = 0.04
DEFAULT_VRP = 0.15
CSP_DTE_DAYS = 35
CSP_DELTA = 0.25

# 反向波动率 ETF（优先 SVIX：-1x 单日，VRP 更厚）
INVERSE_VOL_ETFS: dict[str, str] = {
    "SVIX": "反向波动率 -1x（推荐）",
    "SVXY": "反向波动率 -0.5x（更温和）",
}

# 默认 CSP 扫描池：高成交额 + 常见高波动优质股
DEFAULT_CSP_UNIVERSE: list[str] = [
    "NVDA", "AMD", "TSLA", "AVGO", "META", "GOOGL", "MU", "WDC", "SNDK",
    "COIN", "MARA", "AAPL", "MSFT", "AMZN", "NFLX", "INTC", "QCOM", "CRM",
    "ORCL", "BA", "DIS", "UBER", "PYPL", "SQ", "SHOP", "PLTR", "SMCI",
]


@dataclass
class InverseEtfSignal:
    ticker: str
    label: str
    as_of: str
    close: float
    ma: float
    ma_window: int
    pct_vs_ma: float
    action: str
    detail: str


@dataclass
class VixAlert:
    vix: float
    vix_ma20: float
    daily_chg_pct: float
    level: str
    message: str


@dataclass
class CspFilters:
    min_dollar_vol_m: float = 500.0
    min_rv_pct: float = 30.0
    max_rv_pct: float = 70.0
    min_mcap_b: float = 20.0
    require_above_ma200: bool = True
    vrp: float = DEFAULT_VRP
    put_delta: float = CSP_DELTA
    dte_days: int = CSP_DTE_DAYS


@dataclass
class CspCandidate:
    ticker: str
    close: float
    rv20_pct: float
    dollar_vol_m: float
    mcap_b: float | None
    ma200: float
    above_ma200: bool
    put_strike: float
    est_premium: float
    monthly_yield_pct: float
    score: float
    flags: list[str] = field(default_factory=list)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    lr = np.log(close.astype(float) / close.astype(float).shift(1))
    return lr.rolling(window, min_periods=max(5, window // 2)).std(ddof=0) * np.sqrt(TRADING_DAYS)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_cdf_inv(p: float) -> float:
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bs_put_price(S: float, K: float, T: float, sigma: float, r: float = RFR) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def strike_for_put_delta(S: float, T: float, sigma: float, target_delta: float = CSP_DELTA, r: float = RFR) -> float:
    """|delta|≈target 的认沽行权价（近似）。"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return S * 0.9
    d1 = _norm_cdf_inv(target_delta)
    sqrtT = math.sqrt(T)
    return S * math.exp((r + 0.5 * sigma ** 2) * T + d1 * sigma * sqrtT)


def estimate_csp(
    spot: float,
    rv_annual: float,
    *,
    vrp: float = DEFAULT_VRP,
    delta: float = CSP_DELTA,
    dte_days: int = CSP_DTE_DAYS,
) -> tuple[float, float, float]:
    """估算 CSP：返回 (行权价, 权利金/股, 月化收益率%)。"""
    if spot <= 0 or rv_annual <= 0:
        return spot * 0.9, 0.0, 0.0
    iv = rv_annual * (1 + vrp)
    T = dte_days / TRADING_DAYS
    K = strike_for_put_delta(spot, T, iv, target_delta=delta)
    prem = bs_put_price(spot, K, T, iv)
    yield_pct = (prem / K) * (30.0 / dte_days) * 100.0 if K > 0 else 0.0
    return round(K, 2), round(prem, 2), round(yield_pct, 2)


def inverse_etf_signal(
    df: pd.DataFrame,
    ticker: str,
    ma_window: int = 50,
) -> InverseEtfSignal:
    """反向波动率 ETF 均线择时信号（T 日收盘判断 → 次日执行）。"""
    label = INVERSE_VOL_ETFS.get(ticker, ticker)
    close = df["Close"].astype(float)
    if len(close) < ma_window + 5:
        raise ValueError(f"{ticker} 数据不足（需至少 {ma_window + 5} 根 K 线）。")
    ma = close.rolling(ma_window).mean()
    px = float(close.iloc[-1])
    ma_val = float(ma.iloc[-1])
    pct = (px / ma_val - 1.0) if ma_val > 0 else 0.0
    as_of = pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d")
    above = px > ma_val
    if above:
        action = "🟢 持有 / 可建仓"
        detail = f"收盘 ${px:,.2f} 站上 {ma_window} 日均线 ${ma_val:,.2f}（+{pct:.1%}）→ 次日可持有反向波动率 ETF。"
    else:
        action = "🔴 清仓观望"
        detail = f"收盘 ${px:,.2f} 跌破 {ma_window} 日均线 ${ma_val:,.2f}（{pct:.1%}）→ 次日应清仓转现金。"
    return InverseEtfSignal(
        ticker=ticker, label=label, as_of=as_of, close=px, ma=ma_val,
        ma_window=ma_window, pct_vs_ma=pct, action=action, detail=detail,
    )


def ma_timing_backtest(close: pd.Series, ma_window: int = 50) -> dict[str, float]:
    """均线择时 vs 买入持有（用于 UI 展示参考绩效）。"""
    close = close.astype(float).dropna()
    if len(close) < ma_window + 10:
        return {}
    ret = close.pct_change().fillna(0.0)
    ma = close.rolling(ma_window).mean()
    sig = (close > ma).shift(1).fillna(False).astype(float)
    strat_ret = sig * ret
    eq_bh = (1 + ret).cumprod()
    eq_st = (1 + strat_ret).cumprod()
    years = len(ret) / TRADING_DAYS
    def _stats(eq, r):
        cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
        vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
        sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0
        dd = (eq / eq.cummax() - 1).min()
        return {"年化": cagr, "夏普": sharpe, "最大回撤": dd, "总收益": eq.iloc[-1] - 1}
    bh = _stats(eq_bh, ret)
    st = _stats(eq_st, strat_ret)
    return {"买入持有": bh, "均线择时": st}


def vix_alert(end: str | date | None = None) -> VixAlert | None:
    """VIX 熔断预警：单日急升或绝对水平过高时提示减仓。"""
    try:
        end_d = end or date.today()
        start_d = (pd.Timestamp(end_d) - pd.DateOffset(days=120)).date()
        df = fetch_history("^VIX", start=start_d, end=end_d)
        close = df["Close"].astype(float)
        if len(close) < 5:
            return None
        vix = float(close.iloc[-1])
        vix_ma = float(close.rolling(20).mean().iloc[-1])
        chg = float(close.pct_change().iloc[-1]) if len(close) > 1 else 0.0
        if chg >= 0.30 or vix >= 35:
            level = "🔴 高危"
            msg = "VIX 急升或处于高位 → 波动率 sleeve 建议减半或全部清仓。"
        elif chg >= 0.15 or vix >= 28:
            level = "🟡 警戒"
            msg = "波动率抬升 → 暂停新开 CSP，已有反向 ETF 考虑减仓。"
        else:
            level = "🟢 正常"
            msg = "VIX 处于常规区间，可按计划执行。"
        return VixAlert(vix=vix, vix_ma20=vix_ma, daily_chg_pct=chg, level=level, message=msg)
    except Exception:  # noqa: BLE001
        return None


def _mcap_b(ticker: str) -> float | None:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        mcap = getattr(info, "market_cap", None)
        return float(mcap) / 1e9 if mcap else None
    except Exception:  # noqa: BLE001
        return None


def analyze_csp_ticker(
    ticker: str,
    df: pd.DataFrame,
    filters: CspFilters,
) -> CspCandidate | None:
    """单标的 CSP  suitability 分析。"""
    if df is None or df.empty or len(df) < 220:
        return None
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    px = float(close.iloc[-1])
    rv = float(realized_vol(close).iloc[-1])
    if not np.isfinite(rv) or rv <= 0:
        return None
    rv_pct = rv * 100
    dollar_vol = float((close * vol).tail(20).mean()) / 1e6
    ma200 = float(close.rolling(200).mean().iloc[-1])
    above = px >= ma200 * 0.98  # 允许 2% 缓冲
    mcap = _mcap_b(ticker)

    flags: list[str] = []
    if dollar_vol < filters.min_dollar_vol_m:
        return None
    if rv_pct < filters.min_rv_pct or rv_pct > filters.max_rv_pct:
        return None
    if filters.require_above_ma200 and not above:
        flags.append("低于200日均线")
    if mcap is not None and mcap < filters.min_mcap_b:
        flags.append("市值偏小")
    if len(df) < 252:
        flags.append("上市时间短")
    if rv_pct > 65:
        flags.append("波动偏高，控制仓位")

    K, prem, yld = estimate_csp(px, rv, vrp=filters.vrp, delta=filters.put_delta, dte_days=filters.dte_days)
    # 综合得分：月化 yield × RV 适中度 × 流动性
    rv_score = 1.0 - abs(rv_pct - 45) / 45  # 45% RV 附近最优
    rv_score = max(0.2, min(1.0, rv_score))
    liq_score = min(1.0, dollar_vol / 2000.0)
    trend_score = 1.0 if above else 0.5
    score = yld * rv_score * liq_score * trend_score

    return CspCandidate(
        ticker=ticker, close=px, rv20_pct=round(rv_pct, 1),
        dollar_vol_m=round(dollar_vol, 1), mcap_b=round(mcap, 1) if mcap else None,
        ma200=round(ma200, 2), above_ma200=above,
        put_strike=K, est_premium=prem, monthly_yield_pct=yld,
        score=round(score, 2), flags=flags,
    )


def scan_csp_candidates(
    tickers: list[str],
    start: str | date,
    end: str | date,
    filters: CspFilters | None = None,
) -> pd.DataFrame:
    """批量扫描 CSP 候选。"""
    filters = filters or CspFilters()
    syms = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not syms:
        return pd.DataFrame()
    batch = fetch_history_batch(syms, start=start, end=end)
    rows: list[dict] = []
    for t, df in batch.items():
        cand = analyze_csp_ticker(t, df, filters)
        if cand is None:
            continue
        rows.append({
            "代码": cand.ticker,
            "最新价": cand.close,
            "RV20%": cand.rv20_pct,
            "成交额M": cand.dollar_vol_m,
            "市值B": cand.mcap_b,
            "200MA": cand.ma200,
            "站上200MA": "是" if cand.above_ma200 else "否",
            "建议Put行权": cand.put_strike,
            "估算权利金": cand.est_premium,
            "月化收益%": cand.monthly_yield_pct,
            "综合分": cand.score,
            "提示": "；".join(cand.flags) if cand.flags else "",
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("综合分", ascending=False).reset_index(drop=True)
    return out


def daily_playbook(
    etf_sig: InverseEtfSignal | None,
    vix: VixAlert | None,
    csp_table: pd.DataFrame,
    max_csp: int = 5,
) -> list[str]:
    """生成每日执行清单（文字步骤）。"""
    steps: list[str] = []
    if vix and vix.level.startswith("🔴"):
        steps.append(f"1. 【熔断】{vix.message}（VIX={vix.vix:.1f}，日变{vix.daily_chg_pct:+.0%}）")
    elif vix:
        steps.append(f"1. VIX 状态：{vix.level} — {vix.message}")
    else:
        steps.append("1. VIX 数据暂不可用，请手动查看 ^VIX。")
    if etf_sig:
        steps.append(f"2. 反向 ETF（{etf_sig.ticker}）：{etf_sig.action} — {etf_sig.detail}")
        steps.append("   · 建议仓位 ≤ 总资金 12%；与 CSP 合计 ≤ 30%。")
    if csp_table is not None and not csp_table.empty:
        top = csp_table.head(max_csp)
        codes = "、".join(top["代码"].tolist())
        steps.append(f"3. CSP 候选（Top {len(top)}）：{codes}")
        steps.append(
            "   · 每单：30–45 天到期、Delta≈0.25 认沽；单票保证金 ≤ 总资金 5%；"
            "权利金赚到 50–70% 止盈；财报前 1 周不开新仓。"
        )
    else:
        steps.append("3. 今日无符合筛选的 CSP 候选，或尚未扫描。")
    steps.append("4. 禁止：裸卖 call、裸宽跨式；两类策略在崩盘日同向，务必控制总敞口。")
    return steps
