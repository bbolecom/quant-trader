"""技术指标计算模块（纯 pandas/numpy 实现，无第三方 TA 库依赖）。"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """指数移动平均。"""
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标 (RSI)，采用 Wilder 平滑。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100 - (100 / (1 + rs))
    # 当平均亏损为 0 时，RSI 视为 100。
    out = out.fillna(100.0)
    return out


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD 指标，返回 macd / signal / hist 三列。"""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}
    )


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """布林带，返回 mid / upper / lower 三列。"""
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def momentum(series: pd.Series, window: int = 20) -> pd.Series:
    """动量：当前价格相对 window 日前的涨跌幅。"""
    return series.pct_change(periods=window)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅 (ATR)，采用 Wilder 平滑。需要 High/Low/Close 列。"""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """唐奇安通道：过去 window 日的最高/最低（不含当日，避免自我突破）。"""
    upper = df["High"].rolling(window).max().shift(1)
    lower = df["Low"].rolling(window).min().shift(1)
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})


def keltner(
    df: pd.DataFrame, window: int = 20, atr_window: int = 10, mult: float = 2.0
) -> pd.DataFrame:
    """肯特纳通道：以 EMA 为中轨，上下各 mult 倍 ATR。"""
    mid = ema(df["Close"], window)
    rng = atr(df, atr_window) * mult
    return pd.DataFrame({"mid": mid, "upper": mid + rng, "lower": mid - rng})


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """平均趋向指标 (ADX) 及 +DI / -DI，用于衡量趋势强度。

    ADX > 25 通常视为趋势行情，< 20 视为震荡行情。
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1 / window
    atr_ = tr.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=window, adjust=False).mean() / atr_

    denom = (plus_di + minus_di).replace(0.0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx_ = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()

    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R，范围 [-100, 0]；≥ -20 视为超买区。"""
    high = df["High"].rolling(period, min_periods=period).max()
    low = df["Low"].rolling(period, min_periods=period).min()
    close = df["Close"]
    denom = (high - low).replace(0.0, pd.NA)
    return (-100 * (high - close) / denom).fillna(-50.0)


def efficiency_ratio(series: pd.Series, window: int = 20) -> pd.Series:
    """考夫曼效率比 (ER)：净位移 / 路程总和，∈ [0, 1]。

    越接近 1 说明走势越"笔直"（强趋势），越接近 0 说明越"曲折"（震荡）。
    """
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum()
    return (change / volatility.replace(0.0, pd.NA)).fillna(0.0)
