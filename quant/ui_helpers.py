"""纯展示层助手：数字格式化 + Plotly 图表构建。

从 app.py 抽离的**无 Streamlit 依赖**纯函数，可独立 import 与单测
（app.py 仍以原名 import 使用，保持调用点不变）。
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quant import backtest
from quant import indicators as ind
import ths_theme as theme

__all__ = [
    "fmt_pct",
    "fmt_num",
    "fmt_mcap",
    "fmt_dollar_m",
    "parse_tickers",
    "price_chart",
    "equity_chart",
    "compare_chart",
]


# ---------------- 数字格式化 ----------------

def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def fmt_num(x: float) -> str:
    return f"{x:,.2f}"


def fmt_mcap(x: float) -> str:
    if pd.isna(x):
        return "-"
    b = float(x) / 1e9
    if b >= 1:
        return f"{b:,.1f}B"
    return f"{float(x)/1e6:,.0f}M"


def fmt_dollar_m(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"${float(x)/1e6:,.1f}M"


def parse_tickers(raw: str) -> list[str]:
    """把逗号/空格/换行分隔的代码串解析为去重后的列表。"""
    out: list[str] = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        t = chunk.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


# ---------------- Plotly 图表构建 ----------------

def price_chart(df: pd.DataFrame, strat_name: str, params: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="K线", increasing_line_color=theme.UP, decreasing_line_color=theme.DOWN,
        ),
        row=1, col=1,
    )
    close = df["Close"]
    if strat_name == "双均线交叉":
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("fast", 20))),
                                 name=f"MA{int(params.get('fast', 20))}", line=dict(color=theme.GOLD, width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("slow", 60))),
                                 name=f"MA{int(params.get('slow', 60))}", line=dict(color=theme.BLUE, width=1.3)), row=1, col=1)
    elif strat_name == "布林带回归":
        b = ind.bollinger_bands(close, int(params.get("window", 20)), float(params.get("num_std", 2.0)))
        fig.add_trace(go.Scatter(x=df.index, y=b["upper"], name="上轨", line=dict(color=theme.PURPLE, width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=b["mid"], name="中轨", line=dict(color=theme.GOLD, width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=b["lower"], name="下轨", line=dict(color=theme.PURPLE, width=1)), row=1, col=1)
    elif strat_name == "唐奇安通道突破（海龟）":
        d = ind.donchian(df, int(params.get("entry", 20)))
        de = ind.donchian(df, int(params.get("exit", 10)))
        fig.add_trace(go.Scatter(x=df.index, y=d["upper"], name="入场上轨", line=dict(color=theme.ORANGE, width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=de["lower"], name="离场下轨", line=dict(color=theme.UP, width=1)), row=1, col=1)
    elif strat_name == "肯特纳通道突破":
        k = ind.keltner(df, int(params.get("window", 20)), int(params.get("atr_window", 10)), float(params.get("mult", 2.0)))
        fig.add_trace(go.Scatter(x=df.index, y=k["upper"], name="上轨", line=dict(color=theme.PURPLE, width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k["mid"], name="中轨", line=dict(color=theme.GOLD, width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k["lower"], name="下轨", line=dict(color=theme.PURPLE, width=1)), row=1, col=1)
    elif strat_name == "ATR 跟踪止损趋势":
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("ma_window", 50))),
                                 name=f"MA{int(params.get('ma_window', 50))}", line=dict(color=theme.GOLD, width=1.3)), row=1, col=1)
    elif strat_name == "趋势+动量双确认":
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("ma_window", 100))),
                                 name=f"MA{int(params.get('ma_window', 100))}", line=dict(color=theme.GOLD, width=1.3)), row=1, col=1)
    colors = [theme.UP if c >= o else theme.DOWN for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(height=520, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def equity_chart(result: backtest.BacktestResult, strat_name: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("净值曲线", "回撤"))
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name=f"{strat_name}",
                             line=dict(color=theme.ORANGE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark, name="买入持有基准",
                             line=dict(color=theme.MUTED, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="回撤",
                             fill="tozeroy", line=dict(color=theme.BLUE, width=1)), row=2, col=1)
    fig.update_layout(height=460, template="tiger", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    return fig


def compare_chart(curves: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    palette = theme.PALETTE
    for i, (name, eq) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name,
                                 line=dict(width=2, color=palette[i % len(palette)])))
    fig.update_layout(height=480, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.06), title="各策略净值曲线对比")
    fig.update_yaxes(title_text="净值")
    return fig
