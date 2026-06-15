"""美股量化交易策略回测平台 — Streamlit 应用入口。

运行方式:
    streamlit run app.py

三大功能（顶部标签页）：
    1. 单策略回测：选定策略与参数，查看净值、绩效、交易明细。
    2. 参数寻优：对策略参数做网格搜索，自动找出最优组合。
    3. 策略对比：多个策略横向对比净值曲线与绩效。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quant import (
    backtest,
    indicators as ind,
    optimize,
    paper,
    portfolio,
    probability,
    regime,
    report as report_mod,
    signals,
    strategies,
    validation,
)
from quant.data import DataError, fetch_history

ROOT_DIR = Path(__file__).resolve().parent
PAPER_ACCOUNT_FILE = ROOT_DIR / "paper_account.json"


def _page_icon():
    """优先用内置图标作为页面图标，缺失时回退到 emoji。"""
    icon_path = ROOT_DIR / "assets" / "icon.png"
    if icon_path.exists():
        try:
            from PIL import Image

            return Image.open(icon_path)
        except Exception:  # noqa: BLE001
            return "📈"
    return "📈"

st.set_page_config(
    page_title="美股量化策略回测平台",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1500px;}
    div[data-testid="stMetric"] {
        background: #ffffff0d;
        border: 1px solid #ffffff1a;
        border-radius: 12px;
        padding: 14px 16px;
    }
    h1, h2, h3 {letter-spacing: .3px;}
    /* 让顶部标签可横向滑动，适配手机窄屏 */
    div[data-testid="stTabs"] div[role="tablist"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
    }
    div[data-testid="stTabs"] button[role="tab"] {white-space: nowrap;}
    /* 手机端收窄内边距、缩小标题，提升可读性 */
    @media (max-width: 640px) {
        .main .block-container {padding: 1rem 0.6rem !important;}
        div[data-testid="stMetricValue"] {font-size: 1.1rem !important;}
        h1 {font-size: 1.5rem !important;}
        h2 {font-size: 1.2rem !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_history(ticker, start=start, end=end)


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def fmt_num(x: float) -> str:
    return f"{x:,.2f}"


def get_data(cfg: dict) -> pd.DataFrame | None:
    """按当前配置拉取数据，失败时在界面提示并返回 None。"""
    try:
        with st.spinner(f"正在拉取 {cfg['ticker']} 行情数据…"):
            return load_data(cfg["ticker"], cfg["start"], cfg["end"])
    except DataError as e:
        st.error(f"❌ {e}")
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ 数据获取失败：{e}")
    return None


def get_multi_data(tickers: list[str], cfg: dict) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """批量拉取多个标的数据，返回 (成功字典, 失败代码列表)。"""
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    progress = st.progress(0.0, text="正在拉取行情数据…")
    for i, t in enumerate(tickers):
        try:
            data[t] = load_data(t, cfg["start"], cfg["end"])
        except Exception:  # noqa: BLE001
            failed.append(t)
        progress.progress((i + 1) / len(tickers), text=f"已拉取 {i + 1}/{len(tickers)}：{t}")
    progress.empty()
    return data, failed


def parse_tickers(raw: str) -> list[str]:
    """把逗号/空格/换行分隔的代码串解析为去重后的列表。"""
    out: list[str] = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        t = chunk.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# 侧边栏 — 共享配置（标的、区间、资金、成本）
# ---------------------------------------------------------------------------
def sidebar() -> dict:
    st.sidebar.title("⚙️ 全局配置")

    st.sidebar.subheader("标的与区间")
    ticker = st.sidebar.text_input(
        "股票代码", value="AAPL", help="美股代码，如 AAPL、MSFT、NVDA、SPY"
    ).strip().upper()

    col1, col2 = st.sidebar.columns(2)
    default_start = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
    start = col1.date_input("开始日期", value=default_start, max_value=date.today())
    end = col2.date_input("结束日期", value=date.today(), max_value=date.today())

    allow_short = st.sidebar.checkbox(
        "允许做空", value=False, help="开启后，离场信号将转为反向做空"
    )

    st.sidebar.subheader("资金与成本")
    capital = st.sidebar.number_input("初始资金 (USD)", value=100_000, step=10_000, min_value=1_000)
    fee_bps = st.sidebar.slider("手续费 (基点)", 0.0, 30.0, 5.0, 0.5, help="1 基点 = 0.01%")
    slippage_bps = st.sidebar.slider("滑点 (基点)", 0.0, 30.0, 2.0, 0.5)

    st.sidebar.divider()
    st.sidebar.caption("⚠️ 回测基于历史数据，不构成投资建议。")

    return {
        "ticker": ticker,
        "start": str(start),
        "end": str(end),
        "allow_short": allow_short,
        "capital": float(capital),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "cost": {"capital": float(capital), "fee_bps": fee_bps, "slippage_bps": slippage_bps},
    }


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def price_chart(df: pd.DataFrame, strat_name: str, params: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="K线", increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71",
        ),
        row=1, col=1,
    )
    close = df["Close"]
    if strat_name == "双均线交叉":
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("fast", 20))),
                                 name=f"MA{int(params.get('fast', 20))}", line=dict(color="#f39c12", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("slow", 60))),
                                 name=f"MA{int(params.get('slow', 60))}", line=dict(color="#3498db", width=1.3)), row=1, col=1)
    elif strat_name == "布林带回归":
        b = ind.bollinger_bands(close, int(params.get("window", 20)), float(params.get("num_std", 2.0)))
        fig.add_trace(go.Scatter(x=df.index, y=b["upper"], name="上轨", line=dict(color="#9b59b6", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=b["mid"], name="中轨", line=dict(color="#f39c12", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=b["lower"], name="下轨", line=dict(color="#9b59b6", width=1)), row=1, col=1)
    elif strat_name == "唐奇安通道突破（海龟）":
        d = ind.donchian(df, int(params.get("entry", 20)))
        de = ind.donchian(df, int(params.get("exit", 10)))
        fig.add_trace(go.Scatter(x=df.index, y=d["upper"], name="入场上轨", line=dict(color="#e74c3c", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=de["lower"], name="离场下轨", line=dict(color="#2ecc71", width=1)), row=1, col=1)
    elif strat_name == "肯特纳通道突破":
        k = ind.keltner(df, int(params.get("window", 20)), int(params.get("atr_window", 10)), float(params.get("mult", 2.0)))
        fig.add_trace(go.Scatter(x=df.index, y=k["upper"], name="上轨", line=dict(color="#9b59b6", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k["mid"], name="中轨", line=dict(color="#f39c12", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=k["lower"], name="下轨", line=dict(color="#9b59b6", width=1)), row=1, col=1)
    elif strat_name == "ATR 跟踪止损趋势":
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("ma_window", 50))),
                                 name=f"MA{int(params.get('ma_window', 50))}", line=dict(color="#f39c12", width=1.3)), row=1, col=1)
    elif strat_name == "趋势+动量双确认":
        fig.add_trace(go.Scatter(x=df.index, y=ind.sma(close, int(params.get("ma_window", 100))),
                                 name=f"MA{int(params.get('ma_window', 100))}", line=dict(color="#f39c12", width=1.3)), row=1, col=1)
    colors = ["#e74c3c" if c >= o else "#2ecc71" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="成交量", marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(height=520, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def equity_chart(result: backtest.BacktestResult, strat_name: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("净值曲线", "回撤"))
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name=f"{strat_name}",
                             line=dict(color="#e74c3c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark, name="买入持有基准",
                             line=dict(color="#7f8c8d", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="回撤",
                             fill="tozeroy", line=dict(color="#3498db", width=1)), row=2, col=1)
    fig.update_layout(height=460, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    return fig


def compare_chart(curves: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for i, (name, eq) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name,
                                 line=dict(width=2, color=palette[i % len(palette)])))
    fig.update_layout(height=480, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.06), title="各策略净值曲线对比")
    fig.update_yaxes(title_text="净值")
    return fig


# ---------------------------------------------------------------------------
# 标签页 1：单策略回测
# ---------------------------------------------------------------------------
def tab_single(cfg: dict) -> None:
    st.subheader("单策略回测")
    c1, c2 = st.columns([1, 2])
    strat_name = c1.selectbox("交易策略", strategies.list_strategies(), key="single_strat")
    strat = strategies.get_strategy(strat_name)
    c2.caption(strat.description)

    params: dict[str, float] = {}
    if strat.params:
        cols = st.columns(min(len(strat.params), 4))
        for i, p in enumerate(strat.params):
            with cols[i % len(cols)]:
                if p.is_int:
                    params[p.key] = st.slider(p.label, int(p.min_value), int(p.max_value),
                                              int(p.default), int(p.step), key=f"s_{p.key}")
                else:
                    params[p.key] = st.slider(p.label, float(p.min_value), float(p.max_value),
                                              float(p.default), float(p.step), key=f"s_{p.key}")

    if not st.button("🚀 运行回测", type="primary", key="run_single"):
        return

    df = get_data(cfg)
    if df is None:
        return

    position = strat.generate(df, allow_short=cfg["allow_short"], **params)
    result = backtest.run_backtest(df, position, initial_capital=cfg["capital"],
                                   fee_bps=cfg["fee_bps"], slippage_bps=cfg["slippage_bps"])

    _render_metrics(result)
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{cfg['ticker']} 行情与指标**")
        st.plotly_chart(price_chart(df, strat.name, params), use_container_width=True)
    with right:
        st.markdown("**策略表现**")
        st.plotly_chart(equity_chart(result, strat.name), use_container_width=True)
    st.divider()
    _render_trades(result)


# ---------------------------------------------------------------------------
# 标签页 2：参数寻优
# ---------------------------------------------------------------------------
def tab_optimize(cfg: dict) -> None:
    st.subheader("参数寻优（网格搜索）")
    st.caption("对所选策略的参数进行网格搜索，按目标指标找出历史表现最优的参数组合。")

    strat_name = st.selectbox("交易策略", strategies.list_strategies(), key="opt_strat")
    strat = strategies.get_strategy(strat_name)

    if not strat.params:
        st.info("该策略没有可调参数，无需寻优。")
        return

    param_grid: dict[str, list] = {}
    st.markdown("**设置每个参数的搜索范围**")
    for p in strat.params:
        c1, c2, c3 = st.columns([2, 2, 1])
        if p.is_int:
            lo = c1.number_input(f"{p.label} · 最小", int(p.min_value), int(p.max_value),
                                 int(p.min_value), key=f"o_{p.key}_lo")
            hi = c2.number_input(f"{p.label} · 最大", int(p.min_value), int(p.max_value),
                                 int(p.default), key=f"o_{p.key}_hi")
            step = c3.number_input("步长", 1, 50, max(1, int((hi - lo) / 6) or 1), key=f"o_{p.key}_st")
            values = list(range(int(lo), int(hi) + 1, int(step))) or [int(lo)]
        else:
            lo = c1.number_input(f"{p.label} · 最小", float(p.min_value), float(p.max_value),
                                 float(p.min_value), key=f"o_{p.key}_lo")
            hi = c2.number_input(f"{p.label} · 最大", float(p.min_value), float(p.max_value),
                                 float(p.default), key=f"o_{p.key}_hi")
            step = c3.number_input("步长", float(p.step), 10.0, float(p.step), key=f"o_{p.key}_st")
            values = list(np.round(np.arange(lo, hi + 1e-9, step), 4)) or [lo]
        param_grid[p.key] = [float(v) if not p.is_int else int(v) for v in values]

    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    sort_by = st.selectbox("优化目标（排序指标）", optimize.SORT_METRICS, key="opt_sort")
    st.caption(f"待评估参数组合数：约 {n_combos} 组")

    if not st.button("🔍 开始寻优", type="primary", key="run_opt"):
        return

    df = get_data(cfg)
    if df is None:
        return

    try:
        with st.spinner("正在网格搜索…"):
            res = optimize.grid_search(df, strat_name, param_grid, sort_by=sort_by,
                                       allow_short=cfg["allow_short"], cost=cfg["cost"])
    except ValueError as e:
        st.error(f"❌ {e}")
        return

    st.success(f"完成！最优参数：{res.best_params}")
    cols = st.columns(4)
    cols[0].metric("夏普比率", fmt_num(res.best_stats["夏普比率"]))
    cols[1].metric("累计收益率", fmt_pct(res.best_stats["累计收益率"]))
    cols[2].metric("最大回撤", fmt_pct(res.best_stats["最大回撤"]))
    cols[3].metric("交易次数", f"{int(res.best_stats['交易次数'])}")

    keys = list(param_grid.keys())
    if len(keys) == 2:
        st.markdown("**参数热力图**")
        st.plotly_chart(_heatmap(res.table, keys[0], keys[1], sort_by), use_container_width=True)

    st.markdown("**全部结果（按目标指标排序）**")
    display = res.table.copy()
    for c in ["累计收益率", "年化收益率", "最大回撤"]:
        if c in display:
            display[c] = display[c].map(fmt_pct)
    for c in ["夏普比率", "索提诺比率", "卡尔玛比率"]:
        if c in display:
            display[c] = display[c].map(fmt_num)
    st.dataframe(display, use_container_width=True, hide_index=True)


def _heatmap(table: pd.DataFrame, x: str, y: str, metric: str) -> go.Figure:
    pivot = table.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                               colorscale="RdYlGn", colorbar=dict(title=metric)))
    fig.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=x, yaxis_title=y, title=f"{metric} 随参数变化")
    return fig


# ---------------------------------------------------------------------------
# 标签页 3：策略对比
# ---------------------------------------------------------------------------
def tab_compare(cfg: dict) -> None:
    st.subheader("多策略对比")
    st.caption("用各策略的默认参数，在同一标的与区间上横向对比表现。")

    names = st.multiselect("选择要对比的策略", strategies.list_strategies(),
                           default=strategies.list_strategies(), key="cmp_names")
    if not names:
        st.info("请至少选择一个策略。")
        return

    if not st.button("📊 开始对比", type="primary", key="run_cmp"):
        return

    df = get_data(cfg)
    if df is None:
        return

    table, curves = optimize.compare_strategies(df, names, allow_short=cfg["allow_short"], cost=cfg["cost"])
    st.plotly_chart(compare_chart(curves), use_container_width=True)

    st.markdown("**绩效对比表**")
    display = table.copy()
    for c in ["累计收益率", "年化收益率", "年化波动率", "最大回撤", "胜率"]:
        display[c] = display[c].map(fmt_pct)
    for c in ["夏普比率", "卡尔玛比率"]:
        display[c] = display[c].map(fmt_num)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 标签页 4：组合回测
# ---------------------------------------------------------------------------
def _strategy_param_inputs(strat, key_prefix: str) -> dict:
    """为策略生成参数滑块，返回参数字典。"""
    params: dict[str, float] = {}
    if not strat.params:
        return params
    cols = st.columns(min(len(strat.params), 4))
    for i, p in enumerate(strat.params):
        with cols[i % len(cols)]:
            if p.is_int:
                params[p.key] = st.slider(p.label, int(p.min_value), int(p.max_value),
                                          int(p.default), int(p.step), key=f"{key_prefix}_{p.key}")
            else:
                params[p.key] = st.slider(p.label, float(p.min_value), float(p.max_value),
                                          float(p.default), float(p.step), key=f"{key_prefix}_{p.key}")
    return params


def tab_portfolio(cfg: dict) -> None:
    st.subheader("多标的组合回测")
    st.caption("对组合内每个标的应用同一策略，按权重每日再平衡，合成组合层面的净值与绩效。")

    c1, c2 = st.columns([2, 1])
    raw = c1.text_input("组合标的（逗号分隔）", value="AAPL, MSFT, NVDA, GOOGL",
                        key="pf_tickers", help="例如：AAPL, MSFT, SPY")
    weight_mode = c2.selectbox("权重方式", ["等权", "自定义", "逆波动率", "风险平价"], key="pf_wmode")

    tickers = parse_tickers(raw)
    if not tickers:
        st.info("请至少输入一个标的代码。")
        return

    custom: dict[str, float] = {t: 1.0 for t in tickers}
    if weight_mode == "自定义":
        st.markdown("**设置权重（无需归一化，系统会自动归一）**")
        wcols = st.columns(min(len(tickers), 5))
        for i, t in enumerate(tickers):
            with wcols[i % len(wcols)]:
                custom[t] = st.number_input(t, min_value=0.0, value=1.0, step=0.5, key=f"pf_w_{t}")

    strat_name = st.selectbox("交易策略", strategies.list_strategies(), key="pf_strat")
    strat = strategies.get_strategy(strat_name)
    st.caption(strat.description)
    params = _strategy_param_inputs(strat, "pf")

    st.markdown("**风险控制**")
    r1, r2, r3 = st.columns(3)
    cap_pct = r1.slider("单标的权重上限", 10, 100, 100, 5, key="pf_cap",
                        help="限制单只股票占比，分散集中度风险") / 100.0
    use_vol_target = r2.checkbox("启用目标波动率", value=False, key="pf_usevt",
                                 help="动态调节整体仓位，使组合年化波动率贴近目标值")
    vol_target = None
    max_lev = 1.5
    if use_vol_target:
        vol_target = r2.slider("目标年化波动率", 5, 40, 15, 1, key="pf_vt") / 100.0
        max_lev = r3.slider("最大仓位系数", 1.0, 3.0, 1.5, 0.1, key="pf_lev",
                            help="目标波动率模式下允许的最高杠杆/仓位")

    if not st.button("🚀 运行组合回测", type="primary", key="run_pf"):
        return

    data, failed = get_multi_data(tickers, cfg)
    if failed:
        st.warning(f"以下标的获取失败，已忽略：{', '.join(failed)}")
    if not data:
        st.error("❌ 没有可用的标的数据。")
        return

    weights = portfolio.compute_weights(data, mode=weight_mode, custom=custom, cap=cap_pct)

    result = portfolio.run_portfolio(
        data, weights, strat_name, params=params, allow_short=cfg["allow_short"],
        fee_bps=cfg["fee_bps"], slippage_bps=cfg["slippage_bps"], initial_capital=cfg["capital"],
        vol_target=vol_target, max_leverage=max_lev,
    )

    s = result.stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("组合累计收益", fmt_pct(s["累计收益率"]))
    c2.metric("年化收益率", fmt_pct(s["年化收益率"]))
    c3.metric("夏普比率", fmt_num(s["夏普比率"]))
    c4.metric("最大回撤", fmt_pct(s["最大回撤"]))
    c5.metric("年化波动率", fmt_pct(s["年化波动率"]))
    if "平均仓位系数" in s:
        st.caption(f"目标波动率 {vol_target:.0%} ｜ 平均仓位系数 {s['平均仓位系数']:.2f} ｜ 期末资金 ${s['期末资金']:,.0f}")
    else:
        st.caption(f"期末资金 ${s['期末资金']:,.0f}")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**组合净值与回撤**")
        st.plotly_chart(_portfolio_chart(result), use_container_width=True)
    with right:
        st.markdown("**权重分布**")
        st.plotly_chart(_weight_pie(result.weights), use_container_width=True)
        if result.leverage is not None:
            st.markdown("**动态仓位系数**")
            lev_fig = go.Figure(go.Scatter(x=result.leverage.index, y=result.leverage,
                                           line=dict(color="#1abc9c", width=1.4)))
            lev_fig.update_layout(height=200, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis_title="仓位系数")
            st.plotly_chart(lev_fig, use_container_width=True)

    st.markdown("**各标的表现**")
    disp = result.asset_stats.copy()
    for col in ["权重", "累计收益率", "年化收益率", "最大回撤"]:
        disp[col] = disp[col].map(fmt_pct)
    disp["夏普比率"] = disp["夏普比率"].map(fmt_num)
    st.dataframe(disp, use_container_width=True, hide_index=True)


def _portfolio_chart(result: portfolio.PortfolioResult) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("组合净值曲线", "回撤"))
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="组合",
                             line=dict(color="#e74c3c", width=2.2)), row=1, col=1)
    palette = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
    for i, (t, eq) in enumerate(result.asset_equity.items()):
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=t,
                                 line=dict(width=1, color=palette[i % len(palette)], dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="回撤",
                             fill="tozeroy", line=dict(color="#3498db", width=1)), row=2, col=1)
    fig.update_layout(height=480, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    return fig


def _weight_pie(weights: dict[str, float]) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(weights.keys()), values=list(weights.values()),
                           hole=0.45, textinfo="label+percent"))
    fig.update_layout(height=480, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 标签页 5：信号扫描
# ---------------------------------------------------------------------------
def tab_signals(cfg: dict) -> None:
    st.subheader("自选股当日信号扫描")
    st.caption("对一组自选股应用同一策略，列出每只股票最新一个交易日应执行的买/卖/持有动作。")

    raw = st.text_input("自选股（逗号分隔）", value="AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, SPY, QQQ",
                        key="sig_tickers")
    tickers = parse_tickers(raw)
    if not tickers:
        st.info("请至少输入一个标的代码。")
        return

    strat_name = st.selectbox("交易策略", strategies.list_strategies(), key="sig_strat")
    strat = strategies.get_strategy(strat_name)
    st.caption(strat.description)
    params = _strategy_param_inputs(strat, "sig")

    if not st.button("🔔 扫描今日信号", type="primary", key="run_sig"):
        return

    data, failed = get_multi_data(tickers, cfg)
    if failed:
        st.warning(f"以下标的获取失败，已忽略：{', '.join(failed)}")
    if not data:
        st.error("❌ 没有可用的标的数据。")
        return

    table = signals.scan(data, strat_name, params=params, allow_short=cfg["allow_short"])

    changed = table[table["今日动作"].str.contains("🟢|🔴|🟡", regex=True)]
    if changed.empty:
        st.info("📭 今日没有标的触发新的买卖信号。")
    else:
        st.success(f"📬 今日有 {len(changed)} 只标的触发信号变动：")
        for _, r in changed.iterrows():
            st.markdown(f"- **{r['代码']}** · {r['今日动作']} · 最新价 ${r['最新价']}（{r['最新日期']}）")

    st.divider()
    st.markdown("**全部扫描结果**")
    st.dataframe(table, use_container_width=True, hide_index=True)
    csv = table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载信号表 (CSV)", csv, file_name="signals.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# 标签页 6：样本外验证
# ---------------------------------------------------------------------------
def _param_grid_inputs(strat, key_prefix: str) -> tuple[dict, int]:
    """生成参数搜索范围输入，返回 (param_grid, 组合数)。"""
    param_grid: dict[str, list] = {}
    for p in strat.params:
        c1, c2, c3 = st.columns([2, 2, 1])
        if p.is_int:
            lo = c1.number_input(f"{p.label} · 最小", int(p.min_value), int(p.max_value),
                                 int(p.min_value), key=f"{key_prefix}_{p.key}_lo")
            hi = c2.number_input(f"{p.label} · 最大", int(p.min_value), int(p.max_value),
                                 int(p.default), key=f"{key_prefix}_{p.key}_hi")
            step = c3.number_input("步长", 1, 50, max(1, int((hi - lo) / 5) or 1), key=f"{key_prefix}_{p.key}_st")
            values = list(range(int(lo), int(hi) + 1, int(step))) or [int(lo)]
            param_grid[p.key] = [int(v) for v in values]
        else:
            lo = c1.number_input(f"{p.label} · 最小", float(p.min_value), float(p.max_value),
                                 float(p.min_value), key=f"{key_prefix}_{p.key}_lo")
            hi = c2.number_input(f"{p.label} · 最大", float(p.min_value), float(p.max_value),
                                 float(p.default), key=f"{key_prefix}_{p.key}_hi")
            step = c3.number_input("步长", float(p.step), 10.0, float(p.step), key=f"{key_prefix}_{p.key}_st")
            values = list(np.round(np.arange(lo, hi + 1e-9, step), 4)) or [lo]
            param_grid[p.key] = [float(v) for v in values]
    n_combos = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 0
    return param_grid, n_combos


def tab_validation(cfg: dict) -> None:
    st.subheader("样本外验证（防过拟合）")
    st.caption("先在历史前段寻优，再用最优参数在「没见过」的后段数据上检验。样本外表现远差于样本内，往往意味着参数过拟合。")

    strat_name = st.selectbox("交易策略", strategies.list_strategies(), key="val_strat")
    strat = strategies.get_strategy(strat_name)
    if not strat.params:
        st.info("该策略没有可调参数，无需样本外验证。")
        return
    st.caption(strat.description)

    method = st.radio("验证方式", ["单次划分（训练/测试）", "滚动前向（Walk-Forward）"],
                      horizontal=True, key="val_method")
    sort_by = st.selectbox("寻优目标", optimize.SORT_METRICS, key="val_sort")

    st.markdown("**参数搜索范围**")
    param_grid, n_combos = _param_grid_inputs(strat, "val")

    if method == "单次划分（训练/测试）":
        train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.7, 0.05, key="val_tr1")
    else:
        c1, c2 = st.columns(2)
        n_splits = c1.slider("样本外窗口数", 2, 8, 4, 1, key="val_nsplits")
        train_ratio = c2.slider("首个测试段起点（数据占比）", 0.3, 0.7, 0.5, 0.05, key="val_tr2")
    st.caption(f"每次寻优评估约 {n_combos} 组参数组合")

    if not st.button("🧪 开始验证", type="primary", key="run_val"):
        return

    df = get_data(cfg)
    if df is None:
        return

    try:
        if method == "单次划分（训练/测试）":
            _render_holdout(df, strat_name, param_grid, sort_by, train_ratio, cfg)
        else:
            _render_walk_forward(df, strat_name, param_grid, sort_by, n_splits, train_ratio, cfg)
    except ValueError as e:
        st.error(f"❌ {e}")


def _render_holdout(df, strat_name, param_grid, sort_by, train_ratio, cfg) -> None:
    with st.spinner("正在训练集寻优并在测试集检验…"):
        res = validation.holdout_validate(
            df, strat_name, param_grid, sort_by=sort_by, train_ratio=train_ratio,
            allow_short=cfg["allow_short"], cost=cfg["cost"],
        )

    st.success(f"训练集最优参数：{res.best_params} ｜ 测试集起始：{res.split_date.strftime('%Y-%m-%d')}")

    metrics = ["累计收益率", "年化收益率", "夏普比率", "最大回撤"]
    comp = pd.DataFrame(
        {
            "样本内（训练）": [res.is_stats[m] for m in metrics],
            "样本外（测试）": [res.oos_stats[m] for m in metrics],
        },
        index=metrics,
    )
    cols = st.columns(4)
    for i, m in enumerate(metrics):
        delta = res.oos_stats[m] - res.is_stats[m]
        fmt = fmt_pct if m != "夏普比率" else fmt_num
        cols[i].metric(f"{m}(样本外)", fmt(res.oos_stats[m]), delta=f"{'+' if delta>=0 else ''}{fmt(delta)} vs 样本内")

    overfit = res.is_stats["夏普比率"] - res.oos_stats["夏普比率"]
    if overfit > 1.0:
        st.warning(f"⚠️ 样本外夏普比样本内低 {overfit:.2f}，过拟合风险较高，建议简化策略或扩大数据。")
    else:
        st.info(f"样本内外夏普差距 {overfit:.2f}，相对稳健。")

    disp = comp.copy()
    for c in disp.columns:
        disp[c] = [fmt_pct(v) if m != "夏普比率" else fmt_num(v) for m, v in zip(disp.index, disp[c])]
    st.dataframe(disp, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.is_equity.index, y=res.is_equity, name="样本内（训练）",
                             line=dict(color="#7f8c8d", width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=res.oos_equity.index, y=res.oos_equity, name="样本外（测试）",
                             line=dict(color="#e74c3c", width=2.2)))
    fig.add_vline(x=res.split_date, line=dict(color="#f1c40f", width=1, dash="dash"))
    fig.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08), title="样本内 vs 样本外净值")
    st.plotly_chart(fig, use_container_width=True)


def _render_walk_forward(df, strat_name, param_grid, sort_by, n_splits, train_ratio, cfg) -> None:
    with st.spinner("正在滚动寻优与样本外检验…"):
        res = validation.walk_forward(
            df, strat_name, param_grid, sort_by=sort_by, n_splits=n_splits,
            train_ratio=train_ratio, allow_short=cfg["allow_short"], cost=cfg["cost"],
        )

    s = res.oos_stats
    excess = s["累计收益率"] - s["基准收益率"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本外累计收益", fmt_pct(s["累计收益率"]), delta=f"超额 {fmt_pct(excess)}")
    c2.metric("样本外年化", fmt_pct(s["年化收益率"]))
    c3.metric("样本外夏普", fmt_num(s["夏普比率"]))
    c4.metric("样本外最大回撤", fmt_pct(s["最大回撤"]))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("连续样本外净值", "回撤"))
    fig.add_trace(go.Scatter(x=res.oos_equity.index, y=res.oos_equity, name="滚动样本外策略",
                             line=dict(color="#e74c3c", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.oos_benchmark.index, y=res.oos_benchmark, name="买入持有基准",
                             line=dict(color="#7f8c8d", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.drawdown.index, y=res.drawdown, name="回撤",
                             fill="tozeroy", line=dict(color="#3498db", width=1)), row=2, col=1)
    fig.update_layout(height=460, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**各滚动窗口明细**")
    disp = res.windows.copy()
    disp["样本外收益"] = disp["样本外收益"].map(fmt_pct)
    disp["最大回撤"] = disp["最大回撤"].map(fmt_pct)
    disp["夏普比率"] = disp["夏普比率"].map(fmt_num)
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 标签页 7：模拟交易（Paper Trading）
# ---------------------------------------------------------------------------
def tab_paper(cfg: dict) -> None:
    st.subheader("模拟交易账户（Paper Trading）")
    st.caption("本地虚拟账户、零资金风险。按策略信号在多头标的间等权调仓，长期跟踪模拟盘表现。仅做多 + 现金。")

    account = paper.load_account(PAPER_ACCOUNT_FILE)

    with st.expander("⚙️ 账户设置 / 重置", expanded=account is None):
        init_cap = st.number_input("初始资金 (USD)", value=100_000, step=10_000, min_value=1_000, key="paper_init")
        cc1, cc2 = st.columns(2)
        if cc1.button("🆕 新建 / 重置账户", key="paper_new"):
            account = paper.new_account(float(init_cap))
            paper.save_account(account, PAPER_ACCOUNT_FILE)
            st.success("已创建新的模拟账户。")
        if cc2.button("🗑️ 删除账户", key="paper_del") and PAPER_ACCOUNT_FILE.exists():
            PAPER_ACCOUNT_FILE.unlink()
            account = None
            st.warning("账户已删除。")

    if account is None:
        st.info("还没有模拟账户，请在上方「账户设置」中新建一个。")
        return

    raw = st.text_input("自选股（逗号分隔）", value="AAPL, MSFT, NVDA, GOOGL, AMZN, SPY", key="paper_tickers")
    tickers = parse_tickers(raw)
    strat_name = st.selectbox("交易策略", strategies.list_strategies(), key="paper_strat")
    strat = strategies.get_strategy(strat_name)
    st.caption(strat.description)
    params = _strategy_param_inputs(strat, "paper")
    max_pos = st.slider("最大持仓数量（0 = 不限制）", 0, 20, 0, 1, key="paper_maxpos")

    run = st.button("▶️ 按今日信号调仓", type="primary", key="paper_run")

    if run and tickers:
        data, failed = get_multi_data(tickers, cfg)
        if failed:
            st.warning(f"以下标的获取失败，已忽略：{', '.join(failed)}")
        if data:
            sig = signals.scan(data, strat_name, params=params, allow_short=False)
            targets = paper.targets_from_signals(sig, max_positions=max_pos)
            prices = {t: float(df["Close"].iloc[-1]) for t, df in data.items()}
            as_of = max(pd.Timestamp(df.index[-1]) for df in data.values()).strftime("%Y-%m-%d")
            trades = paper.rebalance(account, targets, prices, as_of=as_of,
                                     fee_bps=cfg["fee_bps"], slippage_bps=cfg["slippage_bps"])
            paper.save_account(account, PAPER_ACCOUNT_FILE)
            if trades:
                st.success(f"已按 {as_of} 信号调仓，产生 {len(trades)} 笔成交。")
            else:
                st.info(f"按 {as_of} 信号无需调仓（目标与当前持仓一致）。")

    # 估值用最近一次成交价 / 成本价兜底（避免每次刷新都联网）。
    prices = {}
    for t, p in account.positions.items():
        prices[t] = p["avg_cost"]
    for h in reversed(account.history):
        if h["标的"] not in prices or prices[h["标的"]] == 0:
            prices[h["标的"]] = h["成交价"]

    s = paper.summary(account, prices)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总权益", f"${s['总权益']:,.0f}", delta=f"{fmt_pct(s['累计收益率'])}")
    c2.metric("现金", f"${s['现金']:,.0f}")
    c3.metric("持仓市值", f"${s['持仓市值']:,.0f}")
    c4.metric("累计盈亏", f"${s['盈亏金额']:,.0f}")
    c5.metric("持仓数量", f"{int(s['持仓数量'])}")
    st.caption(f"账户创建于 {account.created_at} ｜ 估值按最近成交价，实际权益以实时行情为准")

    if account.equity_curve:
        eq_df = pd.DataFrame(account.equity_curve)
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        fig = go.Figure(go.Scatter(x=eq_df["date"], y=eq_df["equity"],
                                   line=dict(color="#e74c3c", width=2), name="权益"))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                          title="模拟账户权益曲线", yaxis_title="权益 (USD)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**当前持仓**")
    hold = paper.holdings_table(account, prices)
    if hold.empty:
        st.info("当前空仓。")
    else:
        disp = hold.copy()
        disp["盈亏%"] = disp["盈亏%"].map(fmt_pct)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("**成交流水**")
    if account.history:
        hist = pd.DataFrame(account.history)
        st.dataframe(hist.iloc[::-1], use_container_width=True, hide_index=True)
        csv = hist.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载成交流水 (CSV)", csv, file_name="paper_trades.csv", mime="text/csv")
    else:
        st.info("还没有成交记录。")


# ---------------------------------------------------------------------------
# 标签页 8：赚钱概率
# ---------------------------------------------------------------------------
def _applicability_card(strat) -> None:
    st.markdown(
        f"""
> **类别**：{strat.category}　|　**最适用**：{strat.best_market}
>
> **应避免**：{strat.avoid_market}
>
> {strat.applicability}
"""
    )


def tab_probability(cfg: dict) -> None:
    st.subheader("赚钱策略与赚钱概率")
    st.caption("查看每个策略的适用条件，并用历史数据测算其「赚钱概率」：随机进场持有一段时间为正的概率、单笔胜率、跑赢基准概率、多标的盈利占比。")

    with st.expander("📚 全部策略 · 适用条件速查", expanded=False):
        rows = []
        for name in strategies.list_strategies():
            s = strategies.get_strategy(name)
            rows.append({"策略": name, "类别": s.category, "最适用": s.best_market,
                         "应避免": s.avoid_market})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    mode = st.radio("分析范围", ["单标的（滚动持有期概率）", "多标的（盈利占比）"],
                    horizontal=True, key="prob_mode")
    strat_name = st.selectbox("交易策略", strategies.list_strategies(), key="prob_strat")
    strat = strategies.get_strategy(strat_name)
    _applicability_card(strat)
    params = _strategy_param_inputs(strat, "prob")

    if mode.startswith("单标的"):
        _prob_single(cfg, strat_name, params)
    else:
        _prob_basket(cfg, strat_name, params)


def _prob_single(cfg: dict, strat_name: str, params: dict) -> None:
    if not st.button("💰 测算赚钱概率", type="primary", key="run_prob_single"):
        return
    df = get_data(cfg)
    if df is None:
        return
    a = probability.analyze_single(df, strat_name, params=params,
                                   allow_short=cfg["allow_short"], cost=cfg["cost"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("单笔胜率", fmt_pct(a["win_rate"]))
    c2.metric("盈亏比", fmt_num(a["payoff"]))
    c3.metric("总收益率", fmt_pct(a["total_return"]), delta=f"基准 {fmt_pct(a['benchmark_return'])}")
    c4.metric("夏普比率", fmt_num(a["sharpe"]))
    st.caption(
        f"交易 {a['num_trades']} 笔 ｜ 平均盈利 {fmt_pct(a['avg_win'])} ／ 平均亏损 {fmt_pct(a['avg_loss'])} "
        f"｜ 凯利建议仓位 ≈ {fmt_pct(a['kelly'])}（仅参考，实盘务必打折）"
    )

    if a["horizons"].empty:
        st.info("数据不足以计算滚动持有期概率，请扩大日期范围。")
        return

    st.markdown("**随机进场 · 持有不同期限的赚钱概率**")
    h = a["horizons"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h["持有期"], y=h["赚钱概率"], name="赚钱概率",
                         marker_color="#e74c3c", text=[fmt_pct(v) for v in h["赚钱概率"]], textposition="outside"))
    fig.add_trace(go.Bar(x=h["持有期"], y=h["跑赢基准概率"], name="跑赢基准概率",
                         marker_color="#3498db", text=[fmt_pct(v) for v in h["跑赢基准概率"]], textposition="outside"))
    fig.add_hline(y=0.5, line=dict(color="#f1c40f", width=1, dash="dash"))
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                      barmode="group", yaxis_tickformat=".0%", yaxis_title="概率",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    disp = h.copy()
    disp["赚钱概率"] = disp["赚钱概率"].map(fmt_pct)
    disp["跑赢基准概率"] = disp["跑赢基准概率"].map(fmt_pct)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption("解读：「赚钱概率」= 历史上任意一天按该策略进场、持有该期限后账户为正的比例；持有越久通常越高，但不代表未来必然如此。")


def _prob_basket(cfg: dict, strat_name: str, params: dict) -> None:
    raw = st.text_input("一篮子标的（逗号分隔）",
                        value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, SPY, QQQ, TSLA, JPM",
                        key="prob_tickers")
    tickers = parse_tickers(raw)
    if not st.button("💰 测算盈利占比", type="primary", key="run_prob_basket"):
        return
    if not tickers:
        st.info("请至少输入一个标的。")
        return
    data, failed = get_multi_data(tickers, cfg)
    if failed:
        st.warning(f"以下标的获取失败，已忽略：{', '.join(failed)}")
    if not data:
        st.error("❌ 没有可用的标的数据。")
        return

    summ, table = probability.analyze_basket(data, strat_name, params=params,
                                             allow_short=cfg["allow_short"], cost=cfg["cost"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("盈利标的占比", fmt_pct(summ["盈利概率"]))
    c2.metric("跑赢基准占比", fmt_pct(summ["跑赢基准概率"]))
    c3.metric("平均策略收益", fmt_pct(summ["平均策略收益"]))
    c4.metric("平均超额收益", fmt_pct(summ["平均超额收益"]))
    st.caption(f"共测试 {int(summ['标的数'])} 只标的 ｜ 中位策略收益 {fmt_pct(summ['中位策略收益'])}")

    disp = table.copy()
    for c in ["策略收益", "买入持有", "超额收益", "最大回撤"]:
        disp[c] = disp[c].map(fmt_pct)
    disp["夏普比率"] = disp["夏普比率"].map(fmt_num)
    disp["是否盈利"] = disp["是否盈利"].map(lambda x: "✅" if x else "❌")
    disp["是否跑赢"] = disp["是否跑赢"].map(lambda x: "✅" if x else "❌")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 标签页 9：策略推荐
# ---------------------------------------------------------------------------
def tab_recommend(cfg: dict) -> None:
    st.subheader("智能策略推荐")
    st.caption("自动诊断标的当前是趋势市还是震荡市（含方向与波动水平），再结合策略适用条件与近一年实测表现，推荐最适配的策略。")

    if not st.button("🧭 诊断市场并推荐策略", type="primary", key="run_rec"):
        return

    df = get_data(cfg)
    if df is None:
        return
    if len(df) < 60:
        st.error("❌ 数据量过少，无法诊断市场状态，请扩大日期范围。")
        return

    reg, table = regime.recommend(df, allow_short=cfg["allow_short"], cost=cfg["cost"])

    # 市场状态卡片。
    color = {"趋势市": "#e74c3c", "震荡市": "#3498db", "过渡": "#f39c12"}.get(reg.trend_label, "#95a5a6")
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;background:{color}22;border:1px solid {color}55'>"
        f"<h3 style='margin:0;color:{color}'>当前市场状态：{reg.summary}</h3></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("趋势强度 ADX", fmt_num(reg.adx), help="≥25 趋势市，<18 震荡市")
    c2.metric("效率比 ER", fmt_num(reg.er), help="越接近 1 走势越笔直（趋势越强）")
    c3.metric("年化波动率", fmt_pct(reg.annual_vol))
    c4.metric("波动历史分位", fmt_pct(reg.vol_pct))

    # 诊断解读。
    if reg.trend_label == "趋势市":
        advice = "趋势行情，宜用**趋势跟踪 / 突破 / 动量**类策略顺势而为，避免逆势抄底摸顶。"
    elif reg.trend_label == "震荡市":
        advice = "震荡行情，宜用**均值回归**类策略高抛低吸，趋势突破类容易反复被套。"
    else:
        advice = "趋势尚不明朗（过渡期），建议降低仓位或等待信号明确，可优先看近一年表现稳健的策略。"
    if reg.vol_label == "高波动":
        advice += " 当前波动偏高，建议配合 ATR 跟踪止损并适当降低仓位。"
    st.info(advice)

    st.divider()
    st.markdown("**策略推荐排名**（契合度优先，近一年夏普为辅）")

    top = table.iloc[0]
    st.success(f"🏆 首选推荐：**{top['策略']}**（{top['类别']} · {top['契合度']}）"
               f" — 近一年收益 {fmt_pct(top['近一年收益'])}，夏普 {fmt_num(top['近一年夏普'])}")

    disp = table.copy()
    for c in ["近一年收益", "近一年最大回撤"]:
        disp[c] = disp[c].map(fmt_pct)
    disp["近一年夏普"] = disp["近一年夏普"].map(fmt_num)

    def _hl(row):
        if row["契合度"] == "高度契合":
            return ["background-color: #2ecc7122"] * len(row)
        if row["契合度"] == "不契合":
            return ["background-color: #e74c3c22"] * len(row)
        return [""] * len(row)

    st.dataframe(disp.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
    st.caption("绿色 = 与当前市场高度契合；红色 = 不契合（不建议在当前环境使用）。推荐基于历史，仅供参考。")


# ---------------------------------------------------------------------------
# 标签页 10：一键体检
# ---------------------------------------------------------------------------
def _score_gauge(score: float, grade: str) -> go.Figure:
    color = "#2ecc71" if score >= 72 else "#f39c12" if score >= 55 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " 分", "font": {"size": 36}},
        title={"text": f"综合评分 · {grade}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#e74c3c22"},
                {"range": [40, 55], "color": "#e67e2222"},
                {"range": [55, 72], "color": "#f39c1222"},
                {"range": [72, 100], "color": "#2ecc7122"},
            ],
        },
    ))
    fig.update_layout(height=260, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=10))
    return fig


def tab_report(cfg: dict) -> None:
    st.subheader("一键体检 · 全流程决策报告")
    st.caption("输入一只股票，自动跑完整条决策链：判市 → 推荐策略 → 自动寻优 → 样本外验证 → 赚钱概率 → 综合评分与可执行结论。")

    if not st.button("📋 开始一键体检", type="primary", key="run_report"):
        return

    df = get_data(cfg)
    if df is None:
        return
    if len(df) < 120:
        st.error("❌ 数据量过少（建议至少 1 年以上），无法完成完整体检。")
        return

    with st.spinner("正在判市、推荐、寻优、验证…（约数秒）"):
        rep = report_mod.run_full_report(df, ticker=cfg["ticker"], allow_short=cfg["allow_short"], cost=cfg["cost"])

    # 顶部：评分 + 结论。
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(_score_gauge(rep.score, rep.grade), use_container_width=True)
    with g2:
        st.markdown(f"### {cfg['ticker']} 体检结论")
        st.markdown(f"**市场状态**：{rep.regime.summary}　|　**推荐策略**：{rep.strategy}")
        if rep.best_params:
            st.markdown(f"**最优参数**：`{rep.best_params}`")
        st.success(rep.verdict)
        if rep.flags:
            for f in rep.flags:
                st.warning("⚠️ " + f)

    st.divider()

    # 关键指标。
    s = rep.final_result.stats
    st.markdown("**全样本表现（最优参数）**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("累计收益率", fmt_pct(s["累计收益率"]), delta=f"基准 {fmt_pct(s['基准收益率'])}")
    c2.metric("年化收益率", fmt_pct(s["年化收益率"]))
    c3.metric("夏普比率", fmt_num(s["夏普比率"]))
    c4.metric("最大回撤", fmt_pct(s["最大回撤"]))
    c5.metric("单笔胜率", fmt_pct(rep.prob["win_rate"]))

    # 样本外验证。
    if rep.oos_stats is not None and rep.is_stats is not None:
        st.markdown("**样本外验证（防过拟合）**")
        v1, v2, v3 = st.columns(3)
        v1.metric("样本内夏普", fmt_num(rep.is_stats["夏普比率"]))
        v2.metric("样本外夏普", fmt_num(rep.oos_stats["夏普比率"]),
                  delta=f"{'+' if -rep.overfit_gap>=0 else ''}{fmt_num(-rep.overfit_gap)}")
        v3.metric("样本外收益", fmt_pct(rep.oos_stats["累计收益率"]))

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**净值曲线**")
        st.plotly_chart(equity_chart(rep.final_result, rep.strategy), use_container_width=True)
    with right:
        st.markdown("**随机进场 · 持有期赚钱概率**")
        h = rep.prob["horizons"]
        if not h.empty:
            fig = go.Figure(go.Bar(x=h["持有期"], y=h["赚钱概率"], marker_color="#e74c3c",
                                   text=[fmt_pct(v) for v in h["赚钱概率"]], textposition="outside"))
            fig.add_hline(y=0.5, line=dict(color="#f1c40f", width=1, dash="dash"))
            fig.update_layout(height=460, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_tickformat=".0%", yaxis_title="赚钱概率")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 查看所有策略在该标的上的推荐排名"):
        disp = rep.recommend_table.copy()
        for c in ["近一年收益", "近一年最大回撤"]:
            disp[c] = disp[c].map(fmt_pct)
        disp["近一年夏普"] = disp["近一年夏普"].map(fmt_num)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.caption("本报告全部基于历史回测，是「过去的规律」，不构成投资建议。实盘前请务必用「💼 模拟交易」跑一段确认。")


# ---------------------------------------------------------------------------
# 渲染辅助
# ---------------------------------------------------------------------------
def _render_metrics(result: backtest.BacktestResult) -> None:
    s = result.stats
    st.markdown("**绩效指标**")
    c1, c2, c3, c4, c5 = st.columns(5)
    excess = s["累计收益率"] - s["基准收益率"]
    c1.metric("累计收益率", fmt_pct(s["累计收益率"]), delta=f"超额 {fmt_pct(excess)}")
    c2.metric("年化收益率", fmt_pct(s["年化收益率"]))
    c3.metric("夏普比率", fmt_num(s["夏普比率"]))
    c4.metric("最大回撤", fmt_pct(s["最大回撤"]))
    c5.metric("期末资金", f"${s['期末资金']:,.0f}")
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("年化波动率", fmt_pct(s["年化波动率"]))
    c7.metric("索提诺比率", fmt_num(s["索提诺比率"]))
    c8.metric("卡尔玛比率", fmt_num(s["卡尔玛比率"]))
    c9.metric("交易次数", f"{int(s['交易次数'])}")
    c10.metric("胜率", fmt_pct(s["胜率"]))


def _render_trades(result: backtest.BacktestResult) -> None:
    st.markdown("**交易明细**")
    trades = result.trades
    if trades.empty:
        st.info("回测区间内没有产生交易。")
        return
    display = trades.copy()
    display["开仓日期"] = pd.to_datetime(display["开仓日期"]).dt.strftime("%Y-%m-%d")
    display["平仓日期"] = pd.to_datetime(display["平仓日期"]).dt.strftime("%Y-%m-%d")
    display["收益率"] = display["收益率"].map(fmt_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)
    csv = trades.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载交易明细 (CSV)", csv, file_name="trades.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("📈 美股量化交易策略回测平台")
    st.caption("自用 · 体检 / 推荐 / 回测 / 寻优 / 对比 / 组合 / 信号 / 验证 / 模拟 / 赚钱概率 · 数据来源：Yahoo Finance")

    cfg = sidebar()
    tabs = st.tabs(
        ["📋 一键体检", "🧭 策略推荐", "🎯 单策略回测", "🔍 参数寻优", "📊 策略对比", "🧺 组合回测",
         "🔔 信号扫描", "🧪 样本外验证", "💼 模拟交易", "💰 赚钱概率"]
    )
    with tabs[0]:
        tab_report(cfg)
    with tabs[1]:
        tab_recommend(cfg)
    with tabs[2]:
        tab_single(cfg)
    with tabs[3]:
        tab_optimize(cfg)
    with tabs[4]:
        tab_compare(cfg)
    with tabs[5]:
        tab_portfolio(cfg)
    with tabs[6]:
        tab_signals(cfg)
    with tabs[7]:
        tab_validation(cfg)
    with tabs[8]:
        tab_paper(cfg)
    with tabs[9]:
        tab_probability(cfg)


if __name__ == "__main__":
    main()
