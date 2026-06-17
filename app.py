"""美股量化交易策略回测平台 — Streamlit 应用入口。

运行方式:
    streamlit run app.py

三大功能（顶部标签页）：
    1. 单策略回测：选定策略与参数，查看净值、绩效、交易明细。
    2. 参数寻优：对策略参数做网格搜索，自动找出最优组合。
    3. 策略对比：多个策略横向对比净值曲线与绩效。
"""

from __future__ import annotations

from datetime import date, timedelta
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
    options as options_mod,
    paper,
    portfolio,
    precursor,
    probability,
    regime,
    report as report_mod,
    screener,
    screen_strategies,
    signals,
    strategies,
    validation,
)
from quant.data import DataError, fetch_history, get_data_source_info
import tiger_theme as theme

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
    page_title="量化策略 · Tiger Style",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(theme.inject_css(), unsafe_allow_html=True)


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
def _render_brand_header() -> None:
    """老虎证券风格顶栏。"""
    icon_path = ROOT_DIR / "assets" / "icon.png"
    logo_html = ""
    if icon_path.exists():
        import base64

        b64 = base64.b64encode(icon_path.read_bytes()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{b64}" width="48" height="48" '
            f'style="border-radius:12px;flex-shrink:0" alt="logo"/>'
        )
    st.markdown(
        f'<div class="tiger-topbar">'
        f'<div class="brand">{logo_html}'
        f'<div><div class="brand-name">量化策略</div>'
        f'<div class="brand-sub">研究 · 选股 · 回测 · 期权</div></div></div>'
        f'<span class="brand-tag">PRO</span></div>',
        unsafe_allow_html=True,
    )


def sidebar() -> dict:
    st.sidebar.markdown("### 交易设置")

    st.sidebar.markdown("**标的**")
    ticker = st.sidebar.text_input(
        "代码", value="AAPL", label_visibility="collapsed",
        help="美股代码，如 AAPL、MSFT、NVDA",
        key="sidebar_ticker",
    ).strip().upper()

    col1, col2 = st.sidebar.columns(2)
    default_start = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
    start = col1.date_input("开始", value=default_start, max_value=date.today(), key="sidebar_start")
    end = col2.date_input("结束", value=date.today(), max_value=date.today(), key="sidebar_end")

    allow_short = st.sidebar.checkbox(
        "允许做空", value=False, help="开启后，离场信号将转为反向做空",
        key="sidebar_allow_short",
    )

    st.sidebar.markdown("**资金与成本**")
    capital = st.sidebar.number_input(
        "初始资金 (USD)", value=100_000, step=10_000, min_value=1_000, key="sidebar_capital",
    )
    fee_bps = st.sidebar.slider("手续费 (bp)", 0.0, 30.0, 5.0, 0.5, key="sidebar_fee_bps")
    slippage_bps = st.sidebar.slider("滑点 (bp)", 0.0, 30.0, 2.0, 0.5, key="sidebar_slippage_bps")

    st.sidebar.divider()
    src = get_data_source_info()
    st.sidebar.markdown("**行情数据源**")
    st.sidebar.caption(f"当前：**{src['label']}**")
    if src["provider"] == "yahoo":
        with st.sidebar.expander("切换到专业数据源", expanded=False):
            st.markdown(
                "在 `.streamlit/secrets.toml`（本地）或 Streamlit Cloud Secrets 中配置：\n\n"
                "```toml\n[data]\nprovider = \"polygon\"   # 或 alpaca\n"
                "polygon_api_key = \"你的Key\"\n```\n\n"
                "**Polygon.io**（推荐）：[polygon.io](https://polygon.io) 注册免费 Key\n\n"
                "**Alpaca**（免费）：[alpaca.markets](https://alpaca.markets) 开户后获取 API Key"
            )
    st.sidebar.caption("数据仅供参考，不构成投资建议")

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
    fig.update_layout(height=420, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
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
                                           line=dict(color=theme.UP, width=1.4)))
            lev_fig.update_layout(height=200, template="tiger", margin=dict(l=10, r=10, t=10, b=10),
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
                             line=dict(color=theme.ORANGE, width=2.2)), row=1, col=1)
    palette = theme.PALETTE + [theme.MUTED]
    for i, (t, eq) in enumerate(result.asset_equity.items()):
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=t,
                                 line=dict(width=1, color=palette[i % len(palette)], dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="回撤",
                             fill="tozeroy", line=dict(color=theme.BLUE, width=1)), row=2, col=1)
    fig.update_layout(height=480, template="tiger", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    return fig


def _weight_pie(weights: dict[str, float]) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(weights.keys()), values=list(weights.values()),
                           hole=0.45, textinfo="label+percent"))
    fig.update_layout(height=480, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
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
                             line=dict(color=theme.MUTED, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=res.oos_equity.index, y=res.oos_equity, name="样本外（测试）",
                             line=dict(color=theme.ORANGE, width=2.2)))
    fig.add_vline(x=res.split_date, line=dict(color=theme.GOLD, width=1, dash="dash"))
    fig.update_layout(height=420, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
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
                             line=dict(color=theme.ORANGE, width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.oos_benchmark.index, y=res.oos_benchmark, name="买入持有基准",
                             line=dict(color=theme.MUTED, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=res.drawdown.index, y=res.drawdown, name="回撤",
                             fill="tozeroy", line=dict(color=theme.BLUE, width=1)), row=2, col=1)
    fig.update_layout(height=460, template="tiger", margin=dict(l=10, r=10, t=40, b=10),
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
                                   line=dict(color=theme.ORANGE, width=2), name="权益"))
        fig.update_layout(height=300, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
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
                         marker_color=theme.ORANGE, text=[fmt_pct(v) for v in h["赚钱概率"]], textposition="outside"))
    fig.add_trace(go.Bar(x=h["持有期"], y=h["跑赢基准概率"], name="跑赢基准概率",
                         marker_color=theme.BLUE, text=[fmt_pct(v) for v in h["跑赢基准概率"]], textposition="outside"))
    fig.add_hline(y=0.5, line=dict(color=theme.GOLD, width=1, dash="dash"))
    fig.update_layout(height=400, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
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
# 标签页：策略选股（条件筛选 + 批量回测）
# ---------------------------------------------------------------------------
def _fmt_mcap(x: float) -> str:
    if pd.isna(x):
        return "-"
    b = float(x) / 1e9
    if b >= 1:
        return f"{b:,.1f}B"
    return f"{float(x)/1e6:,.0f}M"


def _fmt_dollar_m(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"${float(x)/1e6:,.1f}M"


DEFAULT_WATCHLIST = "SNDK, MU, WDC, NVDA, AMD, AVGO, SMCI, PLTR, COIN, TSLA, META, AAPL"


def _tab_screen_preset_backtest(cfg: dict) -> None:
    st.markdown("### 📚 命名选股策略库 · 近 3 年回测")
    st.caption("每套策略均有名称与选股依据，可一键回测近 3 年调仓表现（盈利周期占比、夏普、回撤等）。")

    preset_list = screen_strategies.list_presets()
    preset_names = {p.id: f"[{getattr(p, 'horizon', '中线')}] {p.name}" for p in preset_list}
    sel_id = st.selectbox(
        "选择命名策略",
        list(preset_names.keys()),
        format_func=lambda k: preset_names[k],
        key="scr_preset_id",
    )
    preset = screen_strategies.get_preset(sel_id)
    st.info(f"**策略依据：** {preset.rationale}")
    pc1, pc2, pc3 = st.columns(3)
    pc1.markdown(f"**股票池：** {screener.UNIVERSE_PRESETS.get(preset.pool, preset.pool)}")
    pc2.markdown(f"**交易策略：** {preset.trading_strategy}")
    pc3.markdown(f"**每 {preset.rebalance_days} 日选 {preset.top_picks} 只 · 后 {preset.forward_eval_days} 日评估**")

    # ---- 指定某年某月某日：单日交易计划 + 导出 ----
    with st.expander("📆 查看某一天的交易计划（方向 / 仓位 / 20日兑现 · 可导出）", expanded=False):
        dcol1, dcol2 = st.columns([2, 1])
        plan_date = dcol1.date_input(
            "选股日期", value=date.today() - timedelta(days=40),
            max_value=date.today(), key="scr_plan_date",
        )
        run_day = dcol2.button("查看该日计划", key="run_scr_day_plan")
        if run_day:
            with st.spinner(f"正在生成 {plan_date} 的交易计划…"):
                try:
                    sel = plan_date.isoformat()
                    fetch_start = (pd.Timestamp(sel) - pd.DateOffset(days=260)).strftime("%Y-%m-%d")
                    fetch_end = min(
                        (pd.Timestamp(sel) + pd.DateOffset(days=45)).strftime("%Y-%m-%d"),
                        date.today().isoformat(),
                    )
                    if preset.pool == "custom":
                        tks = preset.custom_tickers
                    elif preset.pool == "sp500":
                        tks = screener.fetch_sp500_tickers()[: preset.pool_size]
                    else:
                        tks = screener.fetch_sp500_tickers()[:50]
                        st.caption("提示：涨幅榜/活跃榜历史池用标普50成分作代理。")
                    day_data, day_failed = get_multi_data(tks, {**cfg, "start": fetch_start, "end": fetch_end})
                    if not day_data:
                        st.error("❌ 无可用数据。")
                    else:
                        day_plan = screen_strategies.trade_plan_at_date(
                            preset, day_data, sel,
                            capital=cfg["capital"], allow_short=cfg["allow_short"],
                            fee_bps=cfg["fee_bps"], slippage_bps=cfg["slippage_bps"],
                        )
                        if day_plan.empty:
                            st.info(f"{sel} 当日按该策略无入选标的（或数据不足）。")
                        else:
                            eff_date = day_plan["选股日期"].iloc[0]
                            if eff_date != sel:
                                st.caption(f"注：{sel} 非交易日，已对齐到最近交易日 {eff_date}。")
                            st.markdown(f"**{eff_date} · {preset.name} 交易计划**")
                            csv = day_plan.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "⬇️ 导出该日计划 (CSV)", data=csv,
                                file_name=f"交易计划_{preset.id}_{eff_date}.csv",
                                mime="text/csv", key="dl_scr_day_plan",
                            )
                            st.dataframe(day_plan, use_container_width=True, hide_index=True)
                            st.caption("『后N日收益%』=按方向买入持有口径；『策略后向收益%』=策略择时口径。NaN 表示该日后续不足 N 个交易日、尚未兑现。仅供研究。")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ 生成失败：{e}")

    # ---- 整段时间：批量每日交易计划 + 一次性 CSV 导出 ----
    with st.expander("📦 批量导出整段时间的每日交易计划（方向 / 仓位 / 20日兑现）", expanded=False):
        st.caption(
            f"按策略调仓周期（每 {preset.rebalance_days} 个交易日）逐日生成交易计划，"
            f"汇总为一张 CSV：含选股日期、理由、做多/做空、建议金额、后 {preset.forward_eval_days} 日盈亏与回撤。"
        )
        r1, r2, r3 = st.columns(3)
        bulk_start = r1.date_input(
            "起始日期", value=date.today() - timedelta(days=365),
            max_value=date.today(), key="scr_bulk_start",
        )
        bulk_end = r2.date_input(
            "结束日期", value=date.today() - timedelta(days=20),
            max_value=date.today(), key="scr_bulk_end",
        )
        run_bulk = r3.button("生成批量计划", key="run_scr_bulk_plan")
        if run_bulk:
            if bulk_start > bulk_end:
                st.error("❌ 起始日期不能晚于结束日期。")
            else:
                with st.spinner(f"正在生成 {bulk_start} ~ {bulk_end} 的每日交易计划…"):
                    try:
                        bs = bulk_start.isoformat()
                        be = bulk_end.isoformat()
                        fetch_start = (pd.Timestamp(bs) - pd.DateOffset(days=260)).strftime("%Y-%m-%d")
                        fetch_end = min(
                            (pd.Timestamp(be) + pd.DateOffset(days=45)).strftime("%Y-%m-%d"),
                            date.today().isoformat(),
                        )
                        if preset.pool == "custom":
                            tks = preset.custom_tickers
                        elif preset.pool == "sp500":
                            tks = screener.fetch_sp500_tickers()[: preset.pool_size]
                        else:
                            tks = screener.fetch_sp500_tickers()[:50]
                            st.caption("提示：涨幅榜/活跃榜历史池用标普50成分作代理。")
                        bulk_data, bulk_failed = get_multi_data(
                            tks, {**cfg, "start": fetch_start, "end": fetch_end},
                        )
                        if bulk_failed:
                            st.warning(f"部分标的拉取失败已忽略：{', '.join(bulk_failed[:8])}")
                        if not bulk_data:
                            st.error("❌ 无可用数据。")
                        else:
                            bulk_res = screen_strategies.daily_trade_plan(
                                preset, bulk_data, start=bs, end=be,
                                capital=cfg["capital"], allow_short=cfg["allow_short"],
                                fee_bps=cfg["fee_bps"], slippage_bps=cfg["slippage_bps"],
                            )
                            if "error" in bulk_res:
                                st.info(f"暂无计划：{bulk_res['error']}")
                            else:
                                plan_df = bulk_res["plan"]
                                s = bulk_res["summary"]
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("选股批次数", int(plan_df["选股日期"].nunique()))
                                m2.metric("累计盈亏", f"${s['累计盈亏USD']:,.0f}")
                                m3.metric("胜率", fmt_pct(s["胜率"]))
                                m4.metric("做多/做空", f"{int(s['做多笔数'])} / {int(s['做空笔数'])}")
                                csv_all = plan_df.to_csv(index=False).encode("utf-8-sig")
                                st.download_button(
                                    "⬇️ 导出整段计划 (CSV)", data=csv_all,
                                    file_name=f"交易计划_{preset.id}_{bs}_{be}.csv",
                                    mime="text/csv", key="dl_scr_bulk_plan",
                                )
                                st.markdown("**预览（最近 80 条）**")
                                st.dataframe(plan_df.tail(80).iloc[::-1], use_container_width=True, hide_index=True)
                                st.caption(
                                    f"共 {len(plan_df)} 条记录 · 评估窗口 {int(s['评估窗口(交易日)'])} 个交易日 · "
                                    "仅供研究，不构成投资建议。"
                                )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"❌ 批量生成失败：{e}")

    if not st.button("📈 回测此选股策略（近 3 年）", type="primary", key="run_scr_preset_bt"):
        return

    with st.spinner("正在拉取行情并滚动回测（可能需要 1～2 分钟）…"):
        try:
            end_d = cfg["end"]
            start_d = (pd.Timestamp(end_d) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
            if preset.pool == "custom":
                tickers = preset.custom_tickers
            elif preset.pool == "sp500":
                tickers = screener.fetch_sp500_tickers()[: preset.pool_size]
            else:
                tickers = screener.fetch_sp500_tickers()[:50]
                st.caption("提示：涨幅榜/活跃榜历史池用标普50成分作回测代理。")
            data, failed = get_multi_data(tickers, {**cfg, "start": start_d, "end": end_d})
            if failed:
                st.warning(f"部分标的拉取失败已忽略：{', '.join(failed[:8])}")
            if not data:
                st.error("❌ 无可用数据。")
                return
            bt_res = screen_strategies.backtest_screen_preset(
                preset, data, max_years=3.0,
                allow_short=cfg["allow_short"],
                initial_capital=cfg["capital"],
                fee_bps=cfg["fee_bps"],
                slippage_bps=cfg["slippage_bps"],
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ 回测失败：{e}")
            return

    if "error" in bt_res:
        st.error(f"❌ {bt_res['error']}")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("累计收益", fmt_pct(bt_res["累计收益率"]))
    m2.metric("年化收益", fmt_pct(bt_res["年化收益率"]))
    m3.metric("夏普比率", fmt_num(bt_res["夏普比率"]))
    m4.metric("最大回撤", fmt_pct(bt_res["最大回撤"]))
    m5.metric("盈利周期占比", fmt_pct(bt_res["盈利周期占比"]))
    st.caption(
        f"调仓 {bt_res['调仓次数']} 次 ｜ 期末权益 ${bt_res['期末权益']:,.0f} ｜ "
        f"策略：{preset.name} + {preset.trading_strategy}"
    )

    eq = bt_res["权益曲线"]
    if not eq.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq["日期"], y=eq["权益"], name="策略权益", line=dict(color=theme.ORANGE, width=2)))
        fig_eq.update_layout(height=360, template="tiger", title=f"{preset.name} · 近3年权益曲线",
                             margin=dict(l=10, r=10, t=40, b=10), yaxis_title="权益 (USD)")
        st.plotly_chart(fig_eq, use_container_width=True)

    detail = bt_res["调仓明细"].tail(20).copy()
    detail["本期收益"] = detail["本期收益"].map(fmt_pct)
    detail["累计权益"] = detail["累计权益"].map(lambda x: f"${x:,.0f}")
    st.markdown("**最近调仓明细**")
    st.dataframe(detail.iloc[::-1], use_container_width=True, hide_index=True)

    picks_detail = bt_res.get("选股明细")
    if picks_detail is not None and not picks_detail.empty:
        st.markdown("**每日选股明细（含理由 · 前后收益/回撤 · 策略后向回测）**")
        pdisp = picks_detail.copy()
        for c in pdisp.columns:
            if "收益" in c or "回撤" in c:
                pdisp[c] = pd.to_numeric(pdisp[c], errors="coerce").map(
                    lambda x: fmt_pct(x) if pd.notna(x) else "-"
                )
            elif c == "涨幅%":
                pdisp[c] = pd.to_numeric(pdisp[c], errors="coerce").map(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
                )
            elif c == "入选价":
                pdisp[c] = pd.to_numeric(pdisp[c], errors="coerce").map(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
                )
        show_p = [c for c in picks_detail.columns if c not in ("_hits",)]
        prefer = ["选股日期", "代码", "选股理由", "涨幅%", "入选价"]
        show_p = [c for c in prefer if c in show_p] + [c for c in show_p if c not in prefer]
        st.dataframe(pdisp[show_p].tail(50).iloc[::-1], use_container_width=True, hide_index=True)

    # ---- 每日短线交易计划：方向 / 仓位 / 金额 / 20日盈亏 ----
    st.divider()
    st.markdown(f"**📋 每日交易计划（方向 · 仓位 · 选股后 {preset.forward_eval_days} 日盈亏）**")
    st.caption(
        "每个选股日给出：做多/做空方向（按策略信号）、建议仓位与金额（等权分配本金）、"
        f"选股理由，以及选股后 {preset.forward_eval_days} 个交易日的实际盈亏与最大回撤。做空需在侧边栏开启『允许做空』。"
    )
    try:
        tp = screen_strategies.daily_trade_plan(
            preset, data,
            capital=cfg["capital"],
            allow_short=cfg["allow_short"],
            fee_bps=cfg["fee_bps"],
            slippage_bps=cfg["slippage_bps"],
        )
    except Exception as e:  # noqa: BLE001
        st.warning(f"交易计划生成失败：{e}")
        tp = {"error": str(e)}

    if "error" in tp:
        st.info(f"暂无交易计划：{tp['error']}")
    else:
        s = tp["summary"]
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("累计盈亏", f"${s['累计盈亏USD']:,.0f}")
        t2.metric("胜率", fmt_pct(s["胜率"]))
        t3.metric("平均单笔收益", f"{s['平均单笔收益%']:.2f}%")
        t4.metric("做多/做空", f"{int(s['做多笔数'])} / {int(s['做空笔数'])}")
        plan = tp["plan"].copy()
        if "盈亏金额USD" in plan.columns:
            plan["盈亏金额USD"] = pd.to_numeric(plan["盈亏金额USD"], errors="coerce").map(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "-"
            )
        if "建议金额USD" in plan.columns:
            plan["建议金额USD"] = pd.to_numeric(plan["建议金额USD"], errors="coerce").map(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "-"
            )
        st.dataframe(plan.tail(80).iloc[::-1], use_container_width=True, hide_index=True)
        st.caption("说明：『后N日收益%』为按方向持有的买入持有口径；『策略后向收益%』为策略自行择时（含中途离场/反手）的口径。仅供研究，不构成投资建议。")


def _tab_historical_daily_screen(cfg: dict) -> None:
    st.markdown("### 📅 历史每日选股回测")
    st.caption(
        "按交易日回放选股：每个选股日列出入选标的、**选股理由**，并计算"
        "**选股前/后**持有期收益与最大回撤；可选对每只入选股做**策略后向回测**（从选股日次日按策略交易）。"
    )

    c1, c2, c3, c4 = st.columns(4)
    pool_options = {**screener.UNIVERSE_PRESETS, "sp500": "标普500成分", "custom": "自定义列表"}
    pool_key = c1.selectbox(
        "股票池", list(pool_options.keys()),
        format_func=lambda k: pool_options[k], key="hds_pool",
    )
    pool_size = c2.number_input("池规模", 10, 150, 40, 10, key="hds_size")
    rebalance = c3.number_input("每 N 日选股", 1, 30, 5, 1, key="hds_rebal")
    top_picks = c4.number_input("每次选几只", 1, 20, 5, 1, key="hds_top")

    custom_raw = ""
    if pool_key == "custom":
        custom_raw = st.text_input("自定义代码", value=DEFAULT_WATCHLIST, key="hds_custom")

    d1, d2, d3, d4 = st.columns(4)
    back_days = d1.selectbox("前向统计(日)", [5, 10, 20, 60], index=2, key="hds_back")
    fwd_days = d2.selectbox("后向统计(日)", [5, 10, 20, 60], index=2, key="hds_fwd")
    lookback = d3.selectbox("涨幅周期", [1, 5, 10, 20, 60], index=3, key="hds_lb")
    gain_range = d4.slider("涨幅区间%", -50.0, 200.0, (-10.0, 100.0), 1.0, key="hds_gain")

    f1, f2 = st.columns(2)
    min_dvol = f1.number_input("成交额下限(M USD)", 0.0, 500.0, 10.0, 5.0, key="hds_dvol")
    strat_name = f2.selectbox("策略后向回测", strategies.list_strategies(), key="hds_strat")
    strat = strategies.get_strategy(strat_name)
    params = _strategy_param_inputs(strat, "hds")

    bt_start = st.date_input(
        "回测起始日", value=(pd.Timestamp(cfg["end"]) - pd.DateOffset(years=2)).date(),
        key="hds_start",
    )
    bt_end = st.date_input("回测结束日", value=pd.Timestamp(cfg["end"]).date(), key="hds_end")

    if not st.button("📅 开始历史每日选股回测", type="primary", key="run_hds"):
        return

    filters = screener.ScreenFilters(
        min_gain_pct=gain_range[0], max_gain_pct=gain_range[1],
        min_dollar_vol_m=min_dvol, lookback_days=int(lookback),
    )
    start_s = str(bt_start)
    end_s = str(bt_end)
    fetch_start = (pd.Timestamp(start_s) - pd.DateOffset(days=400)).strftime("%Y-%m-%d")

    with st.spinner("拉取行情并按日回放选股（可能需要 1～3 分钟）…"):
        try:
            if pool_key == "custom":
                tickers = parse_tickers(custom_raw)
            elif pool_key == "sp500":
                tickers = screener.fetch_sp500_tickers()[: int(pool_size)]
            else:
                st.info("历史回放建议使用 标普500 或 自选 池；『涨/跌幅榜、活跃榜』只有实时名单、无历史成分，已自动改用标普500前 N 只作代理。注：这只影响『选股名单来源』，行情价格仍走你配置的数据源（如 Polygon）。")
                tickers = screener.fetch_sp500_tickers()[: int(pool_size)]
            if not tickers:
                st.error("❌ 股票池为空。")
                return
            data, failed = get_multi_data(tickers, {**cfg, "start": fetch_start, "end": end_s})
            if failed:
                st.warning(f"部分标的拉取失败：{', '.join(failed[:10])}")
            if not data:
                st.error("❌ 无可用行情。")
                return
            hres = screener.run_historical_daily_screen(
                data, filters,
                start=start_s, end=end_s,
                rebalance_days=int(rebalance),
                top_picks=int(top_picks),
                forward_days=int(fwd_days),
                backward_days=int(back_days),
                strategy_name=strat_name,
                params=params,
                allow_short=cfg["allow_short"],
                fee_bps=cfg["fee_bps"],
                slippage_bps=cfg["slippage_bps"],
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ {e}")
            return

    if hres.get("error"):
        st.error(f"❌ {hres['error']}")
        return

    daily = hres.get("daily_picks", pd.DataFrame())
    by_date = hres.get("by_date", pd.DataFrame())
    summary = hres.get("summary", {})
    if daily.empty:
        st.warning("回测期内没有产生有效选股批次，请放宽条件或扩大日期范围。")
        return

    st.success(f"共 **{int(summary.get('选股批次数', 0))}** 个选股日、**{int(summary.get('入选总人次', 0))}** 条入选记录")
    m1, m2, m3, m4 = st.columns(4)
    fwd_col = f"后{fwd_days}日收益"
    m1.metric(f"平均后{fwd_days}日收益", fmt_pct(summary.get("平均后向收益", 0)))
    m2.metric("后向盈利占比", fmt_pct(summary.get("后向盈利占比", 0)))
    if "平均策略后向收益" in summary:
        m3.metric("平均策略后向收益", fmt_pct(summary["平均策略后向收益"]))
        m4.metric("策略后向盈利占比", fmt_pct(summary.get("策略后向盈利占比", 0)))

    st.markdown("**按选股日汇总**")
    bd = by_date.copy()
    for c in ["平均后向收益", "后向盈利占比", "平均策略后向收益"]:
        if c in bd.columns:
            bd[c] = pd.to_numeric(bd[c], errors="coerce").map(
                lambda x: fmt_pct(x) if pd.notna(x) else "-"
            )
    st.dataframe(bd.iloc[::-1], use_container_width=True, hide_index=True)

    st.markdown("**选股明细（可筛选某日查看）**")
    dates = sorted(daily["选股日期"].unique(), reverse=True)
    sel_date = st.selectbox("查看选股日", dates, key="hds_sel_date")
    day_df = daily[daily["选股日期"] == sel_date].copy()
    disp = day_df.copy()
    back_col = f"前{back_days}日"
    for c in disp.columns:
        if "收益" in c or "回撤" in c:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                lambda x: fmt_pct(x) if pd.notna(x) else "-"
            )
        elif c == "涨幅%":
            disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
            )
        elif c == "入选价":
            disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
            )
    show_cols = [c for c in [
        "选股日期", "代码", "选股理由", "涨幅%", "入选价",
        f"{back_col}收益", f"{back_col}最大回撤",
        fwd_col, f"后{fwd_days}日最大回撤",
        "策略后向收益", "策略后向最大回撤",
    ] if c in disp.columns]
    st.dataframe(disp[show_cols], use_container_width=True, hide_index=True)

    if len(day_df) >= 1 and fwd_col in day_df.columns:
        fig = go.Figure()
        vals = pd.to_numeric(day_df[fwd_col], errors="coerce")
        fig.add_trace(go.Bar(
            x=day_df["代码"], y=vals,
            marker_color=[theme.UP if (pd.notna(v) and v >= 0) else theme.DOWN for v in vals],
            text=[fmt_pct(v) if pd.notna(v) else "-" for v in vals],
            textposition="outside",
        ))
        fig.update_layout(
            height=320, template="tiger",
            title=f"{sel_date} 入选 · 后{fwd_days}日收益",
            yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    csv = daily.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载全部每日选股回测 (CSV)", csv, file_name="daily_screen_backtest.csv", mime="text/csv")
    st.caption(
        "说明：「前 N 日」为选股日前的走势背景；「后 N 日」为选股日收盘买入的持有表现；"
        "「策略后向」为选股日次日按所选策略交易的回测结果。历史榜单池无法精确还原，建议用 sp500/custom。"
    )


def tab_screener(cfg: dict) -> None:
    st.subheader("策略选股 · 条件筛选 + 批量回测")
    st.caption(
        "所有选股结果均绑定明确的 **选股日期（某年某月某日）**，并附选股理由；"
        "可对每个选股日做前后收益/回撤与策略回测验证。"
    )

    with st.expander("📜 历史选股记录", expanded=False):
        hist = screener.load_screen_history(ROOT_DIR / "screen_history.csv")
        if hist.empty:
            st.caption("暂无记录。双击「每日选股_运行一次.command」后会写入 screen_history.csv。")
        else:
            summ = screener.summarize_screen_history(hist)
            disp_h = summ.copy()
            disp_h["平均策略收益"] = disp_h["平均策略收益"].map(fmt_pct)
            disp_h["盈利占比"] = disp_h["盈利占比"].map(fmt_pct)
            st.dataframe(disp_h, use_container_width=True, hide_index=True)
            show_cols = [c for c in ["选股日期", "选股时间", "代码", "名称", "选股理由", "涨幅%", "行业", "策略累计收益", "当前信号"]
                         if c in hist.columns]
            st.caption(f"最近明细（共 {len(hist)} 条）")
            recent = hist[show_cols].tail(30).copy()
            if "策略累计收益" in recent.columns:
                recent["策略累计收益"] = pd.to_numeric(recent["策略累计收益"], errors="coerce").map(fmt_pct)
            st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)

    st.divider()
    _tab_screen_preset_backtest(cfg)

    st.divider()
    _tab_historical_daily_screen(cfg)

    st.divider()
    st.markdown("### 🔎 自定义条件选股（指定日期）")
    sel_date = st.date_input(
        "选股日期（确定入选名单的交易日）",
        value=min(pd.Timestamp(cfg["end"]).date(), date.today()),
        max_value=date.today(),
        key="scr_sel_date",
        help="行情与筛选指标均截至该日，不含之后数据。",
    )
    sel_str = str(sel_date)
    fetch_start = (pd.Timestamp(sel_str) - pd.DateOffset(days=400)).strftime("%Y-%m-%d")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    pool_options = {**screener.UNIVERSE_PRESETS, "sp500": "标普500成分", "custom": "自定义列表"}
    pool_key = c1.selectbox(
        "股票池来源",
        list(pool_options.keys()),
        format_func=lambda k: pool_options[k],
        key="scr_pool",
    )
    pool_size = c2.number_input("初选数量", min_value=10, max_value=250, value=50, step=10, key="scr_size")
    max_bt = c3.number_input("最多回测", min_value=5, max_value=50, value=20, step=5, key="scr_max_bt",
                               help="筛选后按涨幅排序，取前 N 只做策略回测")

    custom_raw = ""
    if pool_key == "custom":
        custom_raw = st.text_input(
            "自定义代码（逗号分隔）",
            value=DEFAULT_WATCHLIST,
            key="scr_custom",
        )

    st.markdown("**筛选条件**")
    f1, f2, f3, f4 = st.columns(4)
    lookback = f1.selectbox("涨幅统计周期", [1, 5, 10, 20, 60], index=3,
                            format_func=lambda x: f"近 {x} 日", key="scr_lb")
    gain_range = f2.slider("涨幅区间 (%)", -50.0, 200.0, (-10.0, 100.0), 1.0, key="scr_gain")
    min_dollar_m = f3.number_input("成交额下限 (百万USD)", min_value=0.0, value=10.0, step=5.0, key="scr_dvol",
                                   help="近均日成交额 = 收盘价 × 成交量 的均值")
    turnover_range = f4.slider("换手率区间 (%)", 0.0, 50.0, (0.0, 20.0), 0.5, key="scr_to")

    f5, f6 = st.columns(2)
    mcap_range = f5.slider("市值区间 (十亿美元)", 0.0, 3000.0, (0.0, 500.0), 1.0, key="scr_mcap")
    strat_name = f6.selectbox("回测策略", strategies.list_strategies(), key="scr_strat")

    sector_labels = screener.SECTORS  # 英文 → 中文
    selected_sectors_en = st.multiselect(
        "行业筛选（留空 = 不限）",
        options=list(sector_labels.keys()),
        format_func=lambda k: f"{sector_labels[k]}（{k}）",
        key="scr_sectors",
        help="自定义/标普500池会逐只补拉行业数据，速度略慢；当日榜单池自带行业。",
    )

    strat = strategies.get_strategy(strat_name)
    st.caption(strat.description)
    params = _strategy_param_inputs(strat, "scr")

    if not st.button("🔎 开始选股并回测", type="primary", key="run_scr"):
        st.caption("也可直接使用上方「命名策略库」或「历史每日选股回测」。")
        return

    st.info(f"📅 本次选股日期：**{sel_str}**（指标与筛选均截至该日收盘）")

    filters = screener.ScreenFilters(
        min_gain_pct=gain_range[0],
        max_gain_pct=gain_range[1],
        min_dollar_vol_m=min_dollar_m,
        min_turnover_pct=turnover_range[0],
        max_turnover_pct=turnover_range[1],
        min_mcap_b=mcap_range[0],
        max_mcap_b=mcap_range[1],
        lookback_days=int(lookback),
        sectors=selected_sectors_en or None,
    )
    need_sector = bool(selected_sectors_en)

    with st.spinner("正在拉取股票池并计算行情指标…"):
        try:
            if pool_key == "custom":
                tickers = parse_tickers(custom_raw)
                if not tickers:
                    st.error("❌ 请至少输入一个股票代码。")
                    return
                snapshot = screener.build_snapshot_from_history(
                    tickers, fetch_start, sel_str, lookback_days=filters.lookback_days,
                    with_sector=need_sector,
                )
            elif pool_key == "sp500":
                tickers = screener.fetch_sp500_tickers()[: int(pool_size)]
                snapshot = screener.build_snapshot_from_history(
                    tickers, fetch_start, sel_str, lookback_days=filters.lookback_days,
                    with_sector=need_sector,
                )
            else:
                if sel_str < date.today().isoformat():
                    st.warning(
                        f"『{screener.UNIVERSE_PRESETS.get(pool_key, pool_key)}』只有实时名单、没有 {sel_str} 的历史成分，"
                        f"已自动改用标普500前 {int(pool_size)} 只在该日的行情重算指标。"
                        f"（仅影响选股名单来源；行情价格仍用你配置的数据源 **{get_data_source_info()['label']}**）"
                    )
                    tickers = screener.fetch_sp500_tickers()[: int(pool_size)]
                    snapshot = screener.build_snapshot_from_history(
                        tickers, fetch_start, sel_str,
                        lookback_days=filters.lookback_days,
                        with_sector=need_sector,
                    )
                else:
                    snapshot = screener.fetch_yahoo_screen(pool_key, count=int(pool_size))
                    if snapshot.empty:
                        st.error("❌ 未能从 Yahoo 获取选股数据，请稍后重试或换其他股票池。")
                        return
                    if filters.lookback_days > 1:
                        hist = screener.build_snapshot_from_history(
                            snapshot["代码"].tolist(), fetch_start, sel_str,
                            lookback_days=filters.lookback_days,
                        )
                        if not hist.empty:
                            sector_map = dict(zip(snapshot["代码"], snapshot.get("_行业EN", "")))
                            name_map = dict(zip(snapshot["代码"], snapshot.get("名称", "")))
                            hist["_行业EN"] = hist["代码"].map(sector_map).fillna("")
                            hist["行业"] = hist["_行业EN"].map(screener.sector_cn)
                            hist["名称"] = hist["代码"].map(name_map).fillna(hist["名称"])
                            snapshot = hist
        except DataError as e:
            st.error(f"❌ {e}")
            return

    if snapshot.empty:
        st.error("❌ 股票池为空，请检查网络或更换来源。")
        return

    filtered = screener.stamp_selection_date(screener.apply_filters(snapshot, filters), sel_str)
    st.info(f"📅 选股日 **{sel_str}** ｜ 初选 {len(snapshot)} 只 → 筛选后 **{len(filtered)}** 只符合条件")

    if not filtered.empty and "行业" in filtered.columns:
        sector_counts = filtered["行业"].replace("", "未知").value_counts()
        if len(sector_counts) > 0:
            fig_pie = go.Figure(go.Pie(
                labels=sector_counts.index.tolist(),
                values=sector_counts.values.tolist(),
                hole=0.35,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                height=320, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
                title="筛选结果 · 行业分布",
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    if filtered.empty:
        st.warning("没有标的满足当前筛选条件，请放宽涨幅、成交额、市值或行业条件后重试。")
        disp_pre = snapshot.drop(columns=["_行业EN"], errors="ignore").copy()
        disp_pre["涨幅%"] = disp_pre["涨幅%"].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "-")
        disp_pre["成交额USD"] = disp_pre["成交额USD"].map(_fmt_dollar_m)
        disp_pre["换手率%"] = disp_pre["换手率%"].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "-")
        disp_pre["市值USD"] = disp_pre["市值USD"].map(_fmt_mcap)
        st.markdown("**初选池概览（未通过筛选）**")
        st.dataframe(disp_pre, use_container_width=True, hide_index=True)
        return

    targets = filtered["代码"].head(int(max_bt)).tolist()
    with st.spinner(f"正在对 {len(targets)} 只标的运行「{strat_name}」回测…"):
        bt = screener.backtest_universe(
            targets,
            fetch_start,
            sel_str,
            strat_name,
            params=params,
            allow_short=cfg["allow_short"],
            initial_capital=cfg["capital"],
            fee_bps=cfg["fee_bps"],
            slippage_bps=cfg["slippage_bps"],
        )

    if bt.empty:
        st.error("❌ 回测未产生有效结果，请扩大日期范围或更换策略。")
        return

    merged = screener.add_rationale_to_merged(
        screener.merge_snapshot_backtest(filtered, bt, selection_date=sel_str),
        filters,
        sel_str,
    )
    summ = screener.summarize_backtest(bt)

    st.markdown("**组合汇总（等权视角）**")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("回测标的数", f"{int(summ.get('入选数量', 0))}")
    s2.metric("平均累计收益", fmt_pct(summ.get("平均累计收益", 0)))
    s3.metric("盈利占比", fmt_pct(summ.get("盈利标的占比", 0)))
    s4.metric("平均夏普", fmt_num(summ.get("平均夏普", 0)))
    s5.metric("平均最大回撤", fmt_pct(summ.get("平均最大回撤", 0)))
    st.caption(
        f"策略：{strat_name} ｜ 选股日：{sel_str} ｜ 回测区间：{fetch_start} ~ {sel_str}"
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged["代码"],
        y=merged["策略累计收益"],
        name="策略累计收益",
        marker_color=[theme.UP if v >= 0 else theme.DOWN for v in merged["策略累计收益"]],
        text=[fmt_pct(v) for v in merged["策略累计收益"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=360, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
        yaxis_tickformat=".0%", title="各标的策略累计收益",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**选股 + 回测明细（选股日 {sel_str}）**")
    disp = merged.copy()
    disp["涨幅%"] = disp["涨幅%"].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "-")
    disp["成交额USD"] = disp["成交额USD"].map(_fmt_dollar_m)
    disp["换手率%"] = disp["换手率%"].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "-")
    disp["市值USD"] = disp["市值USD"].map(_fmt_mcap)
    for col in ["策略累计收益", "策略年化收益", "基准收益", "超额收益", "最大回撤", "胜率"]:
        if col in disp.columns:
            disp[col] = disp[col].map(fmt_pct)
    if "夏普比率" in disp.columns:
        disp["夏普比率"] = disp["夏普比率"].map(fmt_num)
    if "最新价" in disp.columns:
        disp["最新价"] = disp["最新价"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    csv = merged.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 下载选股回测结果 (CSV)", csv, file_name="screener_backtest.csv", mime="text/csv")
    st.caption("⚠️ 筛选基于历史与当日行情统计；回测收益不代表未来，小样本更易过拟合，请结合样本外验证使用。")


# ---------------------------------------------------------------------------
# 标签页：期权策略损益计算器
# ---------------------------------------------------------------------------
def _options_payoff_chart(res, spot: float, currency: str = "$") -> go.Figure:
    p = res.prices
    pay = res.payoff
    pos = np.where(pay >= 0, pay, np.nan)
    neg = np.where(pay < 0, pay, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=pos, name="盈利区", line=dict(color=theme.UP, width=2),
                             fill="tozeroy", fillcolor="rgba(0,192,135,0.15)"))
    fig.add_trace(go.Scatter(x=p, y=neg, name="亏损区", line=dict(color=theme.DOWN, width=2),
                             fill="tozeroy", fillcolor="rgba(255,69,69,0.15)"))
    fig.add_hline(y=0, line=dict(color=theme.TEXT_TERTIARY, width=1))
    fig.add_vline(x=spot, line=dict(color=theme.GOLD, width=1, dash="dash"),
                  annotation_text=f"现价 {currency}{spot:,.2f}", annotation_position="top")
    for be in res.breakevens:
        fig.add_vline(x=be, line=dict(color=theme.BLUE, width=1, dash="dot"))
    fig.update_layout(height=440, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="到期股价", yaxis_title="到期盈亏 (USD)",
                      legend=dict(orientation="h", y=1.1))
    return fig


def tab_options(cfg: dict) -> None:
    st.subheader("期权策略损益计算器")
    mode = st.radio("模式", ["单策略损益", "多策略对比"], horizontal=True, key="options_mode")
    if mode == "多策略对比":
        _tab_options_compare(cfg)
        return

    st.caption(
        "输入行权价与权利金（按券商实际报价），画出到期盈亏图，给出最大盈利/亏损与盈亏平衡点。"
        "仅计算到期损益结构，不含定价模型。每张合约 = 100 股。"
    )

    strat_name = st.selectbox("期权策略", options_mod.list_strategies(), key="options_strat")
    info = options_mod.STRATEGY_INFO[strat_name]
    st.info(f"**观点**：{info['view']}　|　**风险**：{info['risk']}　|　**收益**：{info['reward']}\n\n{info['desc']}")

    c0, c1 = st.columns(2)
    spot = c0.number_input("标的现价 (USD)", min_value=0.01, value=float(cfg.get("opt_spot", 100.0)),
                           step=1.0, key="options_spot_in", help="可手动填，或点下方按钮自动拉取当前标的最新价")
    qty = c1.number_input("合约张数", min_value=1, value=1, step=1, key="options_qty")

    if c0.button(f"↻ 拉取 {cfg['ticker']} 最新价", key="options_fetch"):
        df = get_data(cfg)
        if df is not None and len(df):
            st.session_state["options_spot_fetched"] = float(df["Close"].iloc[-1])
            st.rerun()
    if "options_spot_fetched" in st.session_state:
        st.caption(f"已拉取 {cfg['ticker']} 最新价：${st.session_state['options_spot_fetched']:,.2f}（请填入上方现价框）")

    st.markdown("**各腿参数**")
    legs: list = []

    if strat_name == "买入认购 (Long Call)":
        a, b = st.columns(2)
        k = a.number_input("行权价 K", value=round(spot * 1.05, 2), step=1.0, key="oc_k")
        prem = b.number_input("认购权利金/股", min_value=0.0, value=round(spot * 0.05, 2), step=0.1, key="oc_p")
        legs = options_mod.long_call(k, prem, qty)
    elif strat_name == "买入认沽 (Long Put)":
        a, b = st.columns(2)
        k = a.number_input("行权价 K", value=round(spot * 0.95, 2), step=1.0, key="op_k")
        prem = b.number_input("认沽权利金/股", min_value=0.0, value=round(spot * 0.05, 2), step=0.1, key="op_p")
        legs = options_mod.long_put(k, prem, qty)
    elif strat_name == "备兑开仓 (Covered Call)":
        a, b = st.columns(2)
        k = a.number_input("认购行权价 K", value=round(spot * 1.1, 2), step=1.0, key="cc_k")
        prem = b.number_input("收取的认购权利金/股", min_value=0.0, value=round(spot * 0.04, 2), step=0.1, key="cc_p")
        legs = options_mod.covered_call(spot, k, prem, qty)
    elif strat_name == "现金担保认沽 (Cash-Secured Put)":
        a, b = st.columns(2)
        k = a.number_input("认沽行权价 K", value=round(spot * 0.9, 2), step=1.0, key="csp_k")
        prem = b.number_input("收取的认沽权利金/股", min_value=0.0, value=round(spot * 0.04, 2), step=0.1, key="csp_p")
        legs = options_mod.cash_secured_put(k, prem, qty)
    elif strat_name == "牛市认购价差 (Bull Call Spread)":
        a, b, c, d = st.columns(4)
        kl = a.number_input("买入行权价(低)", value=round(spot, 2), step=1.0, key="bcs_kl")
        pl = b.number_input("买入权利金/股", min_value=0.0, value=round(spot * 0.06, 2), step=0.1, key="bcs_pl")
        kh = c.number_input("卖出行权价(高)", value=round(spot * 1.15, 2), step=1.0, key="bcs_kh")
        ph = d.number_input("卖出权利金/股", min_value=0.0, value=round(spot * 0.02, 2), step=0.1, key="bcs_ph")
        legs = options_mod.bull_call_spread(kl, pl, kh, ph, qty)
    elif strat_name == "熊市认沽价差 (Bear Put Spread)":
        a, b, c, d = st.columns(4)
        kh = a.number_input("买入行权价(高)", value=round(spot, 2), step=1.0, key="bps_kh")
        ph = b.number_input("买入权利金/股", min_value=0.0, value=round(spot * 0.06, 2), step=0.1, key="bps_ph")
        kl = c.number_input("卖出行权价(低)", value=round(spot * 0.85, 2), step=1.0, key="bps_kl")
        pl = d.number_input("卖出权利金/股", min_value=0.0, value=round(spot * 0.02, 2), step=0.1, key="bps_pl")
        legs = options_mod.bear_put_spread(kh, ph, kl, pl, qty)
    elif strat_name == "领口 (Collar)":
        a, b, c, d = st.columns(4)
        pk = a.number_input("保护认沽行权价", value=round(spot * 0.9, 2), step=1.0, key="col_pk")
        pp = b.number_input("认沽权利金/股(付)", min_value=0.0, value=round(spot * 0.03, 2), step=0.1, key="col_pp")
        ck = c.number_input("卖出认购行权价", value=round(spot * 1.1, 2), step=1.0, key="col_ck")
        cp = d.number_input("认购权利金/股(收)", min_value=0.0, value=round(spot * 0.03, 2), step=0.1, key="col_cp")
        legs = options_mod.collar(spot, pk, pp, ck, cp, qty)
    elif strat_name == "买入跨式 (Long Straddle)":
        a, b, c = st.columns(3)
        k = a.number_input("行权价 K(同)", value=round(spot, 2), step=1.0, key="ls_k")
        cp = b.number_input("认购权利金/股", min_value=0.0, value=round(spot * 0.05, 2), step=0.1, key="ls_cp")
        pp = c.number_input("认沽权利金/股", min_value=0.0, value=round(spot * 0.05, 2), step=0.1, key="ls_pp")
        legs = options_mod.long_straddle(k, cp, pp, qty)
    elif strat_name == "铁鹰 (Iron Condor)":
        st.caption("从低到高四个行权价：买认沽 < 卖认沽 < 卖认购 < 买认购")
        a, b, c, d = st.columns(4)
        pl_k = a.number_input("买认沽行权", value=round(spot * 0.8, 2), step=1.0, key="ic_plk")
        pl_p = a.number_input("买认沽权利金", min_value=0.0, value=round(spot * 0.01, 2), step=0.1, key="ic_plp")
        ps_k = b.number_input("卖认沽行权", value=round(spot * 0.9, 2), step=1.0, key="ic_psk")
        ps_p = b.number_input("卖认沽权利金", min_value=0.0, value=round(spot * 0.025, 2), step=0.1, key="ic_psp")
        cs_k = c.number_input("卖认购行权", value=round(spot * 1.1, 2), step=1.0, key="ic_csk")
        cs_p = c.number_input("卖认购权利金", min_value=0.0, value=round(spot * 0.025, 2), step=0.1, key="ic_csp")
        cl_k = d.number_input("买认购行权", value=round(spot * 1.2, 2), step=1.0, key="ic_clk")
        cl_p = d.number_input("买认购权利金", min_value=0.0, value=round(spot * 0.01, 2), step=0.1, key="ic_clp")
        legs = options_mod.iron_condor(pl_k, pl_p, ps_k, ps_p, cs_k, cs_p, cl_k, cl_p, qty)

    if not legs:
        return

    try:
        res = options_mod.analyze(legs, spot, width=0.6)
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ 计算失败：{e}")
        return

    m1, m2, m3, m4 = st.columns(4)
    max_p = res.max_profit
    max_l = res.max_loss
    m1.metric("最大盈利", "≈ 无上限" if max_p > 5e7 else f"${max_p:,.0f}")
    m2.metric("最大亏损", f"${max_l:,.0f}")
    be_txt = " / ".join(f"${b:,.2f}" for b in res.breakevens) if res.breakevens else "无"
    m3.metric("盈亏平衡点", be_txt)
    net = res.net_cost
    m4.metric("建仓现金流", f"{'收' if net >= 0 else '付'} ${abs(net):,.0f}",
              help="正=净收权利金；负=净支出（含股票成本）")

    st.plotly_chart(_options_payoff_chart(res, spot), use_container_width=True)

    risk_reward = (max_p / abs(max_l)) if max_l < 0 and max_p < 5e7 else None
    if risk_reward is not None:
        st.caption(f"盈亏比（最大盈利 / 最大亏损）≈ {risk_reward:.2f}")

    st.warning(
        "⚠️ 这是**到期日**的理论损益，未计入时间价值、隐含波动率变化(IV)、提前行权与税费。"
        "高波动个股期权很贵，实际盈亏请以券商报价为准。本工具仅供学习，不构成投资建议。"
    )


def _tab_options_compare(cfg: dict) -> None:
    st.caption("选择多个策略，用相同现价与默认参数并排对比盈亏曲线与最大盈亏。")
    c0, c1 = st.columns(2)
    spot = c0.number_input("标的现价 (USD)", min_value=0.01, value=100.0, step=1.0, key="options_cmp_spot")
    qty = c1.number_input("合约张数", min_value=1, value=1, step=1, key="options_cmp_qty")
    if c0.button(f"↻ 拉取 {cfg['ticker']} 最新价", key="options_cmp_fetch"):
        df = get_data(cfg)
        if df is not None and len(df):
            st.session_state["options_cmp_spot"] = float(df["Close"].iloc[-1])
            st.rerun()
    if "options_cmp_spot" in st.session_state:
        spot = float(st.session_state["options_cmp_spot"])

    picks = st.multiselect(
        "对比策略（选 2~4 个）",
        ["领口 (Collar)", "熊市认沽价差 (Bear Put Spread)", "买入认沽 (Long Put)",
         "牛市认购价差 (Bull Call Spread)", "备兑开仓 (Covered Call)"],
        default=["领口 (Collar)", "熊市认沽价差 (Bear Put Spread)"],
        key="options_cmp_picks",
    )
    if len(picks) < 2:
        st.info("请至少选择 2 个策略进行对比。")
        return

    legs_map: dict[str, list] = {}
    for name in picks:
        if name == "领口 (Collar)":
            legs_map[name] = options_mod.collar(spot, spot * 0.9, spot * 0.03, spot * 1.1, spot * 0.03, qty)
        elif name == "熊市认沽价差 (Bear Put Spread)":
            legs_map[name] = options_mod.bear_put_spread(spot, spot * 0.06, spot * 0.85, spot * 0.02, qty)
        elif name == "买入认沽 (Long Put)":
            legs_map[name] = options_mod.long_put(spot * 0.95, spot * 0.05, qty)
        elif name == "牛市认购价差 (Bull Call Spread)":
            legs_map[name] = options_mod.bull_call_spread(spot, spot * 0.06, spot * 1.15, spot * 0.02, qty)
        elif name == "备兑开仓 (Covered Call)":
            legs_map[name] = options_mod.covered_call(spot, spot * 1.1, spot * 0.04, qty)

    table, results = options_mod.compare_results(legs_map, spot)
    disp = table.copy()
    disp["最大盈利"] = disp["最大盈利"].map(lambda x: "≈无上限" if x > 5e7 else f"${x:,.0f}")
    disp["最大亏损"] = disp["最大亏损"].map(lambda x: f"${x:,.0f}")
    disp["建仓现金流"] = disp["建仓现金流"].map(lambda x: f"{'收' if x >= 0 else '付'} ${abs(x):,.0f}")
    disp["盈亏比"] = disp["盈亏比"].map(lambda x: f"{x:.2f}" if x is not None else "-")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    fig = go.Figure()
    colors = theme.PALETTE[:5]
    for i, (name, res) in enumerate(results.items()):
        fig.add_trace(go.Scatter(x=res.prices, y=res.payoff, name=name, line=dict(color=colors[i % 5], width=2)))
    fig.add_vline(x=spot, line=dict(color=theme.GOLD, width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color=theme.TEXT_TERTIARY, width=1))
    fig.update_layout(height=440, template="tiger", xaxis_title="到期股价", yaxis_title="到期盈亏 (USD)",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("默认权利金为现价的估算比例，请替换为券商真实报价后再决策。")


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
    color = {"趋势市": theme.ORANGE, "震荡市": theme.BLUE, "过渡": theme.GOLD}.get(reg.trend_label, theme.TEXT_SECONDARY)
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
            return [f"background-color: {theme.UP}22"] * len(row)
        if row["契合度"] == "不契合":
            return [f"background-color: {theme.DOWN}22"] * len(row)
        return [""] * len(row)

    st.dataframe(disp.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)
    st.caption("绿色 = 与当前市场高度契合；红色 = 不契合（不建议在当前环境使用）。推荐基于历史，仅供参考。")

    st.divider()
    st.markdown("**期权策略方向参考**（到期损益计算器可进一步模拟）")
    owns = st.checkbox("我已持有该标的", key="rec_own_shares")
    bearish = st.checkbox("我看跌 / 想对冲", key="rec_bearish")
    opt_recs = options_mod.recommend_for_regime(
        trend_label=reg.trend_label,
        direction=reg.direction,
        vol_pct=reg.vol_pct * 100,
        owns_shares=owns,
        bearish_view=bearish,
    )
    for name, reason in opt_recs[:3]:
        st.markdown(f"- **{name}**：{reason}")


# ---------------------------------------------------------------------------
# 标签页：异动前兆选股
# ---------------------------------------------------------------------------
def tab_precursor(cfg: dict) -> None:
    st.subheader("异动前兆选股 · 提前捕捉上涨/下跌迹象")
    st.caption(
        "扫描量能、波动收缩、趋势萌芽、相对强弱、MACD/RSI 等可量化前兆，"
        "在大涨大跌前给出线索。分数越高，触发的看涨/看跌前兆越多。"
    )

    with st.expander("📖 全部前兆信号说明", expanded=False):
        st.dataframe(precursor.list_catalog(), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    pool_options = {**screener.UNIVERSE_PRESETS, "sp500": "标普500成分", "custom": "自定义列表"}
    pool_key = c1.selectbox("扫描股票池", list(pool_options.keys()),
                            format_func=lambda k: pool_options[k], key="pre_pool")
    pool_size = c2.number_input("扫描数量", 10, 150, 40, 10, key="pre_size")
    min_score = c3.slider("最低前兆得分", 0.0, 5.0, 0.8, 0.1, key="pre_min")

    custom_raw = ""
    if pool_key == "custom":
        custom_raw = st.text_input("自定义代码（逗号分隔）",
                                   value=DEFAULT_WATCHLIST,
                                   key="pre_custom")

    use_spy = st.checkbox("相对强弱对比 SPY 基准", value=True, key="pre_spy")

    if not st.button("🔮 扫描异动前兆", type="primary", key="run_pre"):
        return

    with st.spinner("正在拉取行情并扫描前兆信号…"):
        try:
            end = cfg["end"]
            start = cfg["start"]
            if pool_key == "custom":
                tickers = parse_tickers(custom_raw)
            elif pool_key == "sp500":
                tickers = screener.fetch_sp500_tickers()[: int(pool_size)]
            else:
                snap = screener.fetch_yahoo_screen(pool_key, count=int(pool_size))
                tickers = snap["代码"].tolist() if not snap.empty else []
            if not tickers:
                st.error("❌ 股票池为空。")
                return
            data, failed = get_multi_data(tickers[: int(pool_size)], cfg)
            if failed:
                st.warning(f"以下标的拉取失败已忽略：{', '.join(failed)}")
            bench = None
            if use_spy:
                spy_df = fetch_history("SPY", start=start, end=end)
                bench = spy_df["Close"]
            table = precursor.scan_universe(data, bench, min_score=float(min_score))
        except DataError as e:
            st.error(f"❌ {e}")
            return

    if table.empty:
        st.warning("未发现达到得分阈值的异动前兆，请降低最低得分或扩大股票池。")
        return

    st.success(f"发现 **{len(table)}** 只标的存在异动前兆（按得分排序）")
    disp = table.drop(columns=["_hits"], errors="ignore").copy()
    disp["最新价"] = disp["最新价"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
    disp["近5日%"] = disp["近5日%"].map(lambda x: f"{x:+.1f}%")
    disp["近20日%"] = disp["近20日%"].map(lambda x: f"{x:+.1f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    top = table.iloc[0]
    st.markdown(f"**🏆 前兆最强：{top['代码']}**（得分 {top['前兆得分']}，偏向 {top['偏向']}）")
    hits = top.get("_hits") or []
    for h in hits:
        icon = "🟢" if h.direction == "bull" else "🔴" if h.direction == "bear" else "🟡"
        st.markdown(f"- {icon} **{h.name}**（强度 {h.strength:.0%}）：{h.description}")

    st.caption("⚠️ 前兆≠预测；信号可能滞后或失效，请结合样本外验证与仓位管理。")


# ---------------------------------------------------------------------------
# 标签页 10：一键体检
# ---------------------------------------------------------------------------
def _score_gauge(score: float, grade: str) -> go.Figure:
    color = theme.UP if score >= 72 else theme.GOLD if score >= 55 else theme.DOWN
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " 分", "font": {"size": 36}},
        title={"text": f"综合评分 · {grade}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#5c3a3a"},
                {"range": [40, 55], "color": "#5c4a3a"},
                {"range": [55, 72], "color": "#5c503a"},
                {"range": [72, 100], "color": "#3a5c4a"},
            ],
        },
    ))
    fig.update_layout(height=260, template="tiger", margin=dict(l=20, r=20, t=50, b=10))
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
            fig = go.Figure(go.Bar(x=h["持有期"], y=h["赚钱概率"], marker_color=theme.ORANGE,
                                   text=[fmt_pct(v) for v in h["赚钱概率"]], textposition="outside"))
            fig.add_hline(y=0.5, line=dict(color=theme.GOLD, width=1, dash="dash"))
            fig.update_layout(height=460, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
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
    _render_brand_header()
    st.caption("数据来源可配置 Polygon / Alpaca / Yahoo · 自用研究工具，不构成投资建议")

    cfg = sidebar()
    tabs = st.tabs(
        ["体检", "推荐", "前兆", "回测", "寻优", "对比", "组合",
         "信号", "验证", "模拟", "选股", "概率", "期权"]
    )
    with tabs[0]:
        tab_report(cfg)
    with tabs[1]:
        tab_recommend(cfg)
    with tabs[2]:
        tab_precursor(cfg)
    with tabs[3]:
        tab_single(cfg)
    with tabs[4]:
        tab_optimize(cfg)
    with tabs[5]:
        tab_compare(cfg)
    with tabs[6]:
        tab_portfolio(cfg)
    with tabs[7]:
        tab_signals(cfg)
    with tabs[8]:
        tab_validation(cfg)
    with tabs[9]:
        tab_paper(cfg)
    with tabs[10]:
        tab_screener(cfg)
    with tabs[11]:
        tab_probability(cfg)
    with tabs[12]:
        tab_options(cfg)

    st.markdown(
        '<div class="tiger-disclaimer">'
        '数据来源 Yahoo Finance · 仅供个人研究，不构成任何投资建议 · 投资有风险，入市需谨慎'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
