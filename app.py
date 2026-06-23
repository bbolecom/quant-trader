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
    strategy_search,
    validation,
    vol_decay,
)
from quant import decline_income
from quant.daily_screen_fleet import (
    fleet_accounts,
    fleet_stats_table,
    is_csp_account,
    load_fleet_config,
    load_fleet_stats,
    load_market_scan_results,
    load_target_profile,
    market_scan_summary_table,
    meets_target_profile,
    account_strategy_label,
    _stats_from_anchor,
    preset_for_account,
    target_gap_summary,
    tickers_for_preset,
    today_picks_for_account,
    backtest_account,
)
from quant.calendar_spread import scan_calendar_plans
from quant.data import DataError, fetch_history, fetch_history_batch, get_data_source_info
import ths_theme as theme

from research.gainer_daily_backtest import (
    GAINER_MODE_LABELS,
    TOP_N as GAINER_TOP_N,
    backtest_daily_gainer_portfolio,
    compare_gainer_modes,
    fetch_gainer_data_yahoo,
    filters_for_mode,
    live_gainer_picks,
    load_gainer_pool,
)
from research.vol_decay_putspread import (
    PutSpreadConfig,
    compare_structures as vol_decay_compare_structures,
)
from research.income_engine import build_income_plan

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
    page_title="量化策略 · 同花顺风格",
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


@st.cache_data(ttl=600, show_spinner=False)
def load_real_option_chain(ticker: str, min_dte: int, max_dte: int):
    """真实期权链（券商可对照）。缓存 10 分钟。返回 (到期日, dte, calls, puts)。"""
    from quant.option_chain import fetch_chain

    return fetch_chain(ticker, min_dte=min_dte, max_dte=max_dte, use_cache=False)


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
    """同花顺风格顶栏。"""
    from datetime import datetime

    icon_path = ROOT_DIR / "assets" / "icon.png"
    logo_html = ""
    if icon_path.exists():
        import base64

        b64 = base64.b64encode(icon_path.read_bytes()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{b64}" width="44" height="44" '
            f'style="border-radius:10px;flex-shrink:0;box-shadow:0 2px 8px rgba(233,48,48,0.35)" alt="logo"/>'
        )
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(
        f'<div class="ths-topbar">'
        f'<div class="ths-topbar-accent"></div>'
        f'<div class="ths-topbar-body">'
        f'<div class="brand">{logo_html}'
        f'<div><div class="brand-name"><span>量化</span>策略终端</div>'
        f'<div class="brand-sub">研究 · 选股 · 回测 · 期权 · 收入引擎</div></div></div>'
        f'<div style="display:flex;align-items:center;gap:16px">'
        f'<div class="market-strip"><b>行情</b> 实时 · <b>更新</b> {now}</div>'
        f'<span class="brand-tag">PRO</span></div></div></div>',
        unsafe_allow_html=True,
    )


def sidebar() -> dict:
    icon_path = ROOT_DIR / "assets" / "icon.png"
    if icon_path.exists():
        st.sidebar.image(str(icon_path), width=52)
    st.sidebar.markdown(
        '<p style="margin:-4px 0 12px;font-size:0.95rem;font-weight:700;color:#F0F1F5">'
        '<span style="color:#E93030">量化</span>策略终端</p>',
        unsafe_allow_html=True,
    )
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
DAILY_SCREEN_HISTORY = ROOT_DIR / "screen_history.csv"


def _display_picks_with_reason(df: pd.DataFrame, *, limit: int | None = None) -> None:
    """展示选股表，突出「选股理由」列。"""
    if df.empty:
        st.info("暂无选股结果。")
        return
    show = df.head(limit) if limit else df
    prefer = [
        "选股日期", "代码", "名称", "选股理由", "涨幅%", "换手率%", "成交额USD",
        "行业", "策略累计收益", "夏普比率", "当前信号", "入选价",
    ]
    cols = [c for c in prefer if c in show.columns]
    cols += [c for c in show.columns if c not in cols and not str(c).startswith("_")]
    disp = show[cols].copy()
    for c in ["涨幅%", "策略累计收益", "夏普比率", "换手率%"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce")
            if c == "涨幅%":
                disp[c] = disp[c].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
            elif c == "策略累计收益":
                disp[c] = disp[c].map(lambda x: fmt_pct(x) if pd.notna(x) else "-")
            elif c == "夏普比率":
                disp[c] = disp[c].map(lambda x: fmt_num(x) if pd.notna(x) else "-")
            else:
                disp[c] = disp[c].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    if "成交额USD" in disp.columns:
        disp["成交额USD"] = disp["成交额USD"].map(_fmt_dollar_m)
    col_cfg = {}
    if "选股理由" in disp.columns:
        col_cfg["选股理由"] = st.column_config.TextColumn("选股理由", width="large")
    st.dataframe(disp, use_container_width=True, hide_index=True, column_config=col_cfg)
    st.markdown("**入选理由详情**")
    for _, row in show.iterrows():
        code = row.get("代码", "")
        name = row.get("名称", "")
        gain = row.get("涨幅%")
        gain_s = f"{gain:+.1f}%" if pd.notna(gain) else ""
        label = f"**{code}** {name} {gain_s}".strip()
        with st.expander(label, expanded=len(show) <= 5):
            st.markdown(row.get("选股理由") or "（无详细理由）")
            extras = []
            for k in ["行业", "当前信号", "策略累计收益", "夏普比率"]:
                if k in row.index and pd.notna(row[k]):
                    v = fmt_pct(row[k]) if k == "策略累计收益" else row[k]
                    extras.append(f"- **{k}**：{v}")
            if extras:
                st.markdown("\n".join(extras))


def _fmt_fleet_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["年化收益", "最大回撤", "胜率", "选股日胜率"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(
                lambda x: fmt_pct(x) if pd.notna(x) else "—"
            )
    if "夏普" in out.columns:
        out["夏普"] = pd.to_numeric(out["夏普"], errors="coerce").map(
            lambda x: fmt_num(x) if pd.notna(x) else "—"
        )
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_fleet_account_backtest(
    account_id: str,
    preset_id: str,
    tickers_key: str,
    end_d: str,
    years: float,
    account_size: float,
    fee_bps: float,
    slippage_bps: float,
    allow_short: bool,
) -> dict:
    """单账户回测（缓存 24h）。"""
    acct = next(a for a in fleet_accounts() if a["id"] == account_id)
    tickers = tickers_key.split(",")
    fetch_start = (pd.Timestamp(end_d) - pd.DateOffset(years=years + 0.3)).strftime("%Y-%m-%d")
    data = fetch_history_batch(tickers, fetch_start, end_d)
    if not data:
        return {"error": "无可用行情"}
    return backtest_account(
        acct, data, years=years, initial_capital=account_size,
        allow_short=allow_short, fee_bps=fee_bps, slippage_bps=slippage_bps,
    )


def tab_daily_screen(cfg: dict) -> None:
    """5×圣杯舰队：CSP 收入策略 + 历史锚点 + 今日信号。"""
    # ---- 今日选股快照 ----
    _dp_json = ROOT_DIR / "research" / "daily_pick_today.json"
    if _dp_json.exists():
        import json as _json
        _dp = _json.loads(_dp_json.read_text(encoding="utf-8"))
        _ds = _dp.get("summary") or {}
        _reg = _dp.get("regime") or {}
        _ss = _dp.get("strategy_summary") or {}
        d0, d1, d2, d3, d4 = st.columns(5)
        d0.metric("大盘", _reg.get("label", _ds.get("大盘", "—"))[:12])
        d1.metric("模式", _ds.get("模式", "—"))
        d2.metric("今日可开仓", _ds.get("可开仓", 0))
        d3.metric("观望", _ds.get("观望", 0))
        d4.metric("是否空仓日", "是" if _ds.get("是否空仓日") else "否")
        if _reg.get("spy") and _reg.get("ma50"):
            st.caption(f"SPY {_reg['spy']} / MA50 {_reg['ma50']} · {_reg.get('playbook', '')}")
        if _reg.get("bull") is False:
            st.warning("🔴 **弱市模式**：轨迹做多已关闭；优先 **卖Call价差** + **SPY/QQQ 铁鹰**。")
        elif _reg.get("bull") is True:
            st.success("🟢 **牛市模式**：CSP 舰队 + 轨迹高置信 + 卖Call 均可选。")
        if _ds.get("是否空仓日"):
            st.info("今日无符合标的 — **正常空仓**，等待下一信号日。")

        # ---- 全系统策略总览 ----
        _catalog = _ss.get("catalog") or []
        if _catalog:
            with st.expander("📋 全系统策略总览（接入每日选股 + 独立策略）", expanded=True):
                st.caption(
                    f"接入 **{_ss.get('integrated_count', 0)}** 个模块 · "
                    f"独立策略 **{_ss.get('standalone_count', 0)}** 个 · "
                    f"今日有快照 **{_ss.get('integrated_with_data', 0)}** 个 · "
                    f"更新 {_ss.get('updated', '—')}"
                )
                cat_df = pd.DataFrame(_catalog)
                show_cat = [c for c in [
                    "策略", "分类", "已接入每日选股", "模块标签",
                    "今日有数据", "可开仓", "观望", "总条目", "数据日期", "说明",
                ] if c in cat_df.columns]
                st.dataframe(cat_df[show_cat], use_container_width=True, hide_index=True)

        _mods = _dp.get("modules_summary") or {}
        if _mods:
            with st.expander("🧩 今日模块汇总（daily_pick 各引擎）", expanded=True):
                mod_rows = []
                for mod, stt in _mods.items():
                    codes = "、".join(stt.get("代码") or []) or "—"
                    mod_rows.append({
                        "模块": mod,
                        "可开仓": stt.get("可开仓", 0),
                        "观望": stt.get("观望", 0),
                        "总条目": stt.get("总条目", 0),
                        "可开仓代码": codes,
                    })
                st.dataframe(pd.DataFrame(mod_rows), use_container_width=True, hide_index=True)

        _action = [p for p in (_dp.get("high_win") or {}).get("picks") or []]
        if not _action:
            _action = [p for p in (_dp.get("picks") or []) if p.get("状态") == "可开仓"]
        if _action:
            st.markdown("#### ★ 高胜率≥80% · 今日可执行")
            act_df = pd.DataFrame(_action)
            act_cols = [c for c in [
                "模块", "代码", "方向", "策略动作", "历史胜率", "历史年化", "最大回撤",
                "回测摘要", "回测来源", "选股理由",
            ] if c in act_df.columns]
            st.dataframe(act_df[act_cols], use_container_width=True, hide_index=True)
            st.caption("仅展示历史胜率≥80%且今日可开仓的标的；全量见下方。")

        _pick_df = pd.DataFrame(_dp.get("picks") or [])
        if not _pick_df.empty:
            _mod_filter = st.multiselect(
                "筛选模块",
                sorted(_pick_df["模块"].astype(str).unique()) if "模块" in _pick_df.columns else [],
                default=[],
                key="daily_pick_mod_filter",
            )
            if _mod_filter and "模块" in _pick_df.columns:
                _pick_df = _pick_df[_pick_df["模块"].isin(_mod_filter)]
            show_cols = [c for c in [
                "模块", "账户", "代码", "状态", "方向", "策略动作", "历史命中率",
                "建议张数", "权利金$", "选股理由",
            ] if c in _pick_df.columns]
            st.markdown("#### 今日选股快照（全部）")
            st.dataframe(_pick_df[show_cols], use_container_width=True, hide_index=True)
        st.caption("定时：双击「每日选股_运行一次.command」或 `python daily_pick.py`")
    else:
        st.info("尚未生成 `research/daily_pick_today.json` — 请先运行 `python daily_pick.py`。")

    fleet_cfg = load_fleet_config()
    fleet_stats = load_fleet_stats()
    targets = load_target_profile(fleet_cfg)
    acct_size = float(fleet_cfg.get("account_size", 10_000))
    bt_years = float(fleet_cfg.get("backtest_years", 5))
    tgt_ann = float(targets["ann_return"])
    tgt_dd = float(targets["max_dd"])
    tgt_wr = float(targets["win_rate"])

    st.subheader("每日选股 · 有信号才出手")
    st.caption(
        "**不一定每天都有票。** SPY/MA50 牛熊开关：弱市主开卖Call+ETF铁鹰，牛市开 CSP 舰队 + 轨迹高置信；"
        " 条件不满足则 **观望/空仓**。定时：双击「每日选股_运行一次.command」。"
    )

    # ---- 圣杯三标 ----
    st.markdown("### 🎯 目标约束（三标达标）")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("目标 · 胜率", f">{tgt_wr:.0%}")
    t2.metric("目标 · 最大回撤", f"<{abs(tgt_dd):.0%}")
    t3.metric("目标 · 年化", f">{tgt_ann:.0%}")
    acct_list = fleet_stats.get("accounts") or []
    pass_n = sum(
        1 for a in acct_list
        if meets_target_profile(a.get("stats") or {}, targets)
    )
    t4.metric("达标账户", f"{pass_n}/{len(acct_list) or 5}")

    if pass_n < (len(acct_list) or 5):
        st.warning(
            "全库扫描结论：**只有 SNDK 卖 Put（CSP）** 同时满足胜率>85%、回撤<10%、年化>80%。"
            " 动量/超跌等日线选股策略无法同时达标，已切换为 CSP 圣杯舰队。"
        )
    else:
        st.success("当前 5 账户策略均满足三标约束（基于全市场扫描锚点）。")

    # ---- 全市场扫描结果 ----
    mkt_df = load_market_scan_results()
    if not mkt_df.empty:
        with st.expander("🌐 全市场扫描结果（459 只 × CSP/铁鹰网格）", expanded=True):
            st.caption(
                f"扫描脚本 `research/market_triple_scan.py` · 共 **{len(mkt_df)}** 组策略 · "
                f"Tier A **{int((mkt_df['tier'] == 'A').sum())}** 条 · "
                f"达标标的 **{', '.join(sorted(mkt_df[mkt_df['tier'] == 'A']['代码'].unique()))}**"
            )
            mkt_tbl = market_scan_summary_table(mkt_df, targets)
            if not mkt_tbl.empty:
                show = mkt_tbl.copy()
                for c in ["年化", "最大回撤", "胜率", "alloc"]:
                    if c in show.columns:
                        if c == "alloc":
                            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                                lambda x: f"{x:.0%}" if pd.notna(x) else "—"
                            )
                        else:
                            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                                lambda x: fmt_pct(x) if pd.notna(x) else "—"
                            )
                st.dataframe(show, use_container_width=True, hide_index=True)
            st.code(
                f"cd {ROOT_DIR}\n"
                ".venv/bin/python research/market_triple_scan.py "
                f"--ann {tgt_ann} --max-dd {tgt_dd} --win {tgt_wr} --min-dvol-m 30 --pick-fleet 5",
                language="bash",
            )

    # ---- 资金轨迹规律（不限固定标的）----
    _rules_path = ROOT_DIR / "research" / "move_pattern_rules.json"
    _today_path = ROOT_DIR / "research" / "move_pattern_today.csv"
    with st.expander("🔬 资金轨迹规律（成交额 · 量比 · 涨幅 → 涨跌前模式）", expanded=False):
        st.caption(
            "**思路**：大涨大跌需要资金推动，事前往往有放量、动量、价位等轨迹。"
            " **高置信模式**（默认）用次日胜率 + 严格量价模板 + 形态历史验证，目标 **≥62%**。"
        )
        if _rules_path.exists():
            import json as _json
            mp_doc = _json.loads(_rules_path.read_text(encoding="utf-8"))
            st.markdown(
                f"**样本** {mp_doc.get('event_count', '—')} 条 · "
                f"**覆盖标的** {mp_doc.get('ticker_count', '—')} 只 · "
                f"更新 **{mp_doc.get('updated', '—')}**"
            )
            ru = mp_doc.get("rules_up") or []
            rd = mp_doc.get("rules_down") or []
            if ru:
                st.markdown("**📈 上涨前轨迹 / 高置信做多**")
                for r in ru[:6]:
                    wl = r.get("win_label", "胜率")
                    tier = r.get("tier", "")
                    st.markdown(
                        f"- **[{tier}] {r.get('pattern', '')}** — 样本 {r.get('sample_n')} · "
                        f"{wl} **{fmt_pct(r.get('win_rate', 0))}** · {r.get('action', '')}"
                    )
            if rd:
                st.markdown("**📉 下跌前常见轨迹**")
                for r in rd[:5]:
                    st.markdown(
                        f"- **{r.get('pattern', '')}** — 样本 {r.get('sample_n')} · "
                        f"下跌率 {fmt_pct(1 - r.get('win_rate', 0))} · {r.get('action', '')}"
                    )
        else:
            st.info("尚未挖掘。运行下方命令生成全市场规律。")
        if _today_path.exists():
            td = pd.read_csv(_today_path)
            if not td.empty:
                st.markdown("**🎯 今日命中规律的股票**")
                st.dataframe(td, use_container_width=True, hide_index=True)
        st.code(
            f"cd {ROOT_DIR}\n"
            ".venv/bin/python research/move_pattern_mine.py --mode highwin --min-win-rate 0.62\n"
            ".venv/bin/python research/move_pattern_mine.py --today-only",
            language="bash",
        )

    _opt_rules_path = ROOT_DIR / "research" / "pattern_rules_optimized.json"
    with st.expander("📊 三腿策略 · 参数寻优（做多+回避+收租）", expanded=False):
        st.caption(
            "腿①② 用真实量价网格寻优（IS 2019–2023 / OOS 2024+）；腿③ 用 yfinance 真实期权链。"
            " **非 BS 回测绝对胜率**。"
        )
        if _opt_rules_path.exists():
            import json as _json
            from quant.pattern_params import OptimizedPatternRules

            opt = OptimizedPatternRules.from_dict(
                _json.loads(_opt_rules_path.read_text(encoding="utf-8"))
            )
            meta = opt.meta or {}
            ls = meta.get("long_search") or {}
            ds = meta.get("down_search") or {}
            lp, dp = opt.long, opt.down
            c1, c2, c3 = st.columns(3)
            if ls.get("is"):
                c1.metric("做多次日胜率 IS", fmt_pct(ls["is"].get("win_rate", 0)))
            if ls.get("oos"):
                c2.metric("做多次日胜率 OOS", fmt_pct(ls["oos"].get("win_rate", 0)))
            if ds.get("is_down_rate"):
                c3.metric("回避20日下跌率", fmt_pct(ds.get("is_down_rate", 0)))
            st.markdown(
                f"**做多** 涨 {lp.min_gain_pct}–{lp.max_gain_pct}% · 量比 {lp.min_vol_ratio}–{lp.max_vol_ratio} · "
                f"收强≥{lp.min_close_strength:.0%} · 形态≥{lp.min_setup_win_rate:.0%} · "
                f"SPY1d={'≥'+str(lp.min_spy_1d_pct)+'%' if lp.require_spy_positive_1d else '无'}"
            )
            st.markdown(f"**回避** {dp.describe()}")
            if ds.get("active_avoid_rules"):
                st.caption(f"启用规则: {', '.join(ds['active_avoid_rules'])}")
            h5 = ds.get("horizon_5d") or {}
            if h5.get("is_down_rate"):
                st.caption(f"5日合并下跌率 IS={fmt_pct(h5['is_down_rate'])}")
        else:
            st.info("尚未寻优。运行 `./规律寻优_运行一次.command` 或下方命令。")
        st.code(
            f"cd {ROOT_DIR}\n"
            ".venv/bin/python research/pattern_param_search.py\n"
            ".venv/bin/python pattern_daily.py --dry-run",
            language="bash",
        )

    _5d_rules = ROOT_DIR / "research" / "move_pattern_5d_rules.json"
    _5d_today = ROOT_DIR / "research" / "move_pattern_5d_today.csv"
    with st.expander("📅 5日路径规律（真实OHLCV · 换手率 · 流动性）", expanded=False):
        st.caption(
            "未来 5 个交易日内，路径最高价涨≥X% 或最低价跌≥X% 即算命中；"
            " **换手率=成交量/流通股**，成交额+量比双过滤，非 BS 回测。"
        )
        if _5d_rules.exists():
            import json as _json5
            from quant.pattern_5d_params import load_optimized_5d

            d5 = _json5.loads(_5d_rules.read_text(encoding="utf-8"))
            opt5 = load_optimized_5d()
            fu = (opt5.meta.get("final_up") or {}).get("oos") or {}
            fd = (opt5.meta.get("final_down") or {}).get("oos") or {}
            if fu.get("hit_rate"):
                st.success(
                    f"寻优：做多 OOS **{fmt_pct(fu['hit_rate'])}** (n={fu.get('n')}) · "
                    f"回避 OOS **{fmt_pct(fd.get('hit_rate', 0))}** · 路径 **±{opt5.threshold.up_pct}%**"
                )
            st.markdown(f"**方法** {d5.get('method', '—')}")
            st.markdown(
                f"样本 **{d5.get('event_count', '—')}** 条 · "
                f"标的 **{d5.get('ticker_count', '—')}** 只 · 更新 **{d5.get('updated', '—')}**"
            )
            for r in (d5.get("rules_up") or [])[:4]:
                oos = (r.get("conditions") or {}).get("oos_hit_rate", 0)
                st.markdown(
                    f"- 📈 **{r.get('pattern', '')}** — n={r.get('sample_n')} · "
                    f"IS **{fmt_pct(r.get('win_rate', 0))}** OOS **{fmt_pct(oos)}**"
                )
            for r in (d5.get("rules_down") or [])[:3]:
                oos = (r.get("conditions") or {}).get("oos_hit_rate", 0)
                st.markdown(
                    f"- 📉 **{r.get('pattern', '')}** — n={r.get('sample_n')} · "
                    f"IS **{fmt_pct(r.get('win_rate', 0))}** OOS **{fmt_pct(oos)}**"
                )
        else:
            st.info("尚未挖掘。运行下方命令。")
        if _5d_today.exists():
            t5 = pd.read_csv(_5d_today)
            if not t5.empty:
                st.markdown("**🎯 今日 5 日路径命中**")
                st.dataframe(t5.head(30), use_container_width=True, hide_index=True)
        st.code(
            f"cd {ROOT_DIR}\n"
            ".venv/bin/python research/move_pattern_5d_param_search.py\n"
            ".venv/bin/python research/move_pattern_5d_mine.py --from-cache\n"
            ".venv/bin/python research/move_pattern_5d_mine.py --today-only",
            language="bash",
        )

    # ---- 策略矩阵 ----
    st.markdown("### 📋 五账户策略矩阵")
    stats_tbl = fleet_stats_table(fleet_stats, targets) if fleet_stats.get("accounts") else pd.DataFrame()
    if stats_tbl.empty:
        st.info(
            "尚无缓存回测结果。点击下方 **「刷新5年回测统计」** 生成各策略的历史胜率、年化与回撤。"
            "（首次约 3～8 分钟，结果写入 `research/screen_fleet_stats.json`）"
        )
    else:
        gen = fleet_stats.get("generated", "—")
        st.caption(
            f"统计生成日 **{gen}** ｜ 锚点来源 **triple_target_scan** ｜ 目标 **{targets.get('label', '三标')}**"
        )
        st.dataframe(_fmt_fleet_stats_table(stats_tbl), use_container_width=True, hide_index=True)

    r1, r2 = st.columns([1, 2])
    if r1.button("🔄 刷新锚点统计", type="secondary", key="daily_refresh_stats"):
        st.info("CSP 账户使用研究锚点，无需长时回测。更新配置后运行：")
        st.code(
            f"cd {ROOT_DIR}\n.venv/bin/python research/screen_fleet_backtest.py --years 5",
            language="bash",
        )
    r2.caption(
        "扩展搜索更多达标标的："
        f" `.venv/bin/python research/triple_target_scan.py --ann {tgt_ann} --max-dd {tgt_dd} --win {tgt_wr}`"
    )

    # ---- 各账户策略说明 ----
    st.markdown("### 🎯 具体策略说明")
    for acct in fleet_accounts(fleet_cfg):
        strat_name = account_strategy_label(acct)
        cached = next(
            (a for a in (fleet_stats.get("accounts") or []) if a.get("account_id") == acct["id"]),
            {},
        )
        s = cached.get("stats") or {}
        ok = meets_target_profile(s, targets) if s else False
        hdr = f"{'✅' if ok else '❌'} **{acct['label']}** · {strat_name}（{acct.get('role', '')}）"
        with st.expander(hdr, expanded=False):
            st.markdown(acct.get("description") or "")
            if is_csp_account(acct):
                p = acct.get("csp_params") or {}
                st.caption(
                    f"标的 **{acct.get('ticker', 'SNDK')}** ｜ δ={p.get('delta')} ｜ "
                    f"MA={p.get('ma_window', 0) or '无'} ｜ 仓位 {float(p.get('alloc_pct', 0))*100:.0f}% ｜ "
                    f"DTE {p.get('dte_days', 35)} 日"
                )
            elif not is_csp_account(acct):
                preset = preset_for_account(acct)
                st.caption(
                    f"股票池：{screener.UNIVERSE_PRESETS.get(preset.pool, preset.pool)} ｜ "
                    f"每 {preset.rebalance_days} 日选 {preset.top_picks} 只 ｜ "
                    f"交易：{preset.trading_strategy}"
                )
            if s:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("回测年化", fmt_pct(s.get("ann_return", 0)))
                c2.metric("最大回撤", fmt_pct(s.get("max_dd", 0)))
                c3.metric("胜率", fmt_pct(s.get("trade_win_rate", 0)))
                c4.metric("三标", "✅ 达标" if ok else target_gap_summary(s, targets))
            elif acct.get("anchor_stats"):
                a = acct["anchor_stats"]
                st.caption(f"📌 **{a.get('source', '锚点')}**")
            elif cached.get("error"):
                st.warning(f"回测未完成：{cached['error']}")

    st.divider()

    # ---- 今日预测选股 ----
    st.markdown("### 🔮 今日开仓信号（5账户）")
    st.caption("CSP 账户输出 **卖 Put 计划**；条件未满足时显示「观望」及原因。")

    run_fleet_today = st.button("🔴 生成5账户今日信号", type="primary", key="daily_fleet_today")

    if run_fleet_today:
        end_d = cfg["end"]
        screen_accts = [a for a in fleet_accounts(fleet_cfg) if not is_csp_account(a)]
        all_t: list[str] = []
        for acct in screen_accts:
            preset = preset_for_account(acct)
            for t in tickers_for_preset(preset, acct):
                if t not in all_t:
                    all_t.append(t)
        pool_data: dict = {}
        if all_t:
            with st.spinner(f"拉取 {len(all_t)} 只标的…"):
                try:
                    from quant.data import fetch_history_batch
                    fetch_start = (pd.Timestamp(end_d) - pd.DateOffset(days=400)).strftime("%Y-%m-%d")
                    pool_data = fetch_history_batch(all_t, fetch_start, end_d)
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ 数据拉取失败：{e}")

        with st.spinner("生成各账户信号…"):
            for acct in fleet_accounts(fleet_cfg):
                strat_name = account_strategy_label(acct)
                if is_csp_account(acct):
                    plan = today_picks_for_account(acct, {}, end_d, capital=acct_size)
                else:
                    preset = preset_for_account(acct)
                    sub = {t: pool_data[t] for t in tickers_for_preset(preset, acct) if t in pool_data}
                    plan = today_picks_for_account(
                        acct, sub, end_d,
                        capital=acct_size, allow_short=cfg["allow_short"],
                        fee_bps=cfg["fee_bps"], slippage_bps=cfg["slippage_bps"],
                    )
                st.markdown(f"#### {acct['label']} · {strat_name}")
                if plan.empty:
                    st.info("今日该策略无信号（条件未满足或数据不足）。")
                else:
                    sel_date = plan["选股日期"].iloc[0] if "选股日期" in plan.columns else end_d
                    st.success(f"信号日 **{sel_date}** · {plan['方向'].iloc[0] if '方向' in plan.columns else len(plan)}")
                    _display_picks_with_reason(plan)
                    csv = plan.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        f"⬇️ 导出 {acct['label']} 今日信号",
                        csv, file_name=f"daily_{acct['id']}_{sel_date}.csv",
                        mime="text/csv", key=f"dl_fleet_{acct['id']}",
                    )

    st.divider()

    # ---- 单账户深度回测 ----
    st.markdown("### 📊 单账户深度回测（拟合 + 明细）")
    acct_labels = {
        a["id"]: f"{a['label']} · {account_strategy_label(a)}"
        for a in fleet_accounts(fleet_cfg)
    }
    sel_acct = st.selectbox(
        "选择账户查看权益曲线与历史选股明细",
        list(acct_labels.keys()),
        format_func=lambda k: acct_labels[k],
        key="daily_deep_acct",
    )
    run_deep = st.button("📈 运行深度回测", key="daily_run_deep")

    if run_deep:
        acct = next(a for a in fleet_accounts(fleet_cfg) if a["id"] == sel_acct)
        if is_csp_account(acct):
            st.info(
                f"**{acct['label']}** 为 CSP 圣杯策略，历史指标来自 triple_target_scan 锚点，"
                "无日线选股明细。请使用上方「今日开仓信号」查看卖 Put 计划。"
            )
            stats = _stats_from_anchor(acct, years=bt_years) if acct.get("anchor_stats") else {}
            if stats:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("年化收益", fmt_pct(stats.get("ann_return", 0)))
                m2.metric("最大回撤", fmt_pct(stats.get("max_dd", 0)))
                m3.metric("胜率", fmt_pct(stats.get("trade_win_rate", 0)))
                m4.metric("三标", "✅" if meets_target_profile(stats, targets) else "❌")
        else:
            preset = preset_for_account(acct)
            tickers = tickers_for_preset(preset, acct)
            tkey = ",".join(tickers)
            with st.spinner(f"回测 {acct['label']} · 近 {int(bt_years)} 年…"):
                bt_res = _cached_fleet_account_backtest(
                    acct["id"], preset.id, tkey, cfg["end"], bt_years, acct_size,
                    cfg["fee_bps"], cfg["slippage_bps"], cfg["allow_short"],
                )
            if "error" in bt_res:
                st.error(f"❌ {bt_res['error']}")
            else:
                stats = bt_res.get("stats", {})
                btd = bt_res.get("backtest", {})
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("年化收益", fmt_pct(stats.get("ann_return", 0)))
                m2.metric("最大回撤", fmt_pct(stats.get("max_dd", 0)))
                m3.metric("调仓胜率", fmt_pct(stats.get("period_win_rate", 0)))
                wr = stats.get("trade_win_rate")
                m4.metric("单笔胜率", fmt_pct(wr) if wr == wr else "—")
                m5.metric("夏普", fmt_num(stats.get("sharpe", 0)))

                eq = btd.get("权益曲线", pd.DataFrame())
                if isinstance(eq, pd.DataFrame) and not eq.empty:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq["日期"], y=eq["权益"], name=acct["label"],
                        line=dict(color=theme.ACCENT, width=2),
                    ))
                    fig_eq.update_layout(
                        height=360, template="ths",
                        title=f"{acct['label']} · {preset.name} · 近{int(bt_years)}年权益",
                        margin=dict(l=10, r=10, t=40, b=10), yaxis_title="权益 (USD)",
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                picks_detail = btd.get("选股明细")
                if isinstance(picks_detail, pd.DataFrame) and not picks_detail.empty:
                    st.markdown(f"**历史每日选股明细（{len(picks_detail)} 条 · 含选股理由）**")
                    filter_date = st.selectbox(
                        "按选股日筛选",
                        ["全部"] + sorted(picks_detail["选股日期"].unique(), reverse=True),
                        key="daily_deep_filter",
                    )
                    pdf = picks_detail if filter_date == "全部" else picks_detail[picks_detail["选股日期"] == filter_date]
                    _display_picks_with_reason(pdf.tail(200).iloc[::-1], limit=200)
                    st.download_button(
                        "⬇️ 导出选股明细 CSV",
                        picks_detail.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"daily_{sel_acct}_{int(bt_years)}y.csv", mime="text/csv",
                        key="dl_daily_deep",
                    )

    with st.expander("📜 screen_history.csv 历史推送记录", expanded=False):
        hist = screener.load_screen_history(DAILY_SCREEN_HISTORY)
        if hist.empty:
            st.caption("暂无。定时任务「每日选股_运行一次.command」会追加写入。")
        else:
            show_cols = [c for c in [
                "选股日期", "代码", "名称", "选股理由", "涨幅%", "策略累计收益", "当前信号",
            ] if c in hist.columns]
            _display_picks_with_reason(hist[show_cols].tail(30).iloc[::-1], limit=30)


def _tab_screen_preset_backtest(cfg: dict) -> None:
    st.markdown("### 📚 命名选股策略库 · 近 5 年回测")
    st.caption("每套策略均有名称与选股依据，可一键回测近 5 年调仓表现（盈利周期占比、夏普、回撤等）。")

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

    if not st.button("📈 回测此选股策略（近 5 年）", type="primary", key="run_scr_preset_bt"):
        return

    with st.spinner("正在拉取行情并滚动回测（可能需要 2～4 分钟）…"):
        try:
            end_d = cfg["end"]
            start_d = (pd.Timestamp(end_d) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
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
                preset, data, max_years=5.0,
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
        fig_eq.update_layout(height=360, template="tiger", title=f"{preset.name} · 近5年权益曲线",
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
        "回测起始日", value=(pd.Timestamp(cfg["end"]) - pd.DateOffset(years=5)).date(),
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


def _tab_gainer_pro(cfg: dict) -> None:
    st.markdown("### 📈 涨幅榜专业因子 · 高胜率 / 每周方案")
    st.caption(
        "全市场 Yahoo 涨幅/活跃榜 + 趋势/大盘/形态胜率过滤，持有 1 日。"
        " **日胜率≥80% 与每周交易难以兼得** — 可用下方模式切换或开启「四模式对比」。"
        " 行情固定走 Yahoo，不受侧边栏 Polygon 配置影响。"
    )

    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox(
        "策略模式",
        list(GAINER_MODE_LABELS.keys()),
        format_func=lambda k: GAINER_MODE_LABELS[k],
        key="gainer_mode",
    )
    pool = c2.selectbox(
        "股票池",
        ["broad", "sp500", "momentum", "liquid100"],
        format_func=lambda k: {
            "broad": "全市场（缓存400只）",
            "sp500": "标普500",
            "momentum": "动量扩展池",
            "liquid100": "高流动100（快测）",
        }[k],
        key="gainer_pool",
    )
    years = c3.slider("回测年数", 1.0, 5.0, 5.0, 0.5, key="gainer_years")
    do_compare = c4.checkbox("四模式对比", value=False, key="gainer_compare")

    run_bt = st.button("📊 运行回测", type="primary", key="run_gainer_bt")
    run_live = st.button("🔴 今日实时选股", key="run_gainer_live")

    if run_live:
        filt = filters_for_mode(mode)
        with st.spinner("扫描 Yahoo 涨幅榜并计算因子…"):
            try:
                live = live_gainer_picks(filt)
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ 扫描失败：{e}")
                live = pd.DataFrame()
        if live.empty:
            st.info("今日暂无满足条件的标的（或 Yahoo 榜为空）。")
        else:
            st.success(f"**{GAINER_MODE_LABELS[mode]}** · 共 {len(live)} 只")
            disp = live.copy()
            for c in ["涨幅%", "综合分", "量比", "相对SPY20d%"]:
                if c in disp.columns:
                    disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                    )
            if "成交额USD" in disp.columns:
                disp["成交额USD"] = disp["成交额USD"].map(_fmt_dollar_m)
            show = [c for c in [
                "选股日期", "代码", "名称", "涨幅%", "成交额USD", "量比",
                "综合分", "站上MA20", "近8次胜率", "选股理由",
            ] if c in disp.columns]
            st.dataframe(disp[show], use_container_width=True, hide_index=True)

    if not run_bt:
        return

    end_d = date.today().isoformat()
    start_d = (date.today() - timedelta(days=int(years * 365) + 120)).isoformat()
    fee_bps = float(cfg.get("fee_bps", 5.0))

    with st.spinner("拉取 Yahoo 行情并回测（约 1～2 分钟）…"):
        try:
            tickers = load_gainer_pool(pool)
            data, spy = fetch_gainer_data_yahoo(tickers, start_d, end_d)
            if not data:
                st.error("❌ 无可用行情。")
                return
            st.caption(f"有效标的 **{len(data)}** 只 · {start_d} ~ {end_d}")

            if do_compare:
                cmp_df = compare_gainer_modes(
                    data, spy, start=start_d, end=end_d, years=years, fee_bps=fee_bps,
                )
                if not cmp_df.empty:
                    st.markdown("**四模式对比**")
                    disp_cmp = cmp_df.copy()
                    disp_cmp["日胜率"] = disp_cmp["日胜率"].map(fmt_pct)
                    disp_cmp["累计收益"] = disp_cmp["累计收益"].map(fmt_pct)
                    disp_cmp["最大回撤"] = disp_cmp["最大回撤"].map(fmt_pct)
                    disp_cmp["夏普"] = disp_cmp["夏普"].map(fmt_num)
                    disp_cmp["年均次数"] = disp_cmp["年均次数"].map(lambda x: f"{x:.0f}")
                    st.dataframe(disp_cmp, use_container_width=True, hide_index=True)

            if mode == "weekly":
                from research.gainer_weekly_multi import run_weekly_suite
                wres = run_weekly_suite(data, spy, start=start_d, end=end_d, years=years)
                schemes = wres.get("schemes") or []
                if schemes:
                    st.markdown("**五套每周方案对比**")
                    rows = [{
                        "方案": s[0], "方向": s[5], "日胜率": s[1], "交易次": s[2],
                        "年均": s[3], "累计": s[4], "说明": s[6],
                    } for s in schemes]
                    wdf = pd.DataFrame(rows)
                    disp_w = wdf.copy()
                    disp_w["日胜率"] = disp_w["日胜率"].map(fmt_pct)
                    disp_w["累计"] = disp_w["累计"].map(fmt_pct)
                    disp_w["年均"] = disp_w["年均"].map(lambda x: f"{x:.0f}")
                    st.dataframe(disp_w.drop(columns=["说明"]), use_container_width=True, hide_index=True)
                    for _, row in wdf.iterrows():
                        st.caption(f"{row['方案']}：{row['说明']}")
                picks_w = wres.get("weekly_picks", pd.DataFrame())
                if isinstance(picks_w, pd.DataFrame) and not picks_w.empty:
                    st.markdown("**方案2 · 每周高置信 · 最近选股**")
                    st.dataframe(picks_w.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
                sig = wres.get("today_signal")
                if sig is not None:
                    st.success(
                        f"今日信号：**{sig.name}** · "
                        f"{'做多' if sig.side == 'long' else '做空'} · 置信度 {sig.confidence:.1%}"
                    )
                    cols = [c for c in ["代码", "涨幅%", "量比", "近8次胜率"] if c in sig.tickers.columns]
                    st.dataframe(sig.tickers[cols], use_container_width=True, hide_index=True)
                else:
                    st.info("今日暂无满足条件的每周高置信信号。")
                return

            filt = filters_for_mode(mode, top_n=GAINER_TOP_N if mode == "legacy" else 2)
            res = backtest_daily_gainer_portfolio(
                data, spy, start=start_d, end=end_d, filt=filt, fee_bps=fee_bps,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ 回测失败：{e}")
            return

    if res.get("error"):
        st.error(f"❌ {res['error']}")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("累计收益", fmt_pct(res["累计收益率"]))
    m2.metric("年化收益", fmt_pct(res["年化收益率"]))
    m3.metric("日胜率", fmt_pct(res["日胜率"]))
    m4.metric("夏普比率", fmt_num(res["夏普比率"]))
    m5.metric("最大回撤", fmt_pct(res["最大回撤"]))
    st.caption(
        f"**{GAINER_MODE_LABELS[mode]}** · 交易 {res['交易天数']} 天 · "
        f"期末权益 ${res['期末权益']:,.0f}"
    )

    eq = res.get("权益曲线", pd.DataFrame())
    if eq is not None and not eq.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq["日期"], y=eq["权益"], name="策略权益",
            line=dict(color=theme.ORANGE, width=2),
        ))
        fig.update_layout(
            height=360, template="tiger",
            title=f"{GAINER_MODE_LABELS[mode]} · 权益曲线",
            margin=dict(l=10, r=10, t=40, b=10), yaxis_title="权益 (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)

    picks = res.get("选股明细", pd.DataFrame())
    if picks is not None and not picks.empty:
        st.markdown("**最近选股明细**")
        pdisp = picks.tail(20).iloc[::-1].copy()
        for c in ["涨幅%", "量比", "综合分", "次日收益%"]:
            if c in pdisp.columns:
                pdisp[c] = pd.to_numeric(pdisp[c], errors="coerce").map(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
                )
        show = [c for c in ["选股日期", "代码", "涨幅%", "量比", "次日收益%", "选股理由"] if c in pdisp.columns]
        st.dataframe(pdisp[show], use_container_width=True, hide_index=True)
        csv = picks.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 导出选股明细 (CSV)", csv,
            file_name=f"gainer_{mode}_picks.csv", mime="text/csv", key="dl_gainer_picks",
        )


def tab_screener(cfg: dict) -> None:
    st.subheader("策略选股 · 条件筛选 + 批量回测")
    st.caption(
        "高级选股入口。日常选股与 **近5年回测** 请用顶部 **「每日选股」** 标签；"
        "此处提供涨幅榜因子、命名策略库、历史回放与自定义条件筛选。"
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
    _tab_gainer_pro(cfg)

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
# 标签页：缓跌收租 · 闪迪类高波股期权（熊市 Call 价差 / 备兑）
# ---------------------------------------------------------------------------
def tab_decline_income(cfg: dict) -> None:
    st.subheader("稳定收租 · 闪迪类高波股期权（SNDK / WDC）")
    st.caption(
        "经 SNDK/WDC/MU 横向回测：在**强势上涨 + 高 IV** 的闪迪类股上，最稳定的是 **顺势卖下方 Put 的现金担保认沽(CSP)** ——"
        "胜率最高、月度波动最小、回撤最浅；**逆势卖 Call 价差 / 铁鹰是灾难**（回撤 −90%+）。"
        "加「50 日均线过滤 + 50% 权利金止盈」后稳定性再上一个台阶。权利金为 BS+VRP 估算。"
    )

    _tab_csp_income(cfg)
    st.divider()
    _tab_weekly_soup(cfg)
    st.divider()
    _tab_calendar_signal(cfg)
    st.divider()
    with st.expander("🛠 进阶：熊市认购价差 / 备兑扫描（仅适合『确认转弱、缓跌』时）", expanded=False):
        _tab_bear_call_legacy(cfg)


def _tab_csp_income(cfg: dict) -> None:
    st.markdown("### 🏆 稳定收租首选：现金担保认沽 (CSP)")
    cc1, cc2, cc3 = st.columns(3)
    csp_focus = cc1.text_input("分析代码", value=cfg.get("ticker", "SNDK"), key="csp_focus")
    csp_delta = cc2.select_slider("卖出 Delta（越低越稳）", options=[0.10, 0.15, 0.20, 0.25, 0.30],
                                  value=0.20, key="csp_delta")
    csp_dte = cc3.slider("到期天数", 21, 45, 35, 7, key="csp_dte")

    if st.button("📈 生成 CSP 稳定收租方案", type="primary", key="run_csp"):
        end = cfg["end"]
        start = (pd.Timestamp(end) - pd.DateOffset(years=8)).strftime("%Y-%m-%d")
        with st.spinner("回测 + 生成方案中…"):
            out: dict = {}
            try:
                df = fetch_history(csp_focus.strip().upper(), start=start, end=end)
                out["plan"] = decline_income.csp_income_plan(
                    csp_focus.strip().upper(), df, delta=float(csp_delta), dte_days=int(csp_dte),
                )
                out["compare"] = decline_income.compare_income_strategies(df["Close"])
            except DataError as e:
                out["err"] = str(e)
            st.session_state["csp_payload"] = out

    payload = st.session_state.get("csp_payload")
    if not payload:
        st.info("输入闪迪/存储类代码（SNDK、WDC、MU…）后点击上方按钮，获取数据驱动的稳定收租方案。")
        return
    if payload.get("err"):
        st.error(f"❌ {payload['err']}")
        return

    plan = payload.get("plan")
    if plan:
        ok = plan.can_open
        color = theme.UP if ok else theme.GOLD
        st.markdown(
            f"<div style='padding:14px;border-radius:10px;border:1px solid {color}55;background:{color}15'>"
            f"<b>{plan.ticker} · 现金担保认沽 (CSP)</b> ｜ "
            f"现价 ${plan.close:,.2f} ｜ RV {plan.rv_pct:.0f}% ｜ "
            f"50日均线 ${plan.ma50:,.2f} → {'✅ 站上均线，可开仓' if ok else '❌ 跌破均线，先观望'}</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("卖 Put 行权", f"${plan.put_strike:,.2f}")
        m2.metric("收权利金", f"${plan.premium:.2f}/股")
        m3.metric("止盈买回(赚50%)", f"${plan.take_profit_price:.2f}")
        m4.metric("月化估算", f"{plan.monthly_yield_pct:.2f}%")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("担保金/张", f"${plan.capital_per_contract:,.0f}")
        n2.metric("保本价", f"${plan.breakeven:,.2f}")
        n3.metric("回测胜率", f"{plan.bt_win_rate:.0%}" if plan.bt_win_rate is not None else "-")
        n4.metric("回测最差单笔", f"{plan.bt_worst:+.1%}" if plan.bt_worst is not None else "-")
        if plan.bt_info_ratio is not None:
            st.caption(
                f"回测稳定性：信息比 {plan.bt_info_ratio:.2f} ｜ 合成回撤 "
                f"{plan.bt_max_dd:+.1%} ｜ 年化(近似) {plan.bt_annual:+.1%}"
                f"（含 50 日均线过滤 + 50% 止盈）"
            )
        st.markdown("**执行步骤**")
        for step in plan.playbook:
            st.markdown(f"- {step}")
        if plan.flags:
            st.warning("⚠ " + "；".join(plan.flags))

    cmp = payload.get("compare")
    if cmp is not None and not cmp.empty:
        st.markdown("#### 📊 5 种期权卖方策略稳定性横向对比（信息比降序）")
        show = cmp.copy()
        for c in ["胜率", "平均ROR", "标准差", "最差单笔", "年化", "合成回撤"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
        if "信息比" in show.columns:
            show["信息比"] = show["信息比"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "「信息比」= 平均单笔收益 / 收益标准差，越高越稳。CSP 担保金占用大（年化绝对值看着低），"
            "但胜率最高、回撤最浅，是真正『稳定睡得着』的选择；认沽信用价差(PCS)资金效率更高但回撤更大。"
        )


def _tab_weekly_soup(cfg: dict) -> None:
    st.markdown("### 🍲 周 PUT 价差「喝汤」扫描器")
    st.caption(
        "每周大量闪迪 PUT 归零 → 小资金用 **Put 信用价差** 喝一口：卖 OTM Put + 买更低 Put，"
        "保证金 = 价差宽 × 100（$10k 可开）。权利金 BS+VRP 估算；周期权 gamma 高，务必 50% 止盈。"
    )
    w1, w2, w3, w4 = st.columns(4)
    soup_ticker = w1.text_input("代码", value="SNDK", key="soup_ticker")
    soup_account = w2.number_input("账户资金 ($)", min_value=1000, value=10000, step=1000, key="soup_acct")
    soup_delta = w3.select_slider("卖 Put Delta", options=[0.10, 0.15, 0.20], value=0.10, key="soup_delta")
    soup_width = w4.selectbox("价差宽度 ($)", [25, 50, 100], index=0, key="soup_width")

    if st.button("📅 生成本周喝汤方案", type="primary", key="run_soup"):
        end = cfg["end"]
        start = (pd.Timestamp(end) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        with st.spinner("计算本周方案…"):
            out: dict = {}
            try:
                df = fetch_history(soup_ticker.strip().upper(), start=start, end=end)
                tk = soup_ticker.strip().upper()
                out["plan"] = decline_income.weekly_put_soup_plan(
                    tk, df,
                    account_size=float(soup_account),
                    short_delta=float(soup_delta),
                    width=float(soup_width),
                )
                if out["plan"]:
                    S = out["plan"].close
                    rv = out["plan"].rv_pct / 100
                    out["scan"] = decline_income.scan_weekly_soup_configs(
                        S, rv, account_size=float(soup_account),
                    )
            except DataError as e:
                out["err"] = str(e)
            st.session_state["soup_payload"] = out

    payload = st.session_state.get("soup_payload")
    if not payload:
        st.info("默认 SNDK + $10k + Δ0.10 + 宽$25，点击生成本周具体行权价与止盈目标。")
        return
    if payload.get("err"):
        st.error(f"❌ {payload['err']}")
        return

    plan = payload.get("plan")
    if plan:
        ok = plan.can_open
        color = theme.UP if ok else theme.GOLD
        st.markdown(
            f"<div style='padding:14px;border-radius:10px;border:1px solid {color}55;background:{color}15'>"
            f"<b>{plan.ticker} · 本周 PUT 价差</b> ｜ "
            f"现价 ${plan.close:,.2f} ｜ IV {plan.iv_pct:.0f}% ｜ "
            f"{'✅ 本周可喝汤' if ok else '❌ 本周暂停'}</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("卖 Put", f"${plan.short_strike:,.0f}")
        m2.metric("买 Put", f"${plan.long_strike:,.0f}")
        m3.metric("收租/张", f"${plan.credit_per_contract:,.0f}")
        m4.metric("归零概率", f"{plan.zero_prob:.0%}")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("保证金/张", f"${plan.margin_per_contract:,.0f}")
        n2.metric("止盈买回", f"${plan.take_profit_price:.2f}")
        n3.metric("建议张数", f"{plan.max_contracts or 1}")
        n4.metric("周 ROI", f"{plan.weekly_roi_pct:.1f}%")
        st.caption(
            f"顺的一周（归零）约 +${plan.weekly_profit_if_zero:,.0f} ｜ "
            f"最坏 -${plan.weekly_loss_if_max:,.0f} ｜ "
            f"卖腿距现价 -{plan.otm_pct:.0f}% ｜ 7日1σ ±{plan.one_std_move_pct:.0f}%"
        )
        st.markdown("**本周执行步骤**")
        for step in plan.playbook:
            st.markdown(f"- {step}")
        if plan.flags:
            st.warning("⚠ " + "；".join(plan.flags))

    scan = payload.get("scan")
    if scan is not None and not scan.empty:
        st.markdown("#### 📊 Delta × 价差宽度 横向对比（归零概率降序）")
        show = scan.copy()
        show["归零概率"] = show["归零概率"].map(lambda x: f"{x:.0%}")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Δ 越低、价差越窄 → 归零概率越高、收租越少但越稳。$10k 建议 Δ0.10 宽$25 只开 1 张。")


def _tab_calendar_signal(cfg: dict) -> None:
    st.markdown("### 📅 双日历价差 · IV Rank 择时")
    st.caption(
        "卖 14 天 / 买 21 天双日历，持有约 7 自然日平仓。**只在 IV Rank ≤ 40% 且无财报风险时**才提示可开；"
        "long vega 结构最怕 IV 崩塌，高位不开。"
    )
    c1, c2, c3, c4 = st.columns(4)
    cal_account = c1.number_input("账户 ($)", min_value=1000, value=10000, step=1000, key="cal_acct")
    cal_iv_max = c2.slider("IV Rank 上限", 0.20, 0.60, 0.40, 0.05, key="cal_iv_max",
                           help="越低越保守；回测显示 IV 高位开仓 IV crush 风险最大")
    cal_er = c3.slider("效率比 ER 上限", 0.25, 0.60, 0.45, 0.05, key="cal_er",
                       help="ER 低=横盘，适合日历；ER 高=单边趋势")
    cal_pool = c4.text_input("扫描代码", value="NVDA,PLTR,AMD,META,SNDK,QQQ", key="cal_pool")

    if st.button("📅 扫描今日双日历信号", type="primary", key="run_calendar"):
        end = cfg["end"]
        start = (pd.Timestamp(end) - pd.DateOffset(days=400)).strftime("%Y-%m-%d")
        tickers = parse_tickers(cal_pool)
        with st.spinner("扫描 IV Rank + 财报 + 双日历成本…"):
            try:
                plans, errors = scan_calendar_plans(
                    tickers, start, end,
                    account_size=float(cal_account),
                    iv_pct_max=float(cal_iv_max),
                    max_er=float(cal_er),
                )
                st.session_state["calendar_payload"] = {"plans": plans, "errors": errors}
            except Exception as e:  # noqa: BLE001
                st.session_state["calendar_payload"] = {"plans": [], "errors": [str(e)]}

    payload = st.session_state.get("calendar_payload")
    if not payload:
        st.info("默认扫描 NVDA/PLTR/AMD 等；只有 IV 处于低位且无财报时才显示 ✅ 可开。")
        return

    plans = payload.get("plans") or []
    errors = payload.get("errors") or []
    if errors and not plans:
        st.error("；".join(errors[:3]))
        return

    open_plans = [p for p in plans if p.can_open]
    if open_plans:
        st.success(f"✅ 今日 {len(open_plans)} 只可开（IV Rank ≤ {cal_iv_max:.0%}）")
    else:
        st.warning("⏸ 今日无可开仓 — IV 偏高 / 财报临近 / 趋势过强 / 成本过高")

    rows = []
    for p in plans:
        rows.append({
            "代码": p.ticker,
            "可开": "✅" if p.can_open else "❌",
            "现价": round(p.close, 2),
            "IV Rank": f"{p.iv_rank:.0%}",
            "效率比": round(p.er, 2),
            "净付$/张": round(p.debit_per_contract, 0),
            "7日θ$": round(p.theta_est_contract, 0),
            "Call": round(p.call_strike, 0),
            "Put": round(p.put_strike, 0),
            "原因": p.flags[0] if p.flags else "",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if open_plans:
        p = open_plans[0]
        st.markdown("**首选执行步骤**")
        for step in p.playbook:
            st.markdown(f"- {step}")

    st.caption(
        "想每天收盘自动跑并弹通知：双击项目里的「日历价差_开启定时.command」。"
        " 纪律：IV 高位/财报周不开；赚到约 50% 时间价值可提前平。"
    )


def _tab_bear_call_legacy(cfg: dict) -> None:
    st.caption(
        "⚠️ 仅在标的**确认转弱、缓跌/磨顶**时使用熊市认购价差；强势上涨股用此策略会被打穿（回测已验证）。"
    )
    c1, c2, c3 = st.columns(3)
    focus = c1.text_input("重点分析代码", value=cfg.get("ticker", "SNDK"), key="di_focus")
    owns = c2.checkbox("我已持有该标的（推荐备兑）", value=False, key="di_owns")
    spread_w = c3.slider("价差宽度 %", 3, 12, 5, 1, key="di_width") / 100.0

    pool_mode = st.radio(
        "扫描范围", ["闪迪/存储同类", "自定义列表", "默认高波池"],
        horizontal=True, key="di_pool",
    )
    if pool_mode == "自定义列表":
        custom = st.text_input("代码", value="SNDK,WDC,MU,STX", key="di_custom")
        tickers = parse_tickers(custom)
    elif pool_mode == "闪迪/存储同类":
        tickers = ["SNDK", "WDC", "MU", "STX"]
    else:
        tickers = list(decline_income.DECLINE_INCOME_UNIVERSE)

    if st.button("🔍 分析缓跌收租方案", type="primary", key="run_di"):
        end = cfg["end"]
        start = (pd.Timestamp(end) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        filt = decline_income.DeclineFilters(spread_width_pct=float(spread_w))
        with st.spinner("正在分析…"):
            payload: dict = {"plan": None, "table": pd.DataFrame()}
            try:
                payload["plan"] = decline_income.analyze_single_for_ui(
                    focus.strip().upper(), start, end, owns_shares=owns,
                )
            except DataError as e:
                payload["focus_err"] = str(e)
            try:
                payload["table"] = decline_income.scan_decline_income(
                    tickers[:15], start, end, filt, owns_shares=owns,
                )
            except DataError as e:
                payload["scan_err"] = str(e)
            st.session_state["di_payload"] = payload

    payload = st.session_state.get("di_payload")
    if not payload:
        with st.expander("📖 为什么缓跌要用「熊市认购价差」而不是卖 Put？", expanded=True):
            st.markdown(
                "**卖 Put（CSP）** 适合「愿意低价接货、长期看涨」— 股价慢慢跌会持续逼近行权价，"
                "容易被接货后继续亏。\n\n"
                "**熊市认购价差** 卖 **上方** 的 Call：股价跌或横盘 → 权利金全落袋；"
                "涨破卖出行权才有亏损，且 **最大亏损封顶**（价差宽度 − 权利金）。\n\n"
                "**闪迪 SNDK** 波动极高、历史短 — 仓位减半、价差宜窄（5%）、"
                "财报/存储周期消息前不开仓。"
            )
        return

    plan = payload.get("plan")
    if payload.get("focus_err"):
        st.error(f"❌ {focus}：{payload['focus_err']}")
    elif plan:
        st.markdown(f"### 📌 {plan.ticker} 专属方案")
        color = theme.UP if "缓跌" in plan.trend_label else theme.GOLD
        st.markdown(
            f"<div style='padding:14px;border-radius:10px;border:1px solid {color}55;"
            f"background:{color}15'>"
            f"<b>{plan.primary_strategy}</b> · {plan.trend_label}<br/>"
            f"现价 ${plan.close:,.2f} ｜ 60日 {plan.ret_60d_pct:+.1f}% ｜ RV {plan.rv20_pct:.0f}%</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("卖 Call 行权", f"${plan.short_call_strike:,.2f}")
        if plan.long_call_strike:
            m2.metric("买 Call 行权", f"${plan.long_call_strike:,.2f}")
        m3.metric("净收权利金", f"${plan.net_credit:.2f}/股")
        m4.metric("月化估算", f"{plan.monthly_yield_pct:.2f}%")
        if plan.bt_win_rate is not None:
            b1, b2, b3 = st.columns(3)
            b1.metric("历史胜率(近似)", f"{plan.bt_win_rate:.0%}")
            b2.metric("回测年化(近似)", f"{plan.bt_annual:+.1%}" if plan.bt_annual else "-")
            b3.metric("最差单周期", f"{plan.bt_worst_cycle:+.1%}" if plan.bt_worst_cycle else "-")
        st.markdown("**执行步骤**")
        for step in plan.playbook:
            st.markdown(f"- {step}")
        if plan.flags:
            st.warning("⚠ " + "；".join(plan.flags))

    table = payload.get("table", pd.DataFrame())
    if payload.get("scan_err"):
        st.error(f"❌ 扫描：{payload['scan_err']}")
    if table is not None and not table.empty:
        st.divider()
        st.markdown("### 🧺 同类标的扫描排名")
        show = table.copy()
        for c in ["最新价", "卖Call", "买Call", "净权利金", "最大亏损"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and c != "最新价" else (
                    f"${x:,.2f}" if c == "最新价" and isinstance(x, (int, float)) else x))
        if "回测胜率" in show.columns:
            show["回测胜率"] = show["回测胜率"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
        if "回测年化" in show.columns:
            show["回测年化"] = show["回测年化"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
        if "最差周期" in show.columns:
            show["最差周期"] = show["最差周期"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
        st.dataframe(show, use_container_width=True, hide_index=True)
        csv = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载方案表 (CSV)", csv, file_name="decline_income.csv",
                           mime="text/csv", key="di_csv")

    st.caption(
        "⚠️ 暴涨日（单日 +8%）Call 价差仍会亏；SNDK 样本短、IV 常偏高。"
        "稳定 ≠ 无亏，务必小仓 + 50% 权利金止盈 + 财报回避。"
    )


# ---------------------------------------------------------------------------
# 标签页：波动率衰减（VRP）· 反向 ETF 择时 + CSP 扫描
# ---------------------------------------------------------------------------
def _tab_vol_decay_putspread(cfg: dict) -> None:
    with st.expander("🛡️ 期权吃 vol 衰减（看跌价差 / 保护性做空）· 多结构回测", expanded=False):
        st.caption(
            "做多波动率 ETF 长期 contango 衰减 → 看空有结构优势，但裸空会被 spike 爆仓。"
            "用**定义风险期权结构**吃衰减、剪掉尾部。期权按已实现波动×IV倍数近似定价（真实 IV 更高 → 更保守）。"
        )
        c1, c2, c3, c4 = st.columns(4)
        tickers_raw = c1.text_input("标的（逗号分隔）", value="UVIX, UVXY, VXX", key="vdps_tk")
        hold = c2.selectbox("滚动周期(交易日)", [5, 10, 21, 42], index=2, key="vdps_hold")
        lower_otm = c3.slider("价差下腿 OTM%", 20, 60, 40, 5, key="vdps_lotm") / 100.0
        iv_mult = c4.slider("IV/已实现 倍数", 1.0, 2.0, 1.3, 0.1, key="vdps_iv",
                            help="真实 IV 通常高于已实现波动，倍数越高=保护越贵=越保守")
        start_v = (pd.Timestamp(cfg["end"]) - pd.DateOffset(years=4)).strftime("%Y-%m-%d")

        if not st.button("📊 回测期权保护版做空 vol", key="run_vdps"):
            return
        with st.spinner("拉取 UVIX/UVXY/VXX 并回测多结构…"):
            try:
                vcfg = PutSpreadConfig(hold=int(hold), lower_otm=float(lower_otm), iv_mult=float(iv_mult))
                vdf = vol_decay_compare_structures(
                    parse_tickers(tickers_raw), start=start_v, end=cfg["end"], cfg=vcfg,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ 回测失败：{e}")
                return
        if vdf.empty:
            st.warning("无结果（标的数据拉取失败？）。")
            return
        disp = vdf.copy()
        pct_cols = ["总收益", "CAGR", "最大回撤", "胜率", "最差月"] + [c for c in disp.columns if c.isdigit()]
        for c in pct_cols:
            disp[c] = disp[c].map(lambda x: f"{x*100:+.0f}%" if pd.notna(x) else "-")
        disp["夏普"] = disp["夏普"].map(lambda x: f"{x:.2f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)
        csv = vdf.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载结果 (CSV)", csv, file_name="vol_decay_putspread.csv",
                           mime="text/csv", key="dl_vdps")
        st.caption(
            "⚠️ 期权近似定价、未计点差/佣金；真实 IV 更高会进一步压低买保护类收益。"
            "定义风险结构**永不爆仓**，但保护成本吃掉部分衰减利润——不存在『年年大赚且无回撤』。仅供研究。"
        )


def tab_vol_decay(cfg: dict) -> None:
    st.subheader("波动率衰减 · 穿越牛熊双策略")
    st.caption(
        "策略一：反向波动率 ETF（SVIX/SVXY）+ 均线择时，吃波动率结构性衰减。"
        "策略二：高成交额个股卖认沽(CSP) 吃 theta。"
        "两类策略在崩盘日同向，务必控制合计敞口 ≤ 30%。"
    )

    _tab_vol_decay_putspread(cfg)
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.selectbox(
        "反向波动率 ETF",
        list(vol_decay.INVERSE_VOL_ETFS.keys()),
        format_func=lambda k: f"{k} · {vol_decay.INVERSE_VOL_ETFS[k]}",
        key="vrp_etf",
    )
    c2.slider("均线窗口（日）", 20, 100, 50, 5, key="vrp_ma")
    csp_top_n = c3.slider("CSP 清单 Top N", 3, 15, 8, 1, key="vrp_topn")

    f1, f2, f3 = st.columns(3)
    f1.number_input("成交额下限 (百万USD)", 100.0, 3000.0, 500.0, 50.0, key="vrp_dvol")
    f2.slider("RV 下限 %", 10, 50, 30, 5, key="vrp_rv_lo")
    f3.slider("RV 上限 %", 40, 100, 70, 5, key="vrp_rv_hi")
    pool_mode = st.radio("CSP 股票池", ["默认高波池", "成交活跃榜", "自定义"],
                         horizontal=True, key="vrp_pool")
    if pool_mode == "自定义":
        st.text_input("自定义代码（逗号分隔）",
                      value=",".join(vol_decay.DEFAULT_CSP_UNIVERSE[:12]), key="vrp_custom")

    if st.button("🔄 刷新今日信号与扫描", type="primary", key="run_vrp"):
        end = cfg["end"]
        start_etf = (pd.Timestamp(end) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        start_csp = (pd.Timestamp(end) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
        etf_ticker = st.session_state.get("vrp_etf", "SVIX")
        ma_win = int(st.session_state.get("vrp_ma", 50))
        min_dvol = float(st.session_state.get("vrp_dvol", 500.0))
        rv_lo = float(st.session_state.get("vrp_rv_lo", 30))
        rv_hi = float(st.session_state.get("vrp_rv_hi", 70))
        pool_mode = st.session_state.get("vrp_pool", "默认高波池")
        custom_pool = st.session_state.get("vrp_custom", "")
        tickers = list(vol_decay.DEFAULT_CSP_UNIVERSE)
        if pool_mode == "成交活跃榜":
            try:
                snap = screener.fetch_yahoo_screen("most_actives", count=40)
                if not snap.empty:
                    tickers = snap["代码"].tolist()
            except Exception:  # noqa: BLE001
                pass
        elif pool_mode == "自定义":
            tickers = parse_tickers(custom_pool)
        with st.spinner("正在拉取 VIX、反向 ETF 与 CSP 候选…"):
            payload: dict = {"vix": vol_decay.vix_alert(end=end), "etf_sig": None,
                             "etf_stats": {}, "etf_chart_df": None, "csp_table": pd.DataFrame()}
            try:
                etf_df = fetch_history(etf_ticker, start=start_etf, end=end)
                payload["etf_sig"] = vol_decay.inverse_etf_signal(etf_df, etf_ticker, ma_window=ma_win)
                payload["etf_stats"] = vol_decay.ma_timing_backtest(etf_df["Close"], ma_window=ma_win)
                close = etf_df["Close"].astype(float)
                payload["etf_chart_df"] = pd.DataFrame({
                    "收盘": close, f"{ma_win}日均线": close.rolling(ma_win).mean(),
                })
            except DataError as e:
                payload["etf_err"] = str(e)
            csp_filters = vol_decay.CspFilters(
                min_dollar_vol_m=min_dvol, min_rv_pct=rv_lo, max_rv_pct=rv_hi,
            )
            try:
                payload["csp_table"] = vol_decay.scan_csp_candidates(tickers, start_csp, end, csp_filters)
            except DataError as e:
                payload["csp_err"] = str(e)
            st.session_state["vrp_payload"] = payload

    payload = st.session_state.get("vrp_payload")
    if not payload:
        st.info("调整参数后点击「刷新今日信号与扫描」。")
        with st.expander("📋 策略要点（来自历史回测）", expanded=True):
            st.markdown(
                "- **反向 ETF**：SVIX + 50 日均线 — 站上持有、跌破清仓；"
                "2011–2026 回测年化约 10–26%、最大回撤约 -48%（优于裸持有 -95%）。\n"
                "- **卖认沽 CSP**：选 RV 30–70%、成交额 > 5 亿、200 日均线上方的优质股；"
                "NVDA/AMD 类高波股模拟年化 12–18%、胜率 ~85%；**勿卖裸 call / 宽跨式**。\n"
                "- **CSP 权利金**为 BS+VRP 估算，实盘以券商报价为准。"
            )
        return

    vix_alert = payload.get("vix")
    etf_sig = payload.get("etf_sig")
    etf_stats = payload.get("etf_stats") or {}
    etf_chart_df = payload.get("etf_chart_df")
    csp_table = payload.get("csp_table", pd.DataFrame())
    if payload.get("etf_err"):
        st.error(f"❌ 反向 ETF 数据：{payload['etf_err']}")
    if payload.get("csp_err"):
        st.error(f"❌ CSP 扫描：{payload['csp_err']}")

    st.divider()
    st.markdown("### 📋 今日执行清单")
    steps = vol_decay.daily_playbook(etf_sig, vix_alert, csp_table, max_csp=int(csp_top_n))
    for s in steps:
        st.markdown(s)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### ① 反向波动率 ETF 择时")
        if vix_alert:
            v_color = {"🔴": theme.DOWN, "🟡": theme.GOLD, "🟢": theme.UP}.get(vix_alert.level[:2], theme.TEXT_SECONDARY)
            st.markdown(
                f"<div style='padding:12px;border-radius:10px;border:1px solid {v_color}55;"
                f"background:{v_color}18'><b>{vix_alert.level}</b> VIX {vix_alert.vix:.1f} "
                f"（20MA {vix_alert.vix_ma20:.1f}，日变 {vix_alert.daily_chg_pct:+.0%}）<br/>"
                f"{vix_alert.message}</div>",
                unsafe_allow_html=True,
            )
        if etf_sig:
            a_color = theme.UP if "持有" in etf_sig.action or "建仓" in etf_sig.action else theme.DOWN
            st.markdown(
                f"<div style='padding:12px;border-radius:10px;border:1px solid {a_color}55;"
                f"background:{a_color}18;margin-top:8px'><b>{etf_sig.action}</b> "
                f"{etf_sig.ticker} · {etf_sig.as_of}<br/>{etf_sig.detail}</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("最新价", f"${etf_sig.close:,.2f}")
            m2.metric(f"{etf_sig.ma_window}日均线", f"${etf_sig.ma:,.2f}")
            m3.metric("偏离均线", f"{etf_sig.pct_vs_ma:+.1%}")
            if etf_stats:
                st.caption("**历史参考（同区间，非承诺收益）**")
                tb = pd.DataFrame(etf_stats).T
                disp = tb.copy()
                for c in ["年化", "最大回撤", "总收益"]:
                    if c in disp.columns:
                        disp[c] = disp[c].map(lambda x: f"{x:+.1%}")
                if "夏普" in disp.columns:
                    disp["夏普"] = disp["夏普"].map(lambda x: f"{x:.2f}")
                st.dataframe(disp, use_container_width=True)
        if etf_chart_df is not None and not etf_chart_df.empty and etf_sig:
            mw = etf_sig.ma_window
            ma_col = f"{mw}日均线"
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=etf_chart_df.index, y=etf_chart_df["收盘"],
                                     name="收盘", line=dict(color=theme.ORANGE, width=2)))
            if ma_col in etf_chart_df.columns:
                fig.add_trace(go.Scatter(x=etf_chart_df.index, y=etf_chart_df[ma_col],
                                         name=f"{mw}MA", line=dict(color=theme.BLUE, width=1, dash="dash")))
            fig.update_layout(height=320, template="tiger", margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### ② 卖认沽 CSP 候选")
        st.caption("权利金 = BS × (1+VRP) 估算 · 行权价 ≈ Delta 0.25 · 到期约 35 天")
        if csp_table.empty:
            st.warning("无符合筛选条件的 CSP 候选，请放宽 RV/成交额或扩大股票池。")
        else:
            disp = csp_table.head(int(csp_top_n)).copy()
            show = disp.copy()
            show["最新价"] = show["最新价"].map(lambda x: f"${x:,.2f}")
            show["建议Put行权"] = show["建议Put行权"].map(lambda x: f"${x:,.2f}")
            show["估算权利金"] = show["估算权利金"].map(lambda x: f"${x:.2f}/股")
            show["月化收益%"] = show["月化收益%"].map(lambda x: f"{x:.2f}%")
            show["RV20%"] = show["RV20%"].map(lambda x: f"{x:.1f}%")
            show["成交额M"] = show["成交额M"].map(lambda x: f"${x:,.0f}M")
            st.dataframe(show, use_container_width=True, hide_index=True)
            top = disp.iloc[0]
            st.success(
                f"🏆 首选 **{top['代码']}**：月化估算 {top['月化收益%']:.2f}%，"
                f"建议卖 Put ${top['建议Put行权']:,.2f}（现 ${top['最新价']:,.2f}）"
            )
            csv = disp.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 下载 CSP 清单 (CSV)", csv, file_name="csp_candidates.csv",
                             mime="text/csv", key="vrp_csv")

    st.divider()
    with st.expander("📖 每单执行标准（CSP）", expanded=False):
        st.markdown(
            "| 项目 | 标准 |\n|---|---|\n"
            "| 到期 | 30–45 天 |\n"
            "| 行权价 | Delta 0.20–0.30（OTM 约 5–10%）|\n"
            "| 单票保证金 | ≤ 总资金 5% |\n"
            "| 止盈 | 权利金赚到 50–70% 平仓 |\n"
            "| 止损 | 浮亏达权利金 2 倍 → roll 或平仓 |\n"
            "| 接货后 | 转备兑卖 call（Wheel）|\n"
            "| 禁止 | 财报前 1 周新开仓；裸卖 call / 宽跨式 |"
        )
    st.caption(
        "⚠️ CSP 权利金为模型估算；反向 ETF 有路径依赖衰减。"
        "2018/2020 崩盘日两类策略同向巨亏，务必仓位上限 + VIX 熔断。"
    )


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


def _realchain_panel(cfg: dict) -> None:
    """真实期权链：只显示券商盘上真实挂牌的行权价/到期/bid-ask，并给出可成交结构建议。"""
    from quant.option_chain import (
        DEFAULT_MAX_SPREAD_PCT,
        DEFAULT_MIN_OI,
        _credit_spread_plan,
        _debit_spread_plan,
        pick_bear_call,
        pick_bear_put_debit,
        pick_csp,
        pick_put_credit,
    )

    with st.expander("📡 真实期权链（券商可对照，非模型估值）", expanded=False):
        st.caption(
            "数据源 yfinance，延迟约 15 分钟，但行权价/到期/持仓量真实。"
            "卖腿按 bid、买腿按 ask 保守定价；持仓量/价差不足的合约会被过滤为不可成交。"
        )
        c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
        sym = c0.text_input("标的代码", value=str(cfg.get("ticker", "SPY")).upper(),
                            key="rc_sym").strip().upper()
        min_dte = int(c1.number_input("最短到期(天)", min_value=0, value=2, step=1, key="rc_mindte"))
        max_dte = int(c2.number_input("最长到期(天)", min_value=1, value=45, step=1, key="rc_maxdte"))
        go = c3.button("↻ 拉取真实链", key="rc_fetch")
        if not go and not st.session_state.get("rc_loaded"):
            st.info("输入代码后点「拉取真实链」。没有可成交结构时会如实显示「观望」，不虚构行权价。")
            return
        if go:
            st.session_state["rc_loaded"] = True

        try:
            spot = float(load_data({**cfg, "ticker": sym}["ticker"], cfg["start"], cfg["end"])["Close"].iloc[-1])
        except Exception:  # noqa: BLE001
            try:
                import yfinance as yf
                spot = float(yf.Ticker(sym).history(period="1d")["Close"].iloc[-1])
            except Exception as e:  # noqa: BLE001
                st.error(f"取现价失败：{e}")
                return

        expiry, dte, calls, puts = load_real_option_chain(sym, min_dte, max_dte)
        if expiry is None:
            st.warning(f"{sym}：无可交易到期日（可能没有周/近月期权）。")
            return
        st.markdown(f"**{sym}** 现价 ${spot:,.2f} ｜ 最近可用到期 **{expiry}**（{dte} 天）")

        otm = st.slider("价外幅度 OTM%", 0, 30, 8, key="rc_otm") / 100
        width = st.slider("价差宽度 width%", 1, 30, 10, key="rc_width") / 100
        min_oi = int(st.number_input("最小持仓量(OI)", min_value=0, value=DEFAULT_MIN_OI, step=5, key="rc_oi"))
        max_sp = st.slider("最大买卖价差%", 10, 100, int(DEFAULT_MAX_SPREAD_PCT * 100), key="rc_sp") / 100

        struct = st.selectbox(
            "推荐可成交结构", ["卖看涨价差", "卖看跌价差", "买看跌价差(做空替代)", "现金担保卖Put (CSP)"],
            key="rc_struct",
        )
        account = float(cfg.get("account_size", 10_000))

        if struct == "卖看涨价差":
            s, l, why = pick_bear_call(calls, spot, otm=otm, width_pct=width, min_oi=min_oi, max_spread_pct=max_sp)
            if s is None:
                st.warning(f"观望：{why}（真实盘口无可成交结构）")
            else:
                plan = _credit_spread_plan(sym, "bear_call", expiry, dte, spot, s, l, account, 0.02)
                if plan.net_per_share <= 0:
                    st.warning("观望：真实净权利金≤0（价差太宽或盘口太差）。")
                else:
                    st.success(
                        f"{plan.legs_label()} @{expiry} · 净收 ${plan.net_per_contract:.0f}/张 · "
                        f"最大亏 ${plan.max_loss:.0f}/张 · 账户可开 {plan.contracts} 张（2%风控）"
                    )
        elif struct == "卖看跌价差":
            s, l, why = pick_put_credit(puts, spot, otm=otm, width_pct=width, min_oi=min_oi, max_spread_pct=max_sp)
            if s is None:
                st.warning(f"观望：{why}")
            else:
                plan = _credit_spread_plan(sym, "put_credit", expiry, dte, spot, s, l, account, 0.02)
                if plan.net_per_share <= 0:
                    st.warning("观望：真实净权利金≤0。")
                else:
                    st.success(
                        f"{plan.legs_label()} @{expiry} · 净收 ${plan.net_per_contract:.0f}/张 · "
                        f"最大亏 ${plan.max_loss:.0f}/张 · 账户可开 {plan.contracts} 张（2%风控）"
                    )
        elif struct == "买看跌价差(做空替代)":
            lg, sh, why = pick_bear_put_debit(puts, spot, otm=otm, width_pct=width, min_oi=min_oi, max_spread_pct=max_sp)
            if lg is None:
                st.warning(f"观望：{why}")
            else:
                plan = _debit_spread_plan(sym, "bear_put_debit", expiry, dte, spot, lg, sh, account, 0.02)
                st.success(
                    f"{plan.legs_label()} @{expiry} · 付 ${-plan.net_per_contract:.0f}/张 · "
                    f"最大亏 ${plan.max_loss:.0f}/张 · 最大盈 ${plan.max_profit:.0f}/张 · 可开 {plan.contracts} 张"
                )
        else:
            leg, why = pick_csp(puts, spot, otm=otm, min_oi=min_oi, max_spread_pct=max_sp)
            if leg is None:
                st.warning(f"观望：{why}")
            else:
                collateral = leg.strike * 100
                nc = int(account // collateral) if collateral > 0 else 0
                st.success(
                    f"{leg.label()} @{expiry} · 收 ${leg.bid * 100:.0f}/张 · "
                    f"占用 ${collateral:.0f}/张 · 账户可开 {nc} 张"
                )

        cols = ["strike", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        names = {"strike": "行权价", "bid": "买价", "ask": "卖价", "volume": "成交量",
                 "openInterest": "持仓量", "impliedVolatility": "IV"}
        ta, tp = st.columns(2)
        with ta:
            st.caption("Call 链（近 ATM）")
            cc = calls[[c for c in cols if c in calls.columns]].copy()
            cc = cc[(cc["strike"] >= spot * 0.85) & (cc["strike"] <= spot * 1.3)]
            if "impliedVolatility" in cc:
                cc["impliedVolatility"] = (cc["impliedVolatility"] * 100).round(0)
            st.dataframe(cc.rename(columns=names), use_container_width=True, hide_index=True)
        with tp:
            st.caption("Put 链（近 ATM）")
            pp = puts[[c for c in cols if c in puts.columns]].copy()
            pp = pp[(pp["strike"] >= spot * 0.7) & (pp["strike"] <= spot * 1.15)]
            if "impliedVolatility" in pp:
                pp["impliedVolatility"] = (pp["impliedVolatility"] * 100).round(0)
            st.dataframe(pp.rename(columns=names), use_container_width=True, hide_index=True)


def tab_options(cfg: dict) -> None:
    st.subheader("期权策略损益计算器")
    _realchain_panel(cfg)
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
def tab_strategy_search(cfg: dict) -> None:
    st.markdown("### 🔬 短线策略寻优")
    st.caption(
        "在你选定的候选池与时间段上，枚举『选股过滤 × 交易策略 × 参数 × 调仓周期 × 方向』组合，"
        "按**稳健性**（信息比/胜率/盈亏比）排序，并切分**样本内/样本外**抑制过拟合——"
        "只有样本外仍盈利的组合才标记『✅ 稳健通过』。"
    )

    pool_raw = st.text_area(
        "候选股票池（逗号分隔）",
        value=", ".join(strategy_search.DEFAULT_SHORT_TERM_POOL),
        height=80, key="ss_pool",
        help="建议用高流动性、波动充足的标的。池子越大越慢（Polygon 免费档限速）。",
    )
    c1, c2, c3, c4 = st.columns(4)
    start_d = c1.date_input(
        "起始日期", value=date.today() - timedelta(days=540),
        max_value=date.today(), key="ss_start",
    )
    end_d = c2.date_input(
        "结束日期", value=date.today(), max_value=date.today(), key="ss_end",
    )
    fwd = c3.number_input("评估窗口(交易日)", 5, 60, 20, 1, key="ss_fwd")
    include_short = c4.checkbox("含做空组合", value=True, key="ss_short")

    st.warning(
        "⚠️ 寻优需要逐只拉取行情并跑大量回测，Polygon 免费档限速下可能耗时数分钟，请耐心等待。",
        icon="⏳",
    )
    if not st.button("🔬 开始寻优", type="primary", key="run_ss"):
        return

    tickers = parse_tickers(pool_raw)
    if len(tickers) < 4:
        st.error("❌ 候选池至少需要 4 个标的。")
        return
    if start_d >= end_d:
        st.error("❌ 起始日期必须早于结束日期。")
        return

    data, failed = get_multi_data(
        tickers, {**cfg, "start": start_d.isoformat(), "end": end_d.isoformat()},
    )
    if failed:
        st.warning(f"部分标的拉取失败已忽略：{', '.join(failed[:10])}")
    if len(data) < 4:
        st.error("❌ 可用数据不足（成功标的 < 4），请放宽时间段或更换标的。")
        return

    bar = st.progress(0.0, text="正在寻优…")

    def _progress(k: int, total: int, combo) -> None:
        bar.progress(k / total, text=f"寻优中 {k}/{total}：{combo.idea}·{combo.trading_strategy}")

    try:
        table, results = strategy_search.search_short_term(
            data, forward_days=int(fwd), include_short=include_short, progress=_progress,
        )
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"❌ 寻优失败：{e}")
        return
    bar.empty()

    if table.empty:
        st.error("❌ 没有产生有效组合，请放宽时间段或更换标的。")
        return

    robust = table[table["稳健通过"] == "✅"]
    m1, m2, m3 = st.columns(3)
    m1.metric("评估组合数", len(table))
    m2.metric("稳健通过", len(robust))
    m3.metric("候选标的", len(data))

    st.markdown("**策略排行榜**（按 稳健通过 + 样本内评分）")
    show_cols = [c for c in table.columns if c != "_id"]
    st.dataframe(table[show_cols].head(20), use_container_width=True, hide_index=True)
    csv = table.drop(columns=["_id"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 导出完整排行榜 (CSV)", data=csv,
        file_name=f"短线寻优_{start_d}_{end_d}.csv", mime="text/csv", key="dl_ss",
    )

    if robust.empty:
        st.info(
            "⚠️ 本次没有组合通过样本外验证——说明该池/窗口下短线难有稳定 alpha，"
            "建议更换标的池、拉长评估窗口，或接受更高风险。"
        )
        return

    best_id = robust.iloc[0]["_id"]
    best = next((r["combo"] for r in results if r["combo"].id == best_id), None)
    if best is None:
        return
    ev = next(r for r in results if r["combo"].id == best_id)
    te = ev["test"]
    st.success(f"🏆 最优稳健组合：**{best.label}**")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("样本外胜率", f"{te.get('胜率', 0) * 100:.1f}%")
    b2.metric(f"样本外平均收益({int(fwd)}日)", f"{te.get('平均收益%', 0):.2f}%")
    b3.metric("样本外信息比", f"{te.get('信息比', 0):.3f}")
    b4.metric("样本外盈亏比", f"{te.get('盈亏比', 0):.2f}")
    st.caption(
        f"思路：{best.idea}｜交易策略：{best.trading_strategy}（{best.params}）｜"
        f"选股：近 {best.filters.lookback_days} 日涨幅 "
        f"[{best.filters.min_gain_pct:.0f}%, {best.filters.max_gain_pct:.0f}%]｜"
        f"每 {best.rebalance_days} 个交易日调仓｜{'可做空' if best.allow_short else '仅做多'}。"
        " 可在『推荐』页用同思路预设生成每日交易计划。"
    )
    st.caption("说明：稳健性基于方向调整后的单笔收益（信息比=均值/标准差）。样本外仍盈利才入选，但历史不代表未来，仅供研究。")


def tab_income_engine(cfg: dict) -> None:
    st.subheader("💰 每日收入引擎 · 大盘开关 + 三引擎")
    st.caption(
        "把验证过的有效零件拼成多空+期权稳定收入系统：①卖看涨价差（收入核心，胜率最高）"
        "②高胜率做多（仅牛市增厚）③卖看跌 CSP（收租/接货）。大盘 SPY>MA50 全开，<MA50 转防守。"
    )
    c1, c2, c3 = st.columns(3)
    account = c1.number_input("账户规模（美金）", min_value=1000.0, value=10000.0,
                              step=1000.0, key="ie_account")
    count = int(c2.number_input("榜单扫描数量", min_value=50, max_value=500, value=200,
                                step=50, key="ie_count"))
    top_n = int(c3.number_input("每引擎候选数", min_value=3, max_value=15, value=5,
                                step=1, key="ie_top"))
    st.info("实时拉取榜单 + 历史较慢，首次约 1–3 分钟，请耐心等待。")

    if not st.button("🚀 生成今日交易计划", type="primary", key="ie_run"):
        return

    with st.spinner("正在拉取实时榜单并计算期权参数…"):
        try:
            plan = build_income_plan(account=account, count=count, top_n=top_n)
        except Exception as e:  # noqa: BLE001
            st.error(f"扫描失败：{e}")
            return

    reg = plan["regime"]
    m1, m2, m3 = st.columns(3)
    m1.metric("大盘状态", "牛市 三引擎全开" if reg.bull else "弱市 主卖看涨")
    m2.metric("SPY", f"{reg.spy:.2f}")
    m3.metric("MA50", f"{reg.ma50:.2f}")
    if reg.bull:
        st.success("🟢 SPY 在 MA50 上方 → ①卖看涨价差 ②做多强势股 ③卖看跌CSP 全开")
    else:
        st.warning("🔴 SPY 在 MA50 下方 → 弱市模式：主开①卖看涨价差，②做多关闭，③CSP减量")

    st.markdown("#### 引擎① 卖看涨价差 · 收入核心")
    st.caption("振幅/涨幅榜 Top，卖 +OTM 看涨价差。**真实期权链定价**：盘上真实行权价+bid/ask，无可成交结构则标『观望』。")
    cs = plan["call_spreads"]
    if cs is None or cs.empty:
        st.write("今日无合适标的。")
    else:
        st.dataframe(cs, use_container_width=True, hide_index=True)

    if reg.bull:
        st.markdown("#### 引擎② 高胜率做多 · 牛市增厚")
        st.caption("涨幅榜强势股（趋势/形态过滤）持 1 日，约 80% 日胜率。")
        lg = plan.get("longs")
        if lg is None or lg.empty:
            st.write("今日无满足条件标的。")
        else:
            st.dataframe(lg, use_container_width=True, hide_index=True)

    st.markdown("#### 引擎③ 卖看跌 CSP · 稳定底仓")
    st.caption("价位适配账户、RV 适中的票，卖 put 收租；被指派则按接货成本拿货。")
    csp = plan["csp"]
    if csp is None or csp.empty:
        st.write("今日无价位适配标的。")
    else:
        st.dataframe(csp, use_container_width=True, hide_index=True)

    st.markdown(
        "---\n**纪律**：每笔风险 ≤ 账户 2% · 价差永不裸卖 · 50% 权利金止盈 · 分散 5 只 · 财报回避。  \n"
        "*卖Call/CSP 已用真实期权链（yfinance，延迟约15分钟）定价；下单前仍以券商实时盘口为准。胜率高 ≠ 无亏损。*"
    )
    st.caption("想每天收盘自动跑并弹通知：双击项目里的「收入引擎_开启定时.command」。")
    st.divider()
    st.markdown("#### 🦅 5×$10k 舰队 · mixed_balanced（真实期权链）")
    st.caption(
        "**账户1** SNDK Put价差（顺势，无Call）｜**账户2** SOFI CSP（廉价稳）｜"
        "**账户3–5** QQQ/SPY/IWM 月铁鹰。真实 bid/ask，延迟约15min。"
        " 回测口径：年化~8%、回撤~8%、胜率~96%。"
    )
    fleet_size = st.number_input(
        "单户规模 ($)", min_value=1000.0, value=10000.0, step=1000.0, key="fleet_acct_size",
    )
    fleet_cfg_path = Path(__file__).resolve().parent / "sndk_iron_config.json"
    if st.button("🦅 生成舰队方案", type="secondary", key="run_fleet_iron"):
        with st.spinner("拉取真实期权链（SNDK + ETF）…"):
            try:
                from sndk_iron_daily import load_config, run_fleet, fleet_to_dataframe, fleet_summary_metrics
                cfg_fleet = load_config(fleet_cfg_path)
                if cfg_fleet.get("fleet"):
                    cfg_fleet["fleet"]["account_size"] = float(fleet_size)
                result = run_fleet(cfg_fleet)
                st.session_state["fleet_iron"] = {
                    "result": result,
                    "df": fleet_to_dataframe(result),
                    "summary": fleet_summary_metrics(result),
                }
            except Exception as e:  # noqa: BLE001
                st.error(f"舰队扫描失败：{e}")

    fleet_payload = st.session_state.get("fleet_iron")
    if fleet_payload:
        summ = fleet_payload["summary"]
        fm1, fm2, fm3, fm4 = st.columns(4)
        fm1.metric("可开户数", f"{summ['open_count']}/{summ['total_accounts']}")
        fm2.metric("合计收租", f"${summ['total_credit']:,.0f}")
        fm3.metric("占用保证金", f"${summ['total_margin']:,.0f} ({summ['margin_pct']:.0%})")
        fm4.metric("现金纪律", f"{summ.get('cash_pct', 0):.0%}")
        df_fleet = fleet_payload["df"]
        if df_fleet is not None and not df_fleet.empty:
            st.dataframe(df_fleet, use_container_width=True, hide_index=True)
        errs = fleet_payload["result"].get("errors") or []
        if errs:
            for e in errs:
                st.warning(f"⚠ {e}")
        st.caption(
            "纪律：50%止盈 · SPY>MA50 · 财报前不开 · SNDK只卖Put · 每户留≥55%现金。"
            " 定时推送：双击「闪迪铁鹰_开启定时.command」。"
        )
    else:
        st.info("点击上方按钮，按 sndk_iron_config.json（mixed_balanced）生成 5 户方案。")

    st.divider()
    st.markdown("#### 🏆 每日策略排名 Top3")
    st.caption("汇总收入引擎/周铁鹰/日历/动量，按回测分+当日信号排出最优组合与仓位表。")
    prof = st.selectbox("风格", ["balanced", "income", "growth"],
                        format_func=lambda x: {"balanced": "均衡", "income": "稳定收租", "growth": "偏高收益"}[x],
                        key="strat_prof")
    if st.button("🏆 生成今日策略排名", key="run_strat_rank"):
        with st.spinner("扫描各引擎并排名…"):
            try:
                from research.strategy_ranker import evaluate_strategies, format_playbook
                sr = evaluate_strategies(account=float(account), profile=prof)
                st.session_state["strat_rank"] = sr
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    sr = st.session_state.get("strat_rank")
    if sr:
        for line in format_playbook(sr):
            st.markdown(line)
        pf = sr.get("portfolio") or []
        if pf:
            st.dataframe(pd.DataFrame(pf), use_container_width=True, hide_index=True)
        try:
            from research.holy_grail_search import format_summary_lines
            st.markdown("**圣杯距离**（年化100% / 回撤10% / 胜率80%）")
            for line in format_summary_lines():
                st.caption(line)
        except Exception:  # noqa: BLE001
            pass
        st.caption("定时推送：双击「策略排名_开启定时.command」。圣杯搜索：python research/holy_grail_search.py --mode quick")


def main() -> None:
    _render_brand_header()
    st.caption("数据来源可配置 Polygon / Alpaca / Yahoo · 自用研究工具，不构成投资建议")

    cfg = sidebar()
    tabs = st.tabs(
        ["体检", "推荐", "每日选股", "短线寻优", "前兆", "回测", "参数寻优", "对比", "组合",
         "信号", "验证", "模拟", "选股", "概率", "期权", "波动率", "缓跌收租", "收入引擎"]
    )
    with tabs[0]:
        tab_report(cfg)
    with tabs[1]:
        tab_recommend(cfg)
    with tabs[2]:
        tab_daily_screen(cfg)
    with tabs[3]:
        tab_strategy_search(cfg)
    with tabs[4]:
        tab_precursor(cfg)
    with tabs[5]:
        tab_single(cfg)
    with tabs[6]:
        tab_optimize(cfg)
    with tabs[7]:
        tab_compare(cfg)
    with tabs[8]:
        tab_portfolio(cfg)
    with tabs[9]:
        tab_signals(cfg)
    with tabs[10]:
        tab_validation(cfg)
    with tabs[11]:
        tab_paper(cfg)
    with tabs[12]:
        tab_screener(cfg)
    with tabs[13]:
        tab_probability(cfg)
    with tabs[14]:
        tab_options(cfg)
    with tabs[15]:
        tab_vol_decay(cfg)
    with tabs[16]:
        tab_decline_income(cfg)
    with tabs[17]:
        tab_income_engine(cfg)

    st.markdown(
        '<div class="ths-disclaimer">'
        '数据来源 Yahoo Finance · 仅供个人研究，不构成任何投资建议 · 投资有风险，入市需谨慎'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
