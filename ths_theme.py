"""同花顺风格 UI 主题 — 深色终端 + 品牌红 + 数据卡片层级。

参考同花顺专业版：深蓝灰底、经典红主色、红涨绿跌、Tab 下划线导航。
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# 品牌色（同花顺经典红 #E93030）
ACCENT = "#E93030"
ACCENT_HOVER = "#FF4444"
ACCENT_DARK = "#C41E1E"
ACCENT_DIM = "rgba(233,48,48,0.14)"
ORANGE = ACCENT  # 兼容旧引用

# 背景层级（深蓝灰终端感）
BG_APP = "#0F1219"
BG_CARD = "#1A1E28"
BG_ELEVATED = "#232833"
BG_SIDEBAR = "#141720"
BORDER = "#2E3340"
BORDER_LIGHT = "#3A4050"

# 文字
TEXT_PRIMARY = "#F0F1F5"
TEXT_SECONDARY = "#8B919E"
TEXT_TERTIARY = "#636878"

# 行情色（A 股惯例：红涨绿跌）
UP = "#E93030"
DOWN = "#00A854"

# 辅助色
BLUE = "#3B82F6"
GOLD = "#F5A623"
PURPLE = "#8B5CF6"
MUTED = "#636878"

PALETTE = [ACCENT, BLUE, GOLD, PURPLE, UP, "#FF6B9D", "#5AD8A6"]

_ths_layout = go.Layout(
    paper_bgcolor=BG_APP,
    plot_bgcolor=BG_CARD,
    font=dict(
        color=TEXT_PRIMARY,
        family="PingFang SC, Hiragino Sans GB, Microsoft YaHei, -apple-system, sans-serif",
        size=12,
    ),
    colorway=PALETTE,
    xaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(color=TEXT_SECONDARY),
    ),
    yaxis=dict(
        gridcolor=BORDER,
        linecolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(color=TEXT_SECONDARY),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY)),
    margin=dict(l=12, r=12, t=36, b=12),
)

pio.templates["ths"] = go.layout.Template(layout=_ths_layout)
pio.templates["tiger"] = pio.templates["ths"]  # 兼容旧模板名
pio.templates.default = "ths"


def fig_layout(fig, *, height: int = 480, title: str | None = None) -> go.Figure:
    """统一图表外观。"""
    fig.update_layout(
        height=height,
        template="ths",
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY)) if title else None,
        hovermode="x unified",
    )
    return fig


def inject_css() -> str:
    """注入同花顺风格全局 CSS。"""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'PingFang SC', 'Noto Sans SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    }}
    .stApp {{
        background: linear-gradient(180deg, {BG_APP} 0%, #0B0E14 100%);
    }}
    .main .block-container {{
        padding-top: 0.6rem; max-width: 1280px;
        padding-left: 1.4rem; padding-right: 1.4rem;
    }}
    header[data-testid="stHeader"] {{
        background: {BG_APP} !important;
        border-bottom: 1px solid {BORDER};
    }}
    /* 同花顺顶栏 */
    .ths-topbar {{
        background: linear-gradient(135deg, {BG_CARD} 0%, {BG_ELEVATED} 100%);
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0;
        margin-bottom: 14px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}
    .ths-topbar-accent {{
        height: 3px;
        background: linear-gradient(90deg, {ACCENT_DARK} 0%, {ACCENT} 50%, {ACCENT_HOVER} 100%);
    }}
    .ths-topbar-body {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 20px;
    }}
    .ths-topbar .brand {{
        display: flex; align-items: center; gap: 14px;
    }}
    .ths-topbar .brand-name {{
        font-size: 1.35rem; font-weight: 700; color: {TEXT_PRIMARY};
        letter-spacing: 0.02em;
    }}
    .ths-topbar .brand-name span {{
        color: {ACCENT};
    }}
    .ths-topbar .brand-tag {{
        font-size: 0.7rem; color: #fff;
        background: {ACCENT}; padding: 3px 10px; border-radius: 3px;
        font-weight: 600; letter-spacing: 0.05em;
    }}
    .ths-topbar .brand-sub {{
        font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 3px;
    }}
    .ths-topbar .market-strip {{
        display: flex; gap: 18px; align-items: center;
        font-size: 0.75rem; color: {TEXT_TERTIARY};
    }}
    .ths-topbar .market-strip b {{
        color: {TEXT_SECONDARY}; font-weight: 500;
    }}
    /* 指标卡片 — 同花顺报价块 */
    div[data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 6px;
        padding: 14px 16px 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: border-color 0.15s, box-shadow 0.15s;
    }}
    div[data-testid="stMetric"]:hover {{
        border-left-color: {ACCENT_HOVER};
        box-shadow: 0 4px 16px rgba(233,48,48,0.08);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.76rem !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums;
    }}
    div[data-testid="stMetricDelta"] {{
        font-weight: 600 !important;
        font-variant-numeric: tabular-nums;
    }}
    /* 标签页 — 同花顺下划线导航 */
    div[data-testid="stTabs"] div[role="tablist"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px 8px 0 0;
        padding: 0 6px;
        gap: 0;
        overflow-x: auto;
        flex-wrap: nowrap;
        border-bottom: 2px solid {BORDER};
        -webkit-overflow-scrolling: touch;
    }}
    div[data-testid="stTabs"] button[role="tab"] {{
        background: transparent !important;
        color: {TEXT_SECONDARY} !important;
        border: none !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
        padding: 10px 14px !important;
        white-space: nowrap;
        margin-bottom: -2px;
        transition: color 0.15s, border-color 0.15s;
    }}
    div[data-testid="stTabs"] button[role="tab"]:hover {{
        color: {TEXT_PRIMARY} !important;
        background: transparent !important;
    }}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {ACCENT} !important;
        background: transparent !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {ACCENT} !important;
        box-shadow: none;
    }}
    /* 主按钮 — 同花顺红 */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(180deg, {ACCENT_HOVER} 0%, {ACCENT} 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1.1rem !important;
        box-shadow: 0 2px 8px rgba(233,48,48,0.3) !important;
        transition: filter 0.15s, box-shadow 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.08);
        box-shadow: 0 4px 12px rgba(233,48,48,0.4) !important;
    }}
    .stButton > button[kind="secondary"] {{
        border: 1px solid {BORDER_LIGHT} !important;
        border-radius: 4px !important;
        background: {BG_ELEVATED} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    /* 侧边栏 */
    section[data-testid="stSidebar"] {{
        background: {BG_SIDEBAR} !important;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] > div {{
        background: {BG_SIDEBAR} !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {ACCENT} !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        border-left: 3px solid {ACCENT};
        padding-left: 8px;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {BORDER} !important;
    }}
    section[data-testid="stSidebar"] label {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.82rem !important;
    }}
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 4px !important;
        color: {TEXT_PRIMARY} !important;
    }}
    /* 标题层级 */
    h2, h3 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
    }}
    h2 {{
        font-size: 1.05rem !important;
        border-left: 3px solid {ACCENT};
        padding-left: 10px;
        margin-top: 1.2rem !important;
    }}
    /* 信息块 */
    .stAlert, div[data-testid="stNotification"] {{
        border-radius: 6px !important;
        border: 1px solid {BORDER} !important;
    }}
    div[data-testid="stExpander"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}
    div[data-testid="stExpander"] summary {{
        font-weight: 500;
    }}
    /* 表格 */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        overflow: hidden;
    }}
    /* 输入控件 */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
        background: {BG_CARD} !important;
        border-color: {BORDER} !important;
        border-radius: 4px !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        color: {ACCENT} !important;
    }}
    hr {{
        border-color: {BORDER} !important;
        margin: 1rem 0 !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {TEXT_TERTIARY} !important;
        font-size: 0.76rem !important;
    }}
    footer[data-testid="stFooter"] {{
        visibility: hidden;
    }}
    div[data-baseweb="select"] > div {{
        background: {BG_CARD} !important;
        border-color: {BORDER} !important;
    }}
    /* 底部免责声明 */
    .ths-disclaimer {{
        text-align: center; color: {TEXT_TERTIARY};
        font-size: 0.72rem; padding: 16px 0 8px;
        border-top: 1px solid {BORDER}; margin-top: 24px;
    }}
    /* 行情涨跌色辅助类 */
    .ths-up {{ color: {UP} !important; }}
    .ths-down {{ color: {DOWN} !important; }}
    @media (max-width: 640px) {{
        .main .block-container {{ padding: 0.5rem !important; }}
        .ths-topbar .market-strip {{ display: none; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.15rem !important; }}
        div[data-testid="stTabs"] button[role="tab"] {{
            font-size: 0.76rem !important; padding: 8px 10px !important;
        }}
    }}
    </style>
    """
