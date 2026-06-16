"""老虎证券风格 UI 主题 — 深色底 + 品牌橙 + 卡片化信息层级。

参考 Tiger Trade：深色护眼背景、橙色主操作、扁平卡片、数据优先排版。
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# 品牌色（接近老虎证券橙 #FF6900）
ORANGE = "#FF6900"
ORANGE_HOVER = "#FF8533"
ORANGE_DARK = "#E55A00"
ORANGE_DIM = "rgba(255,105,0,0.15)"

# 背景层级
BG_APP = "#0D0D0D"
BG_CARD = "#161616"
BG_ELEVATED = "#1F1F1F"
BG_SIDEBAR = "#121212"
BORDER = "#2A2A2A"
BORDER_LIGHT = "#333333"

# 文字
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#8E8E93"
TEXT_TERTIARY = "#636366"

# 行情色（美股：绿涨红跌，与老虎美股模式一致）
UP = "#00C087"
DOWN = "#FF4545"

# 辅助色
BLUE = "#4DA3FF"
GOLD = "#FFB020"
PURPLE = "#A78BFA"
MUTED = "#636366"  # 基准线、辅助曲线

PALETTE = [ORANGE, BLUE, UP, GOLD, PURPLE, "#FF6B9D", "#5AD8A6"]

# Plotly 全局模板
_tiger_layout = go.Layout(
    paper_bgcolor=BG_APP,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY, family="Inter, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif", size=12),
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

pio.templates["tiger"] = go.layout.Template(layout=_tiger_layout)
pio.templates.default = "tiger"


def fig_layout(fig, *, height: int = 480, title: str | None = None) -> go.Figure:
    """统一图表外观。"""
    fig.update_layout(
        height=height,
        template="tiger",
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY)) if title else None,
        hovermode="x unified",
    )
    return fig


def inject_css() -> str:
    """注入老虎证券风格全局 CSS。"""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    }}
    .stApp {{
        background: {BG_APP};
    }}
    .main .block-container {{
        padding-top: 0.8rem; max-width: 1200px;
        padding-left: 1.2rem; padding-right: 1.2rem;
    }}
    /* 隐藏 Streamlit 默认顶栏装饰 */
    header[data-testid="stHeader"] {{
        background: {BG_APP} !important;
        border-bottom: 1px solid {BORDER};
    }}
    /* 老虎顶栏 */
    .tiger-topbar {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .tiger-topbar .brand {{
        display: flex; align-items: center; gap: 12px;
    }}
    .tiger-topbar .brand-name {{
        font-size: 1.25rem; font-weight: 700; color: {TEXT_PRIMARY};
        letter-spacing: -0.02em;
    }}
    .tiger-topbar .brand-tag {{
        font-size: 0.75rem; color: {ORANGE};
        background: {ORANGE_DIM}; padding: 2px 8px; border-radius: 4px;
        font-weight: 600;
    }}
    .tiger-topbar .brand-sub {{
        font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 2px;
    }}
    /* 指标卡片 — 仿行情报价块 */
    div[data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 16px 10px;
        box-shadow: none;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: {ORANGE}44;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: none;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    div[data-testid="stMetricDelta"] {{
        font-weight: 600 !important;
    }}
    /* 标签页 — 老虎底部导航风格（顶部横向） */
    div[data-testid="stTabs"] div[role="tablist"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        overflow-x: auto;
        flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
    }}
    div[data-testid="stTabs"] button[role="tab"] {{
        background: transparent !important;
        color: {TEXT_SECONDARY} !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 8px 12px !important;
        white-space: nowrap;
    }}
    div[data-testid="stTabs"] button[role="tab"]:hover {{
        color: {TEXT_PRIMARY} !important;
        background: {BG_ELEVATED} !important;
    }}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {ORANGE} !important;
        background: {ORANGE_DIM} !important;
        font-weight: 600 !important;
        box-shadow: inset 0 -2px 0 {ORANGE};
    }}
    /* 主按钮 — 老虎橙 */
    .stButton > button[kind="primary"] {{
        background: {ORANGE} !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.2rem !important;
        box-shadow: none !important;
        transition: background 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {ORANGE_HOVER} !important;
    }}
    .stButton > button[kind="secondary"] {{
        border: 1px solid {BORDER_LIGHT} !important;
        border-radius: 8px !important;
        background: {BG_ELEVATED} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    /* 侧边栏 */
    section[data-testid="stSidebar"] {{
        background: {BG_SIDEBAR} !important;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {ORANGE} !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
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
        border-radius: 8px !important;
        color: {TEXT_PRIMARY} !important;
    }}
    /* 标题层级 */
    h2, h3 {{
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}
    h2 {{
        font-size: 1.05rem !important;
        border-left: 3px solid {ORANGE};
        padding-left: 10px;
        margin-top: 1.2rem !important;
    }}
    /* 信息块 */
    .stAlert, div[data-testid="stNotification"] {{
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
    }}
    div[data-testid="stExpander"] {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    /* 表格 */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}
    /* 输入控件 */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
        background: {BG_CARD} !important;
        border-color: {BORDER} !important;
        border-radius: 8px !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        color: {ORANGE} !important;
    }}
    /* 分隔线 */
    hr {{
        border-color: {BORDER} !important;
        margin: 1rem 0 !important;
    }}
    /* 说明文字 */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {TEXT_TERTIARY} !important;
        font-size: 0.78rem !important;
    }}
    /* 隐藏 Streamlit 默认页脚 */
    footer[data-testid="stFooter"] {{
        visibility: hidden;
    }}
    /* Radio / Select 选中态 */
    .stRadio label[data-baseweb="radio"] > div:first-child {{
        border-color: {BORDER_LIGHT} !important;
    }}
    div[data-baseweb="select"] > div {{
        background: {BG_CARD} !important;
        border-color: {BORDER} !important;
    }}
    /* 底部免责声明 */
    .tiger-disclaimer {{
        text-align: center; color: {TEXT_TERTIARY};
        font-size: 0.72rem; padding: 16px 0 8px;
        border-top: 1px solid {BORDER}; margin-top: 24px;
    }}
    @media (max-width: 640px) {{
        .main .block-container {{ padding: 0.5rem !important; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.1rem !important; }}
        div[data-testid="stTabs"] button[role="tab"] {{
            font-size: 0.75rem !important; padding: 6px 8px !important;
        }}
    }}
    </style>
    """
