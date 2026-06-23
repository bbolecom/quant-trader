"""iOS / App 功能清单 · 同花顺式菜单结构。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from quant.strategy_catalog import collect_strategy_snapshots, strategy_registry

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "research" / "app_manifest.json"
IOS_JSON = ROOT / "ios" / "Resources" / "app_manifest.json"

# 同花顺式顶栏分类
THS_CATEGORIES: list[dict[str, str]] = [
    {"id": "hub", "name": "聚合", "icon": "star.circle.fill", "color": "#E93030"},
    {"id": "momentum", "name": "动量", "icon": "flame.fill", "color": "#FF6B00"},
    {"id": "flow", "name": "量价", "icon": "arrow.left.arrow.right", "color": "#3B82F6"},
    {"id": "pattern", "name": "规律", "icon": "sparkles", "color": "#A855F7"},
    {"id": "options", "name": "期权", "icon": "chart.line.uptrend.xyaxis", "color": "#10B981"},
    {"id": "composite", "name": "综合", "icon": "square.grid.3x3.fill", "color": "#6366F1"},
    {"id": "screen", "name": "筛选", "icon": "line.3.horizontal.decrease", "color": "#64748B"},
    {"id": "lab", "name": "实验室", "icon": "flask.fill", "color": "#78716C"},
    {"id": "terminal", "name": "终端", "icon": "chart.xyaxis.line", "color": "#0EA5E9"},
]

# registry 未收录的独立模块
EXTRA_FEATURES: list[dict[str, Any]] = [
    {
        "id": "short_fade",
        "name": "Meme弱市Put价差",
        "category": "动量",
        "ths_category": "momentum",
        "icon": "arrow.down.right",
        "script": "daily_pick.py",
        "config": "daily_pick_config.json",
        "description": "超涨回吐 · 弱市 Put 价差做空",
        "integrated_in_daily_pick": True,
        "daily_pick_module": "Meme路由",
        "view_type": "daily_pick_filter",
    },
    {
        "id": "income_engine",
        "name": "收入引擎三引擎",
        "category": "聚合",
        "ths_category": "hub",
        "icon": "dollarsign.circle",
        "script": "research/income_engine.py",
        "history_csv": "income_engine_history.csv",
        "description": "卖Call + 动量 + CSP 三引擎",
        "integrated_in_daily_pick": False,
        "launcher": "收入引擎_运行一次.command",
        "view_type": "terminal_only",
        "terminal_tab": "收入引擎",
    },
    {
        "id": "weekly_soup",
        "name": "闪迪喝汤·周铁鹰",
        "category": "期权",
        "ths_category": "options",
        "icon": "cup.and.saucer",
        "script": "weekly_soup.py",
        "config": "weekly_soup_config.json",
        "history_csv": "weekly_soup_history.csv",
        "description": "高波 Put 价差/铁鹰 MA50 过滤",
        "integrated_in_daily_pick": False,
        "launcher": "闪迪喝汤_运行一次.command",
        "view_type": "history_only",
    },
    {
        "id": "precursor",
        "name": "异动前兆扫描",
        "category": "筛选",
        "ths_category": "screen",
        "icon": "eye.trianglebadge.exclamationmark",
        "script": "app.py",
        "description": "量能/波动收缩/MACD 前兆（Streamlit）",
        "view_type": "terminal_only",
        "terminal_tab": "前兆",
    },
    {
        "id": "backtest_single",
        "name": "单策略回测",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "clock.arrow.circlepath",
        "script": "app.py",
        "description": "单标的策略回测与指标",
        "view_type": "terminal_only",
        "terminal_tab": "回测",
    },
    {
        "id": "backtest_optimize",
        "name": "参数寻优",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "slider.horizontal.3",
        "script": "app.py",
        "description": "网格寻优 · Walk-forward",
        "view_type": "terminal_only",
        "terminal_tab": "参数寻优",
    },
    {
        "id": "backtest_portfolio",
        "name": "组合回测",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "chart.pie",
        "script": "app.py",
        "description": "多标的组合净值回测",
        "view_type": "terminal_only",
        "terminal_tab": "组合",
    },
    {
        "id": "options_chain",
        "name": "期权链分析",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "link",
        "script": "app.py",
        "description": "真实期权链 · 结构分析",
        "view_type": "terminal_only",
        "terminal_tab": "期权",
    },
    {
        "id": "paper_trading",
        "name": "模拟盘",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "play.circle",
        "script": "app.py",
        "description": "Paper trading 模拟交易",
        "view_type": "terminal_only",
        "terminal_tab": "模拟",
    },
    {
        "id": "ticker_pattern_backtest",
        "name": "规律策略回测",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "function",
        "script": "research/ticker_pattern_backtest.py",
        "today_json": "research/ticker_pattern_backtest.json",
        "description": "单标的 S1~S8 规律回测",
        "launcher": "规律策略回测_运行一次.command",
        "view_type": "json_stats",
    },
    {
        "id": "flow_pattern_backtest",
        "name": "资金流向回测",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "arrow.triangle.branch",
        "script": "research/flow_pattern_backtest.py",
        "today_json": "research/flow_pattern_stats.json",
        "launcher": "资金流向回测_运行一次.command",
        "view_type": "json_stats",
    },
]

_CATEGORY_TO_THS = {
    "聚合": "hub",
    "动量": "momentum",
    "量价": "flow",
    "规律": "pattern",
    "期权收入": "options",
    "期权": "options",
    "综合": "composite",
    "筛选": "screen",
    "监控": "screen",
}

_VIEW_TYPE = {
    "daily_pick": "daily_pick",
    "gain15": "gain15",
    "surge_scan": "surge",
    "speculative_pool": "speculative_pool",
    "capital_flow": "generic_picks",
    "meme_pattern": "meme",
    "ticker_pattern": "meme",
    "flow_strategy": "generic_picks",
    "surge_scan": "surge",
    "speculative_pool": "speculative_pool",
}


def _view_type_for(entry_id: str, today_json: str) -> str:
    if entry_id in _VIEW_TYPE:
        return _VIEW_TYPE[entry_id]
    if "surge_scan" in today_json:
        return "surge"
    if "speculative_pool" in today_json:
        return "speculative_pool"
    if "gain15" in today_json:
        return "gain15"
    if "flow_daily" in today_json or "flow_strategy" in today_json:
        return "generic_picks"
    if "ticker_pattern" in today_json:
        return "meme"
    if "playbook" in today_json:
        return "playbook"
    if today_json.endswith(".json"):
        return "json_generic"
    return "history_only"


def _icon_for(entry_id: str, category: str) -> str:
    icons = {
        "daily_pick": "star.circle.fill",
        "gain15": "flame.fill",
        "capital_flow": "arrow.left.arrow.right",
        "meme_pattern": "sparkles",
        "fleet_csp": "ferry.fill",
        "bear_call": "phone.arrow.down.left",
        "pattern_daily": "triangle.fill",
        "flow_strategy": "chart.bar.fill",
        "trajectory_highwin": "chart.line.uptrend.xyaxis",
        "move_pattern_5d": "calendar",
        "s8u_liquid": "checkmark.seal.fill",
        "sndk_iron": "shield.fill",
        "strategy_rank": "list.number",
        "vrp": "waveform.path.ecg",
        "calendar": "calendar.badge.clock",
        "universal_playbook": "globe.americas.fill",
        "screen_daily": "line.3.horizontal.decrease.circle",
        "scan_daily": "dot.radiowaves.left.and.right",
        "ticker_pattern": "sparkles",
        "surge_scan": "bolt.fill",
        "speculative_pool": "airplane",
    }
    return icons.get(entry_id, "square.grid.2x2")


def build_feature_list(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or ROOT
    snap_map = {r["id"]: r for r in collect_strategy_snapshots(root)}
    seen: set[str] = set()
    features: list[dict[str, Any]] = []

    for s in strategy_registry():
        snap = snap_map.get(s.id, {})
        ths = _CATEGORY_TO_THS.get(s.category, "composite")
        feat = {
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "ths_category": ths,
            "icon": _icon_for(s.id, s.category),
            "script": s.script,
            "config": s.config or "",
            "today_json": s.today_json or "",
            "today_csv": s.today_csv or "",
            "history_csv": s.history_csv or "",
            "description": s.description,
            "integrated_in_daily_pick": s.integrated_in_daily_pick,
            "daily_pick_module": s.daily_pick_module or "",
            "launcher": s.launcher or "",
            "view_type": _view_type_for(s.id, s.today_json or ""),
            "actionable": snap.get("可开仓", 0),
            "watching": snap.get("观望", 0),
            "total": snap.get("总条目", 0),
            "has_data": snap.get("今日有数据", False),
            "data_date": snap.get("数据日期", "—"),
        }
        features.append(feat)
        seen.add(s.id)

    for extra in EXTRA_FEATURES:
        if extra["id"] in seen:
            continue
        snap = snap_map.get(extra["id"], {})
        feat = {**extra}
        feat.setdefault("actionable", snap.get("可开仓", 0))
        feat.setdefault("watching", snap.get("观望", 0))
        feat.setdefault("total", snap.get("总条目", 0))
        feat.setdefault("has_data", snap.get("今日有数据", False))
        feat.setdefault("data_date", snap.get("数据日期", "—"))
        features.append(feat)
        seen.add(extra["id"])

    return features


def build_app_manifest(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    features = build_feature_list(root)
    by_cat: dict[str, list[dict]] = {}
    for f in features:
        by_cat.setdefault(f["ths_category"], []).append(f)

    # 同花顺首页金刚区：高优先级快捷入口
    quick_ids = [
        "daily_pick", "gain15", "surge_scan", "speculative_pool",
        "capital_flow", "meme_pattern", "pattern_daily", "fleet_csp",
    ]
    quick_entries = [f for fid in quick_ids for f in features if f["id"] == fid]

    return {
        "version": "3.0",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "app_name": "美股量化",
        "tagline": "同花顺式 · 全策略原生入口",
        "categories": THS_CATEGORIES,
        "quick_entries": quick_entries,
        "features": features,
        "features_by_category": by_cat,
        "json_base_hint": "http://{host}:8502/",
        "streamlit_hint": "http://{host}:8501/",
        "total_features": len(features),
        "with_json_feed": sum(1 for f in features if f.get("today_json")),
    }


def export_app_manifest(root: Path | None = None) -> Path:
    root = root or ROOT
    doc = build_app_manifest(root)
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    IOS_JSON.parent.mkdir(parents=True, exist_ok=True)
    IOS_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return OUT_JSON


if __name__ == "__main__":
    p = export_app_manifest()
    print(f"→ {p}")
    print(f"→ {IOS_JSON}")
