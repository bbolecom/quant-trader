"""iOS / App 功能清单 · 同花顺式菜单结构（核心 9 策略）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from quant.strategy_catalog import (
    CORE_STRATEGY_IDS,
    build_strategy_audit,
    collect_strategy_snapshots,
    strategy_registry,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "research" / "app_manifest.json"
IOS_JSON = ROOT / "ios" / "Resources" / "app_manifest.json"

# 同花顺式顶栏分类（仅保留有内容的分类）
THS_CATEGORIES: list[dict[str, str]] = [
    {"id": "hub", "name": "聚合", "icon": "star.circle.fill", "color": "#E93030"},
    {"id": "momentum", "name": "动量", "icon": "flame.fill", "color": "#FF6B00"},
    {"id": "flow", "name": "量价", "icon": "arrow.left.arrow.right", "color": "#3B82F6"},
    {"id": "pattern", "name": "规律", "icon": "sparkles", "color": "#A855F7"},
    {"id": "options", "name": "期权", "icon": "chart.line.uptrend.xyaxis", "color": "#10B981"},
    {"id": "lab", "name": "实验室", "icon": "flask.fill", "color": "#78716C"},
    {"id": "terminal", "name": "终端", "icon": "chart.xyaxis.line", "color": "#0EA5E9"},
]

# 量化终端工具（非 daily_pick 策略，仅供 Streamlit 跳转）
LAB_FEATURES: list[dict[str, Any]] = [
    {
        "id": "market_scan",
        "name": "全市场快扫",
        "category": "聚合",
        "ths_category": "hub",
        "icon": "dot.radiowaves.left.and.right",
        "script": "market_scan_fast.py",
        "config": "market_scan_config.json",
        "today_json": "research/market_scan_today.json",
        "description": "5 分钟内并行扫描全市场 · Yahoo 多榜 + 动量/Gainer10+ 信号",
        "integrated_in_daily_pick": False,
        "view_type": "json_generic",
        "launcher": "MarketScan_运行一次.command",
    },
    {
        "id": "longshort_combo",
        "name": "多空组合 · 高胜率",
        "category": "动量",
        "ths_category": "momentum",
        "icon": "arrow.up.arrow.down.circle.fill",
        "script": "longshort_combo_daily.py",
        "config": "longshort_combo_config.json",
        "today_json": "research/longshort_combo_today.json",
        "today_csv": "research/longshort_combo_today.csv",
        "history_csv": "longshort_combo_history.csv",
        "description": "Extreme20 L1/S1 + Flow U_S2/D_S2 · 质量分过滤 · 5年高胜率",
        "integrated_in_daily_pick": True,
        "daily_pick_module": "多空组合",
        "launcher": "LongShort_运行一次.command",
        "view_type": "json_generic",
        "win_rate": 0.59,
        "sharpe": 1.91,
    },
    {
        "id": "quantum_watch",
        "name": "量子板块盯盘",
        "category": "规律",
        "ths_category": "pattern",
        "icon": "atom",
        "script": "rgti_daily.py",
        "config": "rgti_config.json",
        "today_json": "research/quantum_watch_today.json",
        "description": "RGTI/IONQ/QUBT/QBTS 每日信号：恐慌反弹·NR7防跌·超买勿空",
        "integrated_in_daily_pick": False,
        "view_type": "json_generic",
    },
    {
        "id": "backtest_single",
        "name": "单策略回测",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "clock.arrow.circlepath",
        "script": "app.py",
        "description": "单标的策略回测与指标",
        "integrated_in_daily_pick": False,
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
        "integrated_in_daily_pick": False,
        "view_type": "terminal_only",
        "terminal_tab": "参数寻优",
    },
    {
        "id": "options_chain",
        "name": "期权链分析",
        "category": "实验室",
        "ths_category": "lab",
        "icon": "link",
        "script": "app.py",
        "description": "真实期权链 · 结构分析",
        "integrated_in_daily_pick": False,
        "view_type": "terminal_only",
        "terminal_tab": "期权",
    },
    {
        "id": "terminal_hub",
        "name": "量化策略终端",
        "category": "终端",
        "ths_category": "terminal",
        "icon": "chart.xyaxis.line",
        "script": "app.py",
        "description": "Streamlit 全功能 · 回测 · 体检 · 期权",
        "integrated_in_daily_pick": False,
        "view_type": "terminal_only",
        "terminal_tab": "回测",
    },
]

_CATEGORY_TO_THS = {
    "聚合": "hub",
    "动量": "momentum",
    "量价": "flow",
    "规律": "pattern",
    "期权收入": "options",
    "期权": "options",
    "实验室": "lab",
    "终端": "terminal",
}

_VIEW_TYPE = {
    "daily_pick": "daily_pick",
    "gain15": "gain15",
    "extreme20": "extreme20",
    "gainer10": "json_generic",
    "capital_flow": "generic_picks",
    "meme_long": "meme",
    "flow_strategy": "generic_picks",
}


def _view_type_for(entry_id: str, today_json: str) -> str:
    if entry_id in _VIEW_TYPE:
        return _VIEW_TYPE[entry_id]
    if "gain15" in today_json:
        return "gain15"
    if "flow_daily" in today_json or "flow_strategy" in today_json:
        return "generic_picks"
    if "ticker_pattern" in today_json:
        return "meme"
    if today_json.endswith(".json"):
        return "json_generic"
    return "history_only"


def _icon_for(entry_id: str, category: str) -> str:
    icons = {
        "daily_pick": "star.circle.fill",
        "gain15": "flame.fill",
        "extreme20": "bolt.circle.fill",
        "longshort_combo": "arrow.up.arrow.down.circle.fill",
        "gainer10": "bolt.fill",
        "capital_flow": "arrow.left.arrow.right",
        "meme_long": "sparkles",
        "fleet_csp": "ferry.fill",
        "bear_call": "phone.arrow.down.left",
        "flow_strategy": "chart.bar.fill",
        "sndk_iron": "shield.fill",
        "vrp": "waveform.path.ecg",
    }
    return icons.get(entry_id, "square.grid.2x2")


def build_feature_list(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or ROOT
    snap_map = {r["id"]: r for r in collect_strategy_snapshots(root)}
    audit = build_strategy_audit(root)
    audit_map = {r["id"]: r for r in audit.get("rows", [])}
    seen: set[str] = set()
    features: list[dict[str, Any]] = []

    for s in strategy_registry():
        snap = snap_map.get(s.id, {})
        ar = audit_map.get(s.id, {})
        ths = _CATEGORY_TO_THS.get(s.category, "hub")
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
            "trades": ar.get("trades"),
            "win_rate": ar.get("win_rate", s.win_rate),
            "ann_return": ar.get("ann_return"),
            "max_dd": ar.get("max_dd"),
            "sharpe": ar.get("sharpe", s.sharpe),
            "audit_rank": ar.get("audit_rank"),
            "audit_score": ar.get("audit_score"),
            "audit_tier": ar.get("audit_tier"),
            "audit_verdict": ar.get("audit_verdict"),
            "audit_action": ar.get("audit_action"),
            "is_core": s.id in CORE_STRATEGY_IDS or s.id == "daily_pick",
        }
        features.append(feat)
        seen.add(s.id)

    for extra in LAB_FEATURES:
        if extra["id"] in seen:
            continue
        feat = {**extra, "is_core": False}
        feat.setdefault("actionable", 0)
        feat.setdefault("watching", 0)
        feat.setdefault("total", 0)
        feat.setdefault("has_data", False)
        feat.setdefault("data_date", "—")
        features.append(feat)
        seen.add(extra["id"])

    return features


def build_app_manifest(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    features = build_feature_list(root)
    by_cat: dict[str, list[dict]] = {}
    for f in features:
        by_cat.setdefault(f["ths_category"], []).append(f)

    quick_ids = [
        "daily_pick", "market_scan", "longshort_combo", "capital_flow", "flow_strategy", "meme_long",
        "gain15", "extreme20", "gainer10", "bear_call", "fleet_csp", "sndk_iron",
    ]
    quick_entries = [f for fid in quick_ids for f in features if f["id"] == fid]
    core_features = [f for f in features if f.get("is_core") and f["id"] != "daily_pick"]

    audit = build_strategy_audit(root)

    return {
        "version": "4.0",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "app_name": "美股量化",
        "tagline": f"核心{len(CORE_STRATEGY_IDS)}策略 · 已审核排名",
        "core_strategy_ids": list(CORE_STRATEGY_IDS),
        "core_count": len(CORE_STRATEGY_IDS),
        "strategy_audit": audit,
        "categories": THS_CATEGORIES,
        "quick_entries": quick_entries,
        "core_features": core_features,
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
