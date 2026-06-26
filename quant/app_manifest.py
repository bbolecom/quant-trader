"""iOS / App 功能清单 · 三分类精简结构（期权策略 / 做多票 / 空多双杀）。"""

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

# 顶栏三分类（App 仅保留这三类）
THS_CATEGORIES: list[dict[str, str]] = [
    {"id": "options", "name": "期权策略", "icon": "chart.line.uptrend.xyaxis", "color": "#10B981"},
    {"id": "long", "name": "做多票", "icon": "flame.fill", "color": "#E93030"},
    {"id": "short", "name": "空多双杀", "icon": "arrow.up.arrow.down.circle.fill", "color": "#A855F7"},
]

# 额外 feature（非 daily_pick 核心策略，但需在三分类中展示）
LAB_FEATURES: list[dict[str, Any]] = [
    {
        "id": "short_squeeze",
        "name": "安飞士做空",
        "category": "空多双杀",
        "ths_category": "short",
        "icon": "arrow.up.arrow.down.circle.fill",
        "script": "short_squeeze_daily.py",
        "config": "blowoff_short_config.json",
        "today_json": "research/short_squeeze_today.json",
        "description": "安飞士/SPCE 式：暴涨乏力→破位大阴确认做空 · 合并过热分+投机池+涨幅榜做空",
        "integrated_in_daily_pick": False,
        "view_type": "json_generic",
        "launcher": "空多双杀_运行一次.command",
    },
]

_CATEGORY_TO_THS = {
    # 新三分类
    "期权策略": "options",
    "做多票": "long",
    "空多双杀": "short",
    # 兼容旧分类名（防止历史数据落到默认桶）
    "期权收入": "options",
    "期权": "options",
    "聚合": "hub",
    "动量": "long",
    "量价": "long",
    "规律": "long",
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
        "whipsaw_short": "arrow.down.right.circle.fill",
        "short_squeeze": "arrow.up.arrow.down.circle.fill",
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
        "daily_pick",
        # 首页主推（金刚区置顶）
        "gainer10", "short_squeeze",
        # ① 期权策略
        "bear_call", "fleet_csp", "sndk_iron", "vrp",
        # ② 做多票
        "capital_flow", "flow_strategy", "extreme20", "meme_long",
        # ③ 空多双杀
        "whipsaw_short",
    ]
    quick_entries = [f for fid in quick_ids for f in features if f["id"] == fid]
    core_features = [f for f in features if f.get("is_core") and f["id"] != "daily_pick"]

    audit = build_strategy_audit(root)

    return {
        "version": "4.0",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "app_name": "美股量化",
        "tagline": "三分类精选 · 期权策略 / 做多票 / 空多双杀",
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
