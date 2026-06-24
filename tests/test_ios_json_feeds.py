"""验证 iOS 模块 JSON 快照均可解析且非空。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "ios" / "Resources"
MANIFEST = RES / "app_manifest.json"

REQUIRED = [
    "daily_pick_today.json",
    "gain15_daily_today.json",
    "surge_scan_today.json",
    "speculative_pool.json",
    "flow_daily_today.json",
    "ticker_pattern_today.json",
    "pattern_daily_today.json",
    "move_pattern_5d_today.json",
    "flow_strategy_today.json",
    "liquid_fleet_picks.json",
    "universal_playbook_today.json",
    "s8u_liquid_universe_backtest.json",
    "ticker_pattern_backtest.json",
    "flow_pattern_stats.json",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_bundled_json_exists_and_parses(name: str) -> None:
    path = RES / name
    assert path.exists(), f"missing ios/Resources/{name}"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    assert doc, f"{name} is empty object"


def test_manifest_json_paths_bundled() -> None:
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = []
    for feat in doc.get("features") or []:
        rel = str(feat.get("today_json") or "").strip()
        if not rel or rel.endswith(".csv"):
            continue
        name = rel.replace("research/", "")
        if not (RES / name).exists():
            missing.append(f"{feat.get('id')}: {name}")
    assert not missing, "bundled missing: " + ", ".join(missing)


def test_flow_daily_has_picks() -> None:
    doc = json.loads((RES / "flow_daily_today.json").read_text(encoding="utf-8"))
    picks = doc.get("picks") or []
    assert len(picks) >= 1


def test_ticker_pattern_has_picks() -> None:
    doc = json.loads((RES / "ticker_pattern_today.json").read_text(encoding="utf-8"))
    picks = doc.get("picks") or []
    assert len(picks) >= 1
