#!/usr/bin/env python3
"""同步 research JSON → ios/Resources，并生成缺失的模块快照。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
RESOURCES = ROOT / "ios" / "Resources"
MANIFEST = RESOURCES / "app_manifest.json"

# manifest 未列但模块会用到的快照
EXTRA = [
    "pattern_daily_today.json",
    "move_pattern_5d_today.json",
]


def _load_manifest_paths() -> list[str]:
    if not MANIFEST.exists():
        return []
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for feat in doc.get("features") or []:
        p = str(feat.get("today_json") or "").strip()
        if p:
            paths.add(p)
    for p in EXTRA:
        paths.add(f"research/{p}")
    return sorted(paths)


def csv_to_json(csv_path: Path, json_path: Path) -> None:
    df = pd.read_csv(csv_path)
    doc = {
        "date": date.today().isoformat(),
        "source_csv": csv_path.name,
        "summary": {"总条目": len(df), "可开仓": 0, "观望": len(df)},
        "rows": df.to_dict(orient="records"),
        "picks": df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  csv→json {json_path.name} ({len(df)} rows)")


def export_pattern_daily(out: Path, *, quick: bool = True) -> None:
    import pattern_daily as pdaily

    cfg = pdaily.load_config(pdaily.DEFAULT_CONFIG)
    if quick:
        cfg["quick"] = True
    plan = pdaily.build_plan(cfg)

    def df_rows(key: str) -> list[dict]:
        df = plan.get(key)
        if df is None or getattr(df, "empty", True):
            return []
        return df.to_dict(orient="records")

    reg = plan.get("regime") or {}
    long_rows = df_rows("long")
    avoid_rows = df_rows("avoid")
    path_rows = df_rows("path5d")
    actionable = len(long_rows) + sum(1 for r in path_rows if r.get("方向") == "偏多")
    doc = {
        "date": plan.get("date") or date.today().isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regime": reg,
        "summary": {
            "可开仓": actionable,
            "观望": len(avoid_rows) + len(path_rows),
            "总条目": len(long_rows) + len(avoid_rows) + len(path_rows),
        },
        "long": long_rows,
        "avoid": avoid_rows,
        "path5d": path_rows,
        "income": plan.get("income") or {},
        "picks": long_rows + path_rows + avoid_rows,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  pattern_daily → {out.name}")


def run_generators(*, fetch: bool) -> None:
    if not fetch:
        return
    cmds = [
        [sys.executable, str(ROOT / "flow_daily.py"), "--dry-run"],
        [sys.executable, str(ROOT / "ticker_pattern_daily.py"), "--dry-run"],
    ]
    for cmd in cmds:
        real = cmd.copy()
        if "--dry-run" in real:
            real.remove("--dry-run")
        print(f"→ {' '.join(real)}")
        subprocess.run(real, cwd=ROOT, check=False)


def ensure_local_files() -> None:
    csv5d = RESEARCH / "move_pattern_5d_today.csv"
    json5d = RESEARCH / "move_pattern_5d_today.json"
    if csv5d.exists() and (not json5d.exists() or json5d.stat().st_mtime < csv5d.stat().st_mtime):
        csv_to_json(csv5d, json5d)

    pattern_json = RESEARCH / "pattern_daily_today.json"
    if not pattern_json.exists():
        try:
            export_pattern_daily(pattern_json, quick=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ pattern_daily export skipped: {exc}")


def sync_to_resources() -> list[str]:
    RESOURCES.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for rel in _load_manifest_paths():
        rel_clean = rel.replace("research/", "")
        if rel_clean.endswith(".csv"):
            json_name = rel_clean.replace(".csv", ".json")
            src = RESEARCH / json_name
            if not src.exists():
                csv_src = RESEARCH / rel_clean
                if csv_src.exists():
                    csv_to_json(csv_src, src)
        else:
            src = RESEARCH / rel_clean
        if not src.exists():
            print(f"  ⚠ missing {rel}")
            continue
        dst = RESOURCES / src.name
        dst.write_bytes(src.read_bytes())
        copied.append(src.name)
        print(f"  ✓ {src.name}")
    for extra in EXTRA:
        src = RESEARCH / extra
        if src.exists() and src.name not in copied:
            (RESOURCES / src.name).write_bytes(src.read_bytes())
            copied.append(src.name)
            print(f"  ✓ {src.name} (extra)")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync iOS bundled JSON snapshots")
    ap.add_argument("--fetch", action="store_true", help="Run flow/ticker generators (network)")
    ap.add_argument("--pattern", action="store_true", help="Regenerate pattern_daily_today.json")
    args = ap.parse_args()

    print("=== sync ios/Resources ===")
    run_generators(fetch=args.fetch)
    if args.pattern or not (RESEARCH / "pattern_daily_today.json").exists():
        try:
            export_pattern_daily(RESEARCH / "pattern_daily_today.json", quick=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ pattern_daily: {exc}")
    ensure_local_files()
    copied = sync_to_resources()
    if args.fetch:
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/export_chart_snapshots.py"), "--limit", "80"],
                cwd=ROOT,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ chart export: {exc}")
    print(f"\nDone: {len(copied)} files → ios/Resources/")


if __name__ == "__main__":
    main()
