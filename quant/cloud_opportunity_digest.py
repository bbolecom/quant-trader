"""盘后汇总全策略云端 JSON feed，统计可开仓机会并统一手机推送。

由 scripts/cloud_post_close.py 在 GitHub Actions 收盘后调用。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "cloud_push_config.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _pick_actionable(row: dict) -> bool:
    for k in ("可开", "status", "状态", "action"):
        v = row.get(k)
        if v is None:
            continue
        s = str(v)
        if s in ("✅", "可开仓", "✅破位可空", "actionable", "open", "buy", "short"):
            return True
        if "可空" in s or "可开" in s:
            return True
    return False


def _count_actionable(doc: dict) -> int:
    if not doc:
        return 0
    for node_key in ("summary", "scan_stats"):
        node = doc.get(node_key) or {}
        for field in ("可开仓", "actionable", "可空", "推送条数"):
            v = node.get(field)
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v)
    triggers = doc.get("triggers") or []
    act_triggers = [t for t in triggers if _pick_actionable(t)]
    if act_triggers:
        return len(act_triggers)
    picks = doc.get("picks") or []
    act_picks = [p for p in picks if _pick_actionable(p) or p.get("代码")]
    if picks and act_picks:
        # picks 非空通常即有机会清单
        explicit = doc.get("scan_stats", {}).get("可开仓")
        if isinstance(explicit, int):
            return explicit
        return len(act_picks)
    rows = doc.get("rows") or []
    act_rows = [r for r in rows if _pick_actionable(r)]
    if act_rows:
        return len(act_rows)
    candidates = doc.get("candidates") or []
    if candidates and (doc.get("scan_stats") or {}).get("可开仓"):
        return int((doc.get("scan_stats") or {}).get("可开仓", 0))
    return 0


def _top_symbols(doc: dict, limit: int = 3) -> list[str]:
    out: list[str] = []
    for key in ("triggers", "picks", "rows", "candidates", "signals"):
        for row in doc.get(key) or []:
            sym = row.get("代码") or row.get("ticker") or row.get("Ticker")
            if sym and sym not in out:
                out.append(str(sym))
            if len(out) >= limit:
                return out
    for key in ("buy_a", "buy_b", "buy_sector", "short_s", "short_sector"):
        for row in doc.get(key) or []:
            sym = row.get("代码")
            if sym and sym not in out:
                out.append(str(sym))
            if len(out) >= limit:
                return out
    return out


def collect_from_manifest(manifest_path: Path = ROOT / "research" / "app_manifest.json") -> list[dict[str, Any]]:
    doc = _load_json(manifest_path)
    if not doc:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for feat in doc.get("features") or []:
        rel = str(feat.get("today_json") or "").strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        path = ROOT / rel
        feed = _load_json(path)
        actionable = _count_actionable(feed or {})
        rows.append({
            "id": feat.get("id"),
            "name": feat.get("name"),
            "path": rel,
            "actionable": actionable,
            "symbols": _top_symbols(feed or {}),
            "has_data": feed is not None,
            "data_date": (feed or {}).get("date") or (feed or {}).get("选股日期"),
        })
    return rows


def build_digest_text(rows: list[dict[str, Any]]) -> tuple[str, str, int]:
    today = date.today().isoformat()
    total = sum(r["actionable"] for r in rows)
    actionable_rows = [r for r in rows if r["actionable"] > 0]
    title = f"📊 量化机会 {total}条 · {today}"
    lines: list[str] = []
    regime = _load_json(ROOT / "research" / "daily_pick_today.json")
    if regime:
        sm = regime.get("summary") or {}
        lines.append(f"大盘 {sm.get('大盘', '—')} · 每日选股可开 {sm.get('可开仓', 0)}")
    for r in actionable_rows:
        syms = "、".join(r["symbols"][:4]) if r["symbols"] else "—"
        lines.append(f"• {r['name']}: {r['actionable']} · {syms}")
    if not actionable_rows:
        watch = [r for r in rows if r["has_data"] and r["actionable"] == 0]
        names = "、".join(r["name"] for r in watch[:6])
        lines.append(f"今日无可开仓信号 → 观望（已扫描 {len(watch)} 策略）")
        if names:
            lines.append(names)
    body = "\n".join(lines)
    if len(body) > 900:
        body = body[:897] + "…"
    return title, body, total


def write_digest_json(rows: list[dict[str, Any]], out: Path) -> None:
    title, body, total = build_digest_text(rows)
    doc = {
        "date": date.today().isoformat(),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "total_actionable": total,
        "strategies": rows,
        "push_preview": {"title": title, "body": body},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    ios = ROOT / "ios" / "Resources" / "cloud_opportunity_digest.json"
    ios.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def push_digest(cfg_path: Path = DEFAULT_CONFIG) -> list[str]:
    cfg = _load_json(cfg_path) or {}
    rows = collect_from_manifest()
    write_digest_json(rows, ROOT / "research" / "cloud_opportunity_digest.json")
    title, body, total = build_digest_text(rows)
    push_when = str(cfg.get("push_when", "actionable"))
    if push_when == "actionable" and total <= 0:
        print("[digest] 无可开仓机会，跳过推送")
        return []
    if os.environ.get("QUANT_SKIP_MOBILE_PUSH") == "1":
        print("[digest] QUANT_SKIP_MOBILE_PUSH=1，跳过推送")
        return []
    from quant.mobile_push import push_mobile

    return push_mobile(cfg, title, body)


def main() -> None:
    rows = collect_from_manifest()
    write_digest_json(rows, ROOT / "research" / "cloud_opportunity_digest.json")
    title, body, total = build_digest_text(rows)
    print(f"=== 云端机会汇总 {date.today()} ===")
    print(f"总可开仓: {total}")
    for r in rows:
        flag = "✅" if r["actionable"] else "—"
        syms = ",".join(r["symbols"][:3]) if r["symbols"] else ""
        print(f"  {flag} {r['name']}: {r['actionable']} {syms}")
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    logs = push_digest(cfg_path)
    for line in logs:
        print(line)


if __name__ == "__main__":
    main()
