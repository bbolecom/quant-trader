"""每日选股推送：只推送真实数据推演结果（真实期权链 / 真实行情），不含模型估价。"""

from __future__ import annotations

from typing import Any

OPTION_DIRECTIONS = (
    "卖Put", "卖Call", "卖Call价差", "买Put价差", "铁鹰", "双日历", "CSP",
)
MODEL_MARKERS = ("模型估值", "模型估算", "Black-Scholes", "BS回测", "仅供回测")
REAL_CHAIN_MARKERS = ("真实链", "真实期权链")


def is_option_pick(row: dict) -> bool:
    direction = str(row.get("方向") or "")
    module = str(row.get("模块") or "")
    if any(k in direction for k in OPTION_DIRECTIONS):
        return True
    if any(k in module for k in ("CSP", "铁鹰", "卖Call", "VRP", "日历", "期权")):
        return True
    return False


def infer_data_source(row: dict) -> str:
    explicit = str(row.get("数据源") or "").strip()
    if explicit:
        return explicit
    reason = str(row.get("选股理由") or "")
    if "真实链不可用" in reason:
        return "真实链不可用"
    if any(m in reason for m in REAL_CHAIN_MARKERS) and "不可用" not in reason:
        return "真实链"
    if any(m in reason for m in MODEL_MARKERS):
        return "模型估算"
    if is_option_pick(row):
        return "模型估算"
    return "真实行情"


def enrich_pick_data_source(row: dict) -> dict:
    out = dict(row)
    out["数据源"] = infer_data_source(out)
    return out


def is_push_eligible(row: dict, *, require_real: bool = True) -> bool:
    if str(row.get("状态") or "") != "可开仓":
        return False
    if not require_real:
        return True
    tier = infer_data_source(row)
    if tier == "真实链":
        return True
    if tier == "真实行情" and not is_option_pick(row):
        return True
    return False


def build_push_picks(
    picks: list[dict],
    *,
    require_real: bool = True,
    max_items: int = 12,
) -> tuple[list[dict], dict[str, int]]:
    enriched = [enrich_pick_data_source(p) for p in picks]
    push: list[dict] = []
    stats = {"total": len(enriched), "eligible": 0, "skipped_model": 0, "skipped_watch": 0}
    for row in enriched:
        if str(row.get("状态") or "") != "可开仓":
            stats["skipped_watch"] += 1
            continue
        if is_push_eligible(row, require_real=require_real):
            push.append(row)
            stats["eligible"] += 1
        else:
            stats["skipped_model"] += 1
    push.sort(key=lambda r: (0 if r.get("数据源") == "真实链" else 1, str(r.get("代码") or "")))
    return push[:max_items], stats


def format_push_line(row: dict) -> str:
    code = row.get("代码") or "—"
    direction = row.get("方向") or ""
    module = row.get("模块") or ""
    reason = str(row.get("选股理由") or "")
    src = row.get("数据源") or ""
    head = f"{code} {direction}".strip()
    if src == "真实链":
        # 推送正文优先链上报价片段
        if "真实链" in reason:
            idx = reason.find("真实链")
            tail = reason[idx:].split(" · 历史回测")[0].split(" · 历史")[0]
            return f"{head} [{module}] {tail[:120]}"
        return f"{head} [{module}] {reason[:100]}"
    bt = row.get("回测摘要") or row.get("历史命中率") or ""
    core = reason[:90] if reason else module
    if bt:
        return f"{head} [{module}] {core}（{bt}）"
    return f"{head} [{module}] {core}"


def build_push_block(doc: dict, cfg: dict) -> dict[str, Any]:
    pcfg = cfg.get("push") or {}
    require_real = bool(pcfg.get("require_real_data", True))
    max_items = int(pcfg.get("max_items", 12))
    picks = doc.get("picks") or []
    push_picks, stats = build_push_picks(
        picks, require_real=require_real, max_items=max_items,
    )
    reg = doc.get("regime") or {}
    s = doc.get("summary") or {}
    lines = [format_push_line(p) for p in push_picks]
    if push_picks:
        headline = f"真实信号 {len(push_picks)} 条 · {reg.get('label', s.get('大盘', ''))}"
    else:
        headline = f"今日无真实链/行情可推送 · {reg.get('label', s.get('大盘', '观望'))}"

    return {
        "require_real_data": require_real,
        "generated_at": doc.get("选股时间"),
        "pick_date": doc.get("选股日期"),
        "headline": headline,
        "regime": {
            "label": reg.get("label"),
            "bull": reg.get("bull"),
            "spy": reg.get("spy"),
            "ma50": reg.get("ma50"),
        },
        "count": len(push_picks),
        "stats": stats,
        "lines": lines,
        "picks": push_picks,
    }
