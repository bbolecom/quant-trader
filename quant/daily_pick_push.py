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


def push_priority(row: dict) -> float:
    """推送排序分：真实数据 + 策略审核排名 + 历史胜率。"""
    src_bonus = 20.0 if row.get("数据源") == "真实链" else 10.0
    try:
        explicit = float(row.get("推送优先级") or 0)
    except (TypeError, ValueError):
        explicit = 0.0
    try:
        rank = int(row.get("策略排名") or 99)
    except (TypeError, ValueError):
        rank = 99
    try:
        wr = float(row.get("历史胜率") or row.get("策略胜率") or 0)
    except (TypeError, ValueError):
        wr = 0.0
    try:
        audit_score = float(row.get("策略分") or 0)
    except (TypeError, ValueError):
        audit_score = 0.0
    rank_bonus = max(0.0, 80.0 - rank * 4.0)
    return round(src_bonus + explicit + rank_bonus + wr * 30.0 + audit_score * 20.0, 2)


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
    for row in push:
        row["机会评分"] = push_priority(row)
    push.sort(key=lambda r: (-float(r.get("机会评分") or 0), str(r.get("代码") or "")))
    return push[:max_items], stats


def format_push_line(row: dict) -> str:
    code = row.get("代码") or "—"
    direction = row.get("方向") or ""
    module = row.get("模块") or ""
    reason = str(row.get("选股理由") or "")
    src = row.get("数据源") or ""
    rank = row.get("策略排名")
    tier = row.get("策略评级")
    audit = f"#{rank}{tier}" if rank and tier else ""
    head = f"{code} {direction}".strip()
    if src == "真实链":
        # 推送正文优先链上报价片段
        if "真实链" in reason:
            idx = reason.find("真实链")
            tail = reason[idx:].split(" · 历史回测")[0].split(" · 历史")[0]
            return f"{head} [{module}{' '+audit if audit else ''}] {tail[:120]}"
        return f"{head} [{module}{' '+audit if audit else ''}] {reason[:100]}"
    bt = row.get("回测摘要") or row.get("历史命中率") or ""
    core = reason[:90] if reason else module
    if bt:
        return f"{head} [{module}{' '+audit if audit else ''}] {core}（{bt}）"
    return f"{head} [{module}{' '+audit if audit else ''}] {core}"


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
        "sort": "机会评分 = 真实数据 + 策略排名 + 胜率/审核分",
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
