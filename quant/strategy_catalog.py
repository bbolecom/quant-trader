"""全系统策略目录 · 精简版（仅保留最优 10 策略 + 统一入口）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# 按 strategy_ranker 回测夏普/胜率 + 去重后的核心 10 策略
CORE_STRATEGY_IDS: tuple[str, ...] = (
    "capital_flow",   # 量价 · U_S2 操盘痕迹
    "flow_strategy",  # 量价 · U_S2 组合回测
    "meme_long",      # 规律 · Meme Ultra80
    "gain15",         # 动量 · 暴涨80%确认
    "extreme20",      # 动量 · 暴涨暴跌20% L1/S1/L2/S2 (OOS 70%胜率)
    "longshort_combo", # 动量 · Extreme20 + Flow 多空组合
    "gainer10",       # 动量 · 日涨10%+亿成交 · 分板块高胜率多空 (~75%)
    "whipsaw_short",  # 动量 · 做空涨幅榜卖Call价差 (76%胜率)
    "bear_call",      # 期权收入 · 卖Call价差 (Sharpe 1.40)
    "fleet_csp",      # 期权收入 · 5×CSP舰队 (Sharpe 1.50)
    "sndk_iron",      # 期权收入 · SNDK铁鹰 (Sharpe 1.20)
    "vrp",            # 期权收入 · VRP波动率溢价
)


@dataclass
class StrategyEntry:
    id: str
    name: str
    category: str
    script: str
    config: str = ""
    today_json: str = ""
    today_csv: str = ""
    history_csv: str = ""
    description: str = ""
    integrated_in_daily_pick: bool = False
    daily_pick_module: str = ""
    launcher: str = ""
    sharpe: float = 0.0
    win_rate: float = 0.0


# 统一入口（不计入核心策略计数）
_HUB = StrategyEntry(
    id="daily_pick",
    name="每日选股 · 统一入口",
    category="聚合",
    script="daily_pick.py",
    config="daily_pick_config.json",
    today_json="research/daily_pick_today.json",
    today_csv="research/daily_pick_today.csv",
    history_csv="daily_pick_history.csv",
    description="汇总核心 9 策略 · 有信号才出手",
    integrated_in_daily_pick=True,
    daily_pick_module="—",
    launcher="每日选股_运行一次.command",
)

# 核心 9 策略元数据
_CORE_ENTRIES: dict[str, StrategyEntry] = {
    "capital_flow": StrategyEntry(
        id="capital_flow",
        name="资金流向操盘痕迹",
        category="量价",
        script="flow_daily.py",
        config="flow_daily_config.json",
        today_json="research/flow_daily_today.json",
        today_csv="research/flow_daily_today.csv",
        history_csv="flow_daily_history.csv",
        description="U_S2/D_S2 量价轨迹 · 做多/做空/回避",
        integrated_in_daily_pick=True,
        daily_pick_module="资金流向",
        launcher="资金流向选股_运行一次.command",
        sharpe=1.30,
        win_rate=0.78,
    ),
    "flow_strategy": StrategyEntry(
        id="flow_strategy",
        name="资金流向组合",
        category="量价",
        script="research/flow_strategy_backtest.py",
        config="flow_strategy_config.json",
        today_json="research/flow_strategy_today.json",
        description="U_S2 7~15% 高胜率组合 · --today 出当日信号",
        integrated_in_daily_pick=True,
        daily_pick_module="资金流向组合",
        launcher="资金流向策略回测_运行一次.command",
        sharpe=1.25,
        win_rate=0.80,
    ),
    "meme_long": StrategyEntry(
        id="meme_long",
        name="Meme规律 · Ultra80",
        category="规律",
        script="ticker_pattern_daily.py",
        config="daily_pick_config.json",
        today_json="research/ticker_pattern_today.json",
        description="OOS 高胜率 Meme 规律 · 路径止盈",
        integrated_in_daily_pick=True,
        daily_pick_module="规律·Ultra80",
        sharpe=1.10,
        win_rate=0.80,
    ),
    "gain15": StrategyEntry(
        id="gain15",
        name="暴涨80%规则",
        category="动量",
        script="gain15_daily.py",
        config="gain15_daily_config.json",
        today_json="research/gain15_daily_today.json",
        today_csv="research/gain15_daily_today.csv",
        history_csv="gain15_daily_history.csv",
        description="涨幅>15%+成交额>5000万 → T+1/T+3确认 → 80%追多/回避",
        integrated_in_daily_pick=True,
        daily_pick_module="暴涨80%",
        launcher="暴涨80%扫描_运行一次.command",
        sharpe=0.95,
        win_rate=0.80,
    ),
    "extreme20": StrategyEntry(
        id="extreme20",
        name="暴涨暴跌20%事件",
        category="动量",
        script="extreme20_daily.py",
        config="extreme20_config.json",
        today_json="research/extreme20_today.json",
        today_csv="research/extreme20_today.csv",
        history_csv="extreme20_history.csv",
        description="高胜率多空：牛市L1+S1 / 熊市L2+S1 · max_short=1 · 5年91笔 胜60.4% 年化+56%",
        integrated_in_daily_pick=True,
        daily_pick_module="Extreme20",
        launcher="Extreme20_运行一次.command",
        sharpe=5.23,
        win_rate=0.59,
    ),
    "longshort_combo": StrategyEntry(
        id="longshort_combo",
        name="多空组合 · 高胜率",
        category="动量",
        script="longshort_combo_daily.py",
        config="longshort_combo_config.json",
        today_json="research/longshort_combo_today.json",
        today_csv="research/longshort_combo_today.csv",
        history_csv="longshort_combo_history.csv",
        description="Extreme20 L1/S1 + Flow U_S2/D_S2 · 质量分≥0.55 · 5年257笔 胜55.6% OOS59%",
        integrated_in_daily_pick=True,
        daily_pick_module="多空组合",
        launcher="LongShort_运行一次.command",
        sharpe=1.91,
        win_rate=0.59,
    ),
    "gainer10": StrategyEntry(
        id="gainer10",
        name="Gainer10+ 分板块高胜率",
        category="动量",
        script="gainer10_daily.py",
        config="gainer10_config.json",
        today_json="research/gainer10_today.json",
        today_csv="research/gainer10_today.csv",
        history_csv="research/gainer10_history.csv",
        description="日涨>10%+亿级成交 · 分板块多/空 · L≥60%+avg≥3 S≥80% · 5年组合胜率~75%",
        integrated_in_daily_pick=True,
        daily_pick_module="Gainer10+",
        launcher="Gainer10_运行一次.command",
        sharpe=0.89,
        win_rate=0.75,
    ),
    "whipsaw_short": StrategyEntry(
        id="whipsaw_short",
        name="做空涨幅榜·卖Call价差",
        category="动量",
        script="whipsaw_short_daily.py",
        config="whipsaw_short_config.json",
        today_json="research/whipsaw_short_today.json",
        today_csv="research/whipsaw_short_today.csv",
        history_csv="whipsaw_short_history.csv",
        description="暴涨乏力卖Call价差(定风险)·弱市加倍·回测76%胜率/年年正",
        integrated_in_daily_pick=True,
        daily_pick_module="做空涨幅榜",
        launcher="WhipsawShort_运行一次.command",
        sharpe=0.29,
        win_rate=0.76,
    ),
    "bear_call": StrategyEntry(
        id="bear_call",
        name="卖Call价差 · 收租",
        category="期权收入",
        script="daily_pick.py",
        config="daily_pick_config.json",
        description="涨幅榜卖 Call / meme 路由弱市 Put 价差",
        integrated_in_daily_pick=True,
        daily_pick_module="收入·卖Call",
        sharpe=1.40,
        win_rate=0.88,
    ),
    "fleet_csp": StrategyEntry(
        id="fleet_csp",
        name="5×CSP 圣杯舰队",
        category="期权收入",
        script="quant/daily_screen_fleet.py",
        config="daily_screen_config.json",
        today_json="research/liquid_fleet_picks.json",
        description="5账户 CSP 卖 Put · 目标胜率/回撤/年化三标",
        integrated_in_daily_pick=True,
        daily_pick_module="5×舰队·CSP",
        sharpe=1.50,
        win_rate=0.97,
    ),
    "sndk_iron": StrategyEntry(
        id="sndk_iron",
        name="SNDK铁鹰收租",
        category="期权收入",
        script="sndk_iron_daily.py",
        config="sndk_iron_config.json",
        history_csv="sndk_iron_history.csv",
        integrated_in_daily_pick=True,
        daily_pick_module="SNDK铁鹰",
        launcher="闪迪铁鹰_运行一次.command",
        sharpe=1.20,
        win_rate=0.87,
    ),
    "vrp": StrategyEntry(
        id="vrp",
        name="VRP波动率溢价",
        category="期权收入",
        script="vrp_daily.py",
        config="vrp_config.json",
        history_csv="vrp_history.csv",
        integrated_in_daily_pick=True,
        daily_pick_module="VRP波动率",
        launcher="VRP信号_运行一次.command",
        sharpe=1.00,
        win_rate=0.82,
    ),
}


def strategy_registry() -> list[StrategyEntry]:
    """核心策略 + 统一入口。"""
    return [_HUB] + [_CORE_ENTRIES[sid] for sid in CORE_STRATEGY_IDS]


def is_core_strategy(strategy_id: str) -> bool:
    return strategy_id in CORE_STRATEGY_IDS


def _metric_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _audit_score(*, sharpe: float, win_rate: float, ann_return: float, max_dd: float) -> float:
    """统一审核分：风险调整收益优先，兼顾胜率、回撤、年化。"""
    dd = abs(max_dd)
    dd_pen = max(0.0, 1.0 - dd)
    return round(
        0.35 * min(max(sharpe, 0.0) / 1.5, 1.0)
        + 0.30 * max(min(win_rate, 1.0), 0.0)
        + 0.20 * dd_pen
        + 0.15 * min(max(ann_return, 0.0) / 0.5, 1.0),
        4,
    )


def _audit_tier(score: float) -> str:
    if score >= 0.75:
        return "S"
    if score >= 0.60:
        return "A"
    if score >= 0.45:
        return "B"
    if score >= 0.25:
        return "C"
    return "D"


def _audit_verdict(row: dict[str, Any]) -> tuple[str, str]:
    sid = str(row.get("id", ""))
    ann = _metric_float(row.get("ann_return"))
    sharpe = _metric_float(row.get("sharpe"))
    win = _metric_float(row.get("win_rate"))
    score = _metric_float(row.get("audit_score"))
    if sid in {"gain15", "whipsaw_short"} or (ann < 0 and sharpe < 0):
        return "禁用", "负期望回测，保留研究但不建议实盘"
    if score >= 0.75:
        return "主力", "优先展示和运行"
    if score >= 0.60:
        return "核心", "可作为主要候选"
    if win >= 0.80 and ann >= 0:
        return "收租", "适合稳态收入或低频配合"
    if score >= 0.45:
        return "观察", "保留入口，等待更好参数或样本"
    return "实验", "低权重展示"


def _load_backtest_rows(root: Path) -> dict[str, dict[str, Any]]:
    suite = _read_json(root / "research" / "strategy_suite_5y.json") or {}
    rows = {str(r.get("id")): dict(r) for r in suite.get("strategies") or []}

    # longshort_combo 是组合策略，来自独立优化/验证结果。
    ls = _read_json(root / "research" / "longshort_combo_rules.json") or {}
    m = ls.get("best_metrics") or {}
    if m:
        rows["longshort_combo"] = {
            "id": "longshort_combo",
            "name": "多空组合 · 高胜率",
            "category": "动量",
            "trades": m.get("交易次数"),
            "win_rate": m.get("胜率"),
            "ann_return": m.get("年化"),
            "max_dd": m.get("回撤"),
            "sharpe": m.get("夏普"),
            "detail": {"source": "research/longshort_combo_rules.json", "oos_win_rate": m.get("OOS胜率")},
        }
    return rows


def build_strategy_audit(root: Path | None = None) -> dict[str, Any]:
    """审核核心策略：合并元数据与回测，输出排名、分层、App 展示字段。"""
    root = root or ROOT
    bt_rows = _load_backtest_rows(root)
    audit_rows: list[dict[str, Any]] = []

    for entry in _CORE_ENTRIES.values():
        bt = bt_rows.get(entry.id, {})
        sharpe = _metric_float(bt.get("sharpe"), entry.sharpe)
        win_rate = _metric_float(bt.get("win_rate"), entry.win_rate)
        ann_return = _metric_float(bt.get("ann_return"), 0.0)
        max_dd = _metric_float(bt.get("max_dd"), 0.0)
        score = _metric_float(bt.get("score"))
        if score <= 0:
            score = _audit_score(
                sharpe=sharpe,
                win_rate=win_rate,
                ann_return=ann_return,
                max_dd=max_dd,
            )
        row = {
            "id": entry.id,
            "name": entry.name,
            "category": entry.category,
            "description": entry.description,
            "integrated_in_daily_pick": entry.integrated_in_daily_pick,
            "daily_pick_module": entry.daily_pick_module,
            "script": entry.script,
            "config": entry.config,
            "trades": bt.get("trades"),
            "win_rate": win_rate,
            "ann_return": ann_return,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "audit_score": round(score, 4),
            "audit_tier": _audit_tier(score),
            "source": (bt.get("detail") or {}).get("source") or "strategy_suite_5y",
        }
        verdict, action = _audit_verdict(row)
        row["audit_verdict"] = verdict
        row["audit_action"] = action
        audit_rows.append(row)

    audit_rows.sort(key=lambda x: (_metric_float(x["audit_score"]), _metric_float(x["win_rate"])), reverse=True)
    for i, row in enumerate(audit_rows, 1):
        row["audit_rank"] = i

    return {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "0.35*Sharpe + 0.30*胜率 + 0.20*回撤惩罚 + 0.15*年化",
        "strategy_count": len(audit_rows),
        "rows": audit_rows,
        "tiers": {
            t: sum(1 for r in audit_rows if r["audit_tier"] == t)
            for t in ("S", "A", "B", "C", "D")
        },
        "disabled": [r for r in audit_rows if r["audit_verdict"] == "禁用"],
    }


def export_strategy_audit(root: Path | None = None) -> Path:
    root = root or ROOT
    doc = build_strategy_audit(root)
    out = root / "research" / "strategy_audit.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _count_actionable(
    doc: dict | None,
    *,
    pick_keys: tuple[str, ...] = ("picks",),
    hub_summary: bool = False,
) -> dict[str, Any]:
    """统计可开仓/观望。hub_summary=True 时用 daily_pick 顶层 summary（统一入口）。"""
    if not doc:
        return {"available": False, "可开仓": 0, "观望": 0, "总条目": 0}
    if doc.get("scan_stats") is not None:
        ss = doc["scan_stats"]
        cands = doc.get("candidates") or []
        actionable = int(ss.get("可开仓") or 0)
        if not actionable and cands:
            actionable = sum(
                1 for x in cands
                if x.get("信号") == "卖Call价差" and int(x.get("建议张数") or 0) > 0
            )
        total = len(cands) if cands else max(actionable, int(ss.get("候选") or 0))
        return {
            "available": True,
            "可开仓": actionable,
            "观望": max(0, total - actionable),
            "总条目": total,
            "date": doc.get("date") or doc.get("选股日期"),
        }
    if hub_summary and doc.get("summary"):
        s = doc["summary"]
        return {
            "available": True,
            "可开仓": int(s.get("可开仓") or 0),
            "观望": int(s.get("观望") or 0),
            "总条目": int(s.get("总条目") or 0),
            "date": doc.get("选股日期") or doc.get("date"),
        }
    items: list = []
    for k in pick_keys:
        v = doc.get(k)
        if isinstance(v, list):
            items.extend(v)
    if not items and isinstance(doc.get("buy_confirmed"), list):
        items = (
            doc.get("buy_confirmed", [])
            + doc.get("avoid_confirmed", [])
            + doc.get("watching", [])
            + doc.get("new_spikes", [])
        )
    actionable = sum(
        1 for x in items
        if str(x.get("状态", "")) == "可开仓"
        or x.get("信号") in ("追多", "回避/做空", "卖Call价差")
    )
    if not items and doc.get("summary"):
        s = doc["summary"]
        return {
            "available": True,
            "可开仓": s.get("可开仓", 0),
            "观望": s.get("观望", 0),
            "总条目": s.get("总条目", 0),
            "date": doc.get("选股日期") or doc.get("date"),
        }
    return {
        "available": True,
        "可开仓": actionable,
        "观望": max(0, len(items) - actionable),
        "总条目": len(items),
        "date": doc.get("选股日期") or doc.get("date"),
    }


def collect_strategy_snapshots(root: Path | None = None) -> list[dict]:
    """读取各策略 today JSON 快照（不重新跑扫描）。"""
    root = root or ROOT
    rows: list[dict] = []
    for s in strategy_registry():
        path = root / s.today_json if s.today_json else None
        doc = _read_json(path) if path else None
        stats = _count_actionable(doc, hub_summary=(s.id == "daily_pick"))
        rows.append({
            "id": s.id,
            "策略": s.name,
            "分类": s.category,
            "已接入每日选股": s.integrated_in_daily_pick,
            "模块标签": s.daily_pick_module or "—",
            "今日有数据": stats["available"],
            "可开仓": stats.get("可开仓", 0),
            "观望": stats.get("观望", 0),
            "总条目": stats.get("总条目", 0),
            "数据日期": stats.get("date", "—"),
            "说明": s.description,
            "脚本": s.script,
            "输出": s.today_json or s.history_csv or "—",
            "夏普": s.sharpe,
            "胜率": s.win_rate,
        })
    return rows


MODULE_ID_ALIASES: dict[str, str] = {
    "暴涨80%": "gain15",
    "暴涨80%·回避": "gain15",
    "暴涨80%·观察": "gain15",
    "暴涨80%·新暴涨": "gain15",
    "5×舰队·CSP": "fleet_csp",
    "资金流向": "capital_flow",
    "规律·Ultra80": "meme_long",
    "规律·纯多头": "meme_long",
    "收入·卖Call": "bear_call",
    "弱市·卖Call": "bear_call",
    "flow_strategy": "flow_strategy",
    "资金流向组合": "flow_strategy",
    "VRP波动率": "vrp",
    "VRP波动率·CSP": "vrp",
    "SNDK铁鹰": "sndk_iron",
    "做空涨幅榜": "whipsaw_short",
    "Extreme20": "extreme20",
    "Gainer10+": "gainer10",
    "分板块多": "gainer10",
    "分板块空": "gainer10",
    "多空组合": "longshort_combo",
}


def enrich_catalog_from_daily_pick(catalog: list[dict], dp: dict | None) -> list[dict]:
    """用 daily_pick 汇总回填各策略可开仓/观望（Cloud 不依赖各模块独立 JSON）。"""
    if not dp:
        return catalog
    mods: dict = dp.get("modules_summary") or {}
    summary: dict = dp.get("summary") or {}
    runs = {r.get("id"): r for r in (dp.get("module_runs") or [])}
    pick_date = dp.get("选股日期") or "—"

    by_id: dict[str, dict] = {str(r.get("id")): dict(r) for r in catalog}

    for mod_key, stats in mods.items():
        sid = MODULE_ID_ALIASES.get(mod_key)
        if not sid:
            for rid, row in by_id.items():
                label = str(row.get("模块标签") or "")
                if label and label != "—" and (label in mod_key or mod_key.startswith(label)):
                    sid = rid
                    break
        if not sid or sid not in by_id:
            continue
        row = by_id[sid]
        row["可开仓"] = int(row.get("可开仓") or 0) + int(stats.get("可开仓") or 0)
        row["观望"] = int(row.get("观望") or 0) + int(stats.get("观望") or 0)
        row["总条目"] = int(row.get("总条目") or 0) + int(stats.get("总条目") or 0)
        row["今日有数据"] = True
        row["数据日期"] = pick_date
        codes = list(row.get("_codes") or [])
        codes.extend(stats.get("代码") or [])
        row["_codes"] = codes[:12]

    if "daily_pick" in by_id and summary:
        by_id["daily_pick"].update({
            "可开仓": int(summary.get("可开仓") or 0),
            "观望": int(summary.get("观望") or 0),
            "总条目": int(summary.get("总条目") or 0),
            "今日有数据": True,
            "数据日期": pick_date,
        })

    for sid, run in runs.items():
        if sid not in by_id:
            continue
        row = by_id[sid]
        row["今日有数据"] = True
        row["今日已运行"] = run.get("ok")
        row["数据日期"] = pick_date
        if run.get("rows"):
            row["总条目"] = max(int(row.get("总条目") or 0), int(run.get("rows") or 0))
        if run.get("可开仓") is not None:
            row["可开仓"] = max(int(row.get("可开仓") or 0), int(run.get("可开仓") or 0))

    return list(by_id.values())


def summarize_picks_by_module(picks: list[dict]) -> dict[str, dict]:
    """按 daily_pick 模块标签汇总。"""
    out: dict[str, dict] = {}
    for p in picks:
        mod = str(p.get("模块", "其他"))
        bucket = out.setdefault(mod, {"总条目": 0, "可开仓": 0, "观望": 0, "代码": []})
        bucket["总条目"] += 1
        if p.get("状态") == "可开仓":
            bucket["可开仓"] += 1
            tk = p.get("代码")
            if tk and tk != "—":
                bucket["代码"].append(str(tk))
        else:
            bucket["观望"] += 1
    for b in out.values():
        b["代码"] = b["代码"][:8]
    return out


def _strategy_id_for_pick(pick: dict[str, Any], audit_rows: list[dict[str, Any]]) -> str | None:
    module = str(pick.get("模块") or "")
    explicit = str(pick.get("strategy_id") or pick.get("策略ID") or "").strip()
    if explicit in _CORE_ENTRIES:
        return explicit
    sid = MODULE_ID_ALIASES.get(module)
    if sid:
        return sid
    for key, val in MODULE_ID_ALIASES.items():
        if key and (module.startswith(key) or key in module):
            return val
    for row in audit_rows:
        label = str(row.get("daily_pick_module") or "")
        if label and label != "—" and (module.startswith(label) or label in module):
            return str(row.get("id"))
    return None


def enrich_picks_with_strategy_audit(
    picks: list[dict],
    *,
    audit_doc: dict[str, Any] | None = None,
    root: Path | None = None,
) -> list[dict]:
    """把策略审核排名/评级挂到每日机会行，供 App 排序与自动推送使用。"""
    audit_doc = audit_doc or build_strategy_audit(root or ROOT)
    audit_rows = list(audit_doc.get("rows") or [])
    audit_map = {str(r.get("id")): r for r in audit_rows}
    out: list[dict] = []
    for pick in picks:
        row = dict(pick)
        sid = _strategy_id_for_pick(row, audit_rows)
        ar = audit_map.get(sid or "")
        if ar:
            row.setdefault("策略ID", sid)
            row["策略排名"] = ar.get("audit_rank")
            row["策略评级"] = ar.get("audit_tier")
            row["策略审核"] = ar.get("audit_verdict")
            row["策略结论"] = ar.get("audit_action")
            row["策略分"] = ar.get("audit_score")
            row["策略胜率"] = ar.get("win_rate")
            row["策略年化"] = ar.get("ann_return")
            row["策略夏普"] = ar.get("sharpe")
            if row.get("历史胜率") is None and ar.get("win_rate") is not None:
                row["历史胜率"] = ar.get("win_rate")
            if row.get("历史年化") is None and ar.get("ann_return") is not None:
                row["历史年化"] = ar.get("ann_return")
            if row.get("最大回撤") is None and ar.get("max_dd") is not None:
                row["最大回撤"] = ar.get("max_dd")
            if not row.get("回测摘要"):
                wr = _metric_float(ar.get("win_rate"))
                ann = _metric_float(ar.get("ann_return"))
                row["回测摘要"] = f"策略#{ar.get('audit_rank')} {ar.get('audit_tier')} · 胜{wr:.0%} · 年化{ann:+.0%}"
            # 机会优先级：越大越靠前，低排名和高评分优先。
            rank = int(ar.get("audit_rank") or 99)
            score = _metric_float(ar.get("audit_score"))
            row["推送优先级"] = round(max(0, 130 - rank * 5) + score * 20, 2)
        out.append(row)
    return out


def build_strategy_summary_doc(
    *,
    picks: list[dict],
    modules_summary: dict[str, dict],
    regime: dict,
    root: Path | None = None,
    module_runs: list[dict] | None = None,
    pick_date: str | None = None,
    summary: dict | None = None,
) -> dict:
    """构建写入 daily_pick_today.json 的策略总览。"""
    root = root or ROOT
    catalog = collect_strategy_snapshots(root)
    run_map = {m["id"]: m for m in (module_runs or [])}
    for row in catalog:
        rid = row.get("id")
        if rid in run_map:
            row["今日已运行"] = run_map[rid].get("ok", False)
            row["运行条目"] = run_map[rid].get("rows", 0)
            row["已接入每日选股"] = True
    catalog = enrich_catalog_from_daily_pick(catalog, {
        "modules_summary": modules_summary,
        "summary": summary or {},
        "module_runs": module_runs or [],
        "选股日期": pick_date or "—",
    })
    integrated = [r for r in catalog if r["已接入每日选股"]]
    return {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regime": regime,
        "modules_summary": modules_summary,
        "catalog": catalog,
        "module_runs": module_runs or [],
        "core_strategy_ids": list(CORE_STRATEGY_IDS),
        "core_count": len(CORE_STRATEGY_IDS),
        "integrated_count": len(integrated),
        "standalone_count": 0,
        "integrated_with_data": sum(1 for r in integrated if r["今日有数据"] or r.get("今日已运行")),
        "actionable_modules": [
            k for k, v in modules_summary.items() if v.get("可开仓", 0) > 0
        ],
        "top_actionable": [
            p for p in picks if p.get("状态") == "可开仓"
        ][:20],
    }
