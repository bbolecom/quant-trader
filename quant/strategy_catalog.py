"""全系统策略目录 · 汇总各模块今日快照。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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


def strategy_registry() -> list[StrategyEntry]:
    """全系统策略清单（扫描脚本 + 输出路径）。"""
    return [
        StrategyEntry(
            id="daily_pick",
            name="每日选股 · 统一入口",
            category="聚合",
            script="daily_pick.py",
            config="daily_pick_config.json",
            today_json="research/daily_pick_today.json",
            today_csv="research/daily_pick_today.csv",
            history_csv="daily_pick_history.csv",
            description="汇总资金流向、Meme规律、CSP舰队、暴涨80%等模块",
            integrated_in_daily_pick=True,
            daily_pick_module="—",
            launcher="每日选股_运行一次.command",
        ),
        StrategyEntry(
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
        ),
        StrategyEntry(
            id="capital_flow",
            name="资金流向操盘痕迹",
            category="量价",
            script="flow_daily.py",
            config="flow_daily_config.json",
            today_json="research/flow_daily_today.json",
            today_csv="research/flow_daily_today.csv",
            history_csv="flow_daily_history.csv",
            description="U_S2/D_S2 等量价轨迹 · 做多/做空/回避",
            integrated_in_daily_pick=True,
            daily_pick_module="资金流向",
            launcher="资金流向选股_运行一次.command",
        ),
        StrategyEntry(
            id="meme_pattern",
            name="Meme规律 · Ultra80",
            category="规律",
            script="ticker_pattern_daily.py",
            config="daily_pick_config.json",
            today_json="research/ticker_pattern_today.json",
            description="MSTR/SMCI/COIN 等 OOS 高胜率规律",
            integrated_in_daily_pick=True,
            daily_pick_module="规律·Ultra80",
        ),
        StrategyEntry(
            id="fleet_csp",
            name="5×CSP 圣杯舰队",
            category="期权收入",
            script="quant/daily_screen_fleet.py",
            config="daily_screen_config.json",
            today_json="research/liquid_fleet_picks.json",
            description="5账户 CSP 卖 Put · 目标胜率/回撤/年化三标",
            integrated_in_daily_pick=True,
            daily_pick_module="5×舰队·CSP",
        ),
        StrategyEntry(
            id="bear_call",
            name="卖Call价差 · 收租",
            category="期权收入",
            script="daily_pick.py",
            config="daily_pick_config.json",
            description="涨幅榜卖 Call / meme 路由弱市 Put 价差",
            integrated_in_daily_pick=True,
            daily_pick_module="收入·卖Call",
        ),
        StrategyEntry(
            id="pattern_daily",
            name="三腿策略 · 轨迹/回避/收租",
            category="综合",
            script="pattern_daily.py",
            config="pattern_config.json",
            history_csv="pattern_daily_history.csv",
            description="腿①做多 腿②回避 腿③SNDK铁鹰舰队",
            integrated_in_daily_pick=True,
            daily_pick_module="三腿策略",
            launcher="三腿策略_运行一次.command",
        ),
        StrategyEntry(
            id="flow_strategy",
            name="资金流向组合回测",
            category="量价",
            script="research/flow_strategy_backtest.py",
            config="flow_strategy_config.json",
            today_json="research/flow_strategy_today.json",
            description="U_S2 7~15% 高胜率组合 · --today 出当日信号",
            integrated_in_daily_pick=True,
            daily_pick_module="资金流向组合",
            launcher="资金流向策略回测_运行一次.command",
        ),
        StrategyEntry(
            id="trajectory_highwin",
            name="涨幅榜高置信动量",
            category="动量",
            script="research/gainer_daily_backtest.py",
            config="daily_pick_config.json",
            description="温和涨幅+量比+MA20 · 目标日胜率≥80%",
            integrated_in_daily_pick=True,
            daily_pick_module="高频·动量/轨迹·高置信",
        ),
        StrategyEntry(
            id="move_pattern_5d",
            name="5日路径规律",
            category="规律",
            script="research/move_pattern_5d_mine.py",
            today_json="research/move_pattern_5d_today.csv",
            description="路径涨/跌≥3% · 含换手率因子",
            integrated_in_daily_pick=True,
            daily_pick_module="三腿策略·5日路径",
            launcher="5日路径规律_运行一次.command",
        ),
        StrategyEntry(
            id="s8u_liquid",
            name="S8U高流通规律",
            category="规律",
            script="research/s8u_liquid_universe_backtest.py",
            today_json="research/s8u_liquid_universe_backtest.json",
            description="高流通 Universe 规律 · 供 Meme Ultra80 准入",
            integrated_in_daily_pick=True,
            daily_pick_module="规律·Ultra80",
            launcher="S8U高流通回测_运行一次.command",
        ),
        StrategyEntry(
            id="sndk_iron",
            name="SNDK铁鹰收租",
            category="期权收入",
            script="sndk_iron_daily.py",
            config="sndk_iron_config.json",
            history_csv="sndk_iron_history.csv",
            integrated_in_daily_pick=True,
            daily_pick_module="SNDK铁鹰",
            launcher="闪迪铁鹰_运行一次.command",
        ),
        StrategyEntry(
            id="strategy_rank",
            name="策略排名",
            category="聚合",
            script="strategy_daily.py",
            config="strategy_config.json",
            history_csv="strategy_history.csv",
            description="全策略夏普/胜率排名 Top3",
            integrated_in_daily_pick=True,
            daily_pick_module="策略排名",
            launcher="策略排名_运行一次.command",
        ),
        StrategyEntry(
            id="vrp",
            name="VRP波动率溢价",
            category="期权收入",
            script="vrp_daily.py",
            config="vrp_config.json",
            history_csv="vrp_history.csv",
            integrated_in_daily_pick=True,
            daily_pick_module="VRP波动率",
            launcher="VRP信号_运行一次.command",
        ),
        StrategyEntry(
            id="calendar",
            name="日历价差",
            category="期权",
            script="calendar_daily.py",
            config="calendar_config.json",
            history_csv="calendar_history.csv",
            integrated_in_daily_pick=True,
            daily_pick_module="日历价差",
            launcher="日历价差_运行一次.command",
        ),
        StrategyEntry(
            id="universal_playbook",
            name="Universal Playbook 舰队",
            category="综合",
            script="research/universal_playbook.py",
            config="tier_a_csp_config.json",
            today_json="research/universal_playbook_today.json",
            today_csv="research/universal_playbook_today.csv",
            integrated_in_daily_pick=True,
            daily_pick_module="Universal舰队",
        ),
        StrategyEntry(
            id="screen_daily",
            name="每日选股器",
            category="筛选",
            script="screen_daily.py",
            config="screen_config.json",
            history_csv="screen_history.csv",
            integrated_in_daily_pick=True,
            daily_pick_module="每日选股器",
        ),
        StrategyEntry(
            id="scan_daily",
            name="自选股信号扫描",
            category="监控",
            script="scan_daily.py",
            config="scan_config.json",
            history_csv="scan_history.csv",
            integrated_in_daily_pick=True,
            daily_pick_module="自选股扫描",
        ),
        StrategyEntry(
            id="surge_scan",
            name="暴涨扫描 A/B/C",
            category="动量",
            script="surge_daily.py",
            config="surge_scan_config.json",
            today_json="research/surge_scan_today.json",
            today_csv="research/surge_scan_today.csv",
            history_csv="surge_scan_history.csv",
            description="A突破 · B延续高潮 · C前兆蓄势",
            integrated_in_daily_pick=False,
            daily_pick_module="—",
        ),
        StrategyEntry(
            id="speculative_pool",
            name="SPCE类投机池",
            category="动量",
            script="speculative_pool_daily.py",
            config="speculative_pool_config.json",
            today_json="research/speculative_pool.json",
            today_csv="research/speculative_pool.csv",
            description="与 SPCE 暴涨画像相似的投机票池",
            integrated_in_daily_pick=True,
            daily_pick_module="投机池",
            launcher="SPCE类投机池_运行一次.command",
        ),
        StrategyEntry(
            id="ticker_pattern",
            name="Meme规律独立扫描",
            category="规律",
            script="ticker_pattern_daily.py",
            today_json="research/ticker_pattern_today.json",
            integrated_in_daily_pick=True,
            daily_pick_module="规律·Meme独立",
        ),
    ]


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
    actionable = sum(1 for x in items if str(x.get("状态", "")) == "可开仓" or x.get("信号") in ("追多", "回避/做空"))
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
        })
    return rows


# daily_pick 模块名 → strategy id（用于 modules_summary 回填）
MODULE_ID_ALIASES: dict[str, str] = {
    "暴涨80%": "gain15",
    "暴涨80%·回避": "gain15",
    "暴涨80%·观察": "gain15",
    "5×舰队·CSP": "fleet_csp",
    "资金流向": "capital_flow",
    "规律·Ultra80": "meme_pattern",
    "收入·卖Call": "bear_call",
    "弱市·卖Call": "bear_call",
    "三腿策略": "pattern_daily",
    "pattern_daily": "pattern_daily",
    "flow_strategy": "flow_strategy",
    "VRP波动率": "vrp",
    "VRP波动率·CSP": "vrp",
    "日历价差": "calendar",
    "SNDK铁鹰": "sndk_iron",
    "strategy_rank": "strategy_rank",
    "screen_daily": "screen_daily",
    "自选股扫描": "scan_daily",
    "universal_playbook": "universal_playbook",
    "高频·动量": "trajectory_highwin",
    "轨迹·高置信": "trajectory_highwin",
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
    standalone = [r for r in catalog if not r["已接入每日选股"]]
    return {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regime": regime,
        "modules_summary": modules_summary,
        "catalog": catalog,
        "module_runs": module_runs or [],
        "integrated_count": len(integrated),
        "standalone_count": len(standalone),
        "integrated_with_data": sum(1 for r in integrated if r["今日有数据"] or r.get("今日已运行")),
        "actionable_modules": [
            k for k, v in modules_summary.items() if v.get("可开仓", 0) > 0
        ],
        "top_actionable": [
            p for p in picks if p.get("状态") == "可开仓"
        ][:20],
    }
