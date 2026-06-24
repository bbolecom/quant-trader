"""每日选股 · 全策略运行器（统一 pick 行格式）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pick_row(
    *,
    module: str,
    account: str,
    ticker: str,
    status: str,
    direction: str,
    action: str = "",
    reason: str = "",
    **extra: Any,
) -> dict:
    row = {
        "模块": module,
        "账户": account,
        "代码": ticker,
        "状态": status,
        "方向": direction,
        "策略动作": action,
        "选股理由": reason,
    }
    row.update(extra)
    return row


def fail_row(module: str, account: str, error: str) -> dict:
    return pick_row(
        module=module,
        account=account,
        ticker="—",
        status="扫描失败",
        direction="—",
        reason=str(error)[:300],
    )


def empty_row(module: str, account: str, reason: str) -> dict:
    return pick_row(
        module=module,
        account=account,
        ticker="—",
        status="观望",
        direction="—",
        reason=reason,
    )


def _load_json_cfg(name: str) -> dict:
    path = ROOT / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def run_pattern_three_leg(cfg: dict, *, bull: bool) -> list[dict]:
    """三腿策略：做多 / 5日路径 / 回避 / 收租。"""
    mod = "三腿策略"
    try:
        import pattern_daily as pdaily

        pcfg = _load_json_cfg(str(cfg.get("pattern_config", "pattern_config.json")))
        if cfg.get("quick"):
            pcfg["quick"] = True
        plan = pdaily.build_plan(pcfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "轨迹/回避/收租", e)]

    rows: list[dict] = []
    long_df = plan.get("long")
    if long_df is not None and not long_df.empty:
        for _, r in long_df.iterrows():
            rows.append(pick_row(
                module=f"{mod}·做多",
                account="腿①轨迹",
                ticker=str(r["代码"]),
                status="可开仓",
                direction="做多",
                action="次日平仓",
                reason=str(r.get("选股理由", "")),
                量比=r.get("量比"),
                近8次胜率=r.get("近8次胜率"),
            ))
    elif not bull:
        rows.append(empty_row(f"{mod}·做多", "腿①轨迹", "弱市关闭轨迹做多"))

    path5d = plan.get("path5d")
    if path5d is not None and not path5d.empty:
        for _, r in path5d.iterrows():
            actionable = str(r.get("方向", "")) == "偏多"
            rows.append(pick_row(
                module=f"{mod}·5日路径",
                account="腿①b路径",
                ticker=str(r["代码"]),
                status="可开仓" if actionable else "观望",
                direction=str(r.get("方向", "—")),
                action=str(r.get("建议", "")),
                reason=f"{r.get('规律', '')} · 命中{r.get('5日命中率', '')}",
                量比=r.get("量比"),
            ))

    avoid_df = plan.get("avoid")
    if avoid_df is not None and not avoid_df.empty:
        for _, r in avoid_df.iterrows():
            rows.append(pick_row(
                module=f"{mod}·回避",
                account="腿②风控",
                ticker=str(r["代码"]),
                status="可开仓",
                direction="回避",
                action=str(r.get("建议", "不做多")),
                reason=f"[{r.get('规则', '')}] {r.get('原因', '')}",
            ))

    income = plan.get("income") or {}
    for ln in (income.get("lines") or [])[:8]:
        if ln and "⏸" not in ln and "无可行" not in ln:
            rows.append(pick_row(
                module=f"{mod}·收租",
                account="腿③期权",
                ticker="—",
                status="观望",
                direction="收租",
                reason=str(ln).strip(),
            ))

    if not rows:
        rows.append(empty_row(mod, "三腿", "今日三腿策略无命中（正常空仓）"))
    return rows


def run_flow_strategy_combo(cfg: dict) -> list[dict]:
    mod = "资金流向组合"
    try:
        from research.flow_strategy_backtest import run_today

        fcfg = _load_json_cfg(str(cfg.get("flow_strategy_config", "flow_strategy_config.json")))
        if cfg.get("quick"):
            fcfg["gainer_count"] = min(int(fcfg.get("gainer_count", 250)), 120)
        doc = run_today(fcfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "U_S2组合", e)]

    picks = doc.get("picks") or []
    if not picks:
        return [empty_row(mod, "U_S2组合", "今日无符合规则的标的（空仓）")]
    rows: list[dict] = []
    for p in picks:
        rows.append(pick_row(
            module=mod,
            account=str(doc.get("strategy", "flow")),
            ticker=str(p.get("代码", "")),
            status="可开仓",
            direction=str(p.get("方向", "")),
            action=str(p.get("策略动作", "")),
            reason=str(p.get("选股理由", "")),
            规律=p.get("规律"),
            量比=p.get("量比"),
        ))
    return rows


def run_vrp_signals(cfg: dict) -> list[dict]:
    mod = "VRP波动率"
    try:
        import vrp_daily as vd

        vcfg = _load_json_cfg(str(cfg.get("vrp_config", "vrp_config.json")))
        result = vd.run_vrp(vcfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "VRP", e)]

    rows: list[dict] = []
    for line in result.get("playbook") or []:
        if not line:
            continue
        actionable = "持有" in line or "Top" in line or "✅" in line
        rows.append(pick_row(
            module=mod,
            account="VRP引擎",
            ticker=str(result.get("etf", "SVIX")),
            status="可开仓" if actionable else "观望",
            direction="配置",
            reason=str(line),
        ))

    csp = result.get("csp_table")
    if isinstance(csp, pd.DataFrame) and not csp.empty:
        for _, r in csp.head(int(_load_json_cfg("vrp_config.json").get("csp", {}).get("top_n", 5))).iterrows():
            tk = str(r.get("代码", r.get("ticker", "")))
            if not tk:
                continue
            rows.append(pick_row(
                module=f"{mod}·CSP",
                account="CSP扫描",
                ticker=tk,
                status="观望",
                direction="卖Put",
                reason=f"RV {r.get('RV%', r.get('rv_pct', '—'))}% · 评分 {r.get('score', '—')}",
            ))

    if not rows:
        return [empty_row(mod, "VRP", "今日 VRP 无特殊信号")]
    return rows


def run_calendar_spread(cfg: dict) -> list[dict]:
    mod = "日历价差"
    try:
        import calendar_daily as cd

        ccfg = _load_json_cfg(str(cfg.get("calendar_config", "calendar_config.json")))
        result = cd.run_calendar_scan(ccfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "双日历", e)]

    plans = result.get("plans") or []
    if not plans:
        return [empty_row(mod, "双日历", "无可用方案")]
    rows: list[dict] = []
    for p in plans:
        rows.append(pick_row(
            module=mod,
            account="双日历",
            ticker=str(p.ticker),
            status="可开仓" if p.can_open else "观望",
            direction="做多波动" if p.can_open else "—",
            action="开双日历" if p.can_open else "暂停",
            reason=(
                f"IV Rank {p.iv_rank:.0%} · 付${p.debit_per_contract:,.0f} "
                f"θ≈+${p.theta_est_contract:,.0f}"
                if p.can_open
                else (p.flags[0] if p.flags else "条件未满足")
            ),
        ))
    return rows


def run_universal_playbook_fleet(cfg: dict) -> list[dict]:
    mod = "Universal舰队"
    try:
        from research.universal_playbook import (
            build_universal_playbook,
            playbook_to_dataframe,
            save_playbook,
        )

        ucfg = _load_json_cfg(str(cfg.get("tier_a_config", "tier_a_csp_config.json")))
        mode = str(cfg.get("playbook_mode", "stable"))
        pb = build_universal_playbook(
            account=float(cfg.get("account_size", 10_000)),
            mode=mode,
            cfg=ucfg,
        )
        if not cfg.get("dry_run"):
            save_playbook(pb)
        df = playbook_to_dataframe(pb)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "Playbook", e)]

    if df.empty:
        return [empty_row(mod, "Playbook", "今日无舰队方案")]
    rows: list[dict] = []
    for _, r in df.iterrows():
        ok = str(r.get("可开仓", "")).startswith("✅")
        rows.append(pick_row(
            module=mod,
            account=str(r.get("账户", "")),
            ticker=str(r.get("代码", "—")),
            status="可开仓" if ok else "观望",
            direction="收租" if ok else "—",
            action=str(r.get("策略", "")),
            reason=str(r.get("选股理由", "")),
            预计收租=r.get("预计收租$"),
        ))
    return rows


def run_sndk_iron_fleet(cfg: dict) -> list[dict]:
    mod = "SNDK铁鹰"
    try:
        import sndk_iron_daily as sid

        icfg = _load_json_cfg(str(cfg.get("sndk_iron_config", "sndk_iron_config.json")))
        result = sid.run_fleet(icfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "铁鹰舰队", e)]

    rows: list[dict] = []
    for r in result.get("fleet_rows") or []:
        p = r.get("plan")
        if p is None:
            rows.append(pick_row(
                module=mod,
                account=str(r.get("label", "")),
                ticker=str(r.get("ticker", "")),
                status="观望",
                direction="—",
                reason="无可行期权方案",
            ))
            continue
        rows.append(pick_row(
            module=mod,
            account=str(r.get("label", "")),
            ticker=str(r.get("ticker", "")),
            status="可开仓",
            direction="收租",
            action=sid.format_pick_action(p),
            reason=sid.format_pick_reason(p),
        ))
    if not rows:
        return [empty_row(mod, "铁鹰舰队", "舰队未启用或无方案")]
    return rows


def run_strategy_ranking(cfg: dict) -> list[dict]:
    mod = "策略排名"
    try:
        from research.strategy_ranker import evaluate_strategies

        scfg = _load_json_cfg(str(cfg.get("strategy_rank_config", "strategy_config.json")))
        result = evaluate_strategies(
            account=float(cfg.get("account_size", scfg.get("account_size", 10_000))),
            profile=str(cfg.get("strategy_profile", scfg.get("profile", "balanced"))),
        )
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "Top3", e)]

    rows: list[dict] = []
    for i, p in enumerate(result.get("top3") or [], 1):
        rows.append(pick_row(
            module=mod,
            account=f"Top{i}·{p.meta.category}",
            ticker="—",
            status="可开仓" if p.signal_ok else "观望",
            direction="配置",
            action=p.meta.name,
            reason=p.detail,
            得分=round(p.score, 2),
        ))
    pf = result.get("portfolio") or []
    for r in pf[:5]:
        if r.get("引擎") in ("现金储备", None):
            continue
        rows.append(pick_row(
            module=f"{mod}·仓位",
            account=str(r.get("引擎", "")),
            ticker=str(r.get("代码", "—")),
            status="可开仓",
            direction=str(r.get("方向", "配置")),
            reason=str(r.get("说明", "")),
            金额=r.get("金额$"),
        ))
    if not rows:
        return [empty_row(mod, "Top3", "今日无策略排名信号")]
    return rows


def run_screen_screener(cfg: dict) -> list[dict]:
    mod = "每日选股器"
    scfg = _load_json_cfg(str(cfg.get("screen_config", "screen_config.json")))
    try:
        import screen_daily as sd

        result = sd.run(scfg)
        merged = result.get("merged", pd.DataFrame())
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "筛选器", e)]

    if merged is None or merged.empty:
        return [empty_row(mod, "筛选器", "筛选后无符合标的")]
    rows: list[dict] = []
    top_n = int(cfg.get("screen_top_n", 10))
    for _, r in merged.head(top_n).iterrows():
        sig = str(r.get("当前信号", ""))
        actionable = any(x in sig for x in ("买", "多", "开"))
        rows.append(pick_row(
            module=mod,
            account=str(scfg.get("strategy", "screen")),
            ticker=str(r.get("代码", "")),
            status="可开仓" if actionable else "观望",
            direction=sig or "—",
            reason=str(r.get("选股理由", "")),
            涨幅=r.get("涨幅%"),
        ))
    return rows


def run_watchlist_scan(cfg: dict) -> list[dict]:
    mod = "自选股扫描"
    scfg = _load_json_cfg(str(cfg.get("scan_config", "scan_config.json")))
    try:
        import scan_daily as sd

        table, _ = sd.run_scan(scfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "自选股", e)]

    if table is None or table.empty:
        return [empty_row(mod, "自选股", "自选股无扫描结果")]
    rows: list[dict] = []
    for _, r in table.iterrows():
        action = str(r.get("今日动作", ""))
        tk = str(r.get("代码", r.get("ticker", "")))
        actionable = any(m in action for m in ("🟢", "🔴"))
        rows.append(pick_row(
            module=mod,
            account=str(scfg.get("strategy", "signal")),
            ticker=tk,
            status="可开仓" if actionable else "观望",
            direction=action,
            reason=f"目标仓位 {r.get('目标仓位', '—')}",
        ))
    return rows


def run_ticker_pattern_standalone(cfg: dict, *, bull: bool) -> list[dict]:
    """独立 Meme 规律扫描（与 meme_long 同源，单独模块标签）。"""
    mod = "规律·Meme独立"
    try:
        import ticker_pattern_daily as tpd

        doc = tpd.run_scan(cfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, "Meme", e)]

    rows: list[dict] = []
    for p in doc.get("picks") or []:
        rows.append(pick_row(
            module=mod,
            account="Meme规律",
            ticker=str(p.get("代码", "")),
            status=str(p.get("状态", "观望")),
            direction=str(p.get("方向", "")),
            action=str(p.get("策略动作", "")),
            reason=str(p.get("选股理由", "")),
        ))
    if not rows:
        return [empty_row(mod, "Meme", "今日无 Meme 规律信号")]
    return rows


# 模块 ID → (runner, needs_bull)
RUNNER_REGISTRY: dict[str, tuple[Callable[..., list[dict]], bool]] = {
    "pattern_daily": (run_pattern_three_leg, True),
    "flow_strategy": (run_flow_strategy_combo, False),
    "vrp": (run_vrp_signals, False),
    "calendar": (run_calendar_spread, False),
    "universal_playbook": (run_universal_playbook_fleet, False),
    "sndk_iron": (run_sndk_iron_fleet, False),
    "strategy_rank": (run_strategy_ranking, False),
    "screen_daily": (run_screen_screener, False),
    "scan_daily": (run_watchlist_scan, False),
    "ticker_pattern": (run_ticker_pattern_standalone, True),
}

# quick 模式跳过的重型模块
HEAVY_MODULES = frozenset({
    "pattern_daily",
    "flow_strategy",
    "screen_daily",
    "universal_playbook",
    "strategy_rank",
    "gain15",
    "capital_flow",
    "meme_long",
})


def run_registered(module_id: str, cfg: dict, *, bull: bool) -> list[dict]:
    if cfg.get("quick") and module_id in HEAVY_MODULES:
        return [empty_row(
            module_id,
            "—",
            "quick 模式跳过重型扫描",
        )]
    entry = RUNNER_REGISTRY.get(module_id)
    if not entry:
        return []
    fn, needs_bull = entry
    if needs_bull:
        return fn(cfg, bull=bull)
    return fn(cfg)
