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


_DIP_SKIP_TICKERS = frozenset({"SPY", "QQQ", "DIA", "IWM", "XLE", "XLF", "XLK", "SOXL", "TQQQ"})


def resolve_dip_universe(cfg: dict) -> list[str]:
    """解析买跌扫描股池：liquid100 / sp500（默认，+纳指100+高贝塔）/ broad（全市场多榜）。

    流动性由 scan_today 的成交额闸门保证，故这里只管「广度」。可离线单测（monkeypatch 拉取函数）。"""
    from research.gainer_daily_backtest import LIQUID100

    name = str(cfg.get("mean_reversion_universe", "sp500")).lower()
    base: list[str] = list(LIQUID100)
    if name == "liquid100":
        base = list(LIQUID100)
    elif name == "broad":
        from quant.screener import fetch_broad_universe
        base = fetch_broad_universe(
            screen_count=int(cfg.get("mean_reversion_screen_count", 200)),
            extra=list(LIQUID100),
        )
    else:  # sp500（默认）：标普500 ∪ 纳指100 ∪ 高贝塔流动性票，覆盖全大中盘
        from quant.screener import fetch_nasdaq100_tickers, fetch_sp500_tickers
        base = list(fetch_sp500_tickers()) + list(fetch_nasdaq100_tickers()) + list(LIQUID100)

    uni = sorted({str(t).strip().upper() for t in base if str(t).strip()} - _DIP_SKIP_TICKERS)
    cap = int(cfg.get("mean_reversion_max_universe", 600))
    return uni[:cap]


def run_mean_reversion_dip(cfg: dict, *, bull: bool) -> list[dict]:
    """短期均值回归·顺势买跌（Connors RSI-2 式）：上升趋势中的超卖回调。

    回测（6 年、196 只流动性标的、含成本）：OOS 胜率 ~73%、年化 ~150%、回撤 ~-17%。
    实验证实加大盘择时(regime)反而损失（最肥的买点在恐慌杀跌期），故全天候运行。
    股池默认扩到 S&P500 ∪ 纳指100（含高贝塔），流动性由成交额闸门保证。"""
    mod = "均值回归·买跌"
    acct = "流动性大盘股"
    try:
        from datetime import date, timedelta

        from research.mean_reversion_dip import PROD_PARAMS, scan_today
        from quant.providers import DataConfig, get_provider, reset_provider_cache

        reset_provider_cache()
        yahoo = get_provider(DataConfig(provider="yahoo"))
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=420)).isoformat()  # >200日均线暖机
        universe = resolve_dip_universe(cfg)
        batch = yahoo.fetch_batch(universe, start, end)
        top_n = int(cfg.get("mean_reversion_top_n", 8))
        cands = scan_today(batch, PROD_PARAMS, top_n=top_n)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, acct, e)]

    if not cands:
        return [empty_row(mod, acct, "今日无上升趋势中的超卖回调标的（正常空仓）")]

    regime_note = "牛市" if bull else "弱市"
    rows: list[dict] = []
    for c in cands:
        reason = (
            f"{regime_note} · 上升趋势(高于200日均线 {c['距SMA200%']:+.1f}%) · "
            f"回调跌破5日均线 {c['距SMA5%']:+.1f}% · RSI(2)={c['RSI2']}（超卖） · "
            f"成交额 ${c['成交额M']:.0f}M"
        )
        rows.append(pick_row(
            module=mod,
            account=acct,
            ticker=str(c["代码"]),
            status="可开仓",
            direction="做多",
            action="次日开盘买入·反弹了结",
            reason=reason,
            现价=c["现价"],
            RSI2=c["RSI2"],
            成交额M=c["成交额M"],
            买进时机="次日开盘买入",
            卖出时机="反弹+10% / 首个收盘转正 / 持满10日，硬止损 8%",
        ))
    return rows


def run_longshort_combo(cfg: dict) -> list[dict]:
    """多空组合 · Extreme20 + Flow · 质量分高胜率过滤。"""
    mod = "多空组合"
    acct = "L/S组合"
    modules = cfg.get("modules") or {}
    if not modules.get("longshort_combo", False):
        return []
    if cfg.get("quick"):
        return [empty_row(mod, acct, "quick 模式跳过重型扫描")]
    try:
        from quant.longshort_combo_strategy import config_from_dict, scan_live

        lpath = str(cfg.get("longshort_combo_config", "longshort_combo_config.json"))
        lcfg = config_from_dict(_load_json_cfg(lpath))
        picks = scan_live(lcfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, acct, e)]

    if picks is None or picks.empty:
        return [empty_row(mod, acct, "今日无高质量多空信号（正常空仓）")]

    rows: list[dict] = []
    for _, s in picks.iterrows():
        sid = str(s.get("策略ID", s.get("leg", "")))
        rows.append(pick_row(
            module=mod,
            account=acct,
            ticker=str(s.get("代码", "")),
            status="可开仓",
            direction=str(s.get("方向", s.get("side", "—"))),
            action=str(s.get("策略", s.get("策略动作", sid))),
            reason=str(s.get("依据", s.get("选股理由", "")))[:400],
            策略ID=sid,
            持有=s.get("持有"),
            入场=s.get("入场"),
            止损价=s.get("止损价≈"),
            止盈价=s.get("止盈价≈"),
            side=s.get("side"),
            质量分=s.get("质量分"),
            leg=s.get("leg"),
        ))
    return rows


def run_extreme20(cfg: dict) -> list[dict]:
    """暴涨/暴跌 ≥20% 事件策略 L1/S1/L2/S2。"""
    mod = "Extreme20"
    acct = "暴涨暴跌20%"
    modules = cfg.get("modules") or {}
    if not modules.get("extreme20", False):
        return []
    if cfg.get("quick"):
        return [empty_row(mod, acct, "quick 模式跳过重型扫描")]
    try:
        from quant.extreme20_strategy import config_from_dict, scan_live

        epath = str(cfg.get("extreme20_config", "extreme20_config.json"))
        ecfg = config_from_dict(_load_json_cfg(epath))
        picks = scan_live(ecfg, screen_count=int(_load_json_cfg(epath).get("screen_count", 300)))
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, acct, e)]

    if picks is None or picks.empty:
        return [empty_row(mod, acct, "今日无命中（约85%交易日空仓，属正常）")]

    rows: list[dict] = []
    for _, s in picks.iterrows():
        sid = str(s.get("策略ID", ""))
        rows.append(pick_row(
            module=f"Extreme20·{sid}" if sid else mod,
            account=acct,
            ticker=str(s.get("代码", "")),
            status="可开仓",
            direction=str(s.get("方向", "—")),
            action=str(s.get("策略", "")),
            reason=str(s.get("依据", ""))[:400],
            策略ID=sid,
            持有=s.get("持有"),
            入场=s.get("入场"),
            止损价=s.get("止损价≈"),
            止盈价=s.get("止盈价≈"),
            止损pct=s.get("止损%"),
            止盈pct=s.get("止盈%"),
            side=s.get("side"),
        ))
    return rows


def run_whipsaw_short(cfg: dict) -> list[dict]:
    """涨幅榜暴涨乏力 → 卖 Call 信用价差（定风险做空）。"""
    mod = "做空涨幅榜"
    acct = "涨幅榜BCS"
    modules = cfg.get("modules") or {}
    if not modules.get("whipsaw_short", False):
        return []
    if cfg.get("quick"):
        return [empty_row(mod, acct, "quick 模式跳过涨幅榜扫描")]
    try:
        import whipsaw_short_daily as ws

        wpath = str(cfg.get("whipsaw_short_config", "whipsaw_short_config.json"))
        wcfg = ws.load_config(ROOT / wpath)
        plan = ws.run_scan(wcfg)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, acct, e)]

    if plan.get("note"):
        return [empty_row(mod, acct, str(plan["note"]))]

    actionable = [
        c for c in plan.get("candidates") or []
        if c.get("信号") == "卖Call价差" and int(c.get("建议张数") or 0) > 0
    ]
    if not actionable:
        stats = plan.get("scan_stats") or {}
        n_cand = int(stats.get("候选") or 0)
        note = "今日无「暴涨乏力+可行价差」标的（正常空仓）"
        if n_cand > 0:
            note = f"候选 {n_cand} 只，无可成交价差（正常观望）"
        return [empty_row(mod, acct, note)]

    weak = bool((plan.get("market") or {}).get("弱市"))
    rows: list[dict] = []
    for c in actionable:
        reason = (
            f"模式[{plan.get('mode', '-')}] · Top{c.get('榜单排名')} · "
            f"涨{c.get('涨幅%')}% · 量比{c.get('量比')} · 收盘强度{c.get('收盘强度')} · "
            f"{'弱市×' + str(c.get('仓位倍数', 1)) if weak else '强市×1'}"
        )
        rows.append(pick_row(
            module=mod,
            account=acct,
            ticker=str(c["代码"]),
            status="可开仓",
            direction="做空",
            action=f"卖Call价差 {c.get('结构', '')}",
            reason=reason,
            **{
                "现价": c.get("现价"),
                "建议张数": c.get("建议张数"),
                "权利金$": c.get("收权利金$"),
                "最大亏$": c.get("最大亏$"),
                "到期": c.get("到期"),
                "结构": c.get("结构"),
            },
        ))
    return rows


def run_gainer10(cfg: dict) -> list[dict]:
    """日涨>10%+亿级成交 · 分板块多空 + 续涨 A/B。"""
    mod = "Gainer10+"
    acct = "动量续涨"
    modules = cfg.get("modules") or {}
    if not modules.get("gainer10", False):
        return []
    if cfg.get("quick"):
        return [empty_row(mod, acct, "quick 模式跳过涨幅榜扫描")]
    try:
        import gainer10_daily as g10

        gpath = str(cfg.get("gainer10_config", "gainer10_config.json"))
        raw = g10.load_config(ROOT / gpath)
        plan = g10.run_gainer10_scan(g10.cfg_from_dict(raw))
        g10.save_outputs(plan, raw)
    except Exception as e:  # noqa: BLE001
        return [fail_row(mod, acct, e)]

    rows: list[dict] = []
    for key, label, direction, status in [
        ("buy_sector", "分板块多", "做多", "可开仓"),
        ("buy_a", "续涨A", "做多", "可开仓"),
        ("buy_b", "续涨B", "做多", "可开仓"),
        ("short_sector", "分板块空", "做空", "可开仓"),
        ("short_s", "做空S", "做空", "可开仓"),
    ]:
        for p in plan.get(key) or []:
            extra = ""
            if p.get("限价入场"):
                extra = f" · 限价${p['限价入场']}"
                if p.get("止盈_pct"):
                    extra += f" TP{p['止盈_pct']}%/SL{p.get('止损_pct')}%"
            rows.append(pick_row(
                module=f"Gainer10+·{p.get('信号', label)}",
                account=acct,
                ticker=str(p["代码"]),
                status=status,
                direction=direction,
                action=str(p.get("动作", "")),
                reason=(
                    f"涨{p.get('涨幅_pct')}% ${p.get('成交额M')}M · {p.get('板块')} · "
                    f"跳空{p.get('跳空_pct')}% 乖离{p.get('乖离20_pct')}% RSI{p.get('RSI')} · "
                    f"{p.get('规则说明')} · 历史{p.get('历史胜率')} {p.get('历史均收益')}{extra}"
                )[:400],
                信号=str(p.get("信号", label)),
                现价=p.get("现价"),
                限价入场=p.get("限价入场"),
                止盈pct=p.get("止盈_pct"),
                止损pct=p.get("止损_pct"),
            ))

    if not rows:
        note = plan.get("note") or "今日无续涨/回避信号（正常空仓）"
        return [empty_row(mod, acct, note)]
    return rows


# 核心策略中经 run_registered 调度的子集（其余在 daily_pick.py 内联）
RUNNER_REGISTRY: dict[str, tuple[Callable[..., list[dict]], bool]] = {
    "longshort_combo": (run_longshort_combo, False),
    "flow_strategy": (run_flow_strategy_combo, False),
    "vrp": (run_vrp_signals, False),
    "sndk_iron": (run_sndk_iron_fleet, False),
    "extreme20": (run_extreme20, False),
    "whipsaw_short": (run_whipsaw_short, False),
    "gainer10": (run_gainer10, False),
}

# quick 模式跳过的重型模块
HEAVY_MODULES = frozenset({
    "longshort_combo",
    "flow_strategy",
    "gain15",
    "extreme20",
    "whipsaw_short",
    "gainer10",
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
