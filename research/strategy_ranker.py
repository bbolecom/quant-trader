"""策略排名器：汇总历史回测分 + 当日信号，输出 Top3 与 $10k 仓位表。

静态分数来自本会话各回测结论；当日分来自收入引擎 / 周铁鹰 / 日历 / 动量等实时扫描。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StrategyMeta:
    id: str
    name: str
    category: str  # income / growth / niche / avoid
    win_rate: float
    ann_return: float
    max_dd: float
    sharpe: float
    note: str = ""


# 历史回测锚点（非每日重算，避免启动过慢）
CATALOG: list[StrategyMeta] = [
    StrategyMeta("tier_a_csp", "Tier A CSP·SNDK圣杯", "income",
                 0.966, 0.567, 0.052, 1.50, "50/15/85达标：δ0.25 MA50 仓位50%"),
    StrategyMeta("call_spread", "卖看涨价差·收入核心", "income",
                 0.88, 0.28, 0.20, 1.40, "高振幅票铁鹰/卖Call价差，弱市亦可用"),
    StrategyMeta("weekly_soup", "周铁鹰·高波喝汤", "income",
                 0.87, 0.22, 0.25, 1.20, "SNDK类站上MA50才开，偏斜铁鹰"),
    StrategyMeta("csp", "卖看跌CSP", "income",
                 0.82, 0.14, 0.31, 1.10, "最稳底仓，需低价股"),
    StrategyMeta("momentum_weekly", "温和动量做多 Top2", "growth",
                 0.585, 0.20, 0.35, 0.95, "每周约1笔，仅牛市"),
    StrategyMeta("panic_rebound", "恐慌反弹做多", "growth",
                 0.544, 0.35, 0.36, 1.30,
                 "深跌≥30%票当日再暴跌≥10%→次日开盘抄底；全市场OOS年化+54~77%"),
    StrategyMeta("medallion_long", "Medallion·做多小盘", "growth",
                 0.51, 0.46, 0.51, 1.00, "系统化横截面，回撤深"),
    StrategyMeta("calendar", "双日历·IV低位", "niche",
                 0.58, 0.12, 0.45, 0.60, "仅IV Rank≤40%且无财报"),
    StrategyMeta("short_overheat", "做空猛涨小盘", "avoid",
                 0.49, -0.50, 0.99, -1.00, "回测-98%，禁用"),
]


@dataclass
class StrategyPick:
    meta: StrategyMeta
    score: float
    signal_ok: bool
    regime_ok: bool
    detail: str
    trades: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def rank_label(self) -> str:
        if not self.signal_ok:
            return "⏸ 今日无信号"
        if self.meta.category == "avoid":
            return "🚫 禁用"
        return "✅ 可执行"


def _static_score(m: StrategyMeta) -> float:
    if m.category == "avoid":
        return -1.0
    dd_pen = max(0.0, 1.0 + m.max_dd)  # max_dd 为负，如 -0.31 → 0.69
    return 0.35 * min(m.sharpe / 1.5, 1.0) + 0.30 * m.win_rate + 0.20 * dd_pen + 0.15 * min(m.ann_return / 0.5, 1.0)


def _load_soup_cfg(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"tickers": ["SNDK"], "account_size": 10_000, "iron_condor": {"enabled": True}}


def evaluate_strategies(
    account: float = 10_000.0,
    profile: str = "balanced",
    soup_cfg_path: Path | None = None,
    calendar_cfg_path: Path | None = None,
) -> dict:
    """拉取当日各引擎信号并打分排名。"""
    from research.income_engine import build_income_plan, get_regime
    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from quant.data import fetch_history
    from quant import decline_income as di
    from quant.calendar_spread import scan_calendar_plans

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    regime = get_regime(yahoo)
    bull = regime.bull
    income = build_income_plan(account=account, count=120, top_n=5)

    picks: list[StrategyPick] = []

    # --- ① 卖看涨价差 ---
    cs = income.get("call_spreads", pd.DataFrame())
    cs_act = cs[cs["建议张数"] > 0] if not cs.empty else pd.DataFrame()
    cs_trades = cs_act.head(3).to_dict("records") if not cs_act.empty else []
    cs_detail = ""
    if not cs_act.empty:
        top = cs_act.iloc[0]
        cs_detail = (f"{top['代码']} 卖C${top['卖Call']:,.0f}/买C${top['买Call']:,.0f} "
                     f"收${top['净权利金$']:,.0f}×{int(top['建议张数'])}张")
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "call_spread"),
        score=_static_score(next(m for m in CATALOG if m.id == "call_spread")) * (1.0 if cs_trades else 0.35),
        signal_ok=bool(cs_trades),
        regime_ok=True,
        detail=cs_detail or "今日榜单无合适卖看涨标的",
        trades=cs_trades,
    ))

    # --- ② 周铁鹰喝汤 ---
    soup_cfg = _load_soup_cfg(soup_cfg_path or ROOT / "weekly_soup_config.json")
    soup_account = float(soup_cfg.get("account_size", account))
    ic = soup_cfg.get("iron_condor", {})
    soup_trades: list[dict] = []
    soup_flags: list[str] = []
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(soup_cfg.get("lookback_days", 550)))).isoformat()
    tickers = soup_cfg.get("tickers") or ["SNDK"]
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace(",", " ").split()]
    for tk in tickers[:5]:
        try:
            df = fetch_history(str(tk).upper(), start, end)
            plan = di.weekly_put_soup_plan(
                str(tk).upper(), df, account_size=soup_account,
                add_call=bool(ic.get("enabled", True)),
                call_delta=float(ic.get("call_delta", 0.05)),
                call_width=float(ic["call_width"]) if ic.get("call_width") else None,
            )
            if plan:
                row = {
                    "代码": plan.ticker, "卖Put": plan.short_strike, "买Put": plan.long_strike,
                    "收租$": plan.credit_per_contract, "保证金$": plan.margin_per_contract,
                    "建议张数": plan.max_contracts or 0, "可开": plan.can_open,
                }
                if plan.iron_condor and plan.call_short_strike:
                    row["卖Call"] = plan.call_short_strike
                    row["买Call"] = plan.call_long_strike
                    row["合计收$"] = plan.total_credit_per_contract
                soup_trades.append(row)
                if plan.can_open and not soup_flags:
                    soup_flags.append(f"{plan.ticker} 可开")
                elif plan.flags:
                    soup_flags.extend(plan.flags[:1])
        except Exception:  # noqa: BLE001
            continue
    open_soup = [t for t in soup_trades if t.get("可开") and t.get("建议张数", 0) > 0]
    soup_detail = ""
    if open_soup:
        t0 = open_soup[0]
        soup_detail = (f"{t0['代码']} 卖P${t0['卖Put']:,.0f} "
                       f"{'铁鹰' if '卖Call' in t0 else 'Put价差'} "
                       f"收${t0.get('合计收$', t0['收租$']):,.0f}×{t0['建议张数']}张")
    elif soup_trades:
        soup_detail = soup_flags[0] if soup_flags else "未站上MA50或条件不足"
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "weekly_soup"),
        score=_static_score(next(m for m in CATALOG if m.id == "weekly_soup")) * (1.0 if open_soup else 0.4),
        signal_ok=bool(open_soup),
        regime_ok=True,
        detail=soup_detail or "无喝汤方案",
        trades=open_soup or soup_trades[:2],
        flags=soup_flags,
    ))

    # --- ③ Tier A CSP（三重目标达标，优先于通用 CSP）---
    tier_df = income.get("tier_a_csp", pd.DataFrame())
    tier_cfg = income.get("tier_a_cfg") or {}
    tier_trades: list[dict] = []
    tier_detail = ""
    if tier_df is not None and not tier_df.empty:
        open_t = tier_df[tier_df["可开仓"] == "✅"] if "可开仓" in tier_df.columns else pd.DataFrame()
        tier_trades = open_t.head(5).to_dict("records") if not open_t.empty else tier_df.head(5).to_dict("records")
        fleet_on = (tier_cfg.get("fleet") or {}).get("enabled")
        if not open_t.empty:
            if fleet_on:
                tier_detail = " · ".join(
                    f"{r.get('账户', '')}{r['代码']}×{int(r['建议张数'])}"
                    for r in open_t.to_dict("records")[:3]
                )
            else:
                t0 = open_t.iloc[0]
                tier_detail = (
                    f"{t0['代码']} 卖P${t0['卖Put']:,.0f}×{int(t0['建议张数'])}张 "
                    f"收${t0['权利金$']:,.0f}"
                )
        elif not tier_df.empty:
            tier_detail = str(tier_df.iloc[0].get("提示", "未站上MA50"))[:80]
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "tier_a_csp"),
        score=_static_score(next(m for m in CATALOG if m.id == "tier_a_csp")) * (1.0 if tier_trades and tier_trades[0].get("可开仓") == "✅" else 0.35),
        signal_ok=bool(tier_trades) and tier_trades[0].get("可开仓") == "✅",
        regime_ok=True,
        detail=tier_detail or "Tier A CSP 无信号",
        trades=tier_trades,
        flags=[tier_cfg.get("name", "Tier A CSP")] if tier_cfg else [],
    ))

    # --- ④ 通用 CSP ---
    csp = income.get("csp", pd.DataFrame())
    csp_trades = csp.head(3).to_dict("records") if not csp.empty else []
    csp_detail = ""
    if not csp.empty:
        t0 = csp.iloc[0]
        csp_detail = f"{t0['代码']} 卖P${t0['卖Put']:,.0f} 收${t0['权利金$']:,.0f}"
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "csp"),
        score=_static_score(next(m for m in CATALOG if m.id == "csp")) * (0.9 if csp_trades else 0.3),
        signal_ok=bool(csp_trades),
        regime_ok=True,
        detail=csp_detail or "无价位适配CSP",
        trades=csp_trades,
    ))

    # --- ⑤ 温和动量 ---
    longs = income.get("longs", pd.DataFrame())
    mom_trades = longs.head(2).to_dict("records") if (bull and longs is not None and not longs.empty) else []
    mom_detail = ""
    if mom_trades:
        codes = "、".join(str(t["代码"]) for t in mom_trades)
        mom_detail = f"做多 {codes}（持1日）"
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "momentum_weekly"),
        score=_static_score(next(m for m in CATALOG if m.id == "momentum_weekly")) * (1.0 if mom_trades else 0.2),
        signal_ok=bool(mom_trades),
        regime_ok=bull,
        detail=mom_detail or ("弱市关闭做多" if not bull else "今日无温和动量信号"),
        trades=mom_trades,
        flags=[] if bull else ["SPY<MA50，动量引擎关闭"],
    ))

    # --- ⑤b 恐慌反弹做多（全市场方向性最优） ---
    pr_trades: list[dict] = []
    pr_detail = ""
    try:
        from quant.panic_rebound import PanicReboundConfig, scan_live
        pr_picks = scan_live(PanicReboundConfig())
        if pr_picks is not None and not pr_picks.empty:
            pr_trades = pr_picks.head(3).to_dict("records")
            t0 = pr_picks.iloc[0]
            pr_detail = (
                f"{t0['代码']} 当日{t0['当日跌%']:+.1f}% 前20日{t0['前20日跌%']:+.0f}% "
                f"→ 次日开盘做多 止损≈${t0['止损价≈']}/止盈≈${t0['止盈价≈']}"
            )
    except Exception as e:  # noqa: BLE001
        pr_detail = f"扫描跳过：{e}"
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "panic_rebound"),
        score=_static_score(next(m for m in CATALOG if m.id == "panic_rebound")) * (1.0 if pr_trades else 0.25),
        signal_ok=bool(pr_trades),
        regime_ok=True,
        detail=pr_detail or "今日无恐慌反弹候选",
        trades=pr_trades,
    ))

    # --- ⑥ 双日历 ---
    cal_cfg_path = calendar_cfg_path or ROOT / "calendar_config.json"
    cal_tickers = ["NVDA", "PLTR", "AMD", "META", "SNDK", "QQQ"]
    if cal_cfg_path.exists():
        cal_cfg = json.loads(cal_cfg_path.read_text(encoding="utf-8"))
        cal_tickers = cal_cfg.get("tickers") or cal_tickers
        if isinstance(cal_tickers, str):
            cal_tickers = [t.strip() for t in cal_tickers.replace(",", " ").split()]
    cal_start = (date.today() - timedelta(days=400)).isoformat()
    cal_plans, _ = scan_calendar_plans(
        [str(t).upper() for t in cal_tickers[:6]], cal_start, end,
        account_size=account, iv_pct_max=0.40, max_er=0.45,
    )
    cal_open = [p for p in cal_plans if p.can_open]
    cal_trades = [{
        "代码": p.ticker, "净付$": p.debit_per_contract, "7日θ$": p.theta_est_contract,
        "Call": p.call_strike, "Put": p.put_strike, "IV Rank": f"{p.iv_rank:.0%}",
    } for p in cal_open[:3]]
    cal_detail = ""
    if cal_open:
        p0 = cal_open[0]
        cal_detail = f"{p0.ticker} IV Rank {p0.iv_rank:.0%} 付${p0.debit_per_contract:,.0f} θ≈+${p0.theta_est_contract:,.0f}"
    elif cal_plans:
        cal_detail = cal_plans[0].flags[0] if cal_plans[0].flags else "IV偏高"
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "calendar"),
        score=_static_score(next(m for m in CATALOG if m.id == "calendar")) * (1.0 if cal_open else 0.15),
        signal_ok=bool(cal_open),
        regime_ok=True,
        detail=cal_detail or "无日历候选",
        trades=cal_trades,
    ))

    # --- 做空猛涨：永久禁用 ---
    picks.append(StrategyPick(
        meta=next(m for m in CATALOG if m.id == "short_overheat"),
        score=-1.0,
        signal_ok=False,
        regime_ok=False,
        detail="回测5年-98%，不做",
        flags=["禁用"],
    ))

    # 按 profile 加权
    weights = {"income": 1.0, "growth": 1.0, "niche": 0.7, "avoid": 0.0}
    if profile == "income":
        weights = {"income": 1.3, "growth": 0.5, "niche": 0.6, "avoid": 0.0}
    elif profile == "growth":
        weights = {"income": 0.7, "growth": 1.4, "niche": 0.5, "avoid": 0.0}

    for i, p in enumerate(picks):
        w = weights.get(p.meta.category, 1.0)
        picks[i] = StrategyPick(
            meta=p.meta, score=p.score * w, signal_ok=p.signal_ok, regime_ok=p.regime_ok,
            detail=p.detail, trades=p.trades, flags=p.flags,
        )

    ranked = sorted(
        [p for p in picks if p.meta.category != "avoid"],
        key=lambda x: (x.signal_ok, x.score),
        reverse=True,
    )
    top3 = ranked[:3]

    portfolio = _build_portfolio(account, profile, ranked, regime.bull)

    return {
        "regime": regime,
        "profile": profile,
        "account": account,
        "ranked": ranked,
        "top3": top3,
        "portfolio": portfolio,
        "income_plan": income,
    }


def _build_portfolio(account: float, profile: str, ranked: list[StrategyPick], bull: bool) -> list[dict]:
    """把 Top 策略落成 $10k 仓位表（保证金/现金占比）。"""
    slots: list[dict] = []
    by_id = {p.meta.id: p for p in ranked}

    # 收入核心：卖看涨价差
    cs = by_id.get("call_spread")
    if cs and cs.signal_ok and cs.trades:
        budget = account * (0.50 if profile == "balanced" else 0.70 if profile == "income" else 0.30)
        used = 0.0
        for tr in cs.trades[:3]:
            margin = float(tr.get("最大亏损$", 0)) * int(tr.get("建议张数", 0))
            if margin <= 0 or used + margin > budget:
                continue
            used += margin
            slots.append({
                "引擎": "卖看涨价差",
                "代码": tr["代码"],
                "结构": f"卖Call ${tr['卖Call']:,.0f} / 买Call ${tr['买Call']:,.0f}",
                "张数": int(tr["建议张数"]),
                "预估收租$": tr.get("预计收租$", tr.get("净权利金$", 0)),
                "最大亏损$": tr.get("最大亏损$", 0),
                "占用$": margin,
                "占比%": round(margin / account * 100, 1),
            })

    # 周铁鹰
    soup = by_id.get("weekly_soup")
    if soup and soup.signal_ok and soup.trades:
        budget = account * (0.30 if profile == "balanced" else 0.25 if profile == "income" else 0.15)
        for tr in soup.trades[:2]:
            margin = float(tr.get("保证金$", 0)) * int(tr.get("建议张数", 0))
            if margin <= 0 or margin > budget:
                continue
            struct = "偏斜铁鹰" if "卖Call" in tr else "Put价差"
            slots.append({
                "引擎": "周铁鹰喝汤",
                "代码": tr["代码"],
                "结构": struct,
                "张数": int(tr["建议张数"]),
                "预估收租$": tr.get("合计收$", tr.get("收租$", 0)),
                "最大亏损$": margin,
                "占用$": margin,
                "占比%": round(margin / account * 100, 1),
            })
            break

    # 动量（现金）
    mom = by_id.get("momentum_weekly")
    if bull and mom and mom.signal_ok and mom.trades and profile != "income":
        cash_each = account * (0.20 if profile == "balanced" else 0.50) / min(len(mom.trades), 2)
        for tr in mom.trades[:2]:
            slots.append({
                "引擎": "温和动量",
                "代码": tr.get("代码", ""),
                "结构": "市价做多持1日",
                "张数": 1,
                "预估收租$": "—",
                "最大亏损$": round(cash_each * 0.05, 0),
                "占用$": round(cash_each, 0),
                "占比%": round(cash_each / account * 100, 1),
            })

    # 日历（ niche，小仓）
    cal = by_id.get("calendar")
    if cal and cal.signal_ok and cal.trades:
        tr = cal.trades[0]
        debit = float(tr.get("净付$", 0))
        if debit <= account * 0.15:
            slots.append({
                "引擎": "双日历",
                "代码": tr["代码"],
                "结构": f"C${tr['Call']:,.0f}/P${tr['Put']:,.0f}",
                "张数": 1,
                "预估收租$": tr.get("7日θ$", 0),
                "最大亏损$": debit,
                "占用$": debit,
                "占比%": round(debit / account * 100, 1),
            })

    total_used = sum(float(s["占用$"]) for s in slots if isinstance(s["占用$"], (int, float)))
    cash_left = account - total_used
    if slots:
        slots.append({
            "引擎": "现金储备",
            "代码": "—",
            "结构": "应对追加保证金/新信号",
            "张数": 0,
            "预估收租$": "—",
            "最大亏损$": 0,
            "占用$": round(max(cash_left, 0), 0),
            "占比%": round(max(cash_left, 0) / account * 100, 1),
        })
    return slots


def format_playbook(result: dict) -> list[str]:
    reg = result["regime"]
    lines = [
        f"大盘：{reg.label}（SPY {reg.spy:.2f} / MA50 {reg.ma50:.2f}）",
        f"账户 ${result['account']:,.0f} · 风格 {result['profile']}",
        "",
        "【今日策略 Top3】",
    ]
    for i, p in enumerate(result["top3"], 1):
        lines.append(f"  {i}. {p.meta.name} {p.rank_label}  分={p.score:.2f}")
        lines.append(f"     {p.detail}")
    lines.append("")
    lines.append("【$10k 仓位表】")
    pf = result.get("portfolio") or []
    if not pf:
        lines.append("  今日无可执行仓位，观望或仅保留现金。")
    else:
        for row in pf:
            if row["引擎"] == "现金储备":
                lines.append(f"  · 现金储备 ${row['占用$']:,.0f} ({row['占比%']}%)")
            else:
                lines.append(
                    f"  · {row['引擎']} {row['代码']} {row['结构']} "
                    f"{row['张数']}张 占用${row['占用$']:,.0f}({row['占比%']}%)"
                )
    lines.extend([
        "",
        "纪律：价差永不裸卖 · 50%权利金止盈 · 财报前1周不开 · 做空猛涨小盘禁用",
    ])
    return lines
