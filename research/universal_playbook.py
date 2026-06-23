"""普适策略 Playbook：选股理由 + 具体执行明细（并行铁鹰舰队 / 组合引擎）。

用法：
    python research/universal_playbook.py
    python research/universal_playbook.py --account 10000 --mode aggressive
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAYBOOK_JSON = ROOT / "research" / "universal_playbook_today.json"
PLAYBOOK_CSV = ROOT / "research" / "universal_playbook_today.csv"

# 全市场 continued_search 验证过的 ETF 舰队 — 每只的选股逻辑
TICKER_RATIONALE: dict[str, dict] = {
    "SPY": {
        "类型": "大盘 ETF",
        "角色": "并行铁鹰核心腿 #1",
        "选股理由": [
            "日均成交额 >$400 亿，全美流动性第一，价差窄、滑点小",
            "RV 通常 14–20%，低波 → 铁鹰 Put Delta 0.10 归零概率高",
            "全市场回测：单腿铁鹰胜率 ~100%、回撤 ~0%（2019 至今）",
            "与 QQQ 行业暴露互补，适合同一账户并行开 2 槽",
        ],
        "排除": "非单票，无 SNDK 式个例风险",
    },
    "QQQ": {
        "类型": "科技 ETF",
        "角色": "并行铁鹰核心腿 #2",
        "选股理由": [
            "日均成交额 >$350 亿，期权链极厚",
            "波动略高于 SPY 但仍在低–中波桶，权利金更厚",
            "continued_search 最优组合：SPY+QQQ 并行 → 年化 ~16.2%",
            "与 SPY 同周开铁鹰，分散行业 beta",
        ],
        "排除": "非杠杆 ETF（排除 TQQQ/SOXL）",
    },
    "IWM": {
        "类型": "小盘 ETF",
        "角色": "舰队分散腿 #3",
        "选股理由": [
            "日均成交额 >$80 亿，小盘 exposure 与 SPY/QQQ 低相关",
            "全市场铁鹰分桶：中低波桶 Tier B 达标率最高",
            "$10k 账户 1 张保证金 $2,500（25% 仓位）",
        ],
        "排除": "波动高于 SPY，单账户最多 1 张",
    },
    "XLF": {
        "类型": "金融 ETF",
        "角色": "舰队分散腿 #4",
        "选股理由": [
            "金融板块与科技低相关，组合降低同向暴雷",
            "流动性 >$20 亿/日，满足 min_dvol $50M 门槛",
            "铁鹰回测胜率中位数 ~100%（全市场分桶）",
        ],
        "排除": "财报季注意银行集中披露期",
    },
    "DIA": {
        "类型": "道指 ETF",
        "角色": "舰队分散腿 #5",
        "选股理由": [
            "蓝筹 30 只，波动低于 QQQ，偏防御",
            "与 SPY 高度相关但期权有时更宽 → 略多权利金",
            "5 账户舰队最后一槽，完成行业/风格分散",
        ],
        "排除": "成分少，极端个股事件仍可能牵动",
    },
}

STRATEGY_MODES = {
    "stable": {
        "name": "路线 A · 稳（并行铁鹰舰队）",
        "summary": "5×$10k 各开 1 张偏斜铁鹰，ETF 分散，不押单票",
        "anchor": "回测 年化~16% · 回撤~0% · 胜率~100%（SPY+QQQ 并行）",
        "iron_weight": 1.0,
        "gainer_weight": 0.0,
        "ma_filter": False,
    },
    "aggressive": {
        "name": "路线 B · 攻（铁鹰 85% + 动量 15%）",
        "summary": "总资金 85% 铁鹰收租 + 15% 高胜率动量（仅 SPY>MA50）",
        "anchor": "回测 年化~60% · 回撤~-0.6% · 胜率~80%",
        "iron_weight": 0.85,
        "gainer_weight": 0.15,
        "ma_filter": True,
    },
}


@dataclass
class ExecutionLeg:
    side: str
    option_type: str
    action: str
    strike: float
    qty_contracts: int
    premium_est: float
    note: str = ""


@dataclass
class SlotPlaybook:
    account: str
    account_size: float
    ticker: str
    strategy: str
    can_open: bool
    selection_reason: list[str]
    execution_steps: list[str]
    legs: list[ExecutionLeg] = field(default_factory=list)
    margin_per_contract: float = 0.0
    total_credit: float = 0.0
    take_profit_rule: str = ""
    stop_rule: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class ComboLeg:
    enabled: bool
    weight_pct: float
    capital: float
    picks: list[dict]
    selection_reason: list[str]
    execution_steps: list[str]


@dataclass
class UniversalPlaybook:
    date: str
    mode: str
    mode_label: str
    total_capital: float
    spy_regime: str
    strategy_summary: str
    backtest_anchor: str
    slots: list[SlotPlaybook]
    combo: ComboLeg | None = None

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "total_capital": self.total_capital,
            "spy_regime": self.spy_regime,
            "strategy_summary": self.strategy_summary,
            "backtest_anchor": self.backtest_anchor,
            "slots": [asdict(s) for s in self.slots],
            "combo": asdict(self.combo) if self.combo else None,
        }


def _plan_to_legs(plan, n: int) -> list[ExecutionLeg]:
    legs: list[ExecutionLeg] = []
    legs.append(ExecutionLeg(
        "Put", "Put", "SELL", plan.short_strike, n,
        plan.credit_per_contract, f"Delta≈{plan.short_delta}",
    ))
    legs.append(ExecutionLeg(
        "Put", "Put", "BUY", plan.long_strike, n, 0.0, f"保护腿 宽${plan.width:.0f}",
    ))
    if plan.iron_condor and plan.call_short_strike > 0:
        legs.append(ExecutionLeg(
            "Call", "Call", "SELL", plan.call_short_strike, n,
            plan.call_credit_per_contract, f"Delta≈{plan.call_delta}",
        ))
        legs.append(ExecutionLeg(
            "Call", "Call", "BUY", plan.call_long_strike, n, 0.0, "Call 保护腿",
        ))
    return legs


def _build_slot(
    account_id: str,
    account_size: float,
    ticker: str,
    *,
    soup_cfg: dict,
    take_profit: float,
    use_ma_gate: bool,
) -> SlotPlaybook:
    from quant.data import fetch_history
    from quant.decline_income import weekly_put_soup_plan

    rationale = TICKER_RATIONALE.get(ticker, {})
    reasons = list(rationale.get("选股理由", [f"流动性扫描选中 {ticker}"]))
    if rationale.get("角色"):
        reasons.insert(0, f"【{rationale['角色']}】{rationale.get('类型', '')}")

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=550)).isoformat()
    try:
        df = fetch_history(ticker, start=start, end=end)
    except Exception:  # noqa: BLE001
        return SlotPlaybook(
            account=account_id, account_size=account_size, ticker=ticker,
            strategy="偏斜铁鹰", can_open=False,
            selection_reason=reasons,
            execution_steps=["❌ 无法获取行情，跳过"],
            flags=["数据获取失败"],
        )

    ic = soup_cfg.get("iron_condor", True)
    if isinstance(ic, dict):
        add_call = bool(ic.get("enabled", True))
        call_delta = float(ic.get("call_delta", 0.05))
    else:
        add_call = bool(ic)
        call_delta = float(soup_cfg.get("call_delta", 0.05))

    plan = weekly_put_soup_plan(
        ticker, df,
        account_size=account_size,
        short_delta=float(soup_cfg.get("short_delta", 0.10)),
        width=float(soup_cfg.get("spread_width", 25)),
        dte_days=int(soup_cfg.get("dte_days", 7)),
        max_margin_pct=float(soup_cfg.get("max_margin_pct", 0.25)),
        add_call=add_call,
        call_delta=call_delta,
        take_profit=take_profit,
    )
    if plan is None:
        return SlotPlaybook(
            account=account_id, account_size=account_size, ticker=ticker,
            strategy="偏斜铁鹰", can_open=False,
            selection_reason=reasons,
            execution_steps=["❌ 无法生成期权方案"],
        )

    can_open = plan.can_open
    if not use_ma_gate:
        # 路线 A：回测最优为无 MA 过滤；仍展示 MA 状态供参考
        if plan.max_contracts >= 1:
            can_open = True
            if not plan.above_ma:
                plan.flags.append("（路线A）回测允许无MA50开，但现价低于MA50需自行减量")

    n = plan.max_contracts if can_open else 0
    steps = list(plan.playbook)
    if not use_ma_gate and not plan.above_ma and n > 0:
        steps.insert(1, "※ 路线A：全市场回测最优为无MA50过滤；保守者可等站上MA50再开")

    return SlotPlaybook(
        account=account_id,
        account_size=account_size,
        ticker=ticker,
        strategy="偏斜铁鹰" if plan.iron_condor else "周Put价差",
        can_open=can_open and n > 0,
        selection_reason=reasons,
        execution_steps=steps,
        legs=_plan_to_legs(plan, max(n, 1)),
        margin_per_contract=plan.margin_per_contract,
        total_credit=(plan.total_credit_per_contract or plan.credit_per_contract) * max(n, 0),
        take_profit_rule=f"总权利金盈利 {take_profit:.0%} 平仓（约 ${plan.take_profit_price * 100:,.0f}/张 剩余价值）",
        stop_rule="到期前 1 天仍亏损 → 评估是否 roll；Put 被击穿 → 不裸扛，整组平仓",
        flags=list(plan.flags),
    )


def _build_combo_leg(total_capital: float, weight: float, bull: bool) -> ComboLeg:
    capital = total_capital * weight
    reasons = [
        "continued_search：铁鹰85%+动量15% 回测年化 ~60%",
        "仅 SPY > MA50 时启用（牛市增厚）",
        f"分配资金 ${capital:,.0f}（总资金 {weight:.0%}）",
        "持 1 日、高胜率过滤器（量比+MA20+相对SPY强度）",
    ]
    steps = [
        f"1. 确认 SPY 在 MA50 上方 → 否则动量腿 **不开**",
        f"2. 从涨幅榜选 Top1–2，各分配 ~${capital / 2:,.0f}",
        "3. **收盘前** 市价/限价买入，**次日收盘** 卖出（不过夜周末）",
        "4. 单笔亏损 > 账户 2% 立即止损",
    ]
    picks: list[dict] = []
    if bull:
        try:
            from research.gainer_daily_backtest import high_win_filters, live_gainer_picks
            df = live_gainer_picks(high_win_filters())
            if not df.empty:
                for _, r in df.head(2).iterrows():
                    picks.append({
                        "代码": r.get("代码", ""),
                        "名称": r.get("名称", ""),
                        "涨幅%": float(r.get("涨幅%", 0)),
                        "综合分": float(r.get("综合分", 0)),
                        "选股理由": str(r.get("选股理由", "")),
                        "建议金额$": round(capital / min(len(df), 2), 0),
                    })
                    steps.append(f"5. 买入 **{r.get('代码')}** ${capital / 2:,.0f} — {str(r.get('选股理由', ''))[:80]}")
        except Exception as exc:  # noqa: BLE001
            steps.append(f"5. 动量扫描失败：{exc}")
    else:
        reasons.append("今日 SPY<MA50 → 动量腿关闭")
        steps.append("2. 今日弱市 → **跳过动量**，100% 铁鹰收租")

    return ComboLeg(
        enabled=bull and weight > 0,
        weight_pct=weight,
        capital=capital,
        picks=picks,
        selection_reason=reasons,
        execution_steps=steps,
    )


def build_universal_playbook(
    account: float = 10_000.0,
    *,
    mode: str = "stable",
    cfg: dict | None = None,
) -> UniversalPlaybook:
    from research.tier_a_csp import load_tier_a_csp_config, _resolve_fleet_slots
    from research.income_engine import get_regime
    from quant.providers import DataConfig, get_provider, reset_provider_cache

    cfg = cfg or load_tier_a_csp_config()
    mode_cfg = STRATEGY_MODES.get(mode, STRATEGY_MODES["stable"])
    fleet = cfg.get("fleet") or {}
    n_accounts = int(fleet.get("count", 5))
    account_size = float(fleet.get("account_size", account))
    total = n_accounts * account_size
    soup_cfg = cfg.get("weekly_soup") or {}
    take_profit = float(cfg.get("take_profit", 0.5))
    anchor = cfg.get("backtest_anchor") or {}

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    regime = get_regime(yahoo)
    bull = regime.bull

    slots: list[SlotPlaybook] = []
    for account_id, size, ticker in _resolve_fleet_slots(cfg):
        slots.append(_build_slot(
            account_id, size, ticker,
            soup_cfg=soup_cfg,
            take_profit=take_profit,
            use_ma_gate=mode_cfg.get("ma_filter", False),
        ))

    combo = None
    if mode_cfg.get("gainer_weight", 0) > 0:
        combo = _build_combo_leg(total, float(mode_cfg["gainer_weight"]), bull)

    return UniversalPlaybook(
        date=date.today().isoformat(),
        mode=mode,
        mode_label=mode_cfg["name"],
        total_capital=total,
        spy_regime=regime.label,
        strategy_summary=mode_cfg["summary"],
        backtest_anchor=anchor.get("note") or mode_cfg.get("anchor", ""),
        slots=slots,
        combo=combo,
    )


def playbook_to_dataframe(pb: UniversalPlaybook) -> pd.DataFrame:
    rows: list[dict] = []
    for s in pb.slots:
        leg_txt = " | ".join(
            f"{lg.action} {lg.option_type} ${lg.strike:,.0f}×{lg.qty_contracts}"
            for lg in s.legs
        )
        rows.append({
            "账户": s.account,
            "规模$": s.account_size,
            "代码": s.ticker,
            "策略": s.strategy,
            "可开仓": "✅" if s.can_open else "⏸",
            "选股理由": "；".join(s.selection_reason[:2]),
            "执行腿": leg_txt,
            "保证金$/张": s.margin_per_contract,
            "预计收租$": s.total_credit,
            "止盈规则": s.take_profit_rule,
            "执行步骤数": len(s.execution_steps),
        })
    if pb.combo and pb.combo.enabled:
        for p in pb.combo.picks:
            rows.append({
                "账户": "动量腿",
                "规模$": p.get("建议金额$", 0),
                "代码": p.get("代码", ""),
                "策略": "高胜率做多1日",
                "可开仓": "✅",
                "选股理由": p.get("选股理由", "")[:120],
                "执行腿": "BUY 股票 → 次日 SELL",
                "保证金$/张": "",
                "预计收租$": "",
                "止盈规则": "次日收盘前平仓",
                "执行步骤数": len(pb.combo.execution_steps),
            })
    return pd.DataFrame(rows)


def format_playbook_text(pb: UniversalPlaybook) -> list[str]:
    lines = [
        "",
        "=" * 72,
        f"【普适策略 Playbook】{pb.mode_label} · {pb.date}",
        "=" * 72,
        f"总资金 ${pb.total_capital:,.0f} · {pb.spy_regime}",
        f"策略：{pb.strategy_summary}",
        f"回测锚点：{pb.backtest_anchor}",
        "",
        "── 为什么选这些标的（全市场规律，非 SNDK 单票）──",
    ]
    seen: set[str] = set()
    for s in pb.slots:
        if s.ticker in seen:
            continue
        seen.add(s.ticker)
        meta = TICKER_RATIONALE.get(s.ticker, {})
        lines.append(f"  ▶ {s.ticker}（{meta.get('类型', '标的')}）")
        for r in s.selection_reason[:4]:
            lines.append(f"      · {r}")

    lines.append("")
    lines.append("── 具体执行明细（按账户）──")
    for s in pb.slots:
        status = "✅ 今日执行" if s.can_open else "⏸ 今日观望"
        lines.append(f"\n  [{s.account}] ${s.account_size:,.0f} → {s.ticker} · {s.strategy} · {status}")
        if s.legs:
            lines.append("    下单明细（1 组合单）：")
            for lg in s.legs:
                prem = f" 收~${lg.premium_est:,.0f}" if lg.premium_est else ""
                lines.append(f"      {lg.action:4s} {lg.option_type:4s} ${lg.strike:>8,.2f} × {lg.qty_contracts} 张{prem}  {lg.note}")
            lines.append(f"    保证金/张 ${s.margin_per_contract:,.0f} · 预计收租 ${s.total_credit:,.0f}")
            lines.append(f"    止盈：{s.take_profit_rule}")
            lines.append(f"    止损：{s.stop_rule}")
        for step in s.execution_steps:
            lines.append(f"    {step.replace('**', '')}")
        for f in s.flags:
            lines.append(f"    ⚠ {f}")

    if pb.combo:
        lines.append("")
        lines.append(f"── 组合腿 · 动量 {pb.combo.weight_pct:.0%}（${pb.combo.capital:,.0f}）──")
        for r in pb.combo.selection_reason:
            lines.append(f"  · {r}")
        for step in pb.combo.execution_steps:
            lines.append(f"  {step.replace('**', '')}")

    lines.append("")
    lines.append("── 纪律 ──")
    lines.append("  · 永不裸卖 · 50% 权利金止盈 · 财报前 1 周不开 · 单票不超过账户 25% 保证金")
    lines.append("  · 默认排除 SNDK/MSTR/SOXL/TQQQ 等极端个例")
    return lines


def save_playbook(pb: UniversalPlaybook) -> None:
    PLAYBOOK_JSON.write_text(
        json.dumps(pb.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    playbook_to_dataframe(pb).to_csv(PLAYBOOK_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    p = argparse.ArgumentParser(description="普适策略 Playbook")
    p.add_argument("--account", type=float, default=10_000.0)
    p.add_argument("--mode", choices=["stable", "aggressive"], default="stable")
    args = p.parse_args()
    pb = build_universal_playbook(account=args.account, mode=args.mode)
    save_playbook(pb)
    for line in format_playbook_text(pb):
        print(line)
    print(f"\n→ {PLAYBOOK_JSON}\n→ {PLAYBOOK_CSV}")


if __name__ == "__main__":
    main()
