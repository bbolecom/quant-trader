"""Tier A CSP 每日信号：三重目标扫描达标方案（50%年化 / 15%回撤 / 85%胜率）。

支持单账户 CSP 与 **5×$10k 舰队**（CSP 放不下时自动降级为周 Put 价差/铁鹰）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = ROOT / "tier_a_csp_config.json"


@dataclass
class TierACspRow:
    ticker: str
    close: float
    ma50: float
    above_ma: bool
    can_open: bool
    put_strike: float
    premium_per_share: float
    premium_per_contract: float
    capital_per_contract: float
    take_profit_price: float
    breakeven: float
    suggested_contracts: int
    alloc_pct: float
    capital_used: float
    min_account_hint: float
    bt_win_rate: float | None
    bt_annual: float | None
    bt_max_dd: float | None
    flags: list[str]
    account_id: str = ""
    account_size: float = 0.0
    strategy: str = "CSP"

    def to_dict(self) -> dict:
        d = {
            "账户": self.account_id or "单账户",
            "规模$": self.account_size or "",
            "策略": self.strategy,
            "代码": self.ticker,
            "现价": self.close,
            "MA50": self.ma50,
            "站上MA50": "✅" if self.above_ma else "❌",
            "可开仓": "✅" if self.can_open else "⏸",
            "卖Put": self.put_strike,
            "权利金$": self.premium_per_contract,
            "担保金$": self.capital_per_contract,
            "止盈权利金$": round(self.take_profit_price * 100, 0) if self.take_profit_price else 0,
            "盈亏平衡": self.breakeven,
            "建议张数": self.suggested_contracts,
            "占用资金$": round(self.capital_used, 0),
            "仓位%": round(self.alloc_pct * 100, 0) if self.alloc_pct else "",
            "最低账户$": round(self.min_account_hint, 0) if self.min_account_hint else "",
            "回测胜率": self.bt_win_rate,
            "回测年化": self.bt_annual,
            "回测回撤": self.bt_max_dd,
            "提示": "；".join(self.flags) if self.flags else "",
        }
        return d


def load_tier_a_csp_config(path: Path | None = None) -> dict:
    p = path or DEFAULT_CFG_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "tickers": ["SNDK"],
        "delta": 0.25,
        "ma_window": 50,
        "dte_days": 35,
        "alloc_pct": 0.50,
        "take_profit": 0.5,
        "lookback_days": 550,
        "max_single_ticker_pct": 0.50,
        "fleet": {"enabled": False, "account_size": 10000, "count": 5},
    }


def _contracts_for_account(
    account: float,
    capital_per_contract: float,
    *,
    alloc_pct: float,
    max_single_pct: float,
) -> int:
    if capital_per_contract <= 0 or account <= 0:
        return 0
    budget = account * min(alloc_pct, max_single_pct)
    return int(budget // capital_per_contract)


def _resolve_fleet_slots(cfg: dict) -> list[tuple[str, float, str]]:
    """返回 [(账户名, 规模, 标的), ...]。"""
    fleet = cfg.get("fleet") or {}
    size = float(fleet.get("account_size", 10_000))
    count = int(fleet.get("count", 5))
    labels = fleet.get("labels") or [f"账户{i + 1}" for i in range(count)]

    liq = fleet.get("liquid_scan") or {}
    min_dvol = float(liq.get("min_dollar_vol_m", 50))
    top_n = int(liq.get("top_n", count))
    use_scan = fleet.get("ticker_source") == "liquid_scan" or liq.get("enabled", False)
    use_patterns = bool(liq.get("use_market_patterns", True))

    if use_scan:
        from research.liquid_tier_a_scan import pick_fleet_tickers
        tickers = pick_fleet_tickers(
            top_n, account_size=size,
            min_dvol_m=min_dvol,
            prefer_weekly_for_small=bool(liq.get("prefer_weekly_for_small", True)),
            use_patterns=use_patterns,
        )
    else:
        tickers = fleet.get("tickers") or ["SNDK", "MU", "INTC", "AMD", "WDC"]
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.replace(",", " ").split()]

    slots: list[tuple[str, float, str]] = []
    for i in range(count):
        label = str(labels[i]) if i < len(labels) else f"账户{i + 1}"
        tk = str(tickers[i % len(tickers)]).strip().upper()
        slots.append((label, size, tk))
    return slots


def _scan_csp_row(
    sym: str,
    df: pd.DataFrame,
    account: float,
    account_id: str,
    *,
    delta: float,
    ma_window: int,
    dte_days: int,
    alloc_pct: float,
    max_single: float,
) -> TierACspRow | None:
    from quant.decline_income import csp_income_plan

    plan = csp_income_plan(sym, df, delta=delta, ma_window=ma_window, dte_days=dte_days)
    if plan is None:
        return None
    n = _contracts_for_account(
        account, plan.capital_per_contract,
        alloc_pct=alloc_pct, max_single_pct=max_single,
    )
    min_acct = plan.capital_per_contract / min(alloc_pct, max_single) if plan.capital_per_contract > 0 else 0
    if plan.rv_pct > 80 and n > 0:
        n = max(1, n // 2)
    flags = list(plan.flags)
    if plan.can_open and n == 0 and min_acct > account:
        flags.append(f"CSP 担保金 ${plan.capital_per_contract:,.0f}，${account:,.0f} 账户不够 1 张")
    return TierACspRow(
        ticker=plan.ticker,
        close=plan.close,
        ma50=plan.ma50,
        above_ma=plan.above_ma,
        can_open=plan.can_open and n > 0,
        put_strike=plan.put_strike,
        premium_per_share=plan.premium,
        premium_per_contract=round(plan.premium * 100, 0),
        capital_per_contract=plan.capital_per_contract,
        take_profit_price=plan.take_profit_price,
        breakeven=plan.breakeven,
        suggested_contracts=n if plan.can_open else 0,
        alloc_pct=alloc_pct,
        capital_used=n * plan.capital_per_contract,
        min_account_hint=min_acct,
        bt_win_rate=plan.bt_win_rate,
        bt_annual=plan.bt_annual,
        bt_max_dd=plan.bt_max_dd,
        flags=flags,
        account_id=account_id,
        account_size=account,
        strategy="CSP",
    )


def _scan_weekly_row(
    sym: str,
    df: pd.DataFrame,
    account: float,
    account_id: str,
    *,
    soup_cfg: dict,
) -> TierACspRow | None:
    from quant.decline_income import weekly_put_soup_plan

    ic = soup_cfg.get("iron_condor", True)
    if isinstance(ic, dict):
        add_call = bool(ic.get("enabled", True))
        call_delta = float(ic.get("call_delta", 0.05))
        call_width = ic.get("call_width")
    else:
        add_call = bool(ic)
        call_delta = float(soup_cfg.get("call_delta", 0.05))
        call_width = soup_cfg.get("call_width")

    plan = weekly_put_soup_plan(
        sym, df,
        account_size=account,
        short_delta=float(soup_cfg.get("short_delta", 0.10)),
        width=float(soup_cfg.get("spread_width", 25)),
        dte_days=int(soup_cfg.get("dte_days", 7)),
        max_margin_pct=float(soup_cfg.get("max_margin_pct", 0.25)),
        add_call=add_call,
        call_delta=call_delta,
        call_width=float(call_width) if call_width else None,
    )
    if plan is None:
        return None
    n = plan.max_contracts if plan.can_open else 0
    credit = plan.total_credit_per_contract if plan.iron_condor else plan.credit_per_contract
    flags = list(plan.flags)
    strat = "偏斜铁鹰" if plan.iron_condor else "周Put价差"
    if plan.above_ma and n == 0:
        flags.append(f"保证金 ${plan.margin_per_contract:,.0f}，当前账户开不出 1 张")
    return TierACspRow(
        ticker=plan.ticker,
        close=plan.close,
        ma50=plan.ma50,
        above_ma=plan.above_ma,
        can_open=plan.can_open and n > 0,
        put_strike=plan.short_strike,
        premium_per_share=plan.credit_per_share,
        premium_per_contract=round(credit, 0),
        capital_per_contract=plan.margin_per_contract,
        take_profit_price=plan.take_profit_price,
        breakeven=plan.short_strike - plan.credit_per_share,
        suggested_contracts=n,
        alloc_pct=float(soup_cfg.get("max_margin_pct", 0.25)),
        capital_used=n * plan.margin_per_contract,
        min_account_hint=plan.margin_per_contract / max(float(soup_cfg.get("max_margin_pct", 0.25)), 0.01),
        bt_win_rate=plan.zero_prob,
        bt_annual=None,
        bt_max_dd=None,
        flags=flags + [f"归零概率≈{plan.zero_prob:.0%}", f"周ROI≈{plan.weekly_roi_pct:.1f}%"],
        account_id=account_id,
        account_size=account,
        strategy=strat,
    )


def scan_tier_a_fleet(
    *,
    cfg: dict | None = None,
    cfg_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """5×$10k（可配置）舰队扫描：每账户一标的，CSP 优先，不够则周 Put 价差。"""
    from quant.data import fetch_history

    cfg = cfg or load_tier_a_csp_config(cfg_path)
    fleet = cfg.get("fleet") or {}
    fallback = str(fleet.get("fallback", "weekly_put_spread"))
    soup_cfg = cfg.get("weekly_soup") or {}

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(cfg.get("lookback_days", 550)))).isoformat()
    delta = float(cfg.get("delta", 0.25))
    ma_window = int(cfg.get("ma_window", 50))
    dte_days = int(cfg.get("dte_days", 35))
    alloc_pct = float(cfg.get("alloc_pct", 0.50))
    max_single = float(cfg.get("max_single_ticker_pct", 0.50))

    rows: list[TierACspRow] = []
    for account_id, account_size, sym in _resolve_fleet_slots(cfg):
        try:
            df = fetch_history(sym, start=start, end=end)
        except Exception:
            continue
        row = _scan_csp_row(
            sym, df, account_size, account_id,
            delta=delta, ma_window=ma_window, dte_days=dte_days,
            alloc_pct=alloc_pct, max_single=max_single,
        )
        if row is None:
            continue
        if not row.can_open and fallback == "weekly_put_spread":
            wk = _scan_weekly_row(sym, df, account_size, account_id, soup_cfg=soup_cfg)
            if wk is not None:
                if row.above_ma and not wk.above_ma:
                    wk.flags.append("CSP 趋势参考：沿用同一 MA50 过滤逻辑")
                rows.append(wk)
                continue
        rows.append(row)

    df_out = pd.DataFrame([r.to_dict() for r in rows])
    if not df_out.empty:
        df_out = df_out.sort_values(["可开仓", "账户"], ascending=[False, True]).reset_index(drop=True)
    return df_out, cfg


def scan_tier_a_csp(
    account: float = 10_000.0,
    *,
    cfg: dict | None = None,
    cfg_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """扫描 Tier A；若 fleet.enabled 则走 5×$10k 舰队模式。"""
    cfg = cfg or load_tier_a_csp_config(cfg_path)
    fleet = cfg.get("fleet") or {}
    if fleet.get("enabled"):
        return scan_tier_a_fleet(cfg=cfg)

    from quant.data import fetch_history
    from quant.decline_income import csp_income_plan

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(cfg.get("lookback_days", 550)))).isoformat()
    tickers = cfg.get("tickers") or ["SNDK"]
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace(",", " ").split()]

    delta = float(cfg.get("delta", 0.25))
    ma_window = int(cfg.get("ma_window", 50))
    dte_days = int(cfg.get("dte_days", 35))
    alloc_pct = float(cfg.get("alloc_pct", 0.50))
    max_single = float(cfg.get("max_single_ticker_pct", 0.50))

    rows: list[TierACspRow] = []
    for tk in tickers:
        sym = str(tk).strip().upper()
        if not sym:
            continue
        try:
            df = fetch_history(sym, start=start, end=end)
        except Exception:
            continue
        row = _scan_csp_row(
            sym, df, account, "单账户",
            delta=delta, ma_window=ma_window, dte_days=dte_days,
            alloc_pct=alloc_pct, max_single=max_single,
        )
        if row:
            rows.append(row)

    df_out = pd.DataFrame([r.to_dict() for r in rows])
    if not df_out.empty:
        df_out = df_out.sort_values(["可开仓", "回测年化"], ascending=[False, False]).reset_index(drop=True)
    return df_out, cfg


def fleet_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"total_accounts": 0, "open_count": 0, "total_premium": 0.0, "total_margin": 0.0}
    open_df = df[df["可开仓"] == "✅"] if "可开仓" in df.columns else pd.DataFrame()
    prem = (
        open_df["权利金$"].astype(float) * open_df["建议张数"].astype(int)
        if not open_df.empty and "权利金$" in open_df.columns else 0.0
    )
    margin = (
        open_df["担保金$"].astype(float) * open_df["建议张数"].astype(int)
        if not open_df.empty and "担保金$" in open_df.columns else 0.0
    )
    return {
        "total_accounts": len(df),
        "open_count": len(open_df),
        "total_premium": float(prem.sum()) if hasattr(prem, "sum") else float(prem),
        "total_margin": float(margin.sum()) if hasattr(margin, "sum") else float(margin),
    }


def format_playbook_lines(df: pd.DataFrame, cfg: dict) -> list[str]:
    """生成可打印的 Tier A / 舰队操作步骤。"""
    name = cfg.get("name", "Tier A CSP")
    anchor = cfg.get("backtest_anchor") or {}
    fleet = cfg.get("fleet") or {}
    lines = [f"【{name}】"]
    if fleet.get("enabled"):
        n = int(fleet.get("count", 5))
        sz = float(fleet.get("account_size", 10_000))
        lines.append(f"  舰队：{n} 账户 × ${sz:,.0f} = ${n * sz:,.0f} 总资金")
        summ = fleet_summary(df)
        lines.append(
            f"  今日可开 {summ['open_count']}/{summ['total_accounts']} 户 · "
            f"合计收租≈${summ['total_premium']:,.0f} · 占用保证金≈${summ['total_margin']:,.0f}"
        )
    else:
        lines.append(
            f"  δ={cfg.get('delta')} · MA{cfg.get('ma_window')} · 仓位{float(cfg.get('alloc_pct', 0.5)):.0%}"
        )
    if anchor:
        lines.append(
            f"  CSP回测锚点：年化≈{anchor.get('ann_return', 0):.0%} "
            f"回撤≈{anchor.get('max_dd', 0):.0%} 胜率≈{anchor.get('win_rate', 0):.0%}"
        )

    if df is None or df.empty:
        lines.append("  今日：无扫描结果")
        return lines

    open_rows = df[df["可开仓"] == "✅"] if "可开仓" in df.columns else pd.DataFrame()
    for _, r in df.iterrows():
        acct = r.get("账户", "")
        prefix = f"  [{acct}]" if acct and acct != "单账户" else " "
        status = "✅" if r.get("可开仓") == "✅" else "⏸"
        strat = r.get("策略", "CSP")
        lines.append(
            f"{prefix}{status} {r['代码']} · {strat} · 卖P${r['卖Put']:,.0f} "
            f"×{int(r['建议张数'])}张 收${float(r['权利金$']):,.0f}/张 "
            f"({'站上MA50' if r.get('站上MA50') == '✅' else '未过MA50'})"
        )
        tip = str(r.get("提示", ""))
        if tip and r.get("可开仓") != "✅":
            lines.append(f"      → {tip[:100]}")

    if open_rows.empty:
        lines.append("  ⚠ 今日舰队无可执行开仓（等 MA50 或检查保证金）")
    lines.append(f"  纪律：{float(cfg.get('take_profit', 0.5)):.0%}权利金止盈 · 跌破MA50停开 · 财报前1周不开")
    return lines
