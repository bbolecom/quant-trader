"""每日「收入引擎」一键扫描：大盘开关 + 三引擎（卖看涨价差 / 做多 / 卖看跌 CSP）。

把本会话验证过的有效零件拼成一套多空+期权的稳定收入系统：

  大盘开关：SPY 在 MA50 上方=牛市（三引擎全开）；下方=弱市（主开卖看涨，胜率更高）。

  引擎①  卖看涨价差（bear call spread）：振幅/涨幅榜 Top 卖 +OTM 看涨价差，
          赌它一周不再暴涨。回测胜率 88–95%、正期望、定义风险。← 收入核心
  引擎②  高胜率做多：涨幅榜强势股（趋势/形态过滤）持 1 日，~80% 日胜率。仅牛市开。
  引擎③  卖看跌 CSP：低价高 IV 优质票卖 put 收租/接货。

期权用「已实现波动 × IV倍数」近似定价（真实 IV 更高 → 收的权利金通常更厚）。
输出每只标的的行权价、估算权利金、最大亏损、按账户规模的建议张数。

用法：
    python research/income_engine.py --account 10000
    python research/income_engine.py --account 10000 --count 200
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from quant.screener import fetch_gainer_universe_live

R = 0.045
IV_MULT = 1.3
WEEK_T = 5 / 252


def _ncdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(S: float, K: float, T: float, r: float, sig: float, typ: str) -> float:
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if typ == "c" else (K - S))
    d1 = (log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    if typ == "c":
        return S * _ncdf(d1) - K * exp(-r * T) * _ncdf(d2)
    return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _round_strike(x: float) -> float:
    if x >= 100:
        return round(x)
    if x >= 25:
        return round(x * 2) / 2
    return round(x * 4) / 4


@dataclass
class Regime:
    spy: float
    ma50: float
    bull: bool

    @property
    def label(self) -> str:
        return "🟢 牛市（SPY>MA50）" if self.bull else "🔴 弱市（SPY<MA50）"


def get_regime(yahoo) -> Regime:
    spy = yahoo.fetch_history("SPY", (date.today() - timedelta(days=160)).isoformat(),
                              date.today().isoformat())["Close"].astype(float)
    ma50 = float(spy.rolling(50).mean().iloc[-1])
    last = float(spy.iloc[-1])
    return Regime(spy=last, ma50=ma50, bull=last > ma50)


def build_movers_panel(yahoo, count: int = 200, min_dvol_m: float = 30.0) -> pd.DataFrame:
    """拉实时榜单 + 近期历史，算涨幅/振幅/RV/成交额。"""
    snap = fetch_gainer_universe_live(count=count)
    if snap.empty:
        return pd.DataFrame()
    tickers = snap["代码"].tolist()
    start = (date.today() - timedelta(days=90)).isoformat()
    end = date.today().isoformat()
    batch = yahoo.fetch_batch(tickers, start, end)
    rows = []
    for t, df in batch.items():
        if df is None or df.empty or len(df) < 25:
            continue
        c = df["Close"].astype(float); h = df["High"]; l = df["Low"]; v = df["Volume"]
        S = float(c.iloc[-1]); prev = float(c.iloc[-2])
        gain = S / prev - 1
        amp = float((h.iloc[-1] - l.iloc[-1]) / prev)
        rv = float(c.pct_change().rolling(20).std().iloc[-1] * sqrt(252))
        dvol = float((c * v).iloc[-20:].mean())
        if dvol < min_dvol_m * 1e6 or not np.isfinite(rv) or S < 3.0:
            continue
        rows.append({"代码": t, "现价": S, "涨幅%": gain * 100, "振幅%": amp * 100,
                     "RV%": rv * 100, "成交额M": dvol / 1e6})
    return pd.DataFrame(rows)


def bear_call_plan(row: pd.Series, account: float, *, otm: float = 0.20,
                   width_pct: float = 0.10, risk_per_trade: float = 0.02) -> dict:
    S = row["现价"]; sig = min(max(row["RV%"] / 100 * IV_MULT, 0.5), 4.0)
    Ks = _round_strike(S * (1 + otm))
    Kl = _round_strike(S * (1 + otm + width_pct))
    if Kl <= Ks:
        Kl = Ks + max(_round_strike(S * 0.05), 0.5)
    credit = (bs(S, Ks, WEEK_T, R, sig, "c") - bs(S, Kl, WEEK_T, R, sig, "c"))
    width = Kl - Ks
    max_loss = (width - credit) * 100
    contracts = int(max(0, (account * risk_per_trade) // max(max_loss, 1)))
    return {
        "代码": row["代码"], "现价": round(S, 2),
        "卖Call": Ks, "买Call": Kl,
        "净权利金$": round(credit * 100, 0),
        "最大亏损$": round(max_loss, 0),
        "建议张数": contracts,
        "预计收租$": round(credit * 100 * contracts, 0),
    }


def csp_plan(row: pd.Series, account: float, *, otm: float = 0.12,
             max_cash: float | None = None) -> dict | None:
    S = row["现价"]; sig = min(max(row["RV%"] / 100 * IV_MULT, 0.4), 4.0)
    K = _round_strike(S * (1 - otm))
    cash = K * 100
    if max_cash and cash > max_cash:
        return None
    prem = bs(S, K, WEEK_T, R, sig, "p")
    return {
        "代码": row["代码"], "现价": round(S, 2), "卖Put": K,
        "权利金$": round(prem * 100, 0),
        "占用现金$": round(cash, 0),
        "年化%": round(prem / K / WEEK_T * 100, 0),
        "接货成本": round(K - prem, 2),
    }


LIVE_CHAIN_DEFAULTS = {
    "otm": 0.08, "width_pct": 0.10, "min_dte": 2, "max_dte": 45,
    "min_oi": 25, "max_spread_pct": 0.60,
}


def bear_call_plan_live(row: pd.Series, account: float, **kw) -> dict:
    """卖看涨价差（真实期权链）：盘上真实行权价 + bid/ask 保守定价；无可成交则标观望。"""
    from quant.option_chain import build_bear_call_spread

    p = {**LIVE_CHAIN_DEFAULTS, **kw}
    sym = str(row["代码"]); S = float(row["现价"])
    plan, why = build_bear_call_spread(
        sym, S, account, otm=p["otm"], width_pct=p["width_pct"],
        min_dte=p["min_dte"], max_dte=p["max_dte"],
        min_oi=p["min_oi"], max_spread_pct=p["max_spread_pct"],
    )
    if plan is None:
        return {
            "代码": sym, "现价": round(S, 2), "卖Call": np.nan, "买Call": np.nan,
            "到期": "-", "净权利金$": np.nan, "最大亏损$": np.nan, "建议张数": 0,
            "预计收租$": np.nan, "数据源": "真实链", "状态": f"观望·{why}",
        }
    short, long = plan.legs[0], plan.legs[1]
    can = plan.contracts >= 1
    return {
        "代码": sym, "现价": round(S, 2),
        "卖Call": short.strike, "买Call": long.strike,
        "到期": f"{plan.expiry}({plan.dte}d)",
        "净权利金$": round(plan.net_per_contract, 0), "最大亏损$": plan.max_loss,
        "建议张数": plan.contracts if can else 0,
        "预计收租$": round(plan.net_per_contract * plan.contracts, 0) if can else 0,
        "数据源": "真实链", "状态": "✅可成交" if can else "观望·账户不够1张",
    }


def csp_plan_live(row: pd.Series, account: float, *, max_cash: float | None = None, **kw) -> dict | None:
    """卖看跌 CSP（真实期权链）：盘上真实行权价 + bid 权利金；无可成交返回 None。"""
    from quant.option_chain import build_csp

    p = {**LIVE_CHAIN_DEFAULTS, **kw}
    sym = str(row["代码"]); S = float(row["现价"])
    plan, _why = build_csp(
        sym, S, account, otm=max(p["otm"], 0.10),
        min_dte=p["min_dte"], max_dte=p["max_dte"],
        min_oi=p["min_oi"], max_spread_pct=p["max_spread_pct"],
    )
    if plan is None:
        return None
    if max_cash and plan.collateral > max_cash:
        return None
    leg = plan.legs[0]
    return {
        "代码": sym, "现价": round(S, 2), "卖Put": leg.strike,
        "到期": f"{plan.expiry}({plan.dte}d)",
        "权利金$": round(leg.bid * 100, 0), "占用现金$": plan.collateral,
        "建议张数": plan.contracts, "接货成本": round(leg.strike - leg.bid, 2),
        "数据源": "真实链",
    }


def build_income_plan(account: float = 10000.0, count: int = 200, top_n: int = 5,
                      use_live_chain: bool = True) -> dict:
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    regime = get_regime(yahoo)
    panel = build_movers_panel(yahoo, count=count)
    out: dict = {"regime": regime, "call_spreads": pd.DataFrame(),
                 "longs": pd.DataFrame(), "csp": pd.DataFrame(),
                 "tier_a_csp": pd.DataFrame(), "tier_a_cfg": {}}
    try:
        from research.tier_a_csp import scan_tier_a_csp
        tier_df, tier_cfg = scan_tier_a_csp(account=account)
        out["tier_a_csp"] = tier_df
        out["tier_a_cfg"] = tier_cfg
    except Exception:  # noqa: BLE001
        pass

    if panel.empty:
        return out

    # 引擎①：卖看涨价差 —— 取振幅榜 + 涨幅榜并集 Top
    amp_top = panel.sort_values("振幅%", ascending=False).head(top_n)
    gain_top = panel.sort_values("涨幅%", ascending=False).head(top_n)
    cand = pd.concat([amp_top, gain_top]).drop_duplicates("代码").head(top_n + 3)
    _bc = bear_call_plan_live if use_live_chain else bear_call_plan
    out["call_spreads"] = pd.DataFrame([_bc(r, account) for _, r in cand.iterrows()])

    # 引擎③：卖看跌 CSP —— 价位适配账户、RV 适中的票
    csp_cand = panel[(panel["现价"] <= account / 100 * 1.2) & (panel["RV%"].between(40, 200))]
    csp_cand = csp_cand.sort_values("RV%", ascending=False).head(top_n)
    _csp = csp_plan_live if use_live_chain else csp_plan
    csp_rows = [_csp(r, account, max_cash=account) for _, r in csp_cand.iterrows()]
    out["csp"] = pd.DataFrame([r for r in csp_rows if r])

    # 引擎②：高胜率做多（仅牛市）
    if regime.bull:
        try:
            from research.gainer_daily_backtest import high_win_filters, live_gainer_picks
            longs = live_gainer_picks(high_win_filters())
            if not longs.empty:
                cols = [c for c in ["代码", "名称", "涨幅%", "综合分", "选股理由"] if c in longs.columns]
                out["longs"] = longs[cols]
        except Exception:  # noqa: BLE001
            pass
    return out


def _print_plan(plan: dict, account: float) -> None:
    reg = plan["regime"]
    tier_cfg = plan.get("tier_a_cfg") or {}
    fleet = tier_cfg.get("fleet") or {}
    if fleet.get("enabled"):
        n = int(fleet.get("count", 5))
        sz = float(fleet.get("account_size", 10_000))
        acct_label = f"舰队 {n}×${sz:,.0f}（合计 ${n * sz:,.0f}）"
    else:
        acct_label = f"账户 ${account:,.0f}"
    print("=" * 72)
    print(f"每日收入引擎 · {date.today().isoformat()} · {acct_label}")
    print("=" * 72)
    print(f"大盘开关：{reg.label}   SPY {reg.spy:.2f} / MA50 {reg.ma50:.2f}")
    if reg.bull:
        print("→ 三引擎全开：①卖看涨价差 ②做多强势股 ③卖看跌CSP\n")
    else:
        print("→ 弱市模式：主开①卖看涨价差（胜率最高），②做多关闭，③CSP减量\n")

    cs = plan["call_spreads"]
    print("【引擎① 卖看涨价差 · 收入核心】(振幅/涨幅榜，赌一周不再暴涨)")
    if cs.empty:
        print("  今日无合适标的。\n")
    else:
        print(cs.to_string(index=False)); print()

    if reg.bull:
        lg = plan["longs"]
        print("【引擎② 高胜率做多 · 牛市增厚】(强势股持1日)")
        print(("  " + lg.to_string(index=False) if not lg.empty else "  今日无满足条件标的。") + "\n")

    csp = plan["csp"]
    print("【引擎③ 卖看跌 CSP · 稳定底仓】(收租/愿接货)")
    if csp.empty:
        print("  今日无价位适配标的。\n")
    else:
        print(csp.to_string(index=False)); print()

    tier = plan.get("tier_a_csp", pd.DataFrame())
    tier_cfg = plan.get("tier_a_cfg") or {}
    if tier is not None and not tier.empty:
        from research.tier_a_csp import format_playbook_lines, fleet_summary
        fleet = tier_cfg.get("fleet") or {}
        if fleet.get("enabled"):
            summ = fleet_summary(tier)
            print(f"【引擎③★ Tier A 舰队 · 5×$10k】  总资金 ${summ['total_accounts'] * float(fleet.get('account_size', 10000)):,.0f}")
        else:
            print("【引擎③★ Tier A CSP · 三重目标达标】(50%年化/15%回撤/85%胜率回测)")
        for line in format_playbook_lines(tier, tier_cfg):
            print(line)
        show_cols = [c for c in tier.columns if c not in ("回测胜率", "回测年化", "回测回撤")]
        print(tier[show_cols].to_string(index=False))
        print()

    # 普适策略：选股理由 + 执行明细
    fleet_on = (tier_cfg.get("fleet") or {}).get("enabled", False)
    if fleet_on or (tier is not None and not tier.empty):
        try:
            from research.universal_playbook import build_universal_playbook, format_playbook_text, save_playbook
            mode = tier_cfg.get("strategy_mode", "stable")
            pb = build_universal_playbook(account=account, mode=mode, cfg=tier_cfg)
            save_playbook(pb)
            for line in format_playbook_text(pb):
                print(line)
        except Exception as exc:  # noqa: BLE001
            print(f"[Playbook 跳过] {exc}\n")

    print("-" * 72)
    print("纪律：每笔风险≤账户2% · 价差永不裸卖 · 50%权利金止盈 · 分散5只 · 财报回避")
    print("注：期权为近似估值，实盘以券商链为准；胜率高≠无亏，定义风险结构永不爆仓。")


def format_notification(plan: dict) -> tuple[str, str]:
    """生成 (标题, 正文) 供桌面/邮件通知。"""
    reg = plan["regime"]
    title = f"收入引擎 · {'牛市三开' if reg.bull else '弱市卖看涨'}"
    parts = [f"SPY {reg.spy:.0f}/{reg.ma50:.0f}"]
    cs = plan.get("call_spreads")
    if cs is not None and not cs.empty:
        actives = cs[cs["建议张数"] > 0]
        names = "、".join(actives["代码"].head(3).tolist()) if not actives.empty else "无可下单"
        parts.append(f"卖看涨: {names}")
    if reg.bull:
        lg = plan.get("longs")
        if lg is not None and not lg.empty:
            parts.append(f"做多: {'、'.join(lg['代码'].head(3).tolist())}")
    csp = plan.get("csp")
    if csp is not None and not csp.empty:
        parts.append(f"CSP: {'、'.join(csp['代码'].head(3).tolist())}")
    tier = plan.get("tier_a_csp")
    if tier is not None and not tier.empty:
        open_t = tier[tier["可开仓"] == "✅"] if "可开仓" in tier.columns else tier
        if not open_t.empty:
            if "账户" in open_t.columns:
                parts.append(f"TierA舰队{len(open_t)}户")
            else:
                parts.append(f"TierA: {'、'.join(open_t['代码'].head(2).tolist())}")
    body = " · ".join(parts)
    return title, body[:217] + "…" if len(body) > 220 else body


def append_history(plan: dict, account: float) -> None:
    from datetime import datetime
    reg = plan["regime"]
    cs = plan.get("call_spreads", pd.DataFrame())
    csp = plan.get("csp", pd.DataFrame())
    tier = plan.get("tier_a_csp", pd.DataFrame())
    lg = plan.get("longs", pd.DataFrame())
    tier_open = tier[tier["可开仓"] == "✅"] if (tier is not None and not tier.empty and "可开仓" in tier.columns) else pd.DataFrame()
    fleet_cfg = (plan.get("tier_a_cfg") or {}).get("fleet") or {}
    row = {
        "扫描时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "账户": account,
        "舰队模式": "5x10k" if fleet_cfg.get("enabled") else "单账户",
        "大盘": reg.label,
        "SPY": round(reg.spy, 2),
        "MA50": round(reg.ma50, 2),
        "卖看涨标的": ",".join(cs["代码"].tolist()) if not cs.empty else "",
        "做多标的": ",".join(lg["代码"].tolist()) if (lg is not None and not lg.empty) else "",
        "CSP标的": ",".join(csp["代码"].tolist()) if not csp.empty else "",
        "TierA明细": ";".join(
            f"{r.get('账户', '')}:{r['代码']}×{int(r['建议张数'])}" for _, r in tier_open.iterrows()
        ) if not tier_open.empty else "",
        "TierA可开户数": len(tier_open),
    }
    f = ROOT / "income_engine_history.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False, encoding="utf-8-sig")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--account", type=float, default=10000.0)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--notify", action="store_true", help="发桌面通知 + 写历史CSV（定时任务用）")
    args = p.parse_args()
    plan = build_income_plan(account=args.account, count=args.count, top_n=args.top)
    _print_plan(plan, args.account)
    if args.notify:
        append_history(plan, args.account)
        try:
            from scan_daily import desktop_notify
            title, body = format_notification(plan)
            desktop_notify(title, body)
        except Exception as e:  # noqa: BLE001
            print(f"[通知跳过] {e}")


if __name__ == "__main__":
    main()
