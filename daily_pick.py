#!/usr/bin/env python3
"""每日选股 · 统一入口：核心策略 + Gainer10+ + 大盘开关。

原则：**不一定每天都有票** —— 条件不满足则「观望」，记入历史，不强行交易。

调度模块：资金流向 / Meme / 暴涨80% / Extreme20 / Gainer10+ / 做空涨幅榜 /
          卖Call / CSP舰队 / SNDK铁鹰 / VRP / 资金流向组合

定时：scripts/install_daily_pick_launchd.sh → 每天 18:30 自动跑（含 Gainer10+）

用法：
    python daily_pick.py
    python daily_pick.py --dry-run
    python daily_pick.py --no-notify
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.io_safe import append_csv_locked, atomic_write_csv, atomic_write_text
from quant.meme_route import parse_meme_route

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "daily_pick_config.json"


def load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def resolve_profile(cfg: dict) -> dict:
    """合并 frequency_profile 与 profiles 子配置。"""
    name = str(cfg.get("frequency_profile", "standard"))
    base = (cfg.get("profiles") or {}).get(name) or {}
    rcfg = cfg.get("regime") or {}
    bcfg = cfg.get("bear_call") or {}
    mods = cfg.get("modules") or {}
    traj_on = bool(base.get("trajectory_enabled", True)) and bool(mods.get("trajectory_highwin", True))
    return {
        "name": name,
        "trajectory_mode": base.get("trajectory_mode", "highwin"),
        "trajectory_top_n": int(base.get("trajectory_top_n", 2)),
        "trajectory_bull_only": bool(base.get(
            "trajectory_bull_only",
            rcfg.get("trajectory_bull_only", True),
        )),
        "bear_call_top_n": int(base.get("bear_call_top_n", bcfg.get("top_n", 5))),
        "etf_iron_always": bool(base.get("etf_iron_always", False)),
        "csp_step_days": int(base.get("csp_step_days", 5)),
        "trajectory_enabled": traj_on,
    }


def get_market_regime(cfg: dict) -> dict:
    """SPY vs MA50 牛熊开关。"""
    rcfg = cfg.get("regime") or {}
    mock = rcfg.get("mock")
    if mock is not None:
        bull = bool(mock.get("bull", True))
        spy = float(mock.get("spy", 500.0))
        ma50 = float(mock.get("ma50", 490.0))
        label = mock.get("label") or ("🟢 牛市（SPY>MA50）" if bull else "🔴 弱市（SPY<MA50）")
        prof = resolve_profile(cfg)
        pb = "高频：卖Call×10 + CSP（无动量/无铁鹰）" if prof["name"] == "high_freq" else (
            "牛市：CSP + 轨迹做多 + 卖Call" if bull else "弱市：卖Call + ETF铁鹰"
        )
        return {
            "bull": bull,
            "label": label,
            "spy": spy,
            "ma50": ma50,
            "mode": "bull" if bull else "bear",
            "playbook": pb,
            "frequency_profile": prof["name"],
        }
    from quant.providers import DataConfig, get_provider, reset_provider_cache
    from research.income_engine import get_regime

    prof = resolve_profile(cfg)
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    try:
        reg = get_regime(yahoo)
    except Exception as exc:  # noqa: BLE001 - daily 生成宁可降级，也不要因行情代理失败中断。
        bull = True
        return {
            "bull": bull,
            "label": "🟡 大盘未知（使用降级牛市配置）",
            "spy": None,
            "ma50": None,
            "mode": "bull",
            "playbook": "行情获取失败：使用保守默认配置",
            "frequency_profile": prof["name"],
            "warning": f"market_regime_fallback: {exc}",
        }
    bull = reg.bull
    if prof["name"] == "high_freq":
        playbook = "高频：卖Call×10 + CSP舰队（动量/铁鹰已关）"
    elif bull:
        playbook = "牛市：CSP + 轨迹做多 + 卖Call"
    else:
        playbook = "弱市：卖Call价差 + ETF铁鹰；轨迹做多关闭"
    return {
        "bull": bull,
        "label": reg.label,
        "spy": round(reg.spy, 2),
        "ma50": round(reg.ma50, 2),
        "mode": "bull" if bull else "bear",
        "playbook": playbook,
        "frequency_profile": prof["name"],
    }


def _run_bear_call(account_size: float, cfg: dict, *, bull: bool, prof: dict) -> list[dict]:
    """卖看涨价差 + meme 路由（极端票观望 / 超涨回吐做空）。"""
    bcfg = cfg.get("bear_call") or {}
    top_n = int(prof.get("bear_call_top_n", bcfg.get("top_n", 5)))
    count = int(bcfg.get("count", 200))
    mod = "弱市·卖Call" if not bull else "收入·卖Call"
    modules = cfg.get("modules") or {}
    mrc = parse_meme_route(cfg)
    use_route = mrc.enabled and bool(modules.get("short_fade", True))
    lc = cfg.get("live_chain") or {}
    live = bool(lc.get("enabled", True))
    try:
        from quant.providers import DataConfig, get_provider, reset_provider_cache
        from quant.meme_route import (
            enrich_movers_panel,
            estimate_bear_put_debit_spread,
            route_action,
            short_fade_direction,
            short_fade_module_label,
        )
        from quant.option_chain import build_bear_call_spread, build_bear_put_debit_spread
        from research.income_engine import bear_call_plan, build_movers_panel

        reset_provider_cache()
        yahoo = get_provider(DataConfig(provider="yahoo"))
        panel = build_movers_panel(yahoo, count=count)
        if panel.empty:
            return [{
                "模块": mod,
                "账户": "收入引擎",
                "代码": "—",
                "状态": "观望",
                "方向": "卖Call价差",
                "选股理由": "今日涨幅/振幅榜无合适标的（正常空仓）",
            }]
        spy_1d = np.nan
        if use_route:
            start = (date.today() - timedelta(days=90)).isoformat()
            end = date.today().isoformat()
            batch = yahoo.fetch_batch(panel["代码"].tolist(), start, end)
            panel = enrich_movers_panel(panel, batch)
            try:
                spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
                if len(spy) >= 2:
                    spy_1d = float(spy.iloc[-1] / spy.iloc[-2] - 1) * 100
            except Exception:  # noqa: BLE001
                pass
            panel["SPY1d涨%"] = spy_1d

        amp_top = panel.sort_values("振幅%", ascending=False).head(top_n * 2)
        gain_top = panel.sort_values("涨幅%", ascending=False).head(top_n * 2)
        cand = pd.concat([amp_top, gain_top]).drop_duplicates("代码")
        cand = cand.assign(_score=pd.to_numeric(cand["涨幅%"], errors="coerce").fillna(0)
                           + pd.to_numeric(cand["振幅%"], errors="coerce").fillna(0) * 0.5)
        cand = cand.sort_values("_score", ascending=False)

        rows: list[dict] = []
        bear_n = 0
        short_n = 0
        short_cap = mrc.short_fade.top_n if use_route else 0

        for _, r in cand.iterrows():
            if use_route:
                action = route_action(
                    r, mrc,
                    spy_1d_pct=spy_1d if np.isfinite(spy_1d) else None,
                    spy_bear=not bull,
                )
                if action == "skip":
                    continue
                if action == "short_fade":
                    if short_n >= short_cap:
                        continue
                    short_n += 1
                    sf = mrc.short_fade
                    mod_sf = short_fade_module_label(sf.structure)
                    dir_sf = short_fade_direction(sf.structure)
                    reason = (
                        f"meme路由 · 弱市SPY<MA50 · 涨幅{float(r['涨幅%']):+.1f}% · "
                        f"收盘强度{float(r.get('收盘强度', 0.5)):.2f} · "
                    )
                    if sf.structure == "put_spread":
                        if live:
                            plan, why = build_bear_put_debit_spread(
                                str(r["代码"]), float(r["现价"]), account_size,
                                otm=float(lc.get("put_debit_otm", 0.0)),
                                width_pct=float(lc.get("put_debit_width_pct", 0.10)),
                                min_dte=int(lc.get("min_dte", 2)), max_dte=int(lc.get("max_dte", 45)),
                                min_oi=int(lc.get("min_open_interest", 25)),
                                max_spread_pct=float(lc.get("max_spread_pct", 0.60)),
                            )
                            if plan is None:
                                rows.append({
                                    "模块": mod_sf, "账户": "meme路由", "代码": r["代码"],
                                    "状态": "观望", "方向": dir_sf,
                                    "选股理由": reason + f"真实期权链：{why}",
                                    "数据源": "真实链不可用",
                                })
                                continue
                            can = plan.contracts >= 1
                            reason += (
                                f"{plan.legs_label()} @{plan.expiry}({plan.dte}d) · "
                                f"付${-plan.net_per_contract:.0f}/张 · 最大亏${plan.max_loss:.0f}"
                                + (f" × {plan.contracts}张" if can else " · 账户不够1张")
                            )
                            rows.append({
                                "模块": mod_sf, "账户": "meme路由", "代码": r["代码"],
                                "状态": "可开仓" if can else "观望", "方向": dir_sf,
                                "选股理由": reason,
                                "建议张数": plan.contracts if can else "",
                                "最大亏损$": plan.max_loss,
                                "数据源": "真实链",
                            })
                            continue
                        sig = float(r.get("RV%", r.get("RV", 50))) / 100
                        if not np.isfinite(sig) or sig <= 0:
                            sig = 0.5
                        kl, ks, debit, max_l, max_p = estimate_bear_put_debit_spread(
                            float(r["现价"]), sig,
                            delta=sf.put_delta, width_pct=sf.put_width_pct, dte_days=sf.dte_days,
                        )
                        reason += f"买P${kl:.0f}/卖P${ks:.0f} · 付${debit * 100:.0f}/张 · 最大亏${max_l * 100:.0f}（模型估值）"
                    else:
                        reason += f"SPY1d{float(spy_1d) if np.isfinite(spy_1d) else 0:+.1f}% · 次日平空"
                    rows.append({
                        "模块": mod_sf,
                        "账户": "meme路由",
                        "代码": r["代码"],
                        "状态": "观望",
                        "方向": dir_sf,
                        "选股理由": reason,
                        "数据源": "模型估算",
                    })
                    continue
            if bear_n >= top_n:
                continue
            bear_n += 1
            base_reason = f"涨幅{float(r['涨幅%']):+.1f}% · 振幅{float(r['振幅%']):.1f}% · "
            if live:
                plan, why = build_bear_call_spread(
                    str(r["代码"]), float(r["现价"]), account_size,
                    otm=float(lc.get("bear_call_otm", 0.08)),
                    width_pct=float(lc.get("bear_call_width_pct", 0.10)),
                    min_dte=int(lc.get("min_dte", 2)), max_dte=int(lc.get("max_dte", 45)),
                    min_oi=int(lc.get("min_open_interest", 25)),
                    max_spread_pct=float(lc.get("max_spread_pct", 0.60)),
                )
                if plan is None:
                    rows.append({
                        "模块": mod, "账户": "收入引擎", "代码": str(r["代码"]),
                        "状态": "观望", "方向": "卖Call价差",
                        "选股理由": base_reason + f"真实期权链：{why}",
                        "数据源": "真实链不可用",
                    })
                    continue
                can = plan.contracts >= 1
                rows.append({
                    "模块": mod, "账户": "收入引擎", "代码": plan.ticker,
                    "状态": "可开仓" if can else "观望", "方向": "卖Call价差",
                    "选股理由": (
                        base_reason
                        + f"**真实链** {plan.legs_label()} @{plan.expiry}({plan.dte}d) · "
                        f"净收${plan.net_per_contract:.0f}/张 · 最大亏${plan.max_loss:.0f}"
                        + (f" × {plan.contracts}张" if can else " · 账户风控不够 1 张")
                    ),
                    "建议张数": plan.contracts if can else "",
                    "权利金$": round(plan.net_per_contract, 0),
                    "最大亏损$": plan.max_loss,
                    "数据源": "真实链",
                })
                continue
            plan = bear_call_plan(r, account_size)
            n = int(plan.get("建议张数") or 0)
            can = n > 0
            rows.append({
                "模块": mod,
                "账户": "收入引擎",
                "代码": plan["代码"],
                "状态": "观望",
                "方向": "卖Call价差",
                "选股理由": (
                    base_reason
                    + f"卖C${plan['卖Call']:,.0f}/买C${plan['买Call']:,.0f} · "
                    f"收${plan['净权利金$']:,.0f}/张（模型估值，不推送）"
                    + (f" × {n}张" if can else " · 账户风控不够 1 张")
                ),
                "建议张数": "",
                "权利金$": "",
                "最大亏损$": plan.get("最大亏损$", ""),
                "数据源": "模型估算",
            })
        return rows or [{
            "模块": mod,
            "账户": "收入引擎",
            "代码": "—",
            "状态": "观望",
            "方向": "卖Call价差",
            "选股理由": "无满足风控的卖Call方案（meme已过滤）",
        }]
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": mod,
            "账户": "收入引擎",
            "代码": "—",
            "状态": "扫描失败",
            "方向": "—",
            "选股理由": str(e),
        }]


def _run_bear_iron_etf(account_size: float, cfg: dict, *, prof: dict) -> list[dict]:
    """ETF 偏斜铁鹰（高频模式全年收租；标准模式仅弱市）。"""
    icfg = cfg.get("bear_iron_etf") or {}
    mod = "并行·ETF铁鹰" if prof.get("etf_iron_always") else "弱市·ETF铁鹰"
    tickers = icfg.get("tickers") or ["SPY", "QQQ"]
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace(",", " ").split()]
    rows: list[dict] = []
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=550)).isoformat()
    try:
        from quant.data import fetch_history
        from quant import decline_income as di

        for tk in tickers[:3]:
            sym = str(tk).upper()
            try:
                df = fetch_history(sym, start=start, end=end)
            except Exception:  # noqa: BLE001
                continue
            plan = di.weekly_put_soup_plan(
                sym, df, account_size=account_size,
                add_call=True, call_delta=0.05,
            )
            if plan is None:
                rows.append({
                    "模块": mod,
                    "账户": "收租锚点",
                    "代码": sym,
                    "状态": "无数据",
                    "方向": "铁鹰",
                    "选股理由": "行情不足或无法定价",
                })
                continue
            n = int(plan.max_contracts or 0)
            can = n >= 1
            credit = plan.total_credit_per_contract if plan.iron_condor else plan.credit_per_contract
            reason = (
                f"{sym} 偏斜铁鹰 · RV {plan.rv_pct}% · 现价 ${plan.close} · "
                f"卖P${plan.short_strike}/买P${plan.long_strike}"
            )
            if plan.iron_condor and plan.call_short_strike:
                reason += f" · 卖C${plan.call_short_strike}/买C${plan.call_long_strike}"
            reason += f" · 收 ${credit:,.0f}/张"
            if can:
                reason += f" × {n}张"
                if not prof.get("etf_iron_always"):
                    reason += " · 弱市不要求MA50"
            else:
                reason += " · 保证金不足 1 张"
            rows.append({
                "模块": mod,
                "账户": "收租锚点",
                "代码": sym,
                "状态": "可开仓" if can else "观望",
                "方向": "铁鹰",
                "选股理由": reason,
                "建议张数": n if can else "",
                "权利金$": credit,
            })
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": mod,
            "账户": "收租锚点",
            "代码": "—",
            "状态": "扫描失败",
            "方向": "—",
            "选股理由": str(e),
        }]
    return rows


def _run_fleet_csp(fleet_cfg: dict, account_size: float, cfg: dict | None = None) -> list[dict]:
    from quant.daily_screen_fleet import fleet_accounts, today_picks_for_account

    lc = (cfg or {}).get("live_chain") or {}
    rows: list[dict] = []
    for acct in fleet_accounts(fleet_cfg):
        plan = today_picks_for_account(
            acct, {}, capital=account_size, live_chain=lc,
        )
        if plan.empty:
            rows.append({
                "模块": "5×舰队·CSP",
                "账户": acct.get("label", acct["id"]),
                "代码": acct.get("ticker", ""),
                "状态": "无数据",
                "方向": "—",
                "选股理由": "行情拉取失败或标的数据不足",
            })
            continue
        r = plan.iloc[0]
        can = str(r.get("可开仓", "")) == "✅"
        rows.append({
            "模块": "5×舰队·CSP",
            "账户": acct.get("label", acct["id"]),
            "代码": str(r.get("代码", acct.get("ticker", ""))),
            "状态": "可开仓" if can else "观望",
            "方向": r.get("方向", "卖Put" if can else "观望"),
            "策略动作": r.get("策略动作", ""),
            "选股理由": r.get("选股理由", acct.get("description", "")),
            "建议张数": r.get("建议张数", ""),
            "卖Put行权价": r.get("卖Put行权价", ""),
            "权利金$": r.get("权利金$", ""),
            "回测胜率": r.get("回测胜率", ""),
            "数据源": r.get("数据源", ""),
            "数据有效": r.get("数据有效"),
            "可交易": r.get("可交易"),
        })
    return rows


def _run_momentum(cfg: dict, *, bull: bool, prof: dict) -> list[dict]:
    """动量扫描：standard=轨迹高置信 high_freq=每日Top5。"""
    mod = "高频·动量" if prof.get("name") == "high_freq" else "轨迹·高置信"
    acct = "每日Top5" if prof.get("name") == "high_freq" else "涨幅榜Top"
    if prof.get("trajectory_bull_only") and not bull:
        return [{
            "模块": mod,
            "账户": "全市场",
            "代码": "—",
            "状态": "观望",
            "方向": "—",
            "选股理由": "弱市（SPY<MA50）关闭动量做多；改用卖Call/铁鹰腿",
        }]
    try:
        from research.gainer_daily_backtest import filters_for_mode, live_gainer_picks
        filt = filters_for_mode(
            prof.get("trajectory_mode", "highwin"),
            top_n=int(prof.get("trajectory_top_n", 2)),
        )
        df = live_gainer_picks(filt)
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": mod, "账户": "全市场", "代码": "—",
            "状态": "扫描失败", "方向": "—", "选股理由": str(e),
        }]
    if df.empty:
        return [{
            "模块": mod, "账户": "全市场", "代码": "—",
            "状态": "观望", "方向": "—",
            "选股理由": "今日无标的满足动量因子（正常空仓日）",
        }]
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append({
            "模块": mod,
            "账户": acct,
            "代码": r.get("代码", ""),
            "状态": "可开仓",
            "方向": "做多",
            "选股理由": r.get("选股理由", ""),
            "买进时机": "今日收盘买入",
            "卖出时机": "次日收盘卖出（持1日）",
            "量比": r.get("量比", ""),
        })
    return rows


def _run_trajectory_highwin(*, bull: bool, bull_only: bool) -> list[dict]:
    if bull_only and not bull:
        return [{
            "模块": "轨迹·高置信",
            "账户": "全市场",
            "代码": "—",
            "状态": "观望",
            "方向": "—",
            "选股理由": "弱市（SPY<MA50）关闭轨迹做多；改用卖Call/铁鹰腿",
        }]
    rules_path = ROOT / "research" / "move_pattern_rules.json"
    if not rules_path.exists():
        return [{
            "模块": "轨迹·高置信",
            "账户": "全市场",
            "代码": "—",
            "状态": "未初始化",
            "方向": "—",
            "选股理由": "请先运行 research/move_pattern_mine.py --mode highwin",
        }]
    try:
        from research.move_pattern_mine import scan_today_highwin
        doc = json.loads(rules_path.read_text(encoding="utf-8"))
        df = scan_today_highwin(doc.get("rules") or [], quick=False)
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": "轨迹·高置信",
            "账户": "全市场",
            "代码": "—",
            "状态": "扫描失败",
            "方向": "—",
            "选股理由": str(e),
        }]
    if df.empty:
        return [{
            "模块": "轨迹·高置信",
            "账户": "全市场",
            "代码": "—",
            "状态": "观望",
            "方向": "—",
            "选股理由": "今日无标的满足温和涨+量比+MA+形态胜率模板（正常空仓日）",
        }]
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append({
            "模块": "轨迹·高置信",
            "账户": "涨幅榜Top",
            "代码": r.get("代码", ""),
            "状态": "可开仓",
            "方向": r.get("方向", "偏多"),
            "选股理由": r.get("建议", r.get("规律", "")),
            "历史胜率": r.get("历史胜率", ""),
            "量比": r.get("量比", ""),
        })
    return rows


def _run_capital_flow(cfg: dict, *, bull: bool) -> list[dict]:
    """资金流向操盘痕迹扫描（quant/capital_flow）。"""
    fcfg_path = ROOT / str(cfg.get("flow_config", "flow_daily_config.json"))
    try:
        import flow_daily as fd

        fcfg = fd.load_config(fcfg_path) if fcfg_path.exists() else {}
        doc = fd.run_flow_scan(fcfg)
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": "资金流向",
            "账户": "操盘痕迹",
            "代码": "—",
            "状态": "扫描失败",
            "方向": "—",
            "选股理由": str(e),
        }]
    rows: list[dict] = []
    for p in doc.get("picks") or []:
        rows.append({
            "模块": "资金流向",
            "账户": p.get("账户", "操盘痕迹"),
            "代码": p.get("代码", ""),
            "状态": p.get("状态", "观望"),
            "方向": p.get("方向", ""),
            "策略动作": p.get("策略动作", ""),
            "选股理由": p.get("选股理由", ""),
            "上涨规律": p.get("上涨规律", ""),
            "下跌规律": p.get("下跌规律", ""),
        })
    if not rows:
        rows.append({
            "模块": "资金流向",
            "账户": "操盘痕迹",
            "代码": "—",
            "状态": "观望",
            "方向": "—",
            "选股理由": "今日无量价操盘痕迹命中（正常空仓）",
        })
    return rows


def _run_meme_long(cfg: dict, *, bull: bool) -> list[dict]:
    """Meme 规律 Ultra80 / S8U（MSTR/SMCI/COIN）。"""
    modules = cfg.get("modules") or {}
    if not modules.get("meme_long", False):
        return []
    from quant.ticker_pattern_strategy import parse_meme_long, scan_meme_long

    mlc = parse_meme_long(cfg)
    mod = "规律·Ultra80准入" if mlc.ticker_source == "oos_approved" else (
        "规律·Ultra80" if mlc.high_win_mode else "规律·纯多头"
    )
    acct = "Meme S8U" if mlc.high_win_mode else "Meme S8"
    if not mlc.enabled:
        return []
    if mlc.bull_only and not bull:
        return [{
            "模块": mod,
            "账户": acct,
            "代码": "—",
            "状态": "观望",
            "方向": "—",
            "选股理由": "弱市（SPY<MA50）关闭 meme 规律做多",
        }]
    try:
        from quant.providers import DataConfig, get_provider, reset_provider_cache

        reset_provider_cache()
        yahoo = get_provider(DataConfig(provider="yahoo"))
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=400)).isoformat()
        batch = yahoo.fetch_batch(mlc.tickers, start, end)
        return scan_meme_long(batch, spy_bull=bull, mlc=mlc)
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": mod,
            "账户": acct,
            "代码": "—",
            "状态": "扫描失败",
            "方向": "—",
            "选股理由": str(e),
        }]


def _run_gain15(cfg: dict) -> list[dict]:
    """暴涨80%规则：观察池 + 追多/回避确认。"""
    modules = cfg.get("modules") or {}
    if not modules.get("gain15", False):
        return []
    gpath = ROOT / str(cfg.get("gain15_config", "gain15_daily_config.json"))
    gcfg: dict = {}
    if gpath.exists():
        gcfg = json.loads(gpath.read_text(encoding="utf-8"))
    try:
        from quant.gain15_scan import run_gain15_scan

        plan = run_gain15_scan(gcfg)
    except Exception as e:  # noqa: BLE001
        return [{
            "模块": "暴涨80%",
            "账户": "动量确认",
            "代码": "—",
            "状态": "扫描失败",
            "方向": "—",
            "选股理由": str(e),
        }]

    rows: list[dict] = []
    for b in plan.get("buy_confirmed") or []:
        rows.append({
            "模块": "暴涨80%·追多",
            "账户": "动量确认",
            "代码": b.get("代码", ""),
            "状态": "可开仓",
            "方向": "做多",
            "策略动作": "追多确认",
            "选股理由": (
                f"暴涨日{b.get('暴涨日')} +{b.get('暴涨日涨幅%')}% Top{b.get('涨幅榜排名')} · "
                f"{b.get('规则')} · 历史命中{b.get('历史命中率')} · 5日均{b.get('历史5日均')}"
            ),
            "历史命中率": b.get("历史命中率"),
            "规则ID": b.get("规则ID"),
        })
    for a in plan.get("avoid_confirmed") or []:
        rows.append({
            "模块": "暴涨80%·回避",
            "账户": "动量确认",
            "代码": a.get("代码", ""),
            "状态": "可开仓",
            "方向": "回避/做空",
            "策略动作": "回避确认",
            "选股理由": (
                f"暴涨日{a.get('暴涨日')} +{a.get('暴涨日涨幅%')}% Top{a.get('涨幅榜排名')} · "
                f"{a.get('规则')} · 历史命中{a.get('历史命中率')} · 5日均{a.get('历史5日均')}"
            ),
            "历史命中率": a.get("历史命中率"),
            "规则ID": a.get("规则ID"),
        })
    for w in plan.get("watching") or []:
        hint = "；".join(w.get("早期提示") or []) or "等T+1/T+3确认"
        rows.append({
            "模块": "暴涨80%·观察",
            "账户": "动量确认",
            "代码": w.get("代码", ""),
            "状态": "观望",
            "方向": "观察",
            "策略动作": "待确认",
            "选股理由": (
                f"暴涨日{w.get('暴涨日')} Top{w.get('涨幅榜排名')} T+{w.get('已过交易日', 0)} · {hint}"
            ),
        })
    for s in plan.get("new_spikes") or []:
        rows.append({
            "模块": "暴涨80%·新暴涨",
            "账户": "动量确认",
            "代码": s.get("代码", ""),
            "状态": "观望",
            "方向": "观察",
            "策略动作": "入观察池",
            "选股理由": (
                f"今日涨{s.get('涨幅_pct')}% Top{s.get('gain_rank')} "
                f"${s.get('成交额M')}M · 等次日/3日确认"
            ),
        })
    if not rows:
        rows.append({
            "模块": "暴涨80%",
            "账户": "动量确认",
            "代码": "—",
            "状态": "观望",
            "方向": "—",
            "选股理由": "今日无暴涨入选或待确认信号（正常空仓）",
        })
    return rows


def run_daily_pick(cfg: dict) -> dict:
    today = date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    account_size = float(cfg.get("account_size", 10_000))
    modules = cfg.get("modules") or {}
    prof = resolve_profile(cfg)

    regime = get_market_regime(cfg)
    bull = regime["bull"]
    all_rows: list[dict] = []

    if modules.get("capital_flow", False):
        if cfg.get("quick"):
            all_rows.append({
                "模块": "资金流向", "账户": "操盘痕迹", "代码": "—",
                "状态": "观望", "方向": "—", "选股理由": "quick 模式跳过",
            })
        else:
            all_rows.extend(_run_capital_flow(cfg, bull=bull))

    if modules.get("meme_long", False):
        if cfg.get("quick"):
            all_rows.append({
                "模块": "规律·Ultra80", "账户": "Meme", "代码": "—",
                "状态": "观望", "方向": "—", "选股理由": "quick 模式跳过",
            })
        else:
            all_rows.extend(_run_meme_long(cfg, bull=bull))

    if modules.get("bear_call", True):
        all_rows.extend(_run_bear_call(account_size, cfg, bull=bull, prof=prof))
    if modules.get("bear_iron_etf", True):
        if prof.get("etf_iron_always") or not bull:
            all_rows.extend(_run_bear_iron_etf(account_size, cfg, prof=prof))

    if modules.get("fleet_csp", True):
        fleet_path = ROOT / cfg.get("fleet_config", "daily_screen_config.json")
        fleet_cfg = json.loads(fleet_path.read_text(encoding="utf-8")) if fleet_path.exists() else {}
        all_rows.extend(_run_fleet_csp(fleet_cfg, account_size, cfg))

    if modules.get("trajectory_highwin", True) and prof.get("trajectory_enabled", True):
        if prof.get("name") == "high_freq" or prof.get("trajectory_mode") == "highfreq":
            all_rows.extend(_run_momentum(cfg, bull=bull, prof=prof))
        else:
            all_rows.extend(_run_trajectory_highwin(
                bull=bull, bull_only=prof.get("trajectory_bull_only", True),
            ))

    if modules.get("gain15", False):
        if cfg.get("quick"):
            all_rows.extend([{
                "模块": "暴涨80%",
                "账户": "动量确认",
                "代码": "—",
                "状态": "观望",
                "方向": "—",
                "选股理由": "quick 模式跳过重型扫描",
            }])
        else:
            all_rows.extend(_run_gain15(cfg))

    from quant.daily_pick_runners import RUNNER_REGISTRY, run_registered

    module_runs: list[dict] = []
    for mod_id in RUNNER_REGISTRY:
        if not modules.get(mod_id, False):
            continue
        try:
            rows = run_registered(mod_id, cfg, bull=bull)
            module_runs.append({
                "id": mod_id,
                "ok": True,
                "rows": len(rows),
                "可开仓": sum(1 for r in rows if r.get("状态") == "可开仓"),
            })
            all_rows.extend(rows)
        except Exception as e:  # noqa: BLE001
            module_runs.append({"id": mod_id, "ok": False, "error": str(e)[:200]})
            all_rows.append({
                "模块": mod_id,
                "账户": "—",
                "代码": "—",
                "状态": "扫描失败",
                "方向": "—",
                "选股理由": str(e)[:200],
            })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.insert(0, "选股日期", today)
        df.insert(1, "选股时间", now)

    actionable = 0
    if not df.empty and "状态" in df.columns:
        actionable = int((df["状态"] == "可开仓").sum())

    mode_label = "高频·收入三引擎" if prof.get("name") == "high_freq" else (
        "牛市三引擎" if bull else "弱市偏空收租"
    )

    from quant.daily_pick_push import build_push_block, sanitize_picks
    from quant.strategy_catalog import (
        build_strategy_summary_doc,
        build_strategy_audit,
        enrich_picks_with_strategy_audit,
        summarize_picks_by_module,
    )

    modules_summary = summarize_picks_by_module(all_rows)
    strategy_audit = build_strategy_audit(ROOT)
    strategy_summary = build_strategy_summary_doc(
        picks=all_rows,
        modules_summary=modules_summary,
        regime=regime,
        root=ROOT,
        module_runs=module_runs,
        pick_date=today,
        summary={
            "总条目": len(df),
            "可开仓": actionable,
            "观望": len(df) - actionable,
        },
    )
    all_rows = enrich_picks_with_strategy_audit(
        all_rows,
        audit_doc=strategy_audit,
        root=ROOT,
    )

    hw_cfg = cfg.get("high_win_filter") or {}
    high_win_doc: dict = {}
    if hw_cfg.get("enabled", True):
        from quant.high_win_pick import build_high_win_doc

        high_win_doc = build_high_win_doc(
            all_rows,
            min_win_rate=float(hw_cfg.get("min_win_rate", 0.80)),
            regime=regime,
        )
        all_rows = high_win_doc.get("all_enriched") or all_rows

    all_rows = sanitize_picks(all_rows)

    doc = {
        "选股日期": today,
        "选股时间": now,
        "philosophy": cfg.get("philosophy", "有信号才出手，无票则空仓"),
        "frequency_profile": prof.get("name", "standard"),
        "regime": regime,
        "summary": {
            "总条目": len(df),
            "可开仓": actionable,
            "观望": len(df) - actionable,
            "是否空仓日": actionable == 0,
            "大盘": regime["label"],
            "模式": mode_label,
            "接入模块数": len(modules_summary),
            "有信号模块": strategy_summary.get("actionable_modules") or [],
            "运行模块数": len(module_runs),
            "高胜率可开仓": len(high_win_doc.get("high_win_actionable") or []),
        },
        "modules_summary": modules_summary,
        "module_runs": module_runs,
        "strategy_audit": strategy_audit,
        "strategy_summary": strategy_summary,
        "high_win": {
            "min_win_rate": hw_cfg.get("min_win_rate", 0.80),
            "summary": high_win_doc.get("summary") or {},
            "picks": high_win_doc.get("high_win_actionable") or [],
            "watch": high_win_doc.get("high_win_watch") or [],
        },
        "picks": all_rows,
    }
    push_block = build_push_block(doc, cfg)
    doc["push"] = push_block
    doc["summary"]["推送条数"] = push_block.get("count", 0)
    doc["summary"]["推送说明"] = push_block.get("headline", "")
    return doc


def _json_default(obj: object) -> object:
    """numpy/pandas 标量 → 原生 JSON 类型。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_outputs(doc: dict, cfg: dict) -> None:
    outs = cfg.get("outputs") or {}
    jpath = ROOT / outs.get("today_json", "research/daily_pick_today.json")
    cpath = ROOT / outs.get("today_csv", "research/daily_pick_today.csv")
    hpath = ROOT / outs.get("history_csv", "daily_pick_history.csv")
    hwpath = ROOT / outs.get("high_win_json", "research/daily_pick_high_win.json")
    pushpath = ROOT / outs.get("push_json", "research/daily_pick_push.json")

    jpath.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        jpath,
        json.dumps(doc, ensure_ascii=False, indent=2, default=_json_default),
    )

    hw = doc.get("high_win") or {}
    if hw.get("picks") is not None:
        hw_doc = {
            "选股日期": doc["选股日期"],
            "选股时间": doc["选股时间"],
            "min_win_rate": hw.get("min_win_rate", 0.80),
            "summary": hw.get("summary"),
            "regime": doc.get("regime"),
            "picks": hw.get("picks"),
            "watch": hw.get("watch"),
        }
        atomic_write_text(
            hwpath,
            json.dumps(hw_doc, ensure_ascii=False, indent=2, default=_json_default),
        )

    push = doc.get("push") or {}
    if push:
        atomic_write_text(
            pushpath,
            json.dumps(push, ensure_ascii=False, indent=2, default=_json_default),
        )

    df = pd.DataFrame(doc.get("picks") or [])
    if not df.empty:
        df.insert(0, "选股日期", doc["选股日期"])
        df.insert(1, "选股时间", doc["选股时间"])
        atomic_write_csv(df, cpath)
        append_csv_locked(df, hpath)

    try:
        from quant.app_manifest import export_app_manifest

        export_app_manifest(ROOT)
    except Exception:  # noqa: BLE001
        pass

    try:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(ROOT / "scripts/export_chart_snapshots.py"), "--limit", "80"],
            cwd=str(ROOT),
            timeout=180,
            check=False,
        )
    except Exception:  # noqa: BLE001
        pass


def notify(doc: dict, cfg: dict) -> None:
    """桌面推送：仅推送真实链/真实行情推演，不含模型估价。"""
    notify_cfg = cfg.get("notify") or {}
    if not notify_cfg.get("desktop", True):
        return
    push = doc.get("push") or {}
    push_picks = push.get("picks") or []
    only_real = bool((cfg.get("push") or {}).get("require_real_data", True))
    if notify_cfg.get("only_when_action") and not push_picks:
        return
    try:
        from scan_daily import desktop_notify
    except ImportError:
        return
    reg = doc.get("regime") or {}
    prefix = "弱市" if not reg.get("bull", True) else "牛市"
    if push_picks:
        title = push.get("headline") or f"每日选股推送 · {prefix} · {len(push_picks)} 条"
        lines = push.get("lines") or [format_push_line(p) for p in push_picks[:6]]
        body = "\n".join(lines[: int((cfg.get("push") or {}).get("max_lines", 6))])
    else:
        title = f"每日选股 · {prefix} · 无真实信号"
        skipped = (push.get("stats") or {}).get("skipped_model", 0)
        extra = f"（已过滤 {skipped} 条模型估价）" if skipped and only_real else ""
        body = f"{reg.get('label', '')} · 真实链/行情暂无可推送标的，正常观望。{extra}"
    desktop_notify(title, body)


def format_push_line(row: dict) -> str:
    from quant.daily_pick_push import format_push_line as _fmt
    return _fmt(row)


def print_report(doc: dict) -> None:
    s = doc["summary"]
    reg = doc.get("regime") or {}
    print(f"\n{'=' * 56}")
    print(f"每日选股  {doc['选股日期']}  {doc['选股时间']}")
    print(f"{'=' * 56}")
    print(f"原则：{doc.get('philosophy', '')}")
    if reg:
        print(f"大盘：{reg.get('label', '')}  SPY {reg.get('spy', '')} / MA50 {reg.get('ma50', '')}")
        print(f"模式：{s.get('模式', '')} — {reg.get('playbook', '')}")
    print(f"可开仓 {s['可开仓']} ｜ 观望 {s['观望']} ｜ {'空仓日 ✓' if s['是否空仓日'] else '有信号'}")
    mods = doc.get("modules_summary") or {}
    if mods:
        print("\n模块汇总：")
        for mod, st in mods.items():
            print(f"  · {mod}: 可开仓{st.get('可开仓', 0)} 观望{st.get('观望', 0)}")
    hw = doc.get("high_win") or {}
    hwp = hw.get("picks") or []
    push = doc.get("push") or {}
    pp = push.get("picks") or []
    if pp:
        print(f"\n📣 推送（真实数据） {len(pp)} 条：")
        for line in (push.get("lines") or [])[:10]:
            print(f"  · {line}")
        st = push.get("stats") or {}
        if st.get("skipped_model"):
            print(f"  （已过滤模型估价 {st['skipped_model']} 条，不推送）")
    if hwp:
        print(f"\n★ 高胜率≥{hw.get('min_win_rate', 0.8):.0%} 可开仓 {len(hwp)} 条：")
        for p in hwp[:12]:
            print(f"  ✅ {p.get('代码')} [{p.get('模块')}] {p.get('回测摘要', '')}")
            print(f"      {str(p.get('选股理由', ''))[:100]}")
    ss = doc.get("strategy_summary") or {}
    if ss.get("catalog"):
        n_data = ss.get("integrated_with_data", 0)
        print(f"\n全系统策略: 接入{ss.get('integrated_count', 0)} · 今日有数据{n_data}")
    print()
    for p in doc.get("picks") or []:
        mark = "✅" if p.get("状态") == "可开仓" else "⏸"
        print(f"  {mark} [{p.get('模块','')}] {p.get('账户','')} · {p.get('代码','')} · {p.get('状态','')}")
        reason = p.get("选股理由", "")
        if reason:
            print(f"      {reason[:120]}{'…' if len(str(reason)) > 120 else ''}")
    print(f"\n→ {ROOT / 'research' / 'daily_pick_today.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="每日选股（可无票）")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CFG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--quick", action="store_true", help="跳过重型网络扫描模块")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.quick:
        cfg["quick"] = True
    doc = run_daily_pick(cfg)
    print_report(doc)
    if not args.dry_run:
        save_outputs(doc, cfg)
        if not args.no_notify:
            notify(doc, cfg)


if __name__ == "__main__":
    main()
