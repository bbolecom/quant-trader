#!/usr/bin/env python3
"""三腿每日策略：上涨规律做多 + 下跌规律回避 + 真实链收租。

全部基于真实量价（腿①②）与 yfinance 期权链 bid/ask（腿③），不用 BS 回测数字。

  腿① 做多：SPY>MA50 时，涨幅榜高置信 TopN（温和涨+量比+形态胜率）
  腿② 回避：全市场扫描 D1–D4 下跌前轨迹（缩量顶/放量杀跌/抛物线）
  腿③ 收租：mixed_balanced 舰队（SNDK Put价差 + SOFI CSP + ETF铁鹰）

用法：
    python pattern_daily.py
    python pattern_daily.py --dry-run
    python pattern_daily.py --quick
    python pattern_daily.py -c pattern_config.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from quant.move_pattern import (
    assess_down_avoidance,
    assess_up_favor,
    extract_trajectory_features,
)
from quant.pattern_params import load_optimized_rules
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.liquid_tier_a_scan import _avg_dollar_vol, build_candidate_pool
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pattern_config.json"
HISTORY_FILE = ROOT / "pattern_daily_history.csv"
RULES_JSON = ROOT / "research" / "move_pattern_rules.json"


def load_config(path: Path) -> dict:
    if not path.exists():
        return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _spy_regime() -> dict:
    import yfinance as yf

    spy = yf.download("SPY", period="6mo", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    close = spy["Close"].astype(float)
    px = float(close.iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    bull = px > ma50
    return {
        "spy": px,
        "ma50": ma50,
        "ma20": ma20,
        "bull": bull,
        "label": f"{'🟢 牛市' if bull else '🔴 弱市'} SPY ${px:.2f} / MA50 ${ma50:.2f}",
    }


def _last_features(batch: dict[str, pd.DataFrame], min_dvol_m: float) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for tk, df in batch.items():
        if df is None or df.empty:
            continue
        dvol_m = _avg_dollar_vol(df["Close"], df["Volume"]) / 1e6
        if dvol_m < min_dvol_m:
            continue
        feat = extract_trajectory_features(df, forward_days=20)
        if feat.empty:
            continue
        last = feat.iloc[-1].copy()
        last["代码"] = tk
        last["dvol_m"] = dvol_m
        out[tk] = last
    return out


def scan_avoid_leg(
    features: dict[str, pd.Series],
    down_params=None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for tk, row in features.items():
        for hit in assess_down_avoidance(row, down_params):
            rows.append({
                "代码": tk,
                "规则": hit["rule_id"],
                "原因": hit["reason"],
                "建议": hit["action"],
                "紧急度": hit.get("urgency", "中"),
                "周期": hit.get("horizon", "20d"),
                "量比": round(float(row.get("vol_ratio", 0)), 2),
                "5日涨跌%": round(float(row.get("ret_5d", 0)) * 100, 1),
                "20日涨跌%": round(float(row.get("ret_20d", 0)) * 100, 1),
                "MA50": "上" if bool(row.get("above_ma50")) else "下",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["代码", "规则"])


def scan_long_leg(
    cfg: dict,
    regime: dict,
    avoid_tickers: set[str],
    long_params=None,
) -> pd.DataFrame:
    from research.gainer_daily_backtest import (
        GAINER_MOMENTUM,
        LIQUID100,
        fetch_gainer_data_yahoo,
        pick_top_gainers,
    )

    if cfg.get("require_spy_ma50_for_long", True) and not regime["bull"]:
        return pd.DataFrame()

    end = date.today().isoformat()
    start = (date.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    pool = LIQUID100 if cfg.get("quick") else GAINER_MOMENTUM
    if long_params is None:
        from quant.pattern_params import LongParams
        long_params = LongParams()
    top_n = int(long_params.top_n or cfg.get("long_top_n", 2))
    data, spy = fetch_gainer_data_yahoo(pool, start, end)
    if not data:
        return pd.DataFrame()

    spy_close = spy["Close"].astype(float) if spy is not None and not spy.empty else None
    if spy_close is not None and isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    as_of = pd.Timestamp(end)
    filters = long_params.build_gainer_filters()
    picks = pick_top_gainers(data, as_of, spy_close, filters)
    if picks.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, r in picks.iterrows():
        tk = str(r["代码"])
        if tk in avoid_tickers:
            continue
        # 轨迹特征（加分标签）
        df = data.get(tk)
        up_tags = []
        if df is not None and not df.empty:
            feat = extract_trajectory_features(df, forward_days=20)
            if not feat.empty:
                up_tags = assess_up_favor(feat.iloc[-1])
        rows.append({
            "代码": tk,
            "方向": "做多",
            "选股理由": r.get("选股理由", ""),
            "1日涨%": round(float(r.get("涨幅%", 0)), 2),
            "量比": round(float(r.get("量比", 0)), 2),
            "近8次胜率": round(float(r.get("近8次胜率", 0) or 0) * 100, 1),
            "轨迹加分": "；".join(t["note"] for t in up_tags) if up_tags else "—",
            "持有": "1日（次日平仓纪律）",
        })
    return pd.DataFrame(rows)


def scan_income_leg(cfg: dict) -> dict:
    from sndk_iron_daily import fleet_lines, load_config as load_fleet_cfg, run_fleet

    fleet_path = ROOT / str(cfg.get("fleet_config", "sndk_iron_config.json"))
    fleet_cfg = load_fleet_cfg(fleet_path)
    if not (fleet_cfg.get("fleet") or {}).get("enabled", True):
        return {"lines": ["舰队未启用"], "result": None, "errors": []}
    result = run_fleet(fleet_cfg)
    return {
        "lines": fleet_lines(result),
        "result": result,
        "errors": result.get("errors") or [],
    }


def _income_tickers(income: dict) -> set[str]:
    rows = (income.get("result") or {}).get("fleet_rows") or []
    return {str(r.get("ticker", "")).upper() for r in rows if r.get("ticker")}


def _fleet_avoid_conflicts(
    avoid_df: pd.DataFrame,
    income: dict,
    down_params=None,
) -> list[dict]:
    """收租标的命中回避规律 → 收租需警惕。"""
    if avoid_df is None or avoid_df.empty:
        return []
    from quant.pattern_params import DownParams

    dp = down_params or DownParams()
    inc = _income_tickers(income)
    if not inc:
        return []
    conflicts: list[dict] = []
    for tk in inc & set(avoid_df["代码"].astype(str)):
        sub = avoid_df[avoid_df["代码"] == tk]
        urgent = sub[sub.get("紧急度", pd.Series(dtype=str)) == "高"] if "紧急度" in sub.columns else sub
        r20 = float(sub["20日涨跌%"].iloc[0]) if "20日涨跌%" in sub.columns and len(sub) else 0
        rules = "；".join(sub["规则"].astype(str).tolist())
        conflicts.append({
            "代码": tk,
            "规则": rules,
            "20日涨跌%": r20,
            "级别": "高" if not urgent.empty or r20 >= dp.income_conflict_min_ret_20d * 100 else "中",
        })
    return conflicts


def scan_5d_path_leg(cfg: dict) -> pd.DataFrame:
    """5 日路径规律（真实 OHLCV + 换手率）。"""
    rules_path = ROOT / "research" / "move_pattern_5d_rules.json"
    if not rules_path.exists():
        return pd.DataFrame()
    import json
    from research.move_pattern_5d_mine import LiquidityFilter, PathThreshold, scan_today_5d

    doc = json.loads(rules_path.read_text(encoding="utf-8-sig"))
    liq = LiquidityFilter(**(doc.get("liquidity") or {}))
    th_d = doc.get("threshold") or {}
    th = PathThreshold(up_pct=float(th_d.get("up_pct", 3)), down_pct=float(th_d.get("down_pct", 3)))
    rules = doc.get("rules") or []
    df = scan_today_5d(rules, liq=liq, th=th, quick=bool(cfg.get("quick")), min_tier_hit=0.62)
    if df.empty:
        return df
    up = df[df["方向"] == "偏多"].head(int(cfg.get("5d_top_n", 5)))
    down = df[df["方向"] == "偏空"].head(int(cfg.get("5d_top_n", 5)))
    return pd.concat([up, down], ignore_index=True)


def build_plan(cfg: dict) -> dict:
    rules = load_optimized_rules()
    regime = _spy_regime()
    min_dvol = float(cfg.get("min_dvol_m", 30))
    quick = bool(cfg.get("quick"))

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    pool = build_candidate_pool(use_broad=not quick, max_names=100 if quick else 0)
    end = date.today().isoformat()
    start = (date.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    batch = yahoo.fetch_batch(pool, start, end)
    features = _last_features(batch, min_dvol)

    avoid_df = scan_avoid_leg(features, rules.down)
    avoid_tickers = set(avoid_df["代码"].astype(str).tolist()) if not avoid_df.empty else set()
    long_df = scan_long_leg(cfg, regime, avoid_tickers, rules.long)
    income = scan_income_leg(cfg)
    path5d = scan_5d_path_leg(cfg)
    conflicts = _fleet_avoid_conflicts(avoid_df, income, rules.down)

    return {
        "date": end,
        "regime": regime,
        "rules": rules,
        "long": long_df,
        "avoid": avoid_df,
        "path5d": path5d,
        "income": income,
        "fleet_conflicts": conflicts,
        "feature_count": len(features),
    }


def format_lines(plan: dict) -> list[str]:
    lines: list[str] = []
    reg = plan["regime"]
    lines.append(f"三腿每日策略 · {plan['date']} · 扫描 {plan['feature_count']} 只")
    lines.append(reg["label"])
    rules = plan.get("rules")
    if rules and rules.meta.get("long_search"):
        ls = rules.meta["long_search"]
        if ls.get("is"):
            lines.append(
                f"  参数寻优: 做多次日胜率 IS={ls['is']['win_rate']:.0%}"
                + (f" OOS={ls['oos']['win_rate']:.0%}" if ls.get("oos") else "")
            )
    lines.append("")

    lines.append("【腿① 上涨规律 · 做多】真实量价 · 涨幅榜高置信")
    lp = rules.long if rules else None
    if lp and lp.require_spy_positive_1d:
        lines.append(f"  过滤: SPY 当日涨≥{lp.min_spy_1d_pct}% 才开仓")
    if not reg["bull"]:
        lines.append("  ⏸ 弱市：SPY<MA50，腿①关闭（不做多）")
    elif plan["long"] is None or plan["long"].empty:
        lines.append("  今日无 Top 命中（或被腿②回避过滤）")
    else:
        for _, r in plan["long"].iterrows():
            lines.append(f"  ✅ {r['代码']} 1日{r['1日涨%']:+.1f}% 量比{r['量比']} 形态胜率{r['近8次胜率']:.0f}%")
            lines.append(f"     {r['选股理由']}")
            if r.get("轨迹加分") and r["轨迹加分"] != "—":
                lines.append(f"     轨迹: {r['轨迹加分']}")

    lines.append("")
    lines.append("【腿①b · 5日路径规律】真实OHLCV · 路径涨/跌≥3% · 含换手率")
    p5 = plan.get("path5d")
    if p5 is None or p5.empty:
        lines.append("  无高置信 5 日路径命中（或尚未运行 move_pattern_5d_mine.py）")
    else:
        for _, r in p5.iterrows():
            wr = float(r.get("5日命中率", 0) or 0) * 100
            lines.append(
                f"  {'📈' if r['方向']=='偏多' else '📉'} {r['代码']} {r['方向']} "
                f"命中{wr:.0f}% n={r.get('样本数')} 量比{r['量比']} 换手{r.get('换手率%', '—')}%"
            )
            lines.append(f"     {r['规律']} → {r['建议']}")

    lines.append("")
    lines.append("【腿② 下跌规律 · 回避】真实量价 · 243k事件提炼")
    if plan["avoid"] is None or plan["avoid"].empty:
        lines.append("  ✅ 全市场无高危回避信号")
    else:
        for _, r in plan["avoid"].iterrows():
            tag = f"[{r.get('紧急度', '中')}/{r.get('周期', '20d')}]"
            lines.append(f"  ⚠ {r['代码']} {tag} [{r['规则']}] {r['原因']} → {r['建议']}")

    conflicts = plan.get("fleet_conflicts") or []
    if conflicts:
        lines.append("")
        lines.append("  🚨 收租×回避冲突（规律命中但舰队仍持有该标的）")
        for c in conflicts:
            lines.append(
                f"     {c['代码']} 20日{c['20日涨跌%']:+.0f}% · {c['规则']} → "
                f"{'暂停新开/考虑减仓' if c['级别']=='高' else '收租缩仓/提高止盈纪律'}"
            )

    lines.append("")
    lines.append("【腿③ 收租 · 真实期权链】mixed_balanced · bid/ask")
    for ln in plan["income"].get("lines") or []:
        lines.append(f"  {ln}" if ln else "")
    for e in plan["income"].get("errors") or []:
        lines.append(f"  ⚠ {e}")

    lines.append("")
    lines.append(
        "纪律：腿①小仓1日 · 腿②命中不做多 · 腿③50%止盈 · 数据=真实量价+真实链（非BS回测）"
    )
    return lines


def format_notification(plan: dict) -> tuple[str, str]:
    n_long = len(plan["long"]) if plan["long"] is not None and not plan["long"].empty else 0
    n_avoid = len(plan["avoid"]) if plan["avoid"] is not None and not plan["avoid"].empty else 0
    inc = plan["income"].get("result") or {}
    n_inc = sum(1 for r in (inc.get("fleet_rows") or []) if r.get("plan") is not None)
    title = f"📊 三腿策略 · 多{n_long} 避{n_avoid} 租{n_inc}"
    body = plan["regime"]["label"][:120]
    return title, body


def append_history(plan: dict) -> None:
    long_t = ",".join(plan["long"]["代码"].astype(str).tolist()) if plan["long"] is not None and not plan["long"].empty else ""
    avoid_t = ",".join(plan["avoid"]["代码"].astype(str).unique().tolist()) if plan["avoid"] is not None and not plan["avoid"].empty else ""
    row = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "大盘": plan["regime"]["label"],
        "做多样": long_t,
        "回避数": len(plan["avoid"]) if plan["avoid"] is not None else 0,
        "回避清单": avoid_t,
        "收租摘要": " | ".join(format_lines(plan))[:2000],
    }
    df = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        df = pd.concat([pd.read_csv(HISTORY_FILE), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def print_report(plan: dict) -> None:
    print("=" * 78)
    for line in format_lines(plan):
        print(line)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="三腿每日策略：做多+回避+收租")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="缩小股票池加速")
    ap.add_argument("--optimize", action="store_true", help="先运行参数寻优再出计划")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.quick:
        cfg["quick"] = True

    if args.optimize:
        from research.pattern_param_search import print_report as print_opt, run_search

        rules = run_search(quick=bool(cfg.get("quick")))
        print_opt(rules)

    plan = build_plan(cfg)
    print_report(plan)
    append_history(plan)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify = cfg.get("notify", {})
    title, body = format_notification(plan)
    if notify.get("desktop"):
        desktop_notify(title, body)
    if notify.get("email", {}).get("enabled"):
        email_notify(notify["email"], title, "\n".join(format_lines(plan)))


if __name__ == "__main__":
    main()
