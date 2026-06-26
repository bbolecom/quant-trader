#!/usr/bin/env python3
"""每日波动率衰减（VRP）信号推送脚本（命令行，可用 launchd / cron 定时执行）。

功能：
    1. 读取 vrp_config.json：反向 ETF 择时、VIX 预警、CSP 候选扫描参数。
    2. 生成今日执行清单（反向 ETF 持有/清仓 + Top CSP 标的 + 风控提示）。
    3. 弹 macOS 桌面通知（可选）并发邮件（可选）。
    4. 追加写入 vrp_history.csv。

用法：
    python vrp_daily.py                 # 默认 vrp_config.json
    python vrp_daily.py -c my.json        # 指定配置
    python vrp_daily.py --dry-run         # 只打印，不通知
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant import vol_decay
from quant.data import DataError, fetch_history
from scan_daily import desktop_notify, email_notify

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "vrp_config.json"
HISTORY_FILE = ROOT / "vrp_history.csv"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_csp_tickers(cfg: dict) -> list[str]:
    csp = cfg.get("csp", {})
    pool = csp.get("pool", "default")
    if pool == "most_actives":
        try:
            from quant import screener
            snap = screener.fetch_yahoo_screen("most_actives", count=40)
            if not snap.empty:
                return snap["代码"].tolist()
        except Exception:  # noqa: BLE001
            pass
    if pool == "custom":
        raw = csp.get("custom_tickers") or []
        if isinstance(raw, str):
            raw = raw.replace(",", " ").split()
        return [str(t).strip().upper() for t in raw if str(t).strip()]
    return list(vol_decay.DEFAULT_CSP_UNIVERSE)


def run_vrp(cfg: dict) -> dict:
    """执行 VRP 扫描，返回结构化结果。"""
    end = date.today().isoformat()
    etf = str(cfg.get("inverse_etf", "SVIX")).upper()
    ma_win = int(cfg.get("ma_window", 50))
    etf_lb = int(cfg.get("lookback_etf_days", 450))
    csp_lb = int(cfg.get("lookback_csp_days", 550))
    csp_cfg = cfg.get("csp", {})

    start_etf = (date.today() - timedelta(days=etf_lb)).isoformat()
    start_csp = (date.today() - timedelta(days=csp_lb)).isoformat()

    result: dict = {
        "etf": etf,
        "ma_window": ma_win,
        "vix": None,
        "etf_sig": None,
        "csp_table": pd.DataFrame(),
        "playbook": [],
        "errors": [],
    }

    result["vix"] = vol_decay.vix_alert(end=end)

    try:
        etf_df = fetch_history(etf, start=start_etf, end=end)
        result["etf_sig"] = vol_decay.inverse_etf_signal(etf_df, etf, ma_window=ma_win)
    except DataError as e:
        result["errors"].append(f"反向 ETF：{e}")

    filters = vol_decay.CspFilters(
        min_dollar_vol_m=float(csp_cfg.get("min_dollar_vol_m", 500)),
        min_rv_pct=float(csp_cfg.get("min_rv_pct", 30)),
        max_rv_pct=float(csp_cfg.get("max_rv_pct", 70)),
    )
    tickers = resolve_csp_tickers(cfg)
    max_scan = int(csp_cfg.get("max_scan", 20))
    if max_scan > 0:
        tickers = tickers[:max_scan]
    try:
        result["csp_table"] = vol_decay.scan_csp_candidates(tickers, start_csp, end, filters)
    except DataError as e:
        result["errors"].append(f"CSP 扫描：{e}")

    top_n = int(csp_cfg.get("top_n", 5))
    result["playbook"] = vol_decay.daily_playbook(
        result["etf_sig"], result["vix"], result["csp_table"], max_csp=top_n,
    )
    return result


def write_today_json(result: dict, cfg: dict) -> Path:
    out = cfg.get("outputs") or {}
    tj = ROOT / out.get("today_json", "research/vrp_today.json")
    tj.parent.mkdir(parents=True, exist_ok=True)

    vix = result.get("vix")
    etf_sig = result.get("etf_sig")
    csp = result.get("csp_table")
    top_n = int((cfg.get("csp") or {}).get("top_n", 5))
    if csp is not None and not csp.empty:
        csp_rows = csp.head(top_n).to_dict(orient="records")
    else:
        csp_rows = []

    actionable = len(csp_rows)
    if etf_sig and "持有" in str(etf_sig.action):
        actionable = max(actionable, 1)

    doc = {
        "date": date.today().isoformat(),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy_id": "vrp",
        "title": "VRP波动率溢价",
        "etf": result.get("etf"),
        "vix": vars(vix) if vix else None,
        "etf_signal": vars(etf_sig) if etf_sig else None,
        "scan_stats": {
            "CSP候选": len(csp_rows),
            "可开仓": actionable,
        },
        "csp_candidates": csp_rows,
        "picks": csp_rows,
        "playbook": result.get("playbook") or [],
        "errors": result.get("errors") or [],
    }
    payload = json.dumps(doc, ensure_ascii=False, indent=2)
    tj.write_text(payload, encoding="utf-8")
    ios = out.get("ios_bundle")
    if ios:
        ip = ROOT / ios
        ip.parent.mkdir(parents=True, exist_ok=True)
        ip.write_text(payload, encoding="utf-8")
    return tj


def format_notification(result: dict) -> tuple[str, str]:
    """生成 (标题, 正文) 供桌面/邮件通知。"""
    etf_sig = result.get("etf_sig")
    vix = result.get("vix")
    csp = result.get("csp_table")

    if etf_sig:
        etf_part = etf_sig.action.replace("🟢 ", "").replace("🔴 ", "")
        title = f"VRP · {etf_sig.ticker} {etf_part.split('/')[0].strip()}"
    elif vix:
        title = f"VRP · VIX {vix.level}"
    else:
        title = "VRP 波动率信号"

    parts: list[str] = []
    if vix:
        parts.append(f"VIX {vix.vix:.1f} {vix.level}（日变 {vix.daily_chg_pct:+.0%}）")
    if etf_sig:
        parts.append(f"{etf_sig.ticker} ${etf_sig.close:,.2f} vs MA{etf_sig.ma_window} ${etf_sig.ma:,.2f}")
        parts.append(etf_sig.action)
    if csp is not None and not csp.empty:
        tops = "、".join(csp.head(3)["代码"].tolist())
        parts.append(f"CSP Top: {tops}")
    else:
        parts.append("CSP: 无候选")

    if result.get("errors"):
        parts.append("⚠ " + "；".join(result["errors"]))

    body = " · ".join(parts)
    if len(body) > 220:
        body = body[:217] + "…"
    return title, body


def build_history_row(result: dict) -> dict:
    """构造写入 CSV 的一行记录。"""
    etf_sig = result.get("etf_sig")
    vix = result.get("vix")
    csp = result.get("csp_table")
    top_codes = ""
    top1 = ""
    if csp is not None and not csp.empty:
        top_codes = ",".join(csp["代码"].tolist())
        top1 = str(csp.iloc[0]["代码"])
    return {
        "扫描时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "反向ETF": result.get("etf", ""),
        "ETF动作": etf_sig.action if etf_sig else "",
        "ETF收盘": etf_sig.close if etf_sig else None,
        "ETF均线": etf_sig.ma if etf_sig else None,
        "VIX": vix.vix if vix else None,
        "VIX状态": vix.level if vix else "",
        "CSP首选": top1,
        "CSP列表": top_codes,
        "执行清单": " | ".join(result.get("playbook", [])),
        "错误": "；".join(result.get("errors", [])),
    }


def append_history(result: dict) -> None:
    row = build_history_row(result)
    df = pd.DataFrame([row])
    header = not HISTORY_FILE.exists()
    df.to_csv(HISTORY_FILE, mode="a", header=header, index=False, encoding="utf-8-sig")


def print_report(result: dict) -> None:
    print(f"=== VRP 波动率信号 {datetime.now():%Y-%m-%d %H:%M} ===")
    if result.get("errors"):
        for e in result["errors"]:
            print(f"[警告] {e}", file=sys.stderr)

    vix = result.get("vix")
    if vix:
        print(f"\n[VIX] {vix.level}  {vix.vix:.1f}（20MA {vix.vix_ma20:.1f}，日变 {vix.daily_chg_pct:+.1%}）")
        print(f"      {vix.message}")

    etf_sig = result.get("etf_sig")
    if etf_sig:
        print(f"\n[反向ETF] {etf_sig.ticker} · {etf_sig.as_of}")
        print(f"  {etf_sig.action}")
        print(f"  {etf_sig.detail}")

    csp = result.get("csp_table")
    if csp is not None and not csp.empty:
        print(f"\n[CSP 候选] {len(csp)} 只（按综合分排序）")
        show = csp.copy()
        for c in ["最新价", "建议Put行权", "估算权利金"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: f"{x:.2f}")
        cols = [c for c in ["代码", "RV20%", "成交额M", "建议Put行权", "估算权利金", "月化收益%", "综合分"]
                if c in show.columns]
        print(show[cols].to_string(index=False))
    else:
        print("\n[CSP 候选] 无符合筛选条件的标的")

    print("\n[今日执行清单]")
    for step in result.get("playbook", []):
        print(f"  {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="每日 VRP 波动率衰减信号推送")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不发送通知")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    result = run_vrp(cfg)
    write_today_json(result, cfg)
    print_report(result)
    append_history(result)

    if args.dry_run:
        print("\n[dry-run] 跳过通知。")
        return

    notify_cfg = cfg.get("notify", {})
    title, body = format_notification(result)
    if notify_cfg.get("desktop"):
        desktop_notify(title, body)
    if notify_cfg.get("email", {}).get("enabled"):
        email_body = "\n".join(result.get("playbook", []))
        if result.get("errors"):
            email_body += "\n\n错误：\n" + "\n".join(result["errors"])
        email_notify(notify_cfg["email"], title, email_body or body)


if __name__ == "__main__":
    main()
