"""资金流向策略参数寻优（基于已缓存信号面板，加速迭代）。

用法：
    python research/flow_strategy_optimize.py --years 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from quant.flow_strategy import FlowStrategyParams, build_signal_panel, run_portfolio_backtest
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

PANEL_CACHE = ROOT / "research" / "flow_strategy_panel.parquet"


def load_or_build_panel(years: float, quick: bool) -> tuple[pd.DataFrame, pd.Series]:
    if PANEL_CACHE.exists():
        meta = json.loads((ROOT / "research/flow_strategy_panel_meta.json").read_text())
        if meta.get("years") == years and meta.get("quick") == quick:
            return pd.read_parquet(PANEL_CACHE), None

    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(years * 365) + 150)).isoformat()
    pool = LIQUID100 if quick else GAINER_MOMENTUM
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    batch = yahoo.fetch_batch(pool, start, end)
    # 宽面板：不做规律过滤，用宽松 params 构建全特征
    wide = FlowStrategyParams(
        name="wide",
        long_patterns=frozenset({"U_S1", "U_S2", "U_A1", "U_A2"}),
        short_patterns=frozenset({"D_S2", "D_A3", "D_B3", "D_OFFERING"}),
        offering_proxy_pct=25.0,
        min_dvol_m=30.0,
        min_price=3.0,
    )
    panel = build_signal_panel(batch, spy, wide)
    panel.to_parquet(PANEL_CACHE, index=False)
    (ROOT / "research/flow_strategy_panel_meta.json").write_text(
        json.dumps({"years": years, "quick": quick, "rows": len(panel)}), encoding="utf-8",
    )
    return panel, spy


def grid_search(panel: pd.DataFrame, *, min_trades: int = 80) -> pd.DataFrame:
    rows: list[dict] = []
    for s2_lo, s2_hi in [(7, 15), (7, 12), (5, 12)]:
        for off in (30, 40, 50):
            for ltop, stop in [(1, 1), (2, 1), (1, 2)]:
                for lp in (["U_S2"], ["U_A2", "U_S2"]):
                    p = FlowStrategyParams(
                        name="grid",
                        long_patterns=frozenset(lp),
                        short_patterns=frozenset({"D_S2", "D_OFFERING"}),
                        offering_proxy_pct=float(off),
                        long_s2_min_gain_pct=float(s2_lo),
                        long_s2_max_gain_pct=float(s2_hi),
                        long_min_close_strength=0.65,
                        long_min_vol_ratio=1.5,
                        long_max_vol_ratio=3.0,
                        long_top_n=ltop,
                        short_top_n=stop,
                        min_dvol_m=50.0,
                        min_price=5.0,
                    )
                    # 在宽面板上按 params 重筛
                    sub_rows = []
                    for _, r in panel.iterrows():
                        feat = {
                            "涨幅%": r["涨幅%"], "量比": r["量比"],
                            "涨幅5d%": r.get("5日涨%"), "前日涨幅%": r.get("前日涨%"),
                            "close_strength": r.get("收盘强度"),
                            "vol_ratio": r["量比"], "dvol_m": 50,
                            "现价": r["现价"], "above_ma50": r.get("spy_bull"),
                        }
                        from quant.flow_strategy import evaluate_actionable_signal
                        sig = evaluate_actionable_signal(
                            feat, p, spy_bull=bool(r.get("spy_bull", True)),
                            spy_1d_pct=float(r.get("spy_1d%", 0)),
                        )
                        if sig["signal"] == "flat":
                            continue
                        if sig["signal"] != r["signal"]:
                            continue
                        sub_rows.append({
                            **r.to_dict(),
                            "规律": sig["规律"],
                            "score": sig["score"],
                        })
                    if not sub_rows:
                        continue
                    sub = pd.DataFrame(sub_rows)
                    res = run_portfolio_backtest(sub, p)
                    if res.get("总笔数", 0) < min_trades:
                        continue
                    rows.append({
                        "long": ",".join(lp),
                        "s2_gain": f"{s2_lo}-{s2_hi}",
                        "off_pct": off,
                        "ltop": ltop,
                        "stop": stop,
                        "笔胜率": res["笔胜率"],
                        "日胜率": res["日胜率"],
                        "累计%": res["累计收益率"] * 100,
                        "夏普": res["夏普比率"],
                        "回撤%": res["最大回撤"] * 100,
                        "笔数": res["总笔数"],
                    })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["笔胜率", "夏普"], ascending=False)
    out.to_csv(ROOT / "research/flow_strategy_grid.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    panel, _ = load_or_build_panel(args.years, args.quick)
    print(f"面板 {len(panel)} 行，开始网格搜索…")
    df = grid_search(panel)
    print(df.head(10).to_string(index=False))
    print(f"\n→ research/flow_strategy_grid.csv")


if __name__ == "__main__":
    main()
