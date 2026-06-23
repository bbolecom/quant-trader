"""网格搜索 v4 极严做空参数（目标笔胜率≥80%）。"""
from __future__ import annotations

import itertools
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.flow_strategy import FlowStrategyParams, build_signal_panel, run_portfolio_backtest
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM


def main() -> None:
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(3 * 365) + 150)).isoformat()
    spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
    batch = yahoo.fetch_batch(GAINER_MOMENTUM, start, end)

    base = FlowStrategyParams(
        name="wide",
        long_patterns=frozenset({"U_S1", "U_S2", "U_A2"}),
        short_patterns=frozenset({"D_S2", "D_OFFERING"}),
        offering_proxy_pct=40.0,
        long_top_n=1,
        short_top_n=1,
        min_dvol_m=50.0,
        min_price=5.0,
    )
    panel = build_signal_panel(batch, spy, base)
    print(f"base panel shorts: {len(panel[panel['signal']=='short'])}")

    rows = []
    for prev in [40, 42, 44, 45, 50]:
        for gmax in [-1.5, -2.0, -2.5, -3.0, -5.0]:
            for spy_bear in [False, True]:
                for max_spy in [999.0, 0.0, -0.3]:
                    if spy_bear and max_spy < 999:
                        continue
                    p = FlowStrategyParams(
                        name="grid",
                        long_patterns=frozenset({"U_S2"}),
                        short_patterns=frozenset({"D_S2", "D_OFFERING"}),
                        offering_proxy_pct=40.0,
                        short_min_prev_pct=float(prev),
                        short_max_today_gain_pct=float(gmax),
                        require_spy_bear_for_short=spy_bear,
                        max_spy_1d_pct_for_short=max_spy,
                        long_top_n=0,
                        short_top_n=1,
                        min_dvol_m=50.0,
                        min_price=5.0,
                    )
                    res = run_portfolio_backtest(panel, p)
                    n = res["总笔数"]
                    wr = res["笔胜率"]
                    if n >= 4:
                        rows.append({
                            "prev": prev, "gmax": gmax, "spy_bear": spy_bear,
                            "max_spy": max_spy, "n": n, "wr": wr,
                            "ret": res["累计收益率"],
                        })

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["wr", "n"], ascending=[False, False])
    print("\nTop 15 by win rate (n>=4):")
    print(df.head(15).to_string(index=False))
    hit = df[df["wr"] >= 0.80]
    print(f"\n>=80% combos: {len(hit)}")
    if not hit.empty:
        print(hit.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
