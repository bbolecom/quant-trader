"""快速预设搜索（LIQUID100，15组以内）。"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import (
    LIQUID100,
    GainerProFilters,
    backtest_daily_gainer_portfolio,
    legacy_filters,
    search_win_rate_params,
)


def main():
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=800)).isoformat()
    reset_provider_cache()
    yahoo = get_provider(DataConfig(provider="yahoo"))
    print("拉取 LIQUID100…")
    data = yahoo.fetch_batch(LIQUID100, start, end)
    spy = yahoo.fetch_history("SPY", start, end)

    leg = backtest_daily_gainer_portfolio(data, spy, start=start, end=end, filt=legacy_filters(top_n=5))
    print(f"legacy: win={leg.get('日胜率', 0):.3f} days={leg.get('交易天数')} tot={leg.get('累计收益率', 0):+.3f}")

    cur = backtest_daily_gainer_portfolio(data, spy, start=start, end=end, filt=GainerProFilters())
    print(f"defaults: win={cur.get('日胜率', 0):.3f} days={cur.get('交易天数')} tot={cur.get('累计收益率', 0):+.3f}")

    print("\n搜索高胜率参数…")
    filt, res = search_win_rate_params(data, spy, start=start, end=end)
    print(
        f"BEST: win={res['日胜率']:.3f} days={res['交易天数']} tot={res['累计收益率']:+.3f} "
        f"max_gain={filt.max_gain_pct} vr={filt.max_vol_ratio} hist={filt.min_setup_win_rate}"
    )


if __name__ == "__main__":
    main()
