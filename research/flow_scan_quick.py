"""快速扫描 v3 面板子集胜率。"""
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.flow_strategy import build_signal_panel, load_strategy_config
from quant.providers import DataConfig, get_provider, reset_provider_cache
from research.gainer_daily_backtest import GAINER_MOMENTUM

reset_provider_cache()
cfg = load_strategy_config()
end = date.today().isoformat()
start = (date.today() - timedelta(days=int(3 * 365) + 150)).isoformat()
yahoo = get_provider(DataConfig(provider="yahoo"))
spy = yahoo.fetch_history("SPY", start, end)["Close"].astype(float)
batch = yahoo.fetch_batch(GAINER_MOMENTUM, start, end)
panel = build_signal_panel(batch, spy, cfg)
print(f"panel rows {len(panel)}")
trades = panel.copy()
trades["win"] = trades.apply(
    lambda r: r["fwd_1d"] > 0 if r["signal"] == "long" else r["fwd_1d"] < 0, axis=1
)


def scan(df, label=""):
    best = []
    for side in ["long", "short"]:
        s = df[df["signal"] == side]
        if len(s) < 5:
            continue
        for prev_min in [0, 30, 40, 45, 50, 55, 60, 65, 70]:
            for g0, g1 in [(0, 999), (-999, 0), (-999, -3), (-999, -5), (7, 15), (8, 12), (9, 11)]:
                for vr0, vr1 in [(0, 999), (1.5, 3), (1.6, 2), (1.7, 2.2)]:
                    for cs in [0, 0.65, 0.7, 0.75, 0.8]:
                        m = s["前日涨%"] >= prev_min
                        m &= (s["涨幅%"] >= g0) & (s["涨幅%"] <= g1)
                        m &= (s["量比"] >= vr0) & (s["量比"] <= vr1)
                        if cs > 0:
                            m &= s["收盘强度"] >= cs
                        sub = s[m]
                        if len(sub) >= 8:
                            wr = sub["win"].mean()
                            if wr >= 0.78:
                                best.append((wr, len(sub), side, prev_min, g0, g1, vr0, vr1, cs))
    best.sort(key=lambda x: (-x[0], -x[1]))
    print(f"\n{label} top combos >=78%:")
    for b in best[:20]:
        print(b)


scan(trades, "all")

for prev in range(40, 75, 5):
    for gmax in [10, 5, 0, -2, -5, -8, -10]:
        sub = trades[(trades["signal"] == "short") & (trades["前日涨%"] >= prev) & (trades["涨幅%"] <= gmax)]
        if len(sub) >= 5:
            wr = sub["win"].mean()
            if wr >= 0.70:
                print(f"short prev>={prev} gain<={gmax}: n={len(sub)} wr={wr:.1%}")
