"""短期均值回归（顺势买跌）规律 → 策略 + 回测（含样本外 OOS）。

动机（诚实）：
  同一流动性宇宙上，"追强势动量持有 5 日"（liquid_momentum_pattern.py）实测为
  负收益（胜率~40%、年化~-21%）——强势短期倾向于均值回归。故本脚本挖掘其**反面**、
  且学术与实务都稳健的异象：**短期反转 / 顺势买跌（Connors RSI-2 式）**。

规律（事件驱动，无未来函数；信号仅用 t 日及之前信息，t+1 开盘成交）：
  进场（t 日收盘判定）：
    · 处于上升趋势：close > SMA(trend_ma)（只在多头结构里买跌，过滤价值陷阱）
    · 超卖：RSI(rsi_n) ≤ rsi_max（极短周期 RSI，Connors 经典 = 2）
    · 回调确认：close < SMA(pullback_ma)（贴近/跌破短均线）
    · 日成交额 ≥ dvol_min_m（流动性闸门）
  离场（真实 High/Low 路径，t+1 开盘为入场基准，最长持有 horizon 日）：
    · 反弹了结：盘中 high ≥ 入场×(1+tp) → 止盈（tp 较小，吃反弹）
    · 或首个收盘转正：close ≥ 入场 → 当日收盘平仓（exit_on_green）
    · 风控：盘中 low ≤ 入场×(1-sl) → 止损（sl=0 表示不设硬止损，提高胜率但放大尾部回撤）
    · 否则 horizon 末日收盘平仓
  组合：K 个并发槽位，等权、按实现盈亏复利；满仓跳过新信号 → 日频权益曲线求最大回撤。

对照用户目标：年化>100% · 胜率>90% · 回撤<10% · 流动性好。
诚实结论会随回测给出（大概率：胜率/回撤可观，但年化与「胜率>90%且回撤<10%」难同时摸到——
若能稳健三达标早被套利抹平）。

用法：
    python research/mean_reversion_dip.py              # 默认参数 + 网格寻优 + OOS
    python research/mean_reversion_dip.py --no-grid    # 仅默认参数
    python research/mean_reversion_dip.py --years 6 --no-cache
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用与 liquid_momentum 相同的缓存文件与流动性宇宙，保证可离线、同口径对照。
from research.gainer_daily_backtest import GAINER_MOMENTUM  # noqa: E402

CACHE = ROOT / "research" / "_liquid_momentum_prices.pkl"
COST_BPS = 15.0  # 单边手续费+滑点（基点），往返按 2× 计


# --------------------------- 数据 ---------------------------

def fetch_prices(tickers: list[str], years: int, *, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """优先复用 liquid_momentum 的本地缓存（离线、零网络）；缺失则用 yfinance 拉取。"""
    if use_cache and CACHE.exists():
        try:
            obj = pickle.loads(CACHE.read_bytes())
            if obj.get("years") == years and set(obj.get("tickers", [])) >= set(tickers):
                return obj["data"]
            # 缓存覆盖不全也先用上（子集即可回测），避免无谓联网。
            if obj.get("data"):
                return obj["data"]
        except Exception:  # noqa: BLE001
            pass
    import yfinance as yf

    raw = yf.download(
        tickers, period=f"{years}y", interval="1d",
        group_by="ticker", auto_adjust=True, threads=True, progress=False,
    )
    data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(subset=["Close"]).copy()
        if len(df) < 220:  # 需要 SMA200 暖机
            continue
        df.index = pd.to_datetime(df.index)
        data[t] = df[["Open", "High", "Low", "Close", "Volume"]]
    CACHE.write_bytes(pickle.dumps({"years": years, "tickers": list(data.keys()), "data": data}))
    return data


# --------------------------- 信号 + 交易 ---------------------------

@dataclass
class Params:
    trend_ma: int = 200       # 趋势过滤：只在 close > SMA(trend_ma) 时买跌
    pullback_ma: int = 5      # 回调确认：close < SMA(pullback_ma)
    rsi_n: int = 2            # 极短 RSI 周期（Connors 经典 = 2）
    rsi_max: float = 10.0     # 超卖阈值
    dvol_min_m: float = 50.0  # 日成交额闸门（百万美元）
    tp: float = 0.05          # 反弹止盈
    sl: float = 0.0           # 硬止损（0 = 不设，吃满反弹、提高胜率）
    horizon: int = 10         # 最长持有日
    exit_on_green: bool = True  # 首个收盘转正即了结（Connors 式）
    slots: int = 10           # 并发槽位
    use_regime: bool = True   # 大盘择时闸门：仅在 SPY 多头时买跌（不接下跌中的飞刀）
    regime_ma: int = 200      # 大盘趋势均线（SPY close > SMA(regime_ma) 视为多头）
    regime_symbol: str = "SPY"  # 大盘基准（缺失时回退 QQQ）


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # 全程无下跌 → RSI=100；全程无上涨 → RSI=0
    rsi = rsi.where(roll_down != 0, 100.0)
    rsi = rsi.where(roll_up != 0, rsi.where(roll_down == 0, 0.0))
    return rsi


def _signal_mask(df: pd.DataFrame, p: Params) -> pd.Series:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    sma_trend = close.rolling(p.trend_ma, min_periods=p.trend_ma // 2).mean()
    sma_pull = close.rolling(p.pullback_ma, min_periods=max(2, p.pullback_ma // 2)).mean()
    rsi = _rsi(close, p.rsi_n)
    dvol_m = close * vol / 1e6

    return (
        (close > sma_trend)
        & (close < sma_pull)
        & (rsi <= p.rsi_max)
        & (dvol_m >= p.dvol_min_m)
    ).fillna(False)


def build_regime(data: dict[str, pd.DataFrame], p: Params) -> pd.Series | None:
    """大盘多头掩码（True=可买跌）：SPY/QQQ close > SMA(regime_ma)，仅用当日及之前信息。"""
    bench = None
    for sym in (p.regime_symbol, "QQQ", "SPY"):
        cand = data.get(sym)
        if cand is not None and not cand.empty:
            bench = cand
            break
    if bench is None:
        return None
    c = bench["Close"].astype(float)
    return c > c.rolling(p.regime_ma, min_periods=p.regime_ma // 2).mean()


def gen_trades(data: dict[str, pd.DataFrame], p: Params) -> pd.DataFrame:
    """生成所有交易：entry_date, exit_date, ret（已扣往返成本）。"""
    cost = 2 * COST_BPS / 1e4
    regime = build_regime(data, p) if p.use_regime else None
    rows = []
    for tk, df in data.items():
        mask = _signal_mask(df, p)
        if regime is not None:
            # 用决策日 t（信号日收盘）的大盘状态闸门；对齐到本标的交易日历。
            reg_aligned = regime.reindex(df.index).ffill().fillna(False).astype(bool)
            mask = mask & reg_aligned.values
        idx = np.where(mask.values)[0]
        n = len(df)
        opens = df["Open"].astype(float).values
        highs = df["High"].astype(float).values
        lows = df["Low"].astype(float).values
        closes = df["Close"].astype(float).values
        dates = df.index
        for i in idx:
            j = i + 1  # t+1 开盘进场
            if j >= n:
                continue
            entry = opens[j]
            if entry <= 0:
                continue
            tp_px = entry * (1 + p.tp)
            sl_px = entry * (1 - p.sl) if p.sl > 0 else -1.0
            exit_ret = None
            exit_k = min(n - 1, j + p.horizon - 1)
            for k in range(j, min(n, j + p.horizon)):
                if p.sl > 0 and lows[k] <= sl_px:          # 同日双触发按止损（保守）
                    exit_ret = -p.sl
                    exit_k = k
                    break
                if highs[k] >= tp_px:                       # 盘中触及反弹目标
                    exit_ret = p.tp
                    exit_k = k
                    break
                if p.exit_on_green and closes[k] >= entry:  # 首个收盘转正
                    exit_ret = closes[k] / entry - 1.0
                    exit_k = k
                    break
            if exit_ret is None:
                exit_ret = closes[exit_k] / entry - 1.0
            rows.append({
                "ticker": tk,
                "entry_date": dates[j],
                "exit_date": dates[exit_k],
                "ret": exit_ret - cost,
            })
    if not rows:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "ret"])
    return pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)


# --------------------------- 组合模拟 ---------------------------

def simulate(trades: pd.DataFrame, slots: int) -> dict:
    """K 槽位等权复利组合：返回 win_rate / 年化 / 最大回撤 / 交易数 等。"""
    if trades.empty:
        return {"n_trades": 0}
    taken = []
    open_pos = []  # (exit_date, alloc, ret)
    equity = 1.0
    free = slots
    all_dates = pd.DatetimeIndex(sorted(set(trades["entry_date"]) | set(trades["exit_date"])))
    by_entry = {d: g for d, g in trades.groupby("entry_date")}
    curve = []
    for d in all_dates:
        still = []
        for exit_d, alloc, ret in open_pos:
            if exit_d == d:
                equity += alloc * ret
                free += 1
            else:
                still.append((exit_d, alloc, ret))
        open_pos = still
        if d in by_entry:
            for _, tr in by_entry[d].iterrows():
                if free <= 0:
                    continue
                alloc = equity / slots
                if tr["exit_date"] == d:
                    equity += alloc * tr["ret"]
                else:
                    open_pos.append((tr["exit_date"], alloc, tr["ret"]))
                    free -= 1
                taken.append(tr["ret"])
        curve.append((d, equity))
    eq = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    taken_arr = np.array(taken) if taken else np.array([0.0])
    span_days = max((all_dates[-1] - all_dates[0]).days, 1)
    years = span_days / 365.25
    cagr = equity ** (1 / years) - 1 if years > 0 and equity > 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0).min()
    wins = taken_arr[taken_arr > 0]
    losses = taken_arr[taken_arr <= 0]
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
    return {
        "n_trades": int(len(taken)),
        "win_rate": float((taken_arr > 0).mean()),
        "avg_ret": float(taken_arr.mean()),
        "payoff": float(payoff),
        "final_equity": float(equity),
        "cagr": float(cagr),
        "max_dd": float(dd),
    }


def _fmt(m: dict) -> str:
    if not m.get("n_trades"):
        return "无交易"
    return (
        f"交易{m['n_trades']:>4} · 胜率{m['win_rate']*100:5.1f}% · "
        f"单笔均值{m['avg_ret']*100:5.2f}% · 盈亏比{m.get('payoff', float('nan')):4.2f} · "
        f"年化{m['cagr']*100:7.1f}% · 最大回撤{m['max_dd']*100:6.1f}%"
    )


# --------------------------- 今日选股扫描 ---------------------------

# 生产参数（= mean_reversion_dip_best.json 的样本内最优，OOS 稳健）：
# trend200 + RSI(2)≤15 + 跌破SMA5 + tp10%/sl8% + H10，不择时。
PROD_PARAMS = Params(
    trend_ma=200, pullback_ma=5, rsi_n=2, rsi_max=15.0,
    dvol_min_m=50.0, tp=0.10, sl=0.08, horizon=10,
    exit_on_green=True, slots=10, use_regime=False,
)


def scan_today(
    data: dict[str, pd.DataFrame],
    p: Params | None = None,
    *,
    top_n: int = 8,
) -> list[dict]:
    """扫描「今日」（每只标的最新一根 K 线）命中买跌信号的候选。

    仅用截至最新交易日的信息；返回按超卖程度（RSI 升序）排序的候选列表，
    供 daily_pick 管道转成 pick 行。"""
    p = p or PROD_PARAMS
    rows: list[dict] = []
    for tk, df in data.items():
        if df is None or len(df) < max(p.trend_ma // 2 + 5, 60):
            continue
        close = df["Close"].astype(float)
        vol = df["Volume"].astype(float)
        sma_trend = close.rolling(p.trend_ma, min_periods=p.trend_ma // 2).mean()
        sma_pull = close.rolling(p.pullback_ma, min_periods=max(2, p.pullback_ma // 2)).mean()
        rsi = _rsi(close, p.rsi_n)
        dvol_m = close * vol / 1e6

        c = float(close.iloc[-1])
        st = float(sma_trend.iloc[-1])
        sp = float(sma_pull.iloc[-1])
        rv = float(rsi.iloc[-1])
        dv = float(dvol_m.iloc[-1])
        if not all(np.isfinite(x) for x in (c, st, sp, rv, dv)):
            continue
        if c > st and c < sp and rv <= p.rsi_max and dv >= p.dvol_min_m:
            rows.append({
                "代码": tk,
                "现价": round(c, 2),
                "RSI2": round(rv, 1),
                "距SMA5%": round((c / sp - 1.0) * 100, 1),
                "距SMA200%": round((c / st - 1.0) * 100, 1),
                "成交额M": round(dv, 0),
                "日期": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
            })
    rows.sort(key=lambda r: r["RSI2"])  # 最超卖优先
    return rows[:top_n]


# --------------------------- 主流程 ---------------------------

def split_oos(trades: pd.DataFrame, ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades, trades
    cut = trades["entry_date"].quantile(ratio)
    return trades[trades["entry_date"] <= cut], trades[trades["entry_date"] > cut]


def evaluate(data, p: Params) -> dict:
    trades = gen_trades(data, p)
    full = simulate(trades, p.slots)
    is_tr, oos_tr = split_oos(trades)
    return {
        "params": asdict(p),
        "full": full,
        "is": simulate(is_tr, p.slots),
        "oos": simulate(oos_tr, p.slots),
    }


def _hit_targets(m: dict) -> str:
    if not m.get("n_trades"):
        return ""
    ann = m["cagr"] >= 1.0
    win = m["win_rate"] >= 0.90
    dd = m["max_dd"] >= -0.10
    return f"[年化>100%:{'✓' if ann else '✗'} 胜率>90%:{'✓' if win else '✗'} 回撤<10%:{'✓' if dd else '✗'}]"


def grid_search(data) -> list[dict]:
    grid = {
        "trend_ma": [100, 200],
        "rsi_max": [5.0, 10.0, 15.0],
        "tp": [0.04, 0.06, 0.10],
        "sl": [0.0, 0.05, 0.08],
        "horizon": [5, 10],
        "use_regime": [True, False],
    }
    keys = list(grid)
    out = []
    for combo in itertools.product(*grid.values()):
        p = Params(**dict(zip(keys, combo)))
        trades = gen_trades(data, p)
        is_tr, oos_tr = split_oos(trades)
        m_is = simulate(is_tr, p.slots)
        if m_is.get("n_trades", 0) < 50:
            continue
        m_oos = simulate(oos_tr, p.slots)
        out.append({"params": asdict(p), "is": m_is, "oos": m_oos})
    # 按样本内 年化/回撤（卡尔玛近似）排序
    out.sort(key=lambda r: r["is"]["cagr"] / max(abs(r["is"]["max_dd"]), 0.02), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    universe = list(dict.fromkeys(GAINER_MOMENTUM))
    print(f"宇宙 {len(universe)} 只流动性标的 · 载入 {args.years} 年日线…")
    data = fetch_prices(universe, args.years, use_cache=not args.no_cache)
    print(f"成功载入 {len(data)} 只。\n")

    print("=" * 86)
    print("默认参数（先验，不调参）：", asdict(Params()))
    for label, p in [("含大盘择时(regime)", Params(use_regime=True)),
                     ("不择时(全天候)", Params(use_regime=False))]:
        ev = evaluate(data, p)
        print(f"\n— {label} —")
        print("  全样本 ", _fmt(ev["full"]), _hit_targets(ev["full"]))
        print("  样本内 ", _fmt(ev["is"]))
        print("  样本外 ", _fmt(ev["oos"]), _hit_targets(ev["oos"]))
    print("=" * 86)

    if not args.no_grid:
        print("\n网格寻优（按样本内 年化/回撤 排序），展示前 10 + 各自样本外：\n")
        results = grid_search(data)
        for r in results[:10]:
            pr = r["params"]
            tag = (f"trend{pr['trend_ma']} rsi≤{pr['rsi_max']:.0f} "
                   f"tp{pr['tp']:.0%}/sl{pr['sl']:.0%} H{pr['horizon']} "
                   f"{'regime' if pr['use_regime'] else 'allwx'}")
            print(f"[{tag}]")
            print("   IS : ", _fmt(r["is"]))
            print("   OOS: ", _fmt(r["oos"]), _hit_targets(r["oos"]))
        if results:
            best = results[0]
            (ROOT / "research" / "mean_reversion_dip_best.json").write_text(
                json.dumps(best, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print("\n最佳（样本内）已存 research/mean_reversion_dip_best.json")


if __name__ == "__main__":
    main()
