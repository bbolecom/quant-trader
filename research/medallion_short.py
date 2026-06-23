"""Medallion 风味·多因子横截面短线策略（回测 + 当日信号）。

思路（借鉴 Medallion 的"系统化 + 多个小赌注 + 统计优势"，但只用日线、含成本）：

  只在**高流动性票池**里玩（日均成交额 ≥ 阈值，过滤掉低成交量的坑），
  每天对每只票用四个维度做横截面 z-score 打分：

    成交额 dvol   —— 流动性（过滤 + 轻微偏好，越大越安全）
    振幅   amp    —— 近 N 日平均日内振幅（机会/波动）
    涨幅   gain   —— 近 lb 日收益（动量=+ / 反转=−，可配）
    换手率 turn   —— 近 N 日平均换手（vol / 流通股，活跃度）

  综合分排序，做多 Top-K（可叠加做空 Bottom-K 做市场中性），持有 H 日，
  每天按持仓变动收**交易成本**。多个小赌注（K 只）+ 系统化轮动 = 大数定律。

诚实预期：净成本后，这类策略目标年化 15~30%、夏普更高、回撤可控；
不会是 Medallion 的 39%（我们没有它的零成本/高杠杆/独家数据）。

用法：
    python research/medallion_short.py                 # 默认对比多模式
    python research/medallion_short.py --years 5
    python research/medallion_short.py --mode reversal --long-short
    python research/medallion_short.py --signal-only    # 只出当日信号
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.providers import DataConfig, get_provider, reset_provider_cache

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from research.gainer_daily_backtest import GAINER_MOMENTUM, LIQUID100

SHARES_CACHE = ROOT / "research" / "shares_cache.json"
TRADING_DAYS = 252


@dataclass
class FactorConfig:
    """四因子权重与方向。gain_sign: +1 动量 / −1 反转。"""

    w_dvol: float = 0.10
    w_amp: float = 0.25
    w_turn: float = 0.25
    w_gain: float = 0.40
    gain_sign: float = -1.0   # 默认反转（净成本后短线更稳）
    gain_lb: int = 3          # 涨幅回看天数
    amp_win: int = 10         # 振幅平均窗口
    turn_win: int = 10        # 换手平均窗口


@dataclass
class BacktestConfig:
    top_k: int = 10
    hold: int = 1
    side: str = "long"         # long / short / ls(多空中性)
    short_target: str = "low"  # low=空综合分最低 / high=空综合分最高(空过热)
    regime_gate: bool = False  # 仅在 SPY 弱市(<MA)才持有空头，否则空仓
    regime_ma: int = 200
    max_mcap_b: float = 10000.0  # 市值上限（十亿美元）——做空只选小盘
    require_mcap: bool = False    # 做空小盘时，市值未知的票直接排除
    min_dvol_m: float = 50.0   # 高流动性过滤（日均成交额，百万美元）
    min_price: float = 5.0
    fee_bps: float = 5.0       # 单边佣金/费用
    slip_bps: float = 5.0      # 单边滑点
    borrow_bps_annual: float = 3.0  # 做空年化借券费（按持有天数摊）
    factors: FactorConfig = field(default_factory=FactorConfig)

    @property
    def long_short(self) -> bool:
        return self.side == "ls"


# ---------- 数据 ----------

def load_universe() -> list[str]:
    """高流动性、非坑的票池：动量扩展池 + 缓存，去重。"""
    uni = set(GAINER_MOMENTUM) | set(LIQUID100)
    cache = ROOT / "research" / "gainer_universe_cache.json"
    if cache.exists():
        try:
            uni |= set(json.loads(cache.read_text()))
        except Exception:  # noqa: BLE001
            pass
    drop = {"SPY", "QQQ", "XLE", "XLF"}  # 指数/板块 ETF 不参与选股
    return sorted(t for t in uni if t and t not in drop)


def fetch_shares(tickers: list[str]) -> dict[str, float]:
    """抓取流通股数（含本地缓存）。失败的留空，换手因子按截面中位数补。"""
    cache: dict[str, float] = {}
    if SHARES_CACHE.exists():
        try:
            cache = json.loads(SHARES_CACHE.read_text())
        except Exception:  # noqa: BLE001
            cache = {}
    missing = [t for t in tickers if t not in cache]
    if missing and yf is not None:
        for t in missing:
            sh = np.nan
            try:
                fi = yf.Ticker(t).fast_info
                sh = float(fi.get("shares") or fi.get("sharesOutstanding") or np.nan)
            except Exception:  # noqa: BLE001
                sh = np.nan
            cache[t] = sh if np.isfinite(sh) and sh > 0 else None
        try:
            SHARES_CACHE.write_text(json.dumps(cache))
        except Exception:  # noqa: BLE001
            pass
    return {t: float(v) for t, v in cache.items() if v}


def _panel(field_name: str, batch: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({t: df[field_name].astype(float) for t, df in batch.items()}).sort_index()


# ---------- 因子 ----------

def _zscore_rows(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """逐日（行）截面 z-score，仅在 mask=True 的票上计算。"""
    x = df.where(mask)
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    z = x.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z.clip(-3, 3)


def build_factors(batch: dict[str, pd.DataFrame], shares: dict[str, float],
                  cfg: BacktestConfig) -> dict[str, pd.DataFrame]:
    closes = _panel("Close", batch)
    highs = _panel("High", batch)
    lows = _panel("Low", batch)
    vols = _panel("Volume", batch)

    dvol = (closes * vols).rolling(20).mean()
    amp = ((highs - lows) / closes.shift(1)).rolling(cfg.factors.amp_win).mean()
    gain = closes.pct_change(cfg.factors.gain_lb, fill_method=None)
    sh = pd.Series({t: shares.get(t, np.nan) for t in closes.columns})
    turn = vols.div(sh, axis=1).rolling(cfg.factors.turn_win).mean() * 100.0
    mcap = closes.mul(sh, axis=1)  # 市值≈收盘价×流通股（流通股未知则 NaN）

    fwd = closes.pct_change(cfg.hold, fill_method=None).shift(-cfg.hold) / cfg.hold  # 日均前向收益（持有期摊平）
    return {"closes": closes, "dvol": dvol, "amp": amp, "gain": gain, "turn": turn,
            "mcap": mcap, "fwd": fwd}


def composite_score(f: dict[str, pd.DataFrame], cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = cfg.factors
    closes = f["closes"]
    eligible = (f["dvol"] >= cfg.min_dvol_m * 1e6) & (closes >= cfg.min_price)
    eligible &= f["amp"].notna() & f["gain"].notna()
    if cfg.max_mcap_b < 10000.0:  # 小盘过滤
        cap = cfg.max_mcap_b * 1e9
        if cfg.require_mcap:
            eligible &= f["mcap"].notna() & (f["mcap"] <= cap)
        else:
            eligible &= (f["mcap"].isna() | (f["mcap"] <= cap))

    z_dvol = _zscore_rows(np.log(f["dvol"].clip(lower=1)), eligible)
    z_amp = _zscore_rows(f["amp"], eligible)
    z_gain = _zscore_rows(f["gain"], eligible)
    z_turn = _zscore_rows(f["turn"], eligible)
    # 换手缺失 → 用 0（截面中性），不污染打分
    z_turn = z_turn.where(eligible).fillna(0.0)

    score = (cfg.factors.w_dvol * z_dvol
             + cfg.factors.w_amp * z_amp
             + cfg.factors.w_turn * z_turn
             + cfg.factors.w_gain * fc.gain_sign * z_gain)
    score = score.where(eligible)
    return score, eligible


# ---------- 回测 ----------

def backtest(f: dict[str, pd.DataFrame], score: pd.DataFrame, cfg: BacktestConfig,
             regime: pd.Series | None = None) -> dict:
    fwd = f["fwd"]
    dates = score.index
    cost_oneway = (cfg.fee_bps + cfg.slip_bps) / 1e4
    borrow_daily = cfg.borrow_bps_annual / 1e4 / TRADING_DAYS * cfg.hold
    reg = regime.reindex(dates).ffill() if regime is not None else None
    use_long = cfg.side in ("long", "ls")
    use_short = cfg.side in ("short", "ls")
    rets: list[float] = []
    idx: list[pd.Timestamp] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for i in range(len(dates) - cfg.hold):
        row = score.iloc[i].dropna()
        need = cfg.top_k * (2 if cfg.side == "ls" else 1)
        if len(row) < need:
            continue
        ranked = row.sort_values(ascending=False)
        fr = fwd.iloc[i]

        # 大盘择时：仅在弱市做空（regime=True 表示弱市，可做空）
        if cfg.regime_gate and use_short:
            short_ok = reg is not None and bool(reg.iloc[i])
        else:
            short_ok = use_short

        longs = set(ranked.head(cfg.top_k).index) if use_long else set()
        if use_short and short_ok:
            shorts = set(ranked.tail(cfg.top_k).index) if cfg.short_target == "low" \
                else set(ranked.head(cfg.top_k).index)
        else:
            shorts = set()

        long_ret = float(fr[list(longs)].mean()) if longs else 0.0
        short_ret = float(fr[list(shorts)].mean()) if shorts else 0.0
        if (longs and not np.isfinite(long_ret)) or (shorts and not np.isfinite(short_ret)):
            continue

        if cfg.side == "ls":
            gross = 0.5 * long_ret - 0.5 * short_ret - (borrow_daily * 0.5 if shorts else 0)
        elif cfg.side == "short":
            gross = -short_ret - (borrow_daily if shorts else 0)
        else:
            gross = long_ret

        turn_units = (len(longs ^ prev_long) + len(shorts ^ prev_short))
        denom = 2 * cfg.top_k * (2 if cfg.side == "ls" else 1)
        turn_frac = turn_units / denom if denom else 0.0
        prev_long, prev_short = longs, shorts
        net = gross - turn_frac * cost_oneway
        rets.append(net)
        idx.append(dates[i + cfg.hold])

    if len(rets) < 30:
        return {}
    r = pd.Series(rets, index=pd.DatetimeIndex(idx))
    # 同日多次（重叠持有）取均值
    r = r.groupby(r.index).mean()
    eq = (1 + r).cumprod()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    total = float(eq.iloc[-1] - 1)
    cagr = (1 + total) ** (1 / years) - 1
    dd = float((eq / eq.cummax() - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    win = float((r > 0).mean())
    yearly = (r.groupby(r.index.year).apply(lambda s: (1 + s).prod() - 1) * 100).round(1)
    return {"equity": eq, "rets": r, "total": total, "cagr": cagr, "maxdd": dd,
            "sharpe": sharpe, "win": win, "yearly": yearly.to_dict(), "n": len(r)}


def spy_benchmark(provider, start: str, end: str) -> dict:
    df = provider.fetch_history("SPY", start, end)
    c = df["Close"].astype(float)
    r = c.pct_change(fill_method=None).dropna()
    eq = (1 + r).cumprod()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    total = float(eq.iloc[-1] - 1)
    return {"cagr": (1 + total) ** (1 / years) - 1,
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))}


# ---------- 模式 ----------

def preset_configs(base: BacktestConfig) -> dict[str, BacktestConfig]:
    import copy

    def mk(**kw) -> BacktestConfig:
        c = copy.deepcopy(base)
        fc_keys = {"gain_sign", "gain_lb", "amp_win", "turn_win",
                   "w_dvol", "w_amp", "w_turn", "w_gain"}
        for k, v in kw.items():
            if k in fc_keys:
                setattr(c.factors, k, v)
            else:
                setattr(c, k, v)
        return c

    # 做空"过热"：综合分偏重涨幅+振幅，short_target=high → 空近期猛涨、波动大的小盘
    def short_overheat(**kw):
        return mk(gain_sign=+1.0, side="short", short_target="high",
                  w_dvol=0.0, w_amp=0.4, w_turn=0.2, w_gain=0.4, gain_lb=5, **kw)

    return {
        "空·猛涨小盘≤5B(全程)": short_overheat(max_mcap_b=5.0, require_mcap=True),
        "空·猛涨小盘≤5B(弱市择时)": short_overheat(max_mcap_b=5.0, require_mcap=True, regime_gate=True),
        "空·猛涨微盘≤2B(全程)": short_overheat(max_mcap_b=2.0, require_mcap=True),
        "空·猛涨微盘≤2B(弱市择时)": short_overheat(max_mcap_b=2.0, require_mcap=True, regime_gate=True),
        "空·猛涨不限市值(对照)": short_overheat(max_mcap_b=10000.0),
        "做多·猛涨小盘≤5B(反向对照)": mk(gain_sign=+1.0, side="long", w_amp=0.4, w_turn=0.2,
                                  w_gain=0.4, w_dvol=0.0, gain_lb=5, max_mcap_b=5.0, require_mcap=True),
    }


def today_signal(f: dict[str, pd.DataFrame], score: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    row = score.iloc[-1].dropna().sort_values(ascending=False)
    if row.empty:
        return pd.DataFrame()
    # 做空看综合分最高的（过热）；做多看最高的
    picks = row.head(cfg.top_k).index
    out = []
    for t in picks:
        mc = float(f["mcap"][t].iloc[-1]) if t in f["mcap"] else np.nan
        out.append({
            "代码": t,
            "现价": round(float(f["closes"][t].iloc[-1]), 2),
            f"涨幅{cfg.factors.gain_lb}d%": round(float(f["gain"][t].iloc[-1]) * 100, 1),
            "振幅%": round(float(f["amp"][t].iloc[-1]) * 100, 1),
            "换手%": round(float(f["turn"][t].iloc[-1]), 2) if np.isfinite(f["turn"][t].iloc[-1]) else None,
            "市值B": round(mc / 1e9, 2) if np.isfinite(mc) else None,
            "综合分": round(float(row[t]), 2),
        })
    return pd.DataFrame(out)


def run(years: int, base: BacktestConfig, mode: str | None, signal_only: bool) -> None:
    reset_provider_cache()
    y = get_provider(DataConfig(provider="yahoo"))
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=int(365.25 * years) + 60)).isoformat()

    uni = load_universe()
    print(f"票池 {len(uni)} 只（高流动性，过滤坑票）｜抓取行情 {start} ~ {end} …")
    batch = y.fetch_batch(uni, start, end)
    batch = {t: d for t, d in batch.items() if d is not None and len(d) > 60}
    print(f"有效 {len(batch)} 只｜抓取流通股（算换手率）…")
    shares = fetch_shares(list(batch.keys()))
    print(f"拿到流通股 {len(shares)} 只\n")

    bench = None
    regime = None
    try:
        spy_df = y.fetch_history("SPY", start, end)
        spy_c = spy_df["Close"].astype(float)
        regime = spy_c < spy_c.rolling(base.regime_ma).mean()  # True=弱市
        r = spy_c.pct_change(fill_method=None).dropna()
        eq = (1 + r).cumprod()
        yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
        tot = float(eq.iloc[-1] - 1)
        bench = {"cagr": (1 + tot) ** (1 / yrs) - 1,
                 "maxdd": float((eq / eq.cummax() - 1).min()),
                 "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))}
    except Exception:  # noqa: BLE001
        pass

    if signal_only:
        cfg = base
        f = build_factors(batch, shares, cfg)
        score, _ = composite_score(f, cfg)
        sig = today_signal(f, score, cfg)
        _print_signal(sig, cfg)
        return

    configs = preset_configs(base) if mode is None else {mode: base}
    rows = []
    best_name, best = None, None
    for name, cfg in configs.items():
        f = build_factors(batch, shares, cfg)
        score, _ = composite_score(f, cfg)
        res = backtest(f, score, cfg, regime=regime)
        if not res:
            continue
        rows.append({"模式": name, "年化%": round(res["cagr"] * 100, 1),
                     "最大回撤%": round(res["maxdd"] * 100, 1),
                     "夏普": round(res["sharpe"], 2),
                     "日胜率%": round(res["win"] * 100, 1),
                     "累计%": round(res["total"] * 100, 0)})
        if best is None or res["sharpe"] > best["sharpe"]:
            best, best_name, best_cfg, best_f, best_score = res, name, cfg, f, score

    print("=" * 78)
    print(f"Medallion 风味·多因子短线｜{years}年回测（含双边成本 "
          f"{base.fee_bps + base.slip_bps:.0f}bp，Top{base.top_k}，持有{base.hold}日）")
    print("=" * 78)
    df = pd.DataFrame(rows).sort_values("夏普", ascending=False)
    print(df.to_string(index=False))
    if bench:
        print(f"\n[基准] SPY 买入持有：年化 {bench['cagr']*100:.1f}%，"
              f"最大回撤 {bench['maxdd']*100:.1f}%，夏普 {bench['sharpe']:.2f}")

    if best:
        print(f"\n>>> 最佳（按夏普）：{best_name}")
        yr = best["yearly"]
        print("逐年收益%：" + "  ".join(f"{k}:{v:+.0f}" for k, v in yr.items()))
        print("\n【今日信号 · " + best_name + "】")
        _print_signal(today_signal(best_f, best_score, best_cfg), best_cfg)
    print("\n注：高流动性票池、含交易成本、无未来函数（收盘排序→次日持有）。")
    print("    胜率/年化为回测值，实盘受滑点、停牌、财报跳空影响，仓位务必分散。")


def _print_signal(sig: pd.DataFrame, cfg: BacktestConfig) -> None:
    side = "多空（多Top/空Bottom）" if cfg.long_short else "做多 Top"
    if sig.empty:
        print(f"  今日无满足流动性条件的标的（{side}）。")
        return
    print(f"  {side}{cfg.top_k}（综合分降序）：")
    print(sig.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Medallion 风味·多因子横截面短线")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--hold", type=int, default=1)
    p.add_argument("--min-dvol-m", type=float, default=50.0)
    p.add_argument("--mode", choices=["reversal", "momentum"], default=None)
    p.add_argument("--long-short", action="store_true")
    p.add_argument("--signal-only", action="store_true")
    args = p.parse_args()

    fc = FactorConfig()
    if args.mode == "reversal":
        fc.gain_sign = -1.0
    elif args.mode == "momentum":
        fc.gain_sign = +1.0
    side = "ls" if args.long_short else "long"
    base = BacktestConfig(top_k=args.top_k, hold=args.hold, side=side,
                          min_dvol_m=args.min_dvol_m, factors=fc)
    mode_name = None
    if args.mode:
        mode_name = ("反转" if args.mode == "reversal" else "动量") + ("·多空" if args.long_short else "·多头")
    run(args.years, base, mode_name, args.signal_only)


if __name__ == "__main__":
    main()
