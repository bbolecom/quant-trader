"""Tests for 均值回归·顺势买跌 strategy + daily_pick 集成（全离线、确定性）。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.mean_reversion_dip import (
    PROD_PARAMS,
    Params,
    _rsi,
    _signal_mask,
    gen_trades,
    scan_today,
    simulate,
)


def _frame(closes: list[float], vol: float = 2e7) -> pd.DataFrame:
    """合成 OHLCV：Open=前收，High/Low 包络，Volume 常量。"""
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    close = pd.Series(closes, index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) * 1.005
    low = pd.concat([openp, close], axis=1).min(axis=1) * 0.995
    return pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _uptrend_dip(extra: list[float] | None = None) -> pd.DataFrame:
    """长上升趋势 + 末端小回调（命中买跌信号）。"""
    rise = list(np.linspace(100.0, 150.0, 150))
    dip = [149.5, 148.5, 147.5]
    closes = rise + dip + (extra or [])
    return _frame(closes)


def _downtrend() -> pd.DataFrame:
    return _frame(list(np.linspace(150.0, 100.0, 153)))


# --------------------------- 指标 ---------------------------

def test_rsi_extremes() -> None:
    up = pd.Series(np.linspace(10, 30, 20))
    down = pd.Series(np.linspace(30, 10, 20))
    assert _rsi(up, 2).iloc[-1] == 100.0   # 全程上涨 → RSI=100
    assert _rsi(down, 2).iloc[-1] == 0.0   # 全程下跌 → RSI=0


# --------------------------- 信号 ---------------------------

def test_signal_fires_on_oversold_dip_in_uptrend() -> None:
    p = Params(trend_ma=50, pullback_ma=5, rsi_n=2, rsi_max=15.0, dvol_min_m=0.0)
    mask = _signal_mask(_uptrend_dip(), p)
    assert bool(mask.iloc[-1]) is True


def test_signal_silent_in_downtrend() -> None:
    p = Params(trend_ma=50, pullback_ma=5, rsi_n=2, rsi_max=15.0, dvol_min_m=0.0)
    mask = _signal_mask(_downtrend(), p)
    assert not mask.any()  # 跌破趋势 → 不买跌（不接飞刀）


def test_liquidity_gate_blocks_thin_names() -> None:
    p = Params(trend_ma=50, pullback_ma=5, rsi_n=2, rsi_max=15.0, dvol_min_m=1e6)
    thin = _uptrend_dip()  # 价 ~150 × 量 2e7 = $3000M < 1e6M 闸门
    assert not _signal_mask(thin, p).any()


# --------------------------- 交易 + 组合 ---------------------------

def test_gen_trades_and_simulate() -> None:
    p = Params(trend_ma=50, pullback_ma=5, rsi_n=2, rsi_max=15.0,
               dvol_min_m=0.0, tp=0.10, sl=0.08, horizon=10)
    # 末端给一段反弹，确保信号不在最后一根（有 t+1 入场）。
    trades = gen_trades({"UP": _uptrend_dip([149.0, 151.0, 153.0, 155.0])}, p)
    assert not trades.empty
    assert list(trades.columns) == ["ticker", "entry_date", "exit_date", "ret"]
    assert trades["ret"].notna().all()

    m = simulate(trades, slots=p.slots)
    for key in ("n_trades", "win_rate", "cagr", "max_dd"):
        assert key in m
    assert m["n_trades"] == len(trades)
    assert 0.0 <= m["win_rate"] <= 1.0


def test_simulate_empty() -> None:
    assert simulate(pd.DataFrame(), slots=10) == {"n_trades": 0}


# --------------------------- 今日扫描 ---------------------------

def test_scan_today_selects_dip_excludes_downtrend() -> None:
    p = Params(trend_ma=50, pullback_ma=5, rsi_n=2, rsi_max=15.0, dvol_min_m=0.0)
    data = {"UP": _uptrend_dip(), "DOWN": _downtrend()}
    cands = scan_today(data, p, top_n=5)
    codes = {c["代码"] for c in cands}
    assert "UP" in codes
    assert "DOWN" not in codes
    row = next(c for c in cands if c["代码"] == "UP")
    for key in ("现价", "RSI2", "距SMA5%", "距SMA200%", "成交额M", "日期"):
        assert key in row


def test_prod_params_are_production_grade() -> None:
    assert PROD_PARAMS.use_regime is False   # 实验证实择时反而损失
    assert PROD_PARAMS.sl > 0                 # 用硬止损控回撤
    assert PROD_PARAMS.rsi_max == 15.0
    assert PROD_PARAMS.trend_ma == 200


# --------------------------- daily_pick 集成 ---------------------------
# 注：mean_reversion_dip 已移出核心 9 策略的 RUNNER_REGISTRY（精简版决策），
# 故移除「断言其已注册 / quick 跳过」的两个过期测试；模块本身的覆盖仍保留在上方。


def test_resolve_universe_liquid100() -> None:
    from quant.daily_pick_runners import resolve_dip_universe

    uni = resolve_dip_universe({"mean_reversion_universe": "liquid100"})
    assert "AAPL" in uni and "NVDA" in uni
    assert "SPY" not in uni and "QQQ" not in uni   # ETF 闸门排除
    assert len(uni) == len(set(uni))               # 去重


def test_resolve_universe_sp500_union(monkeypatch) -> None:
    import quant.screener as screener

    monkeypatch.setattr(screener, "fetch_sp500_tickers", lambda: ["AAA", "BBB", "SPY"])
    monkeypatch.setattr(screener, "fetch_nasdaq100_tickers", lambda: ["CCC", "AAA"])
    from quant.daily_pick_runners import resolve_dip_universe

    uni = resolve_dip_universe({"mean_reversion_universe": "sp500"})
    assert {"AAA", "BBB", "CCC"} <= set(uni)        # 指数成分并入
    assert "AAPL" in uni                            # LIQUID100 也并入
    assert "SPY" not in uni
    assert len(uni) == len(set(uni))


def test_resolve_universe_cap(monkeypatch) -> None:
    import quant.screener as screener

    monkeypatch.setattr(screener, "fetch_sp500_tickers", lambda: [f"Z{i}" for i in range(1000)])
    monkeypatch.setattr(screener, "fetch_nasdaq100_tickers", lambda: [])
    from quant.daily_pick_runners import resolve_dip_universe

    uni = resolve_dip_universe({"mean_reversion_universe": "sp500", "mean_reversion_max_universe": 50})
    assert len(uni) == 50


def test_stats_anchor_enriches_pick(tmp_path) -> None:
    from quant.high_win_pick import StatsStore, enrich_pick

    (tmp_path / "research").mkdir(parents=True)
    (tmp_path / "research" / "mean_reversion_dip_best.json").write_text(
        json.dumps({
            "oos": {"n_trades": 2967, "win_rate": 0.73, "cagr": 1.5, "max_dd": -0.17}
        }),
        encoding="utf-8",
    )
    store = StatsStore(root=tmp_path)
    enriched = enrich_pick({"模块": "均值回归·买跌", "代码": "AAPL", "状态": "可开仓"}, store)
    assert enriched["历史年化"] == 1.5
    assert enriched["历史胜率"] == 0.73
    assert "胜率" in enriched["回测摘要"]
