"""回测引擎与绩效指标测试，重点验证无未来函数与成本影响。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest, metrics as M


def _const_position(df, value=1.0):
    return pd.Series(value, index=df.index)


def test_equity_starts_near_one(ohlcv):
    res = backtest.run_backtest(ohlcv, _const_position(ohlcv), fee_bps=0, slippage_bps=0)
    # 第一天持仓被顺延，等于买入持有；净值首值约为 1。
    assert abs(res.equity.iloc[0] - 1.0) < 0.05


def test_no_lookahead_signal_shifted(ohlcv):
    """只在最后一天给出多头信号，则该信号不应对历史收益产生任何作用。"""
    pos = pd.Series(0.0, index=ohlcv.index)
    pos.iloc[-1] = 1.0
    res = backtest.run_backtest(ohlcv, pos, fee_bps=0, slippage_bps=0)
    # 持仓顺延一天 -> 实际持仓全为 0，策略收益恒为 0。
    assert np.allclose(res.returns.to_numpy(), 0.0)


def test_costs_reduce_returns(ohlcv):
    """频繁换仓时，成本越高累计收益越低。"""
    rng = np.random.default_rng(1)
    pos = pd.Series(rng.integers(0, 2, len(ohlcv)).astype(float), index=ohlcv.index)
    free = backtest.run_backtest(ohlcv, pos, fee_bps=0, slippage_bps=0)
    costly = backtest.run_backtest(ohlcv, pos, fee_bps=20, slippage_bps=10)
    assert costly.stats["累计收益率"] < free.stats["累计收益率"]


def test_buy_and_hold_matches_benchmark(ohlcv):
    res = backtest.run_backtest(ohlcv, _const_position(ohlcv), fee_bps=0, slippage_bps=0)
    # 满仓持有的策略净值应与基准高度一致（仅首日顺延略有差异）。
    assert abs(res.equity.iloc[-1] - res.benchmark.iloc[-1]) < 0.02 * res.benchmark.iloc[-1]


def test_max_drawdown_non_positive(ohlcv):
    res = backtest.run_backtest(ohlcv, _const_position(ohlcv))
    assert res.stats["最大回撤"] <= 0


def test_metrics_total_return():
    eq = pd.Series([1.0, 1.1, 1.21], index=pd.date_range("2020-01-01", periods=3))
    assert np.isclose(M.total_return(eq), 0.21)


def test_metrics_drawdown():
    eq = pd.Series([1.0, 1.2, 0.6, 0.9], index=pd.date_range("2020-01-01", periods=4))
    assert np.isclose(M.max_drawdown(eq), 0.6 / 1.2 - 1.0)


def test_trades_extracted(ohlcv):
    pos = pd.Series(0.0, index=ohlcv.index)
    pos.iloc[50:100] = 1.0
    pos.iloc[200:260] = 1.0
    res = backtest.run_backtest(ohlcv, pos)
    assert len(res.trades) >= 2
    assert {"开仓日期", "平仓日期", "方向", "收益率"}.issubset(res.trades.columns)
