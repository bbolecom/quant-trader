"""样本外验证、赚钱概率、市场状态与一键体检测试。"""

from __future__ import annotations

from quant import probability, regime, report, validation


def test_holdout_validate(ohlcv):
    grid = {"fast": [5, 10, 20], "slow": [40, 60, 90]}
    res = validation.holdout_validate(ohlcv, "双均线交叉", grid, train_ratio=0.7)
    assert set(res.best_params) == {"fast", "slow"}
    assert "夏普比率" in res.is_stats and "夏普比率" in res.oos_stats
    assert res.split_date is not None


def test_walk_forward(ohlcv):
    grid = {"fast": [5, 10, 20], "slow": [40, 60, 90]}
    res = validation.walk_forward(ohlcv, "双均线交叉", grid, n_splits=3, train_ratio=0.5)
    assert not res.windows.empty
    assert len(res.oos_equity) > 0


def test_rolling_positive_prob_bounded(ohlcv):
    from quant import backtest, strategies

    pos = strategies.get_strategy("动量策略").generate(ohlcv)
    res = backtest.run_backtest(ohlcv, pos)
    p, n = probability.rolling_positive_prob(res.returns, 63)
    assert 0.0 <= p <= 1.0
    assert n > 0


def test_analyze_single_keys(ohlcv):
    a = probability.analyze_single(ohlcv, "双均线交叉")
    for key in ["win_rate", "payoff", "horizons", "total_return", "sharpe"]:
        assert key in a
    assert 0.0 <= a["win_rate"] <= 1.0


def test_analyze_basket(multi_data):
    summ, table = probability.analyze_basket(multi_data, "动量策略")
    assert 0.0 <= summ["盈利概率"] <= 1.0
    assert len(table) == len(multi_data)


def test_detect_regime_labels(trending_ohlcv):
    reg = regime.detect_regime(trending_ohlcv)
    assert reg.trend_label in {"趋势市", "震荡市", "过渡"}
    assert reg.direction in {"上行", "下行", "中性"}
    assert reg.vol_label in {"低波动", "中等波动", "高波动"}


def test_trending_data_detected_as_trend(trending_ohlcv):
    reg = regime.detect_regime(trending_ohlcv)
    assert reg.trend_label == "趋势市"
    assert reg.direction == "上行"


def test_recommend_returns_ranking(ohlcv):
    reg, table = regime.recommend(ohlcv)
    assert "契合度" in table.columns
    assert len(table) >= 11


def test_full_report(ohlcv):
    rep = report.run_full_report(ohlcv, ticker="TEST")
    assert 0 <= rep.score <= 100
    assert rep.grade in {"优秀", "中等", "偏弱", "不建议"}
    assert rep.strategy
    assert rep.verdict
