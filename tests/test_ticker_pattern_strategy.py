"""quant/ticker_pattern_strategy 单元测试（无网络）。"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.ticker_pattern_strategy import (
    GovernanceRule,
    MemeLongConfig,
    detect_avoid_tags,
    detect_long_signal,
    governance_blocked,
    long_momentum,
    long_signal_highwin,
    long_signal_ultra,
    long_trend,
    parse_meme_long,
    short_shrink_top,
    trade_return_bracket,
)


def _row(**kw) -> dict:
    base = {
        "vol_ratio": 2.0,
        "ret_5d": 0.08,
        "ret_20d": 0.35,
        "above_ma50": True,
        "换手率%": 3.0,
        "close_strength": 0.6,
        "ret_1d": 0.02,
        "代码": "SMCI",
    }
    base.update(kw)
    return base


def test_long_momentum_smci():
    assert long_momentum(_row(), spy_bull=True)


def test_long_trend():
    assert long_trend(_row(ret_20d=0.32), spy_bull=True)
    assert not long_trend(_row(ret_20d=0.32), spy_bull=False)


def test_detect_long_ultra_default():
    """high_win=True 默认走 Ultra80。"""
    row = _row(ret_5d=0.06, vol_ratio=2.0, ret_20d=0.12)
    sid, name = detect_long_signal(row, spy_bull=True, high_win=True)
    assert sid == "U1"
    assert "精英" in name


def test_detect_long_s7_legacy():
    sid, name = detect_long_signal(_row(ret_5d=0.02, vol_ratio=1.0), spy_bull=True, high_win=False)
    assert sid == "S7"
    assert name == "趋势中继"


def test_shrink_top_avoid():
    top = _row(ret_5d=0.18, vol_ratio=0.8)
    assert short_shrink_top(top)
    assert "S3缩量顶" in detect_avoid_tags(top)


def test_smci_governance_drop():
    rule = GovernanceRule(block_if_5d_drop_pct=15.0)
    blocked, reason = governance_blocked(
        "SMCI", _row(ret_5d=-0.18), rule, as_of=date(2026, 6, 23),
    )
    assert blocked
    assert "事件冲击" in reason


def test_highwin_smci_trend():
    sid, name = long_signal_highwin(_row(ret_20d=0.32, ret_5d=0.06, vol_ratio=1.8), spy_bull=True)
    assert sid == "S7"


def test_ultra_blocks_avoid():
    sid, _ = long_signal_ultra(_row(ret_5d=0.18, vol_ratio=0.8), spy_bull=True)
    assert sid is None


def test_ultra_mstr_momentum():
    row = _row(代码="MSTR", vol_ratio=2.5, ret_5d=0.06, ret_20d=0.12)
    row["换手率%"] = 3.5
    sid, _ = long_signal_ultra(row, spy_bull=True)
    assert sid == "U1"


def test_ultra_coin_requires_high_cs():
    row = _row(代码="COIN", vol_ratio=2.6, ret_5d=0.06, close_strength=0.55)
    assert long_signal_ultra(row, spy_bull=True)[0] is None
    row["close_strength"] = 0.62
    assert long_signal_ultra(row, spy_bull=True)[0] == "U1"


def test_ultra_generic_liquid():
    row = _row(代码="NVDA", vol_ratio=2.2, ret_5d=0.06, close_strength=0.58, ret_20d=0.10)
    row["换手率%"] = 1.5
    sid, name = long_signal_ultra(row, spy_bull=True)
    assert sid == "U1"
    assert "通用" in name


def test_ultra_generic_blocks_weak():
    row = _row(代码="AAPL", vol_ratio=1.2, ret_5d=0.06, close_strength=0.58)
    assert long_signal_ultra(row, spy_bull=True)[0] is None


def test_parse_oos_approved_config(tmp_path=None):
    from quant.ticker_pattern_strategy import resolve_meme_long_tickers, APPROVED_JSON
    assert APPROVED_JSON.exists()
    tickers = resolve_meme_long_tickers({
        "ticker_source": "oos_approved",
        "approved_file": "research/s8u_approved_tickers.json",
    })
    assert "PLTR" in tickers
    assert "SMCI" in tickers
    assert len(tickers) >= 11


def test_parse_ultra_config():
    cfg = parse_meme_long({
        "meme_long": {
            "high_win_mode": True,
            "exit_mode": "path_tp",
            "path_take_profit_pct": 0.02,
        },
    })
    assert cfg.high_win_mode is True
    assert cfg.exit_mode == "path_tp"
    assert cfg.path_take_profit_pct == 0.02


def test_path_trade_bracket_tp():
    close = pd.Series([100.0, 101.0, 102.5, 103.0])
    high = pd.Series([100.0, 101.5, 102.5, 103.0])
    low = pd.Series([100.0, 99.5, 101.0, 102.0])
    ret, held = trade_return_bracket(close, high, low, 0, 3, take_profit=0.02, stop_loss=0.05)
    assert held == 2
    assert ret is not None and ret > 0.015
