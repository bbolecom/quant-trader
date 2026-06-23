#!/bin/bash
cd "$(dirname "$0")"
python3 research/ticker_pattern_backtest.py --ticker MSTR SMCI COIN "$@"
