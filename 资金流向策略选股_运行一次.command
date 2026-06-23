#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python research/flow_strategy_backtest.py --today
read -p "按回车关闭..."
