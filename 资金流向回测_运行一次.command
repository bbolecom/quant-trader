#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/python research/flow_pattern_backtest.py --years 3
read -p "按回车关闭..."
