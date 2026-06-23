#!/bin/bash
cd "$(dirname "$0")"
echo "S8U Ultra80 · 高流通票全市场回测（≥\$50M/日）…"
python3 research/s8u_liquid_universe_backtest.py --min-dvol-m 50
echo ""
read -p "按回车关闭…"
