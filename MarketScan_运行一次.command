#!/usr/bin/env bash
# 双击：全市场 5 分钟快扫
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt -q
else
  . .venv/bin/activate
fi

echo "=========================================="
echo "  全市场快扫 · 目标 5 分钟内完成"
echo "=========================================="
python3 market_scan_fast.py
echo ""
echo "完成 → research/market_scan_today.json"
