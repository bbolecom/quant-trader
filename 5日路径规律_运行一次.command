#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -q
fi
echo "=========================================="
echo "  5日路径规律 · 真实OHLCV + 换手率"
echo "=========================================="
.venv/bin/python research/move_pattern_5d_mine.py "$@"
echo ""
read -r -p "按回车关闭…" _ 2>/dev/null || true
