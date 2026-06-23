#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -q
fi
echo "=========================================="
echo "  特斯拉 TSLA 涨跌规律（真实量价）"
echo "=========================================="
.venv/bin/python research/ticker_pattern_mine.py --ticker TSLA "$@"
echo ""
read -r -p "按回车关闭…" _ 2>/dev/null || true
