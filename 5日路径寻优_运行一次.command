#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -q
fi
echo "=========================================="
echo "  5日路径参数寻优 → 提高命中率"
echo "  输出 → research/move_pattern_5d_optimized.json"
echo "=========================================="
.venv/bin/python research/move_pattern_5d_param_search.py "$@"
echo ""
.venv/bin/python research/move_pattern_5d_mine.py --from-cache
echo ""
read -r -p "按回车关闭…" _ 2>/dev/null || true
