#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -q
fi
echo "=========================================="
echo "  涨跌规律参数寻优（真实量价）"
echo "  输出 → research/pattern_rules_optimized.json"
echo "=========================================="
.venv/bin/python research/pattern_param_search.py "$@"
echo ""
echo "完成。三腿策略将自动读取寻优结果。"
read -r -p "按回车关闭…" _ 2>/dev/null || true
