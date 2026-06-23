#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -q
fi
echo "=========================================="
echo "  三腿策略：做多 + 回避 + 收租"
echo "=========================================="
.venv/bin/python pattern_daily.py "$@"
echo ""
echo "完成。历史 → pattern_daily_history.csv"
read -r -p "按回车关闭…" _ 2>/dev/null || true
