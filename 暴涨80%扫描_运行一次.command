#!/usr/bin/env bash
# 双击：暴涨80%规则每日扫描（观察池 → 追多/回避确认）
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "首次运行，正在创建虚拟环境并安装依赖…"
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi

echo "=========================================="
echo "  暴涨80%规则 · 每日扫描"
echo "=========================================="
python gain15_daily.py
echo ""
echo "完成。结果见 research/gain15_daily_today.json"
echo "（关闭此窗口即可）"
