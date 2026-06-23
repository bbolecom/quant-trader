#!/usr/bin/env bash
# 双击：立即运行一次「双日历价差择时」扫描并弹通知
cd "$(dirname "$0")"
VENV_PY="$(pwd)/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "未找到虚拟环境，正在创建…"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
fi
"$VENV_PY" calendar_daily.py
echo ""
echo "（关闭此窗口即可）"
read -r -p "按回车关闭…" _ 2>/dev/null || sleep 3
