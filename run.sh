#!/usr/bin/env bash
# 一键启动美股量化策略回测平台
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "首次运行，正在创建虚拟环境并安装依赖…"
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi

echo "启动中… 浏览器将自动打开 http://localhost:8501"
streamlit run app.py
