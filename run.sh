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

# 可选：为 iOS 原生 App 提供 research JSON 静态服务（端口 8502）
if [ "${SERVE_DAILY_PICK_JSON:-1}" = "1" ]; then
  echo "启动 App JSON 静态服务 http://0.0.0.0:8502 （daily_pick / manifest / 各模块 today.json）…"
  (cd research && python -m http.server 8502 --bind 0.0.0.0) &
fi

# K 线实时 API（端口 8503）— iOS 优先从此拉取 yfinance 行情
if [ "${SERVE_CHART_API:-1}" = "1" ]; then
  echo "启动 K线实时 API http://0.0.0.0:8503/v1/chart/{TICKER} …"
  (python -m uvicorn cloud.chart_api.main:app --host 0.0.0.0 --port 8503) &
fi

echo "启动中… 浏览器将自动打开 http://localhost:8501"
streamlit run app.py
