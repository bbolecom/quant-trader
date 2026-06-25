#!/usr/bin/env bash
# 双击：暴涨/暴跌 ≥20% 事件策略每日扫描（L1/S1/L2/S2）
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
echo "  Extreme20 · 暴涨暴跌20%事件策略"
echo "=========================================="
python extreme20_daily.py
echo ""
echo "完成。结果见 research/extreme20_today.json"
echo "（关闭此窗口即可）"
