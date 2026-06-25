#!/usr/bin/env bash
# 双击：Gainer10+ 分板块高胜率策略每日扫描
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
echo "  Gainer10+ · 分板块高胜率多空"
echo "  日涨>10% + 成交额>1亿 · 组合胜率~75%"
echo "=========================================="
python3 gainer10_daily.py
echo ""
echo "完成。结果见 research/gainer10_today.json"
echo "（关闭此窗口即可）"
