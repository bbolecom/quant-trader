#!/usr/bin/env bash
# 双击：立即跑一次「闪迪 SNDK 真实期权链铁鹰」提醒（会弹桌面通知）
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
echo "  正在拉取 mixed_balanced 舰队真实期权链…"
echo "=========================================="
python sndk_iron_daily.py
echo ""
echo "完成。结果已追加到 sndk_iron_history.csv"
echo "（关闭此窗口即可）"
