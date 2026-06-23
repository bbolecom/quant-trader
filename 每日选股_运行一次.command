#!/usr/bin/env bash
# 双击：立即跑一次每日选股（用于测试，会弹桌面通知）
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
echo "  正在执行每日选股…"
echo "=========================================="
python daily_pick.py
echo ""
echo "完成。结果见 research/daily_pick_today.json（无票日也会记录「观望」）"
echo "（关闭此窗口即可）"
