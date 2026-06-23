#!/usr/bin/env bash
# 双击：立即跑一次「闪迪周 PUT 喝汤」提醒（测试用，会弹桌面通知）
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
echo "  正在生成本周 PUT 价差喝汤方案…"
echo "=========================================="
python weekly_soup.py
echo ""
echo "完成。结果已追加到 weekly_soup_history.csv"
echo "（关闭此窗口即可）"
