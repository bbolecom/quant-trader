#!/usr/bin/env bash
# 双击：立即跑一次「每日收入引擎」扫描（大盘开关 + 卖看涨价差/做多/CSP 三引擎）
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

# 账户规模可改（默认 1 万美金）
ACCOUNT="${1:-10000}"

echo "=========================================="
echo "  正在执行每日收入引擎扫描（账户 \$$ACCOUNT）…"
echo "  首次拉取实时榜单较慢，请稍候 1-3 分钟"
echo "=========================================="
python research/income_engine.py --account "$ACCOUNT"
echo ""
echo "完成。（关闭此窗口即可）"
