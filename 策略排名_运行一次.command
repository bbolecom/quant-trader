#!/usr/bin/env bash
cd "$(dirname "$0")"
VENV_PY="$(pwd)/.venv/bin/python"
[ -x "$VENV_PY" ] || { python3 -m venv .venv && VENV_PY="$(pwd)/.venv/bin/python" && .venv/bin/pip install -r requirements.txt -q; }
"$VENV_PY" strategy_daily.py "$@"
echo ""
read -r -p "按回车关闭…" _ 2>/dev/null || sleep 2
