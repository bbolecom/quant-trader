#!/bin/bash
cd "$(dirname "$0")"
python3 longshort_combo_daily.py
read -n 1 -s -r -p "按任意键关闭…"
