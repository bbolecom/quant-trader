#!/bin/bash
cd "$(dirname "$0")"
python3 flow_daily.py
read -p "按回车关闭..."
