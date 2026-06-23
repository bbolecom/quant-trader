#!/usr/bin/env bash
# 安装每日自动选股（美东收盘后本地时间 18:30，可按需改 plist）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/scripts/com.quant.daily-pick.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.quant.daily-pick.plist"
mkdir -p "$ROOT/logs"
sed "s|PROJECT_ROOT|$ROOT|g" "$PLIST_SRC" > "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "已安装定时任务：每天 18:30 运行 daily_pick.py"
echo "日志：$ROOT/logs/daily_pick.log"
echo "卸载：launchctl unload $PLIST_DST"
