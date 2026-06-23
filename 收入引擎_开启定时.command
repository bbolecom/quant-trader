#!/usr/bin/env bash
# 双击：开启 / 关闭「每日收入引擎」定时任务（macOS launchd，收盘后弹通知）
cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
LABEL="com.quant.income"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENTS_DIR/$LABEL.plist"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "未找到虚拟环境，正在创建…"
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
fi

mkdir -p "$AGENTS_DIR"

if [ -f "$PLIST_DST" ]; then
  ANSWER="$(osascript -e 'display dialog "每日收入引擎已开启。是否要关闭它？" buttons {"保持开启", "关闭定时"} default button "保持开启"' -e 'button returned of result' 2>/dev/null || echo "保持开启")"
  if [ "$ANSWER" = "关闭定时" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "✅ 已关闭每日收入引擎。"
    osascript -e 'display notification "已关闭每日收入引擎" with title "收入引擎"' 2>/dev/null || true
  else
    echo "保持开启，未做改动。"
  fi
  exit 0
fi

ACCOUNT="$(osascript -e 'text returned of (display dialog "账户规模（美金）？用于计算建议张数" default answer "10000" with title "收入引擎 · 账户")' 2>/dev/null || echo "10000")"
ACCOUNT=$(echo "$ACCOUNT" | tr -dc '0-9'); ACCOUNT=${ACCOUNT:-10000}
HOUR="$(osascript -e 'text returned of (display dialog "每天几点扫描？（0-23，建议 16 美股收盘后）" default answer "16" with title "收入引擎 · 设定时间")' 2>/dev/null || echo "16")"
MIN="$(osascript -e 'text returned of (display dialog "几分？（0-59）" default answer "40" with title "收入引擎 · 设定时间")' 2>/dev/null || echo "40")"
HOUR=$(echo "$HOUR" | tr -dc '0-9'); HOUR=${HOUR:-16}
MIN=$(echo "$MIN" | tr -dc '0-9'); MIN=${MIN:-40}

cat > "$PLIST_DST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PY</string>
        <string>$PROJECT_DIR/research/income_engine.py</string>
        <string>--account</string>
        <string>$ACCOUNT</string>
        <string>--notify</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>$HOUR</integer>
        <key>Minute</key>
        <integer>$MIN</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/income.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/income.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "✅ 已开启每日收入引擎：每天 ${HOUR}:$(printf '%02d' "$MIN") 运行（账户 \$$ACCOUNT）"
echo "   日志：income.log ／ income.err.log ／ 历史：income_engine_history.csv"
osascript -e "display notification \"每天 ${HOUR}:$(printf '%02d' "$MIN") 自动扫描收入引擎\" with title \"收入引擎已开启\"" 2>/dev/null || true
echo ""
echo "（关闭此窗口即可）"
