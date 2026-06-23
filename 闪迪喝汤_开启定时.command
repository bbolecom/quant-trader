#!/usr/bin/env bash
# 双击：开启 / 关闭「每周一 PUT 喝汤提醒」定时任务（macOS launchd）
cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
LABEL="com.quant.weeklysoup"
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
  ANSWER="$(osascript -e 'display dialog "每周一 PUT 喝汤提醒已开启。是否要关闭它？" buttons {"保持开启", "关闭定时"} default button "保持开启"' -e 'button returned of result' 2>/dev/null || echo "保持开启")"
  if [ "$ANSWER" = "关闭定时" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "✅ 已关闭每周 PUT 喝汤提醒。"
    osascript -e 'display notification "已关闭每周 PUT 喝汤提醒" with title "闪迪喝汤"' 2>/dev/null || true
  else
    echo "保持开启，未做改动。"
  fi
  exit 0
fi

HOUR="$(osascript -e 'text returned of (display dialog "每周一几点提醒？（0-23，建议 22 美股开盘前）" default answer "22" with title "闪迪喝汤 · 设定时间")' 2>/dev/null || echo "22")"
MIN="$(osascript -e 'text returned of (display dialog "几分？（0-59）" default answer "30" with title "闪迪喝汤 · 设定时间")' 2>/dev/null || echo "30")"
HOUR=$(echo "$HOUR" | tr -dc '0-9'); HOUR=${HOUR:-22}
MIN=$(echo "$MIN" | tr -dc '0-9'); MIN=${MIN:-30}

cat > "$PLIST_DST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PY</string>
        <string>$PROJECT_DIR/weekly_soup.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>1</integer>
        <key>Hour</key>
        <integer>$HOUR</integer>
        <key>Minute</key>
        <integer>$MIN</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/weekly_soup.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/weekly_soup.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "✅ 已开启每周一 PUT 喝汤提醒：每周一 ${HOUR}:$(printf '%02d' "$MIN") 运行"
echo "   日志：weekly_soup.log ／ weekly_soup.err.log"
echo "   想改条件：编辑 weekly_soup_config.json"
osascript -e "display notification \"每周一 ${HOUR}:$(printf '%02d' "$MIN") 自动推送 PUT 喝汤方案\" with title \"闪迪喝汤已开启\"" 2>/dev/null || true
echo ""
echo "（关闭此窗口即可）"
