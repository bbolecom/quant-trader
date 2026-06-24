#!/usr/bin/env bash
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
LABEL="com.quant.panicrebound"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENTS_DIR/$LABEL.plist"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

[ -x "$VENV_PY" ] || { python3 -m venv .venv; .venv/bin/pip install -r requirements.txt -q; }
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"
mkdir -p "$AGENTS_DIR"

if [ -f "$PLIST_DST" ]; then
  ANSWER="$(osascript -e 'display dialog "恐慌反弹每日扫描已开启。关闭？" buttons {"保持","关闭"} default button "保持"' -e 'button returned of result' 2>/dev/null || echo "保持")"
  if [ "$ANSWER" = "关闭" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null; rm -f "$PLIST_DST"
    echo "✅ 已关闭"; exit 0
  fi
  echo "保持开启"; exit 0
fi

HOUR="$(osascript -e 'text returned of (display dialog "每天几点跑？（收盘后建议16）" default answer "16" with title "恐慌反弹定时")' 2>/dev/null || echo "16")"
MIN="$(osascript -e 'text returned of (display dialog "几分？" default answer "50" with title "恐慌反弹定时")' 2>/dev/null || echo "50")"
HOUR=$(echo "$HOUR" | tr -dc '0-9'); HOUR=${HOUR:-16}
MIN=$(echo "$MIN" | tr -dc '0-9'); MIN=${MIN:-50}

cat > "$PLIST_DST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PY</string>
        <string>$PROJECT_DIR/panic_rebound_daily.py</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MIN</integer></dict>
    <key>StandardOutPath</key><string>$PROJECT_DIR/panic_rebound.log</string>
    <key>StandardErrorPath</key><string>$PROJECT_DIR/panic_rebound.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_DST" 2>/dev/null; launchctl load "$PLIST_DST"
echo "✅ 恐慌反弹每日扫描已开启：每天 ${HOUR}:$(printf '%02d' "$MIN") 自动运行"
chmod +x "$PROJECT_DIR/恐慌反弹做多_运行一次.command" 2>/dev/null
