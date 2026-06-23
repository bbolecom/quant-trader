#!/usr/bin/env bash
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
LABEL="com.quant.strategy"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENTS_DIR/$LABEL.plist"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

[ -x "$VENV_PY" ] || { python3 -m venv .venv; .venv/bin/pip install -r requirements.txt -q; }
mkdir -p "$AGENTS_DIR"

if [ -f "$PLIST_DST" ]; then
  ANSWER="$(osascript -e 'display dialog "每日策略排名已开启。关闭？" buttons {"保持","关闭"} default button "保持"' -e 'button returned of result' 2>/dev/null || echo "保持")"
  if [ "$ANSWER" = "关闭" ]; then
    launchctl unload "$PLIST_DST" 2>/dev/null; rm -f "$PLIST_DST"
    echo "✅ 已关闭"; exit 0
  fi
  echo "保持开启"; exit 0
fi

ACCOUNT="$(osascript -e 'text returned of (display dialog "账户规模（美金）？" default answer "10000" with title "策略排名")' 2>/dev/null || echo "10000")"
PROFILE="$(osascript -e 'text returned of (choose from list {"balanced","income","growth"} with title "策略风格" default items {"balanced"})' 2>/dev/null || echo "balanced")"
HOUR="$(osascript -e 'text returned of (display dialog "每天几点？" default answer "16" with title "策略排名")' 2>/dev/null || echo "16")"
MIN="$(osascript -e 'text returned of (display dialog "几分？" default answer "50" with title "策略排名")' 2>/dev/null || echo "50")"
ACCOUNT=$(echo "$ACCOUNT" | tr -dc '0-9'); ACCOUNT=${ACCOUNT:-10000}
HOUR=$(echo "$HOUR" | tr -dc '0-9'); HOUR=${HOUR:-16}
MIN=$(echo "$MIN" | tr -dc '0-9'); MIN=${MIN:-50}
PROFILE=${PROFILE:-balanced}

python3 <<PY
import json
from pathlib import Path
cfg = json.loads(Path("strategy_config.json").read_text())
cfg["account_size"] = float("$ACCOUNT")
cfg["profile"] = "$PROFILE"
Path("strategy_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False)+"\n")
PY

cat > "$PLIST_DST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PY</string>
        <string>$PROJECT_DIR/strategy_daily.py</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MIN</integer></dict>
    <key>StandardOutPath</key><string>$PROJECT_DIR/strategy.log</string>
    <key>StandardErrorPath</key><string>$PROJECT_DIR/strategy.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST_DST" 2>/dev/null; launchctl load "$PLIST_DST"
echo "✅ 策略排名已开启：每天 ${HOUR}:$(printf '%02d' "$MIN") 风格=$PROFILE 账户=\$${ACCOUNT}"
chmod +x "$PROJECT_DIR/策略排名_运行一次.command" 2>/dev/null
