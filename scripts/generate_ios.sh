#!/usr/bin/env bash
# 一键生成 iOS Xcode 工程并打开
set -euo pipefail
cd "$(dirname "$0")/.."
IOS_DIR="$(pwd)/ios"
TOOLS_DIR="$(pwd)/.tools"
XCODEGEN="$TOOLS_DIR/xcodegen"

mkdir -p "$TOOLS_DIR"

if [[ ! -x "$XCODEGEN" ]]; then
  echo "正在下载 XcodeGen（只需一次）…"
  TMP="$(mktemp -d)"
  curl -fsSL "https://github.com/yonaskolb/XcodeGen/releases/download/2.44.1/xcodegen.zip" -o "$TMP/xcodegen.zip"
  unzip -oq "$TMP/xcodegen.zip" -d "$TMP"
  install -m 755 "$(find "$TMP" -name xcodegen -type f | head -1)" "$XCODEGEN"
  rm -rf "$TMP"
  echo "✓ XcodeGen 已就绪"
fi

echo "正在生成 QuantTrader.xcodeproj …"
cd "$IOS_DIR"
"$XCODEGEN" generate

echo ""
echo "✅ 工程已生成：$IOS_DIR/QuantTrader.xcodeproj"
echo ""
echo "接下来在 Xcode 里："
echo "  1. Signing → 勾选 Automatically manage signing"
echo "  2. Team 选你的 Apple ID"
echo "  3. 连接 iPhone → 点 ▶ Run"
echo ""

open "$IOS_DIR/QuantTrader.xcodeproj"
osascript -e 'display notification "Xcode 工程已打开，请连接 iPhone 并点 Run" with title "美股量化 iOS"' 2>/dev/null || true
