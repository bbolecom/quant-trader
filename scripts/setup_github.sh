#!/usr/bin/env bash
# 创建 GitHub 私有仓库并推送（需已安装 gh 且 gh auth login）
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_NAME="${1:-us-quant-trader}"
VISIBILITY="${2:-private}"   # private 或 public

if ! command -v gh >/dev/null 2>&1; then
  echo "未找到 gh。请先: brew install gh && gh auth login"
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "已存在 remote origin，直接 push…"
else
  echo "创建 GitHub 仓库: $REPO_NAME ($VISIBILITY)…"
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin
fi

git push -u origin main
echo ""
echo "完成。仓库地址:"
gh repo view --web 2>/dev/null || git remote get-url origin
echo ""
echo "下一步: 打开 https://share.streamlit.io 部署 app.py"
