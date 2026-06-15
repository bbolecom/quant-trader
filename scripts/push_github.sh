#!/usr/bin/env bash
# 一键推送到 GitHub（Mac 弹窗粘贴 Token，只需一次）
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="bbolecom/quant-trader"
REMOTE="https://github.com/${REPO}.git"
TOKEN_FILE=".github_token"

echo "=========================================="
echo "  推送项目 → GitHub: ${REPO}"
echo "=========================================="

git remote set-url origin "$REMOTE" 2>/dev/null || git remote add origin "$REMOTE"

TOKEN="${GITHUB_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$TOKEN_FILE" ]]; then
  TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
fi

if [[ -z "$TOKEN" ]]; then
  echo "正在打开 Token 创建页面…"
  open "https://github.com/settings/tokens/new?scopes=repo&description=mac-push" 2>/dev/null || true
  TOKEN="$(osascript -e 'display dialog "请先在浏览器生成 Token（勾选 repo），复制后粘贴到下面：" default answer "" with title "GitHub 推送" buttons {"取消", "推送"} default button "推送"' -e 'text returned of result' 2>/dev/null || true)"
  TOKEN="$(echo "$TOKEN" | tr -d '[:space:]')"
  if [[ -z "$TOKEN" ]]; then
    echo "❌ 未输入 Token，已取消。"
    read -rsp "或在终端粘贴 Token 后回车: " TOKEN
    echo ""
  fi
  if [[ -z "$TOKEN" ]]; then
    echo "❌ 已取消。"
    exit 1
  fi
  echo -n "$TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
  echo "✓ Token 已保存（仅本机，不会上传）"
fi

echo "正在推送…"
if git push "https://${TOKEN}@github.com/${REPO}.git" main; then
  git remote set-url origin "$REMOTE"
  echo ""
  echo "✅ 推送成功！"
  echo "   https://github.com/${REPO}"
  osascript -e 'display notification "代码已成功推送到 GitHub" with title "quant-trader"' 2>/dev/null || true
  open "https://github.com/${REPO}" 2>/dev/null || true
else
  echo "❌ 推送失败。请检查 Token 是否有效、仓库是否已创建。"
  exit 1
fi
