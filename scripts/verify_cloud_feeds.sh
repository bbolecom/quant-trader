#!/usr/bin/env bash
# 验证云端 JSON 可访问（jsDelivr + GitHub Raw）
set -euo pipefail
REPO="${GITHUB_REPO:-bbolecom/quant-trader}"
BRANCH="${GITHUB_BRANCH:-main}"
FILES=(
  "research/daily_pick_today.json"
  "research/market_scan_today.json"
  "research/app_manifest.json"
)
PASS=0
FAIL=0
for rel in "${FILES[@]}"; do
  name="${rel#research/}"
  for base in \
    "https://cdn.jsdelivr.net/gh/${REPO}@${BRANCH}/${rel}" \
    "https://raw.githubusercontent.com/${REPO}/${BRANCH}/${rel}"; do
    code=$(curl -sS -o /dev/null -w "%{http_code}" -m 25 -L "$base" || echo "000")
    if [[ "$code" == "200" ]]; then
      echo "✓ $name @ $(echo "$base" | sed 's|https://||;s|/.*||')"
      PASS=$((PASS + 1))
    else
      echo "✗ $name HTTP $code @ $base"
      FAIL=$((FAIL + 1))
    fi
  done
done
echo "---"
echo "pass=$PASS fail=$FAIL"
[[ "$FAIL" -eq 0 ]]
