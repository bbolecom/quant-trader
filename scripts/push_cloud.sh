#!/usr/bin/env bash
# 一键：导出清单 → 同步 iOS 快照 → 提交 → 推 GitHub（触发 Streamlit / 云端 JSON 更新）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=========================================="
echo "  推送系统 → GitHub 云端"
echo "=========================================="

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "→ 导出 app_manifest …"
python3 -m quant.app_manifest

echo "→ 同步 ios/Resources 快照 …"
python3 scripts/sync_ios_bundles.py

echo "→ 运行核心测试 …"
python3 -m pytest tests/test_gainer10_strategy.py tests/test_strategy_catalog.py tests/test_ios_json_feeds.py tests/test_daily_pick_runners.py -q

if [[ -n "$(git status --porcelain)" ]]; then
  git add \
    quant/ gainer10_* extreme20_* whipsaw_short_* longshort_combo_* rgti_* \
    "Extreme20_运行一次.command" "Gainer10_运行一次.command" "LongShort_运行一次.command" \
    research/gainer10_*.json research/extreme20_today.json research/whipsaw_short_today.json \
    research/quantum_watch_today.json research/app_manifest.json research/sector_map.json \
    research/longshort_combo_*.json research/surge20_refined_playbook.json \
    ios/Resources/*.json ios/Sources/ \
    daily_pick.py daily_pick_config.json app.py \
    tests/test_gainer10_strategy.py tests/test_extreme20_strategy.py tests/test_longshort_combo_strategy.py \
    tests/test_strategy_catalog.py tests/test_ios_json_feeds.py tests/test_daily_pick_runners.py \
    .github/workflows/cloud_sync.yml scripts/push_cloud.sh \
    requirements-dev.txt 2>/dev/null || true

  git add -u quant/ ios/Sources/ ios/Resources/app_manifest.json research/app_manifest.json \
    app.py daily_pick.py daily_pick_config.json quant/daily_pick_runners.py quant/strategy_catalog.py \
    quant/app_manifest.py tests/ .github/workflows/cloud_sync.yml 2>/dev/null || true

  if git diff --staged --quiet; then
    echo "无新变更需提交（可能已在暂存区外）"
  else
    git commit -m "$(cat <<'EOF'
feat: Gainer10+ 分板块高胜率策略 + 云端同步增强

- 核心策略目录接入 gainer10（L≥60%+avg≥3 · S≥80%，组合胜率~75%）
- 每日扫描、daily_pick、iOS manifest 与 JSON 快照对齐
- cloud_sync 盘后同步 ios/Resources 与模块 today JSON
EOF
)"
  fi
else
  echo "工作区干净，跳过提交"
fi

echo "→ 推送到 origin/main …"
exec bash scripts/push_github.sh
