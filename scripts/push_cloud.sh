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

echo "→ 全市场快扫 …"
python3 market_scan_fast.py

echo "→ 重建 daily_pick 高胜率池（剔除无真实链假信号）…"
python3 - <<'PY'
import json
from quant.high_win_pick import build_high_win_doc
from quant.daily_pick_push import sanitize_picks, build_push_block
P = "research/daily_pick_today.json"
doc = json.load(open(P))
hwd = build_high_win_doc(doc.get("picks") or [], min_win_rate=0.80, regime=doc.get("regime") or {})
doc["picks"] = sanitize_picks(hwd.get("all_enriched") or doc.get("picks") or [])
doc["high_win"] = {
    "min_win_rate": 0.80,
    "summary": hwd.get("summary") or {},
    "picks": hwd.get("high_win_actionable") or [],
    "watch": hwd.get("high_win_watch") or [],
}
doc["summary"]["高胜率可开仓"] = len(doc["high_win"]["picks"])
pb = build_push_block(doc, json.load(open("daily_pick_config.json")))
doc["push"] = pb
json.dump(doc, open(P, "w"), ensure_ascii=False, indent=2)
print(f"  高胜率 可开仓={len(doc['high_win']['picks'])} 观察={len(doc['high_win']['watch'])}")
PY

echo "→ 导出 app_manifest …"
python3 -m quant.app_manifest

echo "→ 同步 ios/Resources 快照 …"
python3 scripts/sync_ios_bundles.py

echo "→ 运行核心测试 …"
python3 -m pytest tests/test_gainer10_strategy.py tests/test_strategy_catalog.py tests/test_ios_json_feeds.py tests/test_daily_pick_runners.py tests/test_market_scan_fast.py tests/test_high_win_pick.py tests/test_daily_pick_push.py -q

if [[ -n "$(git status --porcelain)" ]]; then
  git add \
    quant/ market_scan_fast.py market_scan_config.json "MarketScan_运行一次.command" \
    gainer10_* extreme20_* whipsaw_short_* longshort_combo_* rgti_* \
    "Extreme20_运行一次.command" "Gainer10_运行一次.command" "LongShort_运行一次.command" \
    research/gainer10_*.json research/extreme20_today.json research/whipsaw_short_today.json \
    research/market_scan_today.json research/daily_pick_today.json research/daily_pick_high_win.json research/daily_pick_push.json \
    research/quantum_watch_today.json research/app_manifest.json research/sector_map.json \
    research/longshort_combo_*.json research/surge20_refined_playbook.json \
    ios/Resources/*.json ios/Sources/ ios/project.yml \
    daily_pick.py daily_pick_config.json app.py \
    tests/test_gainer10_strategy.py tests/test_extreme20_strategy.py tests/test_longshort_combo_strategy.py \
    tests/test_strategy_catalog.py tests/test_ios_json_feeds.py tests/test_daily_pick_runners.py \
    tests/test_market_scan_fast.py tests/test_high_win_pick.py tests/test_daily_pick_push.py \
    .github/workflows/cloud_sync.yml .github/workflows/market_scan.yml \
    scripts/push_cloud.sh scripts/verify_cloud_feeds.sh scripts/sync_ios_bundles.py \
    requirements-dev.txt 2>/dev/null || true

  git add -u quant/ ios/Sources/ ios/Resources/app_manifest.json research/app_manifest.json \
    app.py daily_pick.py daily_pick_config.json quant/daily_pick_runners.py quant/strategy_catalog.py \
    quant/app_manifest.py quant/high_win_pick.py quant/providers/yahoo.py quant/screener.py \
    tests/ .github/workflows/ scripts/ 2>/dev/null || true

  if git diff --staged --quiet; then
    echo "无新变更需提交（可能已在暂存区外）"
  else
    git commit -m "$(cat <<'EOF'
feat: 全市场5分钟快扫 + 高胜率数据真实化 + 云端连通增强

- market_scan_fast：并行 Yahoo 多榜，808只~13s，真实行情信号
- 高胜率池剔除无真实期权链的假88%标签；iOS 5分钟刷新
- cloud_sync/market_scan 双 workflow；jsDelivr+Raw 双 CDN
EOF
)"
  fi
else
  echo "工作区干净，跳过提交"
fi

echo "→ 推送到 origin/main …"
exec bash scripts/push_github.sh
