#!/usr/bin/env python3
"""盘后一键：跑全策略扫描 → 写云端 JSON → 汇总机会 → 手机推送。

GitHub Actions cloud_sync.yml 在美东收盘后调用本脚本。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 子策略只写 JSON，推送由 digest 统一发
os.environ["QUANT_SKIP_MOBILE_PUSH"] = "1"

STEPS: list[tuple[str, list[str]]] = [
    ("whipsaw_short_daily.py", []),
    ("blowoff_short_daily.py", []),
    ("gainer10_daily.py", []),
    ("extreme20_daily.py", []),
    ("flow_daily.py", []),
    ("ticker_pattern_daily.py", []),
    ("sndk_iron_daily.py", []),
    ("vrp_daily.py", []),
    ("speculative_pool_daily.py", []),
    ("short_squeeze_daily.py", []),
    ("daily_pick.py", ["--quick", "--no-notify"]),
]


def run_step(script: str, args: list[str]) -> bool:
    path = ROOT / script
    if not path.exists():
        print(f"[skip] 不存在: {script}")
        return False
    cmd = [sys.executable, str(path)] + args
    print(f"\n>>> {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=ROOT, check=False, timeout=900)
        return True
    except subprocess.TimeoutExpired:
        print(f"[timeout] {script}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[error] {script}: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("云端盘后全策略扫描")
    print("=" * 60)
    for script, args in STEPS:
        run_step(script, args)

    run_step("scripts/sync_ios_bundles.py", [])
    subprocess.run([sys.executable, "-m", "quant.app_manifest"], cwd=ROOT, check=False)

    # 统一推送（解除 skip）
    del os.environ["QUANT_SKIP_MOBILE_PUSH"]
    from quant.cloud_opportunity_digest import main as digest_main

    digest_main()
    print("\n✓ cloud_post_close 完成")


if __name__ == "__main__":
    main()
