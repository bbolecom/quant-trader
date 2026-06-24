#!/usr/bin/env python3
"""导出 iOS App 功能清单 app_manifest.json。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.app_manifest import export_app_manifest  # noqa: E402

if __name__ == "__main__":
    paths = export_app_manifest()
    print(f"OK manifest → {paths}")
