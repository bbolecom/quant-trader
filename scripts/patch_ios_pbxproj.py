#!/usr/bin/env python3
"""将 Sources/*.swift 与 Resources/* 同步进 QuantTrader.xcodeproj（无 xcodegen 时使用）。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "ios"
PBX = IOS / "QuantTrader.xcodeproj" / "project.pbxproj"

SWIFT_FILES = sorted(p.name for p in (IOS / "Sources").glob("*.swift"))
RESOURCE_FILES = sorted(
    [p.name for p in (IOS / "Resources").glob("*") if p.suffix in {".json"}]
    + ["Assets.xcassets"]
)
CHARTS_DIR = IOS / "Resources" / "charts"


def uid(name: str) -> str:
    import hashlib

    h = hashlib.md5(name.encode()).hexdigest().upper()[:24]
    return h


def main() -> None:
    text = PBX.read_text(encoding="utf-8")

    # collect existing paths
    existing = set(re.findall(r"path = ([^;]+);", text))

    build_files: list[str] = []
    file_refs: list[str] = []
    source_group_children: list[str] = []
    resource_group_children: list[str] = []
    sources_phase: list[str] = []
    resources_phase: list[str] = []

    for name in SWIFT_FILES:
        if name in existing:
            continue
        ref = uid("ref:" + name)
        build = uid("build:" + name)
        file_refs.append(
            f"\t\t{ref} /* {name} */ = {{isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = {name}; sourceTree = \"<group>\"; }};"
        )
        build_files.append(
            f"\t\t{build} /* {name} in Sources */ = {{isa = PBXBuildFile; fileRef = {ref} /* {name} */; }};"
        )
        source_group_children.append(f"\t\t\t\t{ref} /* {name} */,")
        sources_phase.append(f"\t\t\t\t{build} /* {name} in Sources */,")
        print(f"+ Sources {name}")

    for name in RESOURCE_FILES:
        if name in existing:
            continue
        ref = uid("ref:" + name)
        build = uid("build:" + name)
        ftype = "folder.assetcatalog" if name.endswith(".xcassets") else "text.json"
        file_refs.append(
            f"\t\t{ref} /* {name} */ = {{isa = PBXFileReference; lastKnownFileType = {ftype}; path = {name}; sourceTree = \"<group>\"; }};"
        )
        build_files.append(
            f"\t\t{build} /* {name} in Resources */ = {{isa = PBXBuildFile; fileRef = {ref} /* {name} */; }};"
        )
        resource_group_children.append(f"\t\t\t\t{ref} /* {name} */,")
        resources_phase.append(f"\t\t\t\t{build} /* {name} in Resources */,")
        print(f"+ Resources {name}")

    if CHARTS_DIR.is_dir() and "charts" not in existing:
        ref = uid("ref:charts-folder")
        build = uid("build:charts-folder")
        file_refs.append(
            f"\t\t{ref} /* charts */ = {{isa = PBXFileReference; lastKnownFileType = folder; path = charts; sourceTree = \"<group>\"; }};"
        )
        build_files.append(
            f"\t\t{build} /* charts in Resources */ = {{isa = PBXBuildFile; fileRef = {ref} /* charts */; }};"
        )
        resource_group_children.append(f"\t\t\t\t{ref} /* charts */,")
        resources_phase.append(f"\t\t\t\t{build} /* charts in Resources */,")
        print("+ Resources charts/ (folder)")

    if not build_files:
        print("pbxproj already up to date")
        return

    text = text.replace(
        "/* End PBXBuildFile section */",
        "\n".join(build_files) + "\n/* End PBXBuildFile section */",
    )
    text = text.replace(
        "/* End PBXFileReference section */",
        "\n".join(file_refs) + "\n/* End PBXFileReference section */",
    )

    if source_group_children:
        text = text.replace(
            "\t\t\t1381B154322A386706449C7C /* WebView.swift */,",
            "\t\t\t1381B154322A386706449C7C /* WebView.swift */,\n"
            + "\n".join(source_group_children),
        )
        text = text.replace(
            "\t\t\tD6A2B3C4D5E6F7890A1B2C3F /* AppInfo.swift in Sources */,",
            "\t\t\tD6A2B3C4D5E6F7890A1B2C3F /* AppInfo.swift in Sources */,\n"
            + "\n".join(sources_phase),
        )

    if resource_group_children:
        text = text.replace(
            "\t\t\tC2B2C3D4E5F6789012345679 /* daily_pick_today.json */,",
            "\t\t\tC2B2C3D4E5F6789012345679 /* daily_pick_today.json */,\n"
            + "\n".join(resource_group_children),
        )
        text = text.replace(
            "\t\t\tC2A2B3C4D5E6F7890A1B2C3E /* daily_pick_today.json in Resources */,",
            "\t\t\tC2A2B3C4D5E6F7890A1B2C3E /* daily_pick_today.json in Resources */,\n"
            + "\n".join(resources_phase),
        )

    text = re.sub(r"CURRENT_PROJECT_VERSION = \d+;", "CURRENT_PROJECT_VERSION = 10;", text)
    PBX.write_text(text, encoding="utf-8")
    print(f"updated {PBX}")


if __name__ == "__main__":
    main()
