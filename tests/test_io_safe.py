"""Tests for quant.io_safe (atomic write + locked CSV append)."""

from __future__ import annotations

import pandas as pd

from quant.io_safe import append_csv_locked, atomic_write_csv, atomic_write_text


def test_atomic_write_text_roundtrip(tmp_path) -> None:
    target = tmp_path / "sub" / "doc.json"
    atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    # 不留临时文件
    assert [p.name for p in target.parent.iterdir()] == ["doc.json"]


def test_atomic_write_csv_has_bom_and_parses(tmp_path) -> None:
    target = tmp_path / "today.csv"
    df = pd.DataFrame({"代码": ["AAPL", "NVDA"], "动作": ["做多", "观望"]})
    atomic_write_csv(df, target)
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM（Excel 中文友好）
    back = pd.read_csv(target, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(back, df)


def test_append_csv_locked_writes_header_once(tmp_path) -> None:
    target = tmp_path / "history.csv"
    df1 = pd.DataFrame({"日期": ["2026-06-24"], "代码": ["AAPL"]})
    df2 = pd.DataFrame({"日期": ["2026-06-25"], "代码": ["NVDA"]})

    append_csv_locked(df1, target)
    append_csv_locked(df2, target)

    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")            # 仅开头一个 BOM
    assert raw.count(b"\xef\xbb\xbf") == 1            # 追加不再写 BOM
    back = pd.read_csv(target, encoding="utf-8-sig")
    assert list(back.columns) == ["日期", "代码"]
    assert len(back) == 2                              # 表头只一行，数据两行
    assert back["代码"].tolist() == ["AAPL", "NVDA"]


def test_append_csv_locked_skips_empty(tmp_path) -> None:
    target = tmp_path / "empty.csv"
    append_csv_locked(pd.DataFrame(), target)
    assert not target.exists()
