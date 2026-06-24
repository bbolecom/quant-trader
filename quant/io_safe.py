"""文件写入安全工具：原子写 + 跨进程文件锁。

定时任务（launchd）与手动运行可能并发写同一份「今日快照」或「历史 CSV」，
非原子写会让读取方（HTTP :8502 静态服务 / GitHub Actions / iOS）读到半截文件，
而 ``mode="a"`` 的历史追加存在 TOCTOU（两进程都判定「文件不存在」各写一次表头）
与行交错风险。本模块集中解决：

- :func:`atomic_write_text` / :func:`atomic_write_csv`：同目录临时文件 + ``os.replace`` 原子替换；
- :func:`append_csv_locked`：持有跨进程文件锁后追加，表头仅在新建时写一次。
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

try:  # POSIX（macOS / Linux）
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows 兜底
    _HAS_FCNTL = False

_UTF8_BOM = b"\xef\xbb\xbf"


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> None:
    """原子写文本：先写同目录临时文件再 ``os.replace``，读取方永远看到完整文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)  # 同一文件系统上原子生效
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_csv(df: pd.DataFrame, path: str | os.PathLike, *, bom: bool = True, **to_csv_kwargs) -> None:
    """原子写 CSV。``bom=True`` 时加 UTF-8 BOM（Excel 读中文友好，等价旧 utf-8-sig）。

    注意：``DataFrame.to_csv(None)`` 返回字符串时会忽略 ``encoding``，故统一在此层处理 BOM。
    """
    to_csv_kwargs.setdefault("index", False)
    to_csv_kwargs.pop("encoding", None)  # 字符串模式下无意义，避免误导
    text = df.to_csv(**to_csv_kwargs)
    if bom:
        text = "\ufeff" + text
    atomic_write_text(path, text, encoding="utf-8")


@contextmanager
def file_lock(target: str | os.PathLike) -> Iterator[None]:
    """对 ``<target>.lock`` 加独占跨进程锁；无 fcntl 的平台优雅降级为不加锁。"""
    p = Path(target)
    lock_path = p.with_name(p.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_FCNTL:
        yield
        return
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def append_csv_locked(df: pd.DataFrame, path: str | os.PathLike, *, bom: bool = True) -> None:
    """持锁追加 CSV：表头仅在文件不存在/为空时写一次，BOM 仅随表头写一次。"""
    if df is None or df.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(p):
        write_header = not p.exists() or p.stat().st_size == 0
        payload = df.to_csv(index=False, header=write_header).encode("utf-8")
        if write_header and bom:
            payload = _UTF8_BOM + payload
        with open(p, "ab") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
