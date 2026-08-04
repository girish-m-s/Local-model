"""BENCH 4 — mmap / page-fault cost (cached vs oversized files)."""

from __future__ import annotations

import ctypes
import mmap
import os
import shutil
from typing import Any, Optional

from .native import NativeKernels
from .util import (
    effective_mem,
    ensure_quiet_load,
    now,
    read_io_read_bytes,
    read_minflt_majflt,
    snapshot,
)


def run_bench4(profile: Any, kernels: NativeKernels) -> dict[str, Any]:
    eff = effective_mem(profile)
    cached_need = int(0.4 * eff)
    oversized_need = int(1.5 * eff)
    total_need = cached_need + oversized_need

    cache_dir = os.path.expanduser("~/.cache/localmodel")
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, "phase21_mmap_cached.bin")
    oversized_path = os.path.join(cache_dir, "phase21_mmap_oversized.bin")

    gate = ensure_quiet_load("bench4_pre")
    usage = shutil.disk_usage(cache_dir)
    out: dict[str, Any] = {
        "name": "BENCH 4 — mmap / PAGE-FAULT COST",
        "status": gate.status,
        "cache_dir": cache_dir,
        "effective_mem_bytes": eff,
        "cached_file_bytes": cached_need,
        "oversized_file_bytes": oversized_need,
        "required_bytes_total": total_need,
        "disk_total": usage.total,
        "disk_free": usage.free,
        "pre": gate.snapshot,
        "quiet_gate": {
            "status": gate.status,
            "readings": gate.readings,
            "evidence": gate.evidence,
        },
        "files": {},
        "delta": {},
        "post": None,
        "note": (
            "Two files: cached=0.4*effective_mem, oversized=1.5*effective_mem. "
            "Effective read bandwidth = /proc/self/io read_bytes delta / time. "
            "pages_per_second = pages touched / time (not a bandwidth). "
            "minflt + read_bytes reported; majflt alone is insufficient for mmap readahead."
        ),
    }

    if not gate.ok:
        out["post"] = snapshot("bench4_post_blocked")
        return out

    slack = 256 * 1024 * 1024
    if usage.free < total_need + slack:
        out["status"] = "SKIPPED"
        out["skip_reason"] = (
            f"disk free {usage.free} < required {total_need} + slack {slack}; "
            "NOT shrinking files silently"
        )
        out["post"] = snapshot("bench4_post_skipped")
        return out

    if cached_need <= 0 or oversized_need <= 0:
        out["status"] = "SKIPPED"
        out["skip_reason"] = "effective_mem_bytes undetected/zero"
        out["post"] = snapshot("bench4_post_skipped")
        return out

    try:
        out["files"]["cached"] = _run_one_file(
            cached_path, cached_need, kernels, label="cached_0.4x_mem"
        )
        out["files"]["oversized"] = _run_one_file(
            oversized_path, oversized_need, kernels, label="oversized_1.5x_mem"
        )
    finally:
        _cleanup(cached_path)
        _cleanup(oversized_path)

    c = out["files"].get("cached") or {}
    o = out["files"].get("oversized") or {}
    warm_c = (c.get("warm") or {}).get("effective_read_gbs")
    warm_o = (o.get("warm") or {}).get("effective_read_gbs")
    cliff = (warm_c / warm_o) if (warm_c and warm_o and warm_o > 0) else None
    out["delta"] = {
        "cold_read_gbs_cached_minus_oversized": _sub(
            c.get("cold", {}).get("effective_read_gbs"),
            o.get("cold", {}).get("effective_read_gbs"),
        ),
        "warm_read_gbs_cached_minus_oversized": _sub(warm_c, warm_o),
        "warm_cliff_ratio_cached_over_oversized": cliff,
        "cold_pages_per_second_cached_minus_oversized": _sub(
            c.get("cold", {}).get("pages_per_second"),
            o.get("cold", {}).get("pages_per_second"),
        ),
        "deliverable": (
            "cached vs oversized warm effective_read_gbs; cliff_ratio = "
            "warm_cached / warm_oversized (page-cache hit vs RAM-pressure miss). "
            "Both io_basis_gbs and file_basis_gbs printed per pass for audit."
        ),
    }
    out["status"] = "OK"
    out["post"] = snapshot("bench4_post")
    return out


def _run_one_file(
    path: str, need: int, kernels: NativeKernels, *, label: str
) -> dict[str, Any]:
    chunk = 8 * 1024 * 1024
    pattern = bytes((i * 17 + 31) & 0xFF for i in range(65536))
    pattern = (pattern * (chunk // len(pattern) + 1))[:chunk]

    write_t0 = now()
    written = 0
    with open(path, "wb") as fh:
        while written < need:
            to_write = min(chunk, need - written)
            fh.write(pattern[:to_write])
            written += to_write
        fh.flush()
        os.fsync(fh.fileno())
    write_dt = now() - write_t0

    fadvise_ev = _fadvise_dontneed(path)
    stride = 4096
    pages = (need + stride - 1) // stride

    cold = _timed_pass(path, need, stride, pages, kernels)
    warm = _timed_pass(path, need, stride, pages, kernels)

    return {
        "label": label,
        "path": path,
        "file_bytes": need,
        "write_seconds": write_dt,
        "write_gbs_file_size_over_time": (need / write_dt / 1e9) if write_dt else None,
        "fadvise_evidence": fadvise_ev,
        "stride": stride,
        "pages": pages,
        "cold": cold,
        "warm": warm,
    }


def _timed_pass(
    path: str, need: int, stride: int, pages: int, kernels: NativeKernels
) -> dict[str, Any]:
    min0, maj0, flt0_ev = read_minflt_majflt()
    io0, io0_ev = read_io_read_bytes()

    t0 = now()
    acc = _stride_pass(path, need, stride, kernels)
    dt = now() - t0

    min1, maj1, flt1_ev = read_minflt_majflt()
    io1, io1_ev = read_io_read_bytes()

    read_delta: Optional[int] = None
    if io0 is not None and io1 is not None:
        read_delta = io1 - io0

    io_basis_gbs = (
        (read_delta / dt / 1e9)
        if (read_delta is not None and read_delta > 0 and dt > 0)
        else None
    )
    file_basis_gbs = (need / dt / 1e9) if dt > 0 else None
    pages_per_second = (pages / dt) if dt > 0 else None

    # FIX 1 (Phase 2.2): delta==0 means fully page-cached — use file_bytes/time.
    if read_delta == 0:
        effective_gbs = file_basis_gbs
        bandwidth_basis = "file_bytes / time"
        bandwidth_note = "FULLY CACHED (io read_bytes delta = 0 is the proof)"
    elif read_delta is not None and read_delta > 0:
        effective_gbs = io_basis_gbs
        bandwidth_basis = "/proc/self/io read_bytes delta / time"
        bandwidth_note = "block I/O observed via read_bytes delta"
    else:
        effective_gbs = None
        bandwidth_basis = "UNDETECTED"
        bandwidth_note = "read_bytes unavailable; cannot choose basis"

    return {
        "seconds": dt,
        "checksum": acc,
        "minflt_before": min0,
        "minflt_after": min1,
        "minflt_delta": min1 - min0,
        "majflt_before": maj0,
        "majflt_after": maj1,
        "majflt_delta": maj1 - maj0,
        "io_read_bytes_before": io0,
        "io_read_bytes_after": io1,
        "io_read_bytes_delta": read_delta,
        "io_basis_gbs": io_basis_gbs,
        "file_basis_gbs": file_basis_gbs,
        "effective_read_gbs": effective_gbs,
        "pages_per_second": pages_per_second,
        "fault_evidence": [flt0_ev, flt1_ev],
        "io_evidence": [io0_ev, io1_ev],
        "bandwidth_basis": bandwidth_basis,
        "bandwidth_note": bandwidth_note,
    }


def _stride_pass(path: str, need: int, stride: int, kernels: NativeKernels) -> int:
    import numpy as np

    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        arr = np.frombuffer(mm, dtype=np.uint8, count=need)
        try:
            acc = kernels.lib.stride_touch_addr(
                ctypes.c_uint64(arr.ctypes.data),
                ctypes.c_size_t(need),
                ctypes.c_size_t(stride),
            )
            return int(acc)
        finally:
            del arr
            mm.close()


def _fadvise_dontneed(path: str) -> str:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            return f"os.posix_fadvise({path!r}, POSIX_FADV_DONTNEED) OK"
        finally:
            os.close(fd)
    except (AttributeError, OSError) as exc:
        return f"UNDETECTED (reason: posix_fadvise failed: {exc})"


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _sub(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return a - b
