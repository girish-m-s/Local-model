"""BENCH 4 — mmap / page-fault cost (2x effective RAM file)."""

from __future__ import annotations

import ctypes
import mmap
import os
import shutil
from typing import Any

from .native import NativeKernels
from .util import effective_mem, now, read_majflt, snapshot


def run_bench4(profile: Any, kernels: NativeKernels) -> dict[str, Any]:
    eff = effective_mem(profile)
    need = 2 * eff
    cache_dir = os.path.expanduser("~/.cache/localmodel")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "phase2_mmap_probe.bin")

    pre = snapshot("bench4_pre")
    usage = shutil.disk_usage(cache_dir)
    out: dict[str, Any] = {
        "name": "BENCH 4 — mmap / PAGE-FAULT COST",
        "cache_dir": cache_dir,
        "file_path": path,
        "required_bytes": need,
        "effective_mem_bytes": eff,
        "disk_total": usage.total,
        "disk_free": usage.free,
        "pre": pre,
        "status": None,
        "post": None,
    }

    slack = 256 * 1024 * 1024
    if usage.free < need + slack:
        out["status"] = "SKIPPED"
        out["skip_reason"] = (
            f"disk free {usage.free} < required {need} + slack {slack}; "
            "NOT shrinking file silently"
        )
        out["post"] = snapshot("bench4_post_skipped")
        return out

    if need <= 0:
        out["status"] = "SKIPPED"
        out["skip_reason"] = "effective_mem_bytes undetected/zero"
        out["post"] = snapshot("bench4_post_skipped")
        return out

    chunk = 8 * 1024 * 1024
    # Deterministic pattern (faster than urandom for 32 GiB)
    pattern = bytes((i * 17 + 31) & 0xFF for i in range(65536))
    pattern = (pattern * (chunk // len(pattern) + 1))[:chunk]

    write_t0 = now()
    written = 0
    try:
        with open(path, "wb") as fh:
            while written < need:
                to_write = min(chunk, need - written)
                fh.write(pattern[:to_write])
                written += to_write
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        out["status"] = "SKIPPED"
        out["skip_reason"] = f"write failed after {written} bytes: {exc}"
        _cleanup(path)
        out["post"] = snapshot("bench4_post_skipped")
        return out
    write_dt = now() - write_t0

    # Evict file pages from page cache so the first mmap pass is actually cold.
    # Does not require root (unlike drop_caches).
    fadvise_ev = _fadvise_dontneed(path)

    stride = 4096
    pages = (need + stride - 1) // stride
    # One load per page; OS still brings in whole pages → cold-read GB/s
    # is file_bytes / time (not the 1-byte-touched numerator).
    bytes_touched = pages

    maj0, maj0_ev = read_majflt()
    cold_t0 = now()
    cold_acc = _stride_pass(path, need, stride, kernels)
    cold_dt = now() - cold_t0
    maj1, maj1_ev = read_majflt()

    warm_t0 = now()
    warm_acc = _stride_pass(path, need, stride, kernels)
    warm_dt = now() - warm_t0
    maj2, maj2_ev = read_majflt()

    out.update(
        {
            "status": "OK",
            "write_seconds": write_dt,
            "write_gbs": (need / write_dt / 1e9) if write_dt else None,
            "cold_seconds": cold_dt,
            "warm_seconds": warm_dt,
            "stride": stride,
            "pages_touched_per_pass": pages,
            "estimated_bytes_touched_per_pass": bytes_touched,
            # Spec: cold-read GB/s = effective page-fault throughput for the
            # 2x-RAM mapping (whole pages brought in), not 1-byte/page loads.
            "cold_gbs": (need / cold_dt / 1e9) if cold_dt else None,
            "warm_gbs": (need / warm_dt / 1e9) if warm_dt else None,
            "cold_touch_gbs": (bytes_touched / cold_dt / 1e9) if cold_dt else None,
            "warm_touch_gbs": (bytes_touched / warm_dt / 1e9) if warm_dt else None,
            "cold_checksum": cold_acc,
            "warm_checksum": warm_acc,
            "majflt_before_cold": maj0,
            "majflt_after_cold": maj1,
            "majflt_after_warm": maj2,
            "majflt_cold_delta": maj1 - maj0,
            "majflt_warm_delta": maj2 - maj1,
            "majflt_evidence": [maj0_ev, maj1_ev, maj2_ev],
            "fadvise_evidence": fadvise_ev,
            "note": (
                "cold/warm GB/s = file_bytes/time (whole pages faulted via "
                "4KiB-stride touch). touch_gbs = 1-byte-per-page / time. "
                "File size = 2x effective_mem. posix_fadvise(DONTNEED) before "
                "cold pass (not a silent shrink); majflt may under-count if "
                "pages remain cached."
            ),
        }
    )
    _cleanup(path)
    out["post"] = snapshot("bench4_post")
    return out


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
            # Release buffer export before closing mmap
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
