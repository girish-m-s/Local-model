"""BENCH 4 — mmap / page-fault cost (subprocess; FULLY CACHED when io delta==0)."""

from __future__ import annotations

import os
import shutil
from typing import Any

from .native import NativeKernels
from .subprocess_run import run_worker
from .util import effective_mem, ensure_quiet_load, snapshot


def run_bench4(profile: Any, kernels: NativeKernels | None = None) -> dict[str, Any]:
    _ = kernels
    eff = effective_mem(profile)
    cached_need = int(0.4 * eff)
    oversized_need = int(1.5 * eff)
    total_need = cached_need + oversized_need

    cache_dir = os.path.expanduser("~/.cache/localmodel")
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, "phase24_mmap_cached.bin")
    oversized_path = os.path.join(cache_dir, "phase24_mmap_oversized.bin")

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
        "thread_control": (
            "Each file measured in a fresh subprocess; omp_set_num_threads(1) "
            "verified via tiny observed triad before timed serial stride_touch."
        ),
        "note": (
            "Two files: cached=0.4*effective_mem, oversized=1.5*effective_mem. "
            "When io_read_bytes_delta==0 → FULLY CACHED; bandwidth_basis = "
            "file_bytes/time. Else effective = read_bytes delta / time. "
            "pages_per_second = pages touched / time (not a bandwidth)."
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

    fails: list[str] = []
    try:
        out["files"]["cached"] = _run_one_file_subprocess(
            cached_path, cached_need, label="cached_0.4x_mem", fails=fails
        )
        out["files"]["oversized"] = _run_one_file_subprocess(
            oversized_path, oversized_need, label="oversized_1.5x_mem", fails=fails
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
            "warm_cached / warm_oversized. FULLY CACHED passes use file_bytes/time "
            "when io_read_bytes_delta==0 (Phase 2.2/2.4 fix)."
        ),
    }
    if fails:
        out["status"] = "INVALID"
        out["FAIL"] = fails
    else:
        out["status"] = "OK"
    out["post"] = snapshot("bench4_post")
    return out


def _run_one_file_subprocess(
    path: str, need: int, *, label: str, fails: list[str]
) -> dict[str, Any]:
    row = run_worker(
        {
            "mode": "mmap_file",
            "path": path,
            "file_bytes": int(need),
            "label": label,
            "requested_threads": 1,
            "affinity_cpus": [0],
            "omp_proc_bind": "close",
            "omp_places": "cores",
        },
        timeout_s=3600.0,
    )
    if not row.get("thread_count_applied") or row.get("status") == "INVALID":
        fails.append(
            f"{label}: THREAD COUNT NOT APPLIED "
            f"req=1 actual={row.get('omp_num_threads_actual')}; INVALID"
        )
        return {
            "label": label,
            "path": path,
            "file_bytes": need,
            "status": "INVALID",
            "thread_flag": row.get("thread_flag"),
            "requested_threads": 1,
            "omp_num_threads_actual": row.get("omp_num_threads_actual"),
            "n_distinct_cpus": row.get("n_distinct_cpus"),
            "error": row.get("error"),
            "cold": {},
            "warm": {},
        }

    return {
        "label": label,
        "path": path,
        "file_bytes": need,
        "status": "OK",
        "pid": row.get("pid"),
        "requested_threads": row.get("requested_threads"),
        "omp_num_threads_actual": row.get("omp_num_threads_actual"),
        "n_distinct_cpus": row.get("n_distinct_cpus"),
        "cpu_ids": row.get("cpu_ids"),
        "thread_flag": row.get("thread_flag"),
        "thread_count_applied": True,
        "write_seconds": row.get("write_seconds"),
        "write_gbs_file_size_over_time": row.get("write_gbs_file_size_over_time"),
        "fadvise_evidence": row.get("fadvise_evidence"),
        "stride": row.get("stride"),
        "pages": row.get("pages"),
        "cold": row.get("cold") or {},
        "warm": row.get("warm") or {},
        "note": row.get("note"),
    }


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _sub(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return a - b
