"""BENCH 3 — sustained STREAM thermal probe (subprocess + omp_set_num_threads)."""

from __future__ import annotations

import os
from typing import Any

from .native import NativeKernels
from .subprocess_run import run_worker
from .util import ensure_quiet_load, snapshot


def run_bench3(
    profile: Any,
    kernels: NativeKernels | None = None,
    *,
    sat_threads: int | None,
    n_elements: int,
    reps: int,
    bytes_moved_per_iter: int,
    duration_s: float = 180.0,
    bucket_s: float = 10.0,
) -> dict[str, Any]:
    """Run sustained STREAM at BENCH 1 max-efficiency / recommended thread count."""
    _ = kernels

    gate = ensure_quiet_load("bench3_pre")
    on_ac = None
    if profile.thermal_power is not None:
        on_ac = profile.thermal_power.on_ac_power.value

    duration_override = None
    if "PHASE2_BENCH3_DURATION_S" in os.environ:
        duration_override = os.environ["PHASE2_BENCH3_DURATION_S"]

    out: dict[str, Any] = {
        "name": "BENCH 3 — SUSTAINED VS BURST (thermal, STREAM/DRAM-bound)",
        "status": gate.status,
        "kernel": (
            "stream_triad_reps_observed via fresh subprocess + omp_set_num_threads "
            "(same STREAM kernel as BENCH 1)"
        ),
        "sat_threads": sat_threads,
        "duration_s": duration_s,
        "duration_override_env": duration_override,
        "bucket_s": bucket_s,
        "n_elements_per_array": n_elements,
        "reps_per_call": reps,
        "bytes_moved_per_iter": bytes_moved_per_iter,
        "on_ac_power_at_profile": on_ac,
        "battery_dual_run": (
            "NOT DONE: no readable AC/battery sysfs on this host to gate dual runs; "
            "single run only"
        ),
        "pre": gate.snapshot,
        "quiet_gate": {
            "status": gate.status,
            "readings": gate.readings,
            "evidence": gate.evidence,
        },
        "buckets": [],
        "post": None,
        "summary": {},
        "thermal_blind": False,
        "requested_threads": sat_threads,
        "omp_num_threads_actual": None,
        "n_distinct_cpus": None,
        "thread_count_applied": None,
    }

    if not gate.ok:
        out["post"] = snapshot("bench3_post_blocked")
        return out

    if sat_threads is None or sat_threads < 1:
        out["status"] = "SKIPPED"
        out["skip_reason"] = (
            "BENCH 1 max-efficiency / recommended thread count unavailable "
            "(curve INVALID or BLOCKED); cannot choose DRAM-bound thread count"
        )
        out["post"] = snapshot("bench3_post_skipped")
        return out

    if n_elements < 1 or reps < 1:
        out["status"] = "SKIPPED"
        out["skip_reason"] = "missing BENCH 1 geometry (n_elements/reps)"
        out["post"] = snapshot("bench3_post_skipped")
        return out

    aff = list(range(max(int(sat_threads), 1)))
    row = run_worker(
        {
            "mode": "stream_sustained",
            "n": int(n_elements),
            "reps": int(reps),
            "bytes_moved_per_iter": int(bytes_moved_per_iter),
            "scalar": 3.0,
            "requested_threads": int(sat_threads),
            "affinity_cpus": aff,
            "omp_proc_bind": "close",
            "omp_places": "cores",
            "duration_s": float(duration_s),
            "bucket_s": float(bucket_s),
        },
        timeout_s=float(duration_s) + 120.0,
    )

    out["pid"] = row.get("pid")
    out["omp_num_threads_actual"] = row.get("omp_num_threads_actual")
    out["omp_max_threads"] = row.get("omp_max_threads")
    out["n_distinct_cpus"] = row.get("n_distinct_cpus")
    out["cpu_ids"] = row.get("cpu_ids")
    out["thread_flag"] = row.get("thread_flag")
    out["thread_count_applied"] = row.get("thread_count_applied")
    out["thermal_probe"] = row.get("thermal_probe")
    out["buckets"] = row.get("buckets") or []
    out["summary"] = row.get("summary") or {}
    out["notes"] = list(row.get("notes") or [])

    if row.get("thermal_blind"):
        out["thermal_blind"] = True
        out["status"] = "THERMAL_BLIND" if row.get("thread_count_applied") else "INVALID"
        if not row.get("thread_count_applied"):
            out["FAIL"] = [
                f"THREAD COUNT NOT APPLIED: requested={sat_threads} "
                f"actual={row.get('omp_num_threads_actual')}; INVALID"
            ]
        out["post"] = snapshot("bench3_post_thermal_blind")
        return out

    if not row.get("thread_count_applied") or row.get("status") == "INVALID":
        out["status"] = "INVALID"
        out["FAIL"] = [
            f"THREAD COUNT NOT APPLIED: requested={sat_threads} "
            f"actual={row.get('omp_num_threads_actual')} "
            f"n_distinct_cpus={row.get('n_distinct_cpus')}; INVALID "
            f"(error={row.get('error')})"
        ]
        out["post"] = snapshot("bench3_post_invalid")
        return out

    out["status"] = "OK"
    out["post"] = snapshot("bench3_post")
    return out
