"""BENCH 2 — thread saturation for compute (fp32 + int8), per-thread L2 sizing."""

from __future__ import annotations

import os
from typing import Any

from .native import NativeKernels
from .util import (
    check_superlinear,
    effective_cores,
    ensure_quiet_load,
    find_knee,
    l2_bytes_per_core,
    now,
    pin_to_cpus,
    rep_stats,
    set_omp_threads,
    snapshot,
)


def _is_hybrid(profile: Any) -> bool:
    return bool(profile.hybrid and profile.hybrid.is_hybrid.value is True)


def run_bench2(profile: Any, kernels: NativeKernels) -> dict[str, Any]:
    cores = effective_cores(profile)
    l2 = l2_bytes_per_core(profile)

    gate = ensure_quiet_load("bench2_pre")
    out: dict[str, Any] = {
        "name": "BENCH 2 — THREAD SATURATION FOR COMPUTE",
        "status": gate.status,
        "l2_bytes_per_core": l2,
        "sizing_rule": (
            "working_set_bytes(n_threads) = 0.5 * L2_per_core * n_threads; "
            "cache pressure per core constant across sweep"
        ),
        "fit_evidence": (
            f"L2_per_core={l2}; per-thread target = 0.5*L2 = {0.5 * l2}; "
            f"fp32 uses 2 float32 vectors; int8 uses 2 int8 vectors"
        ),
        "isa_verification": (
            "NOT VERIFIED: kernels are scalar C (+OpenMP). They do NOT "
            "dispatch to AMX/AVX512_VNNI/AVX2 specifically. Phase 1.5 "
            "usable_tier is NOT proven by these measurements."
        ),
        "pre": gate.snapshot,
        "quiet_gate": {
            "status": gate.status,
            "readings": gate.readings,
            "evidence": gate.evidence,
        },
        "curves": {},
        "knee": {},
        "validity": {},
        "post": None,
        "hybrid": _is_hybrid(profile),
        "calibration": {},
    }

    if not gate.ok:
        out["post"] = snapshot("bench2_post_blocked")
        return out

    import ctypes
    import numpy as np

    # Allocate for MAX threads; each sweep uses a prefix of length n(t).
    max_ws = int(0.5 * l2 * cores)
    n_fp32_max = max(1024, max_ws // 8)  # 2 * n * 4
    n_i8_max = max(1024, max_ws // 2)  # 2 * n * 1

    rng = np.random.RandomState(42)
    a32 = rng.randn(n_fp32_max).astype(np.float32)
    b32 = rng.randn(n_fp32_max).astype(np.float32)
    a8 = rng.randint(-128, 127, size=n_i8_max, dtype=np.int8)
    b8 = rng.randint(-128, 127, size=n_i8_max, dtype=np.int8)
    # Prefault full buffers
    a32[:] = a32
    b32[:] = b32
    a8[:] = a8
    b8[:] = b8
    _ = float(a32[0] + b32[0] + int(a8[0]) + int(b8[0]))

    # Calibrate reps at 1 thread with 1-thread working set, target ~0.25s
    n1_fp = max(1024, int((0.5 * l2) // 8))
    n1_i8 = max(1024, int((0.5 * l2) // 2))
    reps_fp32 = _calibrate_reps_fp32(kernels, a32, b32, n1_fp)
    reps_i8 = _calibrate_reps_i8(kernels, a8, b8, n1_i8)
    out["reps_fp32"] = reps_fp32
    out["reps_i8"] = reps_i8
    out["calibration"] = {
        "reps_fp32": reps_fp32,
        "reps_i8": reps_i8,
        "n_fp32_at_1thread": n1_fp,
        "n_i8_at_1thread": n1_i8,
        "evidence": (
            f"reps calibrated at 1 thread with per-thread WS; "
            f"reps_fp32={reps_fp32}, reps_i8={reps_i8} held constant across sweep"
        ),
    }

    configs = [("all_cores", list(range(os.cpu_count() or cores)))]
    if _is_hybrid(profile):
        p_ids = (
            profile.hybrid.p_core_logical_ids.value
            if profile.hybrid.p_core_logical_ids
            else []
        )
        e_ids = (
            profile.hybrid.e_core_logical_ids.value
            if profile.hybrid.e_core_logical_ids
            else []
        )
        if p_ids:
            configs.append(("P_cores_only", list(p_ids)))
        if e_ids:
            configs.append(("E_cores_only", list(e_ids)))
    else:
        out["hybrid_note"] = (
            "host is non-hybrid; P-only/E-only pinned curves NOT run "
            "(no core_type/cpu_capacity signals)"
        )

    for label, cpus in configs:
        aff = pin_to_cpus(cpus)
        max_t = min(cores, len(cpus))
        curve: dict[str, Any] = {
            "affinity": aff,
            "cpus": cpus,
            "fp32": [],
            "int8": [],
        }
        for t in range(1, max_t + 1):
            ws = int(0.5 * l2 * t)
            n_fp = max(1024, ws // 8)
            n_i8 = max(1024, ws // 2)
            n_fp = min(n_fp, n_fp32_max)
            n_i8 = min(n_i8, n_i8_max)
            bytes_per_thread_fp = (2 * n_fp * 4) / t
            bytes_per_thread_i8 = (2 * n_i8 * 1) / t
            set_omp_threads(t)
            curve["fp32"].append(
                _sweep_fp32(
                    kernels,
                    a32,
                    b32,
                    n_fp,
                    reps_fp32,
                    t,
                    working_set_bytes=2 * n_fp * 4,
                    bytes_per_thread=bytes_per_thread_fp,
                )
            )
            curve["int8"].append(
                _sweep_i8(
                    kernels,
                    a8,
                    b8,
                    n_i8,
                    reps_i8,
                    t,
                    working_set_bytes=2 * n_i8,
                    bytes_per_thread=bytes_per_thread_i8,
                )
            )
        out["curves"][label] = curve

        ts = [r["threads"] for r in curve["fp32"]]
        meds = [r["median_gflops"] for r in curve["fp32"]]
        validity = check_superlinear(ts, meds)
        out["validity"][label] = validity
        if validity["valid"]:
            knee = find_knee(ts, meds, 0.10)
            out["knee"][label] = {
                "fp32_knee_threads": knee,
                "rule": "last thread count with >=10% gain vs previous median",
                "fp32_medians": meds,
                "int8_medians": [r["median_gops"] for r in curve["int8"]],
                "invalid": False,
            }
        else:
            out["knee"][label] = {
                "fp32_knee_threads": None,
                "rule": "NOT REPORTED — SUPERLINEAR — INVALID",
                "fp32_medians": meds,
                "int8_medians": [r["median_gops"] for r in curve["int8"]],
                "invalid": True,
                "flag": validity["flag"],
                "violations": validity["violations"],
            }

    pin_to_cpus(list(range(os.cpu_count() or cores)))
    out["post"] = snapshot("bench2_post")

    if "P_cores_only" in out["curves"] and "E_cores_only" in out["curves"]:
        all_peak = max(r["median_gflops"] for r in out["curves"]["all_cores"]["fp32"])
        p_peak = max(r["median_gflops"] for r in out["curves"]["P_cores_only"]["fp32"])
        e_peak = max(r["median_gflops"] for r in out["curves"]["E_cores_only"]["fp32"])
        out["hybrid_conclusion"] = {
            "all_peak_fp32_gflops": all_peak,
            "P_peak_fp32_gflops": p_peak,
            "E_peak_fp32_gflops": e_peak,
            "adding_E_vs_P_only": (
                "helps"
                if all_peak > p_peak * 1.05
                else ("hurts" if all_peak < p_peak * 0.95 else "neutral")
            ),
        }

    if out["status"] != "BLOCKED":
        inv = any(k.get("invalid") for k in out["knee"].values())
        out["status"] = "INVALID" if inv else "OK"
    return out


def _calibrate_reps_fp32(kernels, a, b, n) -> int:
    import ctypes

    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    set_omp_threads(1)
    reps = 64
    for _ in range(8):
        t0 = now()
        kernels.lib.dot_fp32_reps_serial(ap, bp, int(n), int(reps))
        dt = now() - t0
        if dt >= 0.20:
            return reps
        reps = max(reps * 2, int(reps * 0.25 / max(dt, 1e-6)))
    return reps


def _calibrate_reps_i8(kernels, a, b, n) -> int:
    import ctypes

    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    set_omp_threads(1)
    reps = 64
    for _ in range(8):
        t0 = now()
        kernels.lib.dot_i8_reps_serial(ap, bp, int(n), int(reps))
        dt = now() - t0
        if dt >= 0.20:
            return reps
        reps = max(reps * 2, int(reps * 0.25 / max(dt, 1e-6)))
    return reps


def _sweep_fp32(
    kernels, a, b, n, reps, threads: int, *, working_set_bytes: int, bytes_per_thread: float
) -> dict[str, Any]:
    import ctypes

    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    set_omp_threads(threads)
    raw_t: list[float] = []
    raw_gflops: list[float] = []
    ops = 2.0 * n * reps
    # Same compiled path for all thread counts when OpenMP is available.
    fn = kernels.lib.dot_fp32_reps if kernels.omp else kernels.lib.dot_fp32_reps_serial

    t0 = now()
    fn(ap, bp, int(n), int(reps))
    warm_dt = now() - t0
    warm_gflops = ops / warm_dt / 1e9 if warm_dt > 0 else float("nan")

    for _ in range(5):
        t0 = now()
        fn(ap, bp, int(n), int(reps))
        dt = now() - t0
        raw_t.append(dt)
        raw_gflops.append(ops / dt / 1e9)
    st = rep_stats(raw_gflops)
    return {
        "threads": threads,
        "reps": reps,
        "n_elements": n,
        "working_set_bytes": working_set_bytes,
        "bytes_per_thread": bytes_per_thread,
        "discarded_warmup": {
            "label": "DISCARDED",
            "time_s": warm_dt,
            "gflops": warm_gflops,
        },
        "raw_times_s": raw_t,
        "raw_gflops": raw_gflops,
        "median_gflops": st.median,
        "min_gflops": st.minimum,
        "max_gflops": st.maximum,
        "cv_pct": st.cv_pct,
        "noisy": st.noisy,
    }


def _sweep_i8(
    kernels, a, b, n, reps, threads: int, *, working_set_bytes: int, bytes_per_thread: float
) -> dict[str, Any]:
    import ctypes

    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    set_omp_threads(threads)
    raw_t: list[float] = []
    raw_gops: list[float] = []
    ops = 2.0 * n * reps
    fn = kernels.lib.dot_i8_reps if kernels.omp else kernels.lib.dot_i8_reps_serial

    t0 = now()
    fn(ap, bp, int(n), int(reps))
    warm_dt = now() - t0
    warm_gops = ops / warm_dt / 1e9 if warm_dt > 0 else float("nan")

    for _ in range(5):
        t0 = now()
        fn(ap, bp, int(n), int(reps))
        dt = now() - t0
        raw_t.append(dt)
        raw_gops.append(ops / dt / 1e9)
    st = rep_stats(raw_gops)
    return {
        "threads": threads,
        "reps": reps,
        "n_elements": n,
        "working_set_bytes": working_set_bytes,
        "bytes_per_thread": bytes_per_thread,
        "discarded_warmup": {
            "label": "DISCARDED",
            "time_s": warm_dt,
            "gops": warm_gops,
        },
        "raw_times_s": raw_t,
        "raw_gops": raw_gops,
        "median_gops": st.median,
        "min_gops": st.minimum,
        "max_gops": st.maximum,
        "cv_pct": st.cv_pct,
        "noisy": st.noisy,
    }
