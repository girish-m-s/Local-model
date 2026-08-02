"""BENCH 2 — thread saturation for compute (fp32 + int8, L2-sized)."""

from __future__ import annotations

import os
from typing import Any, Optional

from .native import NativeKernels
from .util import (
    effective_cores,
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
    # Two vectors fit in L2: 2 * n * 4 <= l2  (fp32) → n <= l2/8
    n_fp32 = max(1024, int((l2 // 8) * 0.8))
    n_i8 = max(1024, int((l2 // 2) * 0.8))  # 2 * n * 1 byte

    pre = snapshot("bench2_pre")
    out: dict[str, Any] = {
        "name": "BENCH 2 — THREAD SATURATION FOR COMPUTE",
        "l2_bytes_per_core": l2,
        "n_fp32": n_fp32,
        "n_i8": n_i8,
        "fit_evidence": (
            f"fp32 vectors: 2*{n_fp32}*4={2*n_fp32*4} bytes "
            f"(target ≤ 0.8*L2={0.8*l2}); "
            f"int8 vectors: 2*{n_i8}*1={2*n_i8} bytes"
        ),
        "isa_verification": (
            "NOT VERIFIED: kernels are scalar C (+OpenMP). They do NOT "
            "dispatch to AMX/AVX512_VNNI/AVX2 specifically. Phase 1.5 "
            "usable_tier is NOT proven by these measurements."
        ),
        "pre": pre,
        "curves": {},
        "knee": {},
        "post": None,
        "hybrid": _is_hybrid(profile),
    }

    import ctypes
    import numpy as np

    # Allocate once
    rng = np.random.RandomState(42)
    a32 = rng.randn(n_fp32).astype(np.float32)
    b32 = rng.randn(n_fp32).astype(np.float32)
    a8 = rng.randint(-128, 127, size=n_i8, dtype=np.int8)
    b8 = rng.randint(-128, 127, size=n_i8, dtype=np.int8)
    # page commit
    _ = float(a32[0] + b32[0] + int(a8[0]) + int(b8[0]))

    # reps chosen so each trial ~0.1-0.3s at 1 thread
    reps_fp32 = _calibrate_reps_fp32(kernels, a32, b32, n_fp32)
    reps_i8 = _calibrate_reps_i8(kernels, a8, b8, n_i8)
    out["reps_fp32"] = reps_fp32
    out["reps_i8"] = reps_i8

    configs = [("all_cores", list(range(os.cpu_count() or cores)))]
    if _is_hybrid(profile):
        p_ids = profile.hybrid.p_core_logical_ids.value if profile.hybrid.p_core_logical_ids else []
        e_ids = profile.hybrid.e_core_logical_ids.value if profile.hybrid.e_core_logical_ids else []
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
        curve = {"affinity": aff, "cpus": cpus, "fp32": [], "int8": []}
        for t in range(1, max_t + 1):
            set_omp_threads(t)
            curve["fp32"].append(_sweep_fp32(kernels, a32, b32, n_fp32, reps_fp32, t))
            curve["int8"].append(_sweep_i8(kernels, a8, b8, n_i8, reps_i8, t))
        out["curves"][label] = curve

        # knee from fp32 median Gflops
        ts = [r["threads"] for r in curve["fp32"]]
        meds = [r["median_gflops"] for r in curve["fp32"]]
        knee = find_knee(ts, meds, 0.10)
        out["knee"][label] = {
            "fp32_knee_threads": knee,
            "rule": "last thread count with >=10% gain vs previous median",
            "fp32_medians": meds,
            "int8_medians": [r["median_gops"] for r in curve["int8"]],
        }

    # restore affinity
    pin_to_cpus(list(range(os.cpu_count() or cores)))
    out["post"] = snapshot("bench2_post")

    # E-core help/hurt summary
    if "P_cores_only" in out["curves"] and "E_cores_only" in out["curves"]:
        all_peak = max(r["median_gflops"] for r in out["curves"]["all_cores"]["fp32"])
        p_peak = max(r["median_gflops"] for r in out["curves"]["P_cores_only"]["fp32"])
        e_peak = max(r["median_gflops"] for r in out["curves"]["E_cores_only"]["fp32"])
        out["hybrid_conclusion"] = {
            "all_peak_fp32_gflops": all_peak,
            "P_peak_fp32_gflops": p_peak,
            "E_peak_fp32_gflops": e_peak,
            "adding_E_vs_P_only": (
                "helps" if all_peak > p_peak * 1.05
                else ("hurts" if all_peak < p_peak * 0.95 else "neutral")
            ),
        }
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
        if dt >= 0.12:
            return reps
        reps = max(reps * 2, int(reps * 0.15 / max(dt, 1e-6)))
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
        if dt >= 0.12:
            return reps
        reps = max(reps * 2, int(reps * 0.15 / max(dt, 1e-6)))
    return reps


def _sweep_fp32(kernels, a, b, n, reps, threads: int) -> dict[str, Any]:
    import ctypes

    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    set_omp_threads(threads)
    raw_t: list[float] = []
    raw_gflops: list[float] = []
    # ops: 2 per element (mul+add) * n * reps
    ops = 2.0 * n * reps
    fn = kernels.lib.dot_fp32_reps if threads > 1 else kernels.lib.dot_fp32_reps_serial
    fn(ap, bp, int(n), int(max(1, reps // 8)))  # warmup
    for _ in range(5):
        t0 = now()
        fn(ap, bp, int(n), int(reps))
        dt = now() - t0
        raw_t.append(dt)
        raw_gflops.append(ops / dt / 1e9)
    st = rep_stats(raw_gflops)
    return {
        "threads": threads,
        "raw_times_s": raw_t,
        "raw_gflops": raw_gflops,
        "median_gflops": st.median,
        "min_gflops": st.minimum,
        "max_gflops": st.maximum,
        "cv_pct": st.cv_pct,
        "noisy": st.noisy,
    }


def _sweep_i8(kernels, a, b, n, reps, threads: int) -> dict[str, Any]:
    import ctypes

    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    set_omp_threads(threads)
    raw_t: list[float] = []
    raw_gops: list[float] = []
    ops = 2.0 * n * reps
    fn = kernels.lib.dot_i8_reps if threads > 1 else kernels.lib.dot_i8_reps_serial
    fn(ap, bp, int(n), int(max(1, reps // 8)))
    for _ in range(5):
        t0 = now()
        fn(ap, bp, int(n), int(reps))
        dt = now() - t0
        raw_t.append(dt)
        raw_gops.append(ops / dt / 1e9)
    st = rep_stats(raw_gops)
    return {
        "threads": threads,
        "raw_times_s": raw_t,
        "raw_gops": raw_gops,
        "median_gops": st.median,
        "min_gops": st.minimum,
        "max_gops": st.maximum,
        "cv_pct": st.cv_pct,
        "noisy": st.noisy,
    }
