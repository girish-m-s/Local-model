"""BENCH 1 — STREAM-triad achievable memory bandwidth."""

from __future__ import annotations

import os
from typing import Any

from .native import NativeKernels, numpy_available
from .util import (
    effective_cores,
    effective_mem,
    now,
    pin_to_cpus,
    rep_stats,
    resolve_working_set_bytes,
    set_omp_threads,
    snapshot,
)


def run_bench1(profile: Any, kernels: NativeKernels) -> dict[str, Any]:
    cores = effective_cores(profile)
    eff_mem = effective_mem(profile)
    max_alloc = eff_mem // 2 if eff_mem else 0

    ws, ws_evid, suspect = resolve_working_set_bytes(profile)
    if max_alloc and ws > max_alloc:
        ws = max_alloc
        ws_evid += f"; capped to 50% effective_mem={max_alloc}"

    n = max(1, int(ws // (3 * 8)))
    actual_ws = n * 3 * 8
    scalar = 3.0

    has_np, np_ver = numpy_available()
    backend = (
        f"numpy {np_ver} allocation + OpenMP-C stream_triad"
        if has_np
        else f"mmap/ctypes allocation + OpenMP-C stream_triad ({kernels.lib_path})"
    )

    pre = snapshot("bench1_pre")
    results: dict[str, Any] = {
        "name": "BENCH 1 — ACHIEVABLE MEMORY BANDWIDTH (STREAM triad a=b+s*c)",
        "backend": backend,
        "backend_evidence": (
            f"numpy_available={has_np} ({np_ver}); compile: {kernels.compile_cmd}; "
            f"omp={kernels.omp}; lib={kernels.lib_path}"
        ),
        "working_set_bytes": actual_ws,
        "working_set_policy": ws_evid,
        "l3_suspect_fallback": suspect,
        "n_elements_per_array": n,
        "bytes_moved_per_iter": n * 8 * 3,
        "pre": pre,
        "thread_sweeps": [],
        "cache_proof": {},
        "saturation": {},
        "post": None,
        "notes": [],
    }

    l3 = None
    if profile.cache.lscpu_l3_bytes and isinstance(profile.cache.lscpu_l3_bytes.value, int):
        l3 = profile.cache.lscpu_l3_bytes.value
    elif isinstance(profile.cache.l3_bytes.value, int):
        l3 = profile.cache.l3_bytes.value
    results["cache_proof"] = {
        "working_set_bytes": actual_ws,
        "l3_bytes": l3,
        "ratio_ws_over_l3": (actual_ws / l3) if l3 else None,
        "evidence": (
            f"working_set={actual_ws} vs L3={l3}; ratio="
            + (f"{actual_ws/l3:.3f}x" if l3 else "n/a")
            + ("; L3 SUSPECT → forced 512 MiB policy" if suspect else "")
            + "; triad always writes `a`, so results cannot be pure register/L1 reuse"
        ),
    }

    import ctypes

    if has_np:
        import numpy as np

        rng0 = np.random.RandomState(0)
        rng1 = np.random.RandomState(1)
        a = np.empty(n, dtype=np.float64)
        b = rng0.rand(n).astype(np.float64)
        c = rng1.rand(n).astype(np.float64)
        a[:] = 0.0
        ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        touch = float(b[0] + c[0])
        discard = np.zeros(min(n, 1024 * 1024), dtype=np.float64)
    else:
        import mmap as mmap_mod

        nbytes = n * 8
        mb = mmap_mod.mmap(-1, nbytes * 3)
        a = (ctypes.c_double * n).from_buffer(memoryview(mb)[0:nbytes])
        b = (ctypes.c_double * n).from_buffer(memoryview(mb)[nbytes : 2 * nbytes])
        c = (ctypes.c_double * n).from_buffer(memoryview(mb)[2 * nbytes : 3 * nbytes])
        for i in range(0, n, max(1, n // 4096)):
            b[i] = 1.0
            c[i] = 2.0
        ap = ctypes.cast(a, ctypes.POINTER(ctypes.c_double))
        bp = ctypes.cast(b, ctypes.POINTER(ctypes.c_double))
        cp = ctypes.cast(c, ctypes.POINTER(ctypes.c_double))
        touch = float(b[0] + c[0])
        discard = None

    results["notes"].append(f"page-commit touch sentinel={touch}")

    bytes_per_iter = n * 8 * 3
    all_cpus = list(range(os.cpu_count() or cores))

    for t in range(1, cores + 1):
        set_omp_threads(t)
        aff_ev = pin_to_cpus(all_cpus)
        samples_gbs: list[float] = []
        raw_times: list[float] = []
        # warmup
        _triad(kernels, ap, bp, cp, scalar, n, t)
        for _rep in range(5):
            if discard is not None:
                discard += 1.0
            iters = 1
            t0 = now()
            _triad(kernels, ap, bp, cp, scalar, n, t)
            dt = now() - t0
            while dt < 0.20 and iters < 128:
                iters *= 2
                t0 = now()
                for _ in range(iters):
                    _triad(kernels, ap, bp, cp, scalar, n, t)
                dt = now() - t0
            gbs = (bytes_per_iter * iters) / dt / 1e9
            samples_gbs.append(gbs)
            raw_times.append(dt)
        stats = rep_stats(samples_gbs)
        results["thread_sweeps"].append(
            {
                "threads": t,
                "affinity_evidence": aff_ev,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "raw_times_s": raw_times,
                "raw_gbs": samples_gbs,
                "median_gbs": stats.median,
                "min_gbs": stats.minimum,
                "max_gbs": stats.maximum,
                "mean_gbs": stats.mean,
                "cv_pct": stats.cv_pct,
                "noisy": stats.noisy,
            }
        )

    sweeps = results["thread_sweeps"]
    meds = [s["median_gbs"] for s in sweeps]
    peak = max(meds) if meds else 0.0
    sat_threads = sweeps[0]["threads"] if sweeps else 1
    for s in sweeps:
        if peak > 0 and s["median_gbs"] >= 0.95 * peak:
            sat_threads = s["threads"]
            break
    sat_gbs = next(s["median_gbs"] for s in sweeps if s["threads"] == sat_threads)
    one = next(s["median_gbs"] for s in sweeps if s["threads"] == 1)
    results["saturation"] = {
        "saturation_thread_count": sat_threads,
        "bandwidth_saturated_median_gbs": sat_gbs,
        "bandwidth_1thread_median_gbs": one,
        "ratio_sat_over_1": (sat_gbs / one) if one else None,
        "peak_median_gbs": peak,
        "rule": "saturation = lowest thread count whose median >= 95% of peak median",
    }
    results["post"] = snapshot("bench1_post")
    return results


def _triad(kernels, ap, bp, cp, scalar, n, threads: int) -> None:
    set_omp_threads(threads)
    if threads == 1 or not kernels.omp:
        kernels.lib.stream_triad_serial(ap, bp, cp, float(scalar), int(n))
    else:
        kernels.lib.stream_triad(ap, bp, cp, float(scalar), int(n))
