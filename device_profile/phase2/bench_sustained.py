"""BENCH 3 — sustained vs burst thermal (180s at saturation threads)."""

from __future__ import annotations

import os
import time
from typing import Any

from .native import NativeKernels
from .util import (
    now,
    pin_to_cpus,
    sample_thermal_and_freq,
    set_omp_threads,
    snapshot,
)


def run_bench3(
    profile: Any,
    kernels: NativeKernels,
    *,
    sat_threads: int,
    n_fp32: int,
    reps_fp32: int,
    duration_s: float = 180.0,
    bucket_s: float = 10.0,
) -> dict[str, Any]:
    import ctypes
    import numpy as np

    cores = os.cpu_count() or sat_threads
    pin_to_cpus(list(range(cores)))
    set_omp_threads(sat_threads)

    rng = np.random.RandomState(7)
    a = rng.randn(n_fp32).astype(np.float32)
    b = rng.randn(n_fp32).astype(np.float32)
    _ = float(a[0] + b[0])
    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Power rails
    on_ac = None
    if profile.thermal_power is not None:
        on_ac = profile.thermal_power.on_ac_power.value

    pre = snapshot("bench3_pre")
    out: dict[str, Any] = {
        "name": "BENCH 3 — SUSTAINED VS BURST (thermal)",
        "sat_threads": sat_threads,
        "duration_s": duration_s,
        "bucket_s": bucket_s,
        "n_fp32": n_fp32,
        "reps_per_call": reps_fp32,
        "on_ac_power_at_profile": on_ac,
        "battery_dual_run": (
            "NOT DONE: no readable AC/battery sysfs on this host to gate dual runs; "
            "single run only"
        ),
        "pre": pre,
        "buckets": [],
        "post": None,
        "summary": {},
    }

    ops_per_call = 2.0 * n_fp32 * reps_fp32
    fn = kernels.lib.dot_fp32_reps if sat_threads > 1 else kernels.lib.dot_fp32_reps_serial

    t_end = time.time() + duration_s
    bucket_idx = 0
    bucket_start = now()
    wall_bucket_start = time.time()
    ops_in_bucket = 0.0
    calls_in_bucket = 0
    peak_gflops = 0.0
    time_to_throttle_s = None
    first_bucket_gflops = None

    while time.time() < t_end:
        t0 = now()
        fn(ap, bp, int(n_fp32), int(reps_fp32))
        dt = now() - t0
        ops_in_bucket += ops_per_call
        calls_in_bucket += 1
        # bucket boundary
        if (time.time() - wall_bucket_start) >= bucket_s:
            elapsed = now() - bucket_start
            gflops = ops_in_bucket / elapsed / 1e9 if elapsed > 0 else 0.0
            if first_bucket_gflops is None:
                first_bucket_gflops = gflops
            peak_gflops = max(peak_gflops, gflops)
            therm = sample_thermal_and_freq()
            out["buckets"].append(
                {
                    "bucket": bucket_idx,
                    "elapsed_s": elapsed,
                    "calls": calls_in_bucket,
                    "gflops": gflops,
                    "raw_last_call_s": dt,
                    "temps_C": therm["temps_C"],
                    "freqs_kHz": therm["freqs_kHz"],
                    "thermal_evidence": therm["evidence"],
                }
            )
            if (
                time_to_throttle_s is None
                and first_bucket_gflops
                and gflops < 0.90 * first_bucket_gflops
            ):
                time_to_throttle_s = bucket_idx * bucket_s
            bucket_idx += 1
            bucket_start = now()
            wall_bucket_start = time.time()
            ops_in_bucket = 0.0
            calls_in_bucket = 0

    # flush partial bucket if meaningful
    if calls_in_bucket > 0:
        elapsed = now() - bucket_start
        gflops = ops_in_bucket / elapsed / 1e9 if elapsed > 0 else 0.0
        peak_gflops = max(peak_gflops, gflops)
        therm = sample_thermal_and_freq()
        out["buckets"].append(
            {
                "bucket": bucket_idx,
                "elapsed_s": elapsed,
                "calls": calls_in_bucket,
                "gflops": gflops,
                "partial": True,
                "temps_C": therm["temps_C"],
                "freqs_kHz": therm["freqs_kHz"],
                "thermal_evidence": therm["evidence"],
            }
        )

    sustained = (
        statistics_median([b["gflops"] for b in out["buckets"][-3:]])
        if out["buckets"]
        else 0.0
    )
    peak = peak_gflops
    out["summary"] = {
        "peak_bucket_gflops": peak,
        "sustained_last3_median_gflops": sustained,
        "sustained_over_peak": (sustained / peak) if peak else None,
        "time_to_throttle_s": time_to_throttle_s,
        "throttle_rule": "first bucket with gflops < 90% of first bucket",
        "first_bucket_gflops": first_bucket_gflops,
    }
    out["post"] = snapshot("bench3_post")
    return out


def statistics_median(xs: list[float]) -> float:
    import statistics

    return statistics.median(xs) if xs else float("nan")
