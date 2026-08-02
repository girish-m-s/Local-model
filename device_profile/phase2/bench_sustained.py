"""BENCH 3 — sustained vs burst thermal under DRAM-bound STREAM load."""

from __future__ import annotations

import os
import time
from typing import Any

from .native import NativeKernels
from .util import (
    ensure_quiet_load,
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
    sat_threads: int | None,
    n_elements: int,
    reps: int,
    bytes_moved_per_iter: int,
    duration_s: float = 180.0,
    bucket_s: float = 10.0,
) -> dict[str, Any]:
    import ctypes

    from .native import numpy_available

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
        "kernel": "stream_triad_reps (same as BENCH 1)",
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
    }

    if not gate.ok:
        out["post"] = snapshot("bench3_post_blocked")
        return out

    if sat_threads is None or sat_threads < 1:
        out["status"] = "SKIPPED"
        out["skip_reason"] = (
            "BENCH 1 saturation_thread_count unavailable (curve INVALID or BLOCKED); "
            "cannot choose DRAM-bound thread count"
        )
        out["post"] = snapshot("bench3_post_skipped")
        return out

    if n_elements < 1 or reps < 1:
        out["status"] = "SKIPPED"
        out["skip_reason"] = "missing BENCH 1 geometry (n_elements/reps)"
        out["post"] = snapshot("bench3_post_skipped")
        return out

    # Probe thermal/cpufreq BEFORE burning 180s
    probe = sample_thermal_and_freq()
    out["thermal_probe"] = probe
    if probe.get("thermal_blind"):
        out["thermal_blind"] = True
        out["status"] = "THERMAL_BLIND"
        out["summary"] = {
            "flag": "THERMAL BLIND — RESULT NOT MEANINGFUL",
            "explanation": (
                "zero thermal zones and zero cpufreq scaling_cur_freq readable; "
                "sustained/peak conclusion SKIPPED (not reported as 0.96-style ratio)"
            ),
            "peak_bucket_gbs": None,
            "sustained_last3_median_gbs": None,
            "sustained_over_peak": None,
            "time_to_throttle_s": None,
            "conclusion_skipped": True,
        }
        out["notes"] = [
            "THERMAL BLIND — RESULT NOT MEANINGFUL; 180s STREAM run SKIPPED to save budget"
        ]
        out["post"] = snapshot("bench3_post_thermal_blind")
        return out

    cores = os.cpu_count() or sat_threads
    pin_to_cpus(list(range(cores)))
    set_omp_threads(sat_threads)

    has_np, _ = numpy_available()
    scalar = 3.0
    n = int(n_elements)
    if has_np:
        import numpy as np

        a = np.empty(n, dtype=np.float64)
        b = np.linspace(0.0, 1.0, n, dtype=np.float64)
        c = np.linspace(1.0, 2.0, n, dtype=np.float64)
        a[:] = 0.0
        ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        import mmap as mmap_mod

        nbytes = n * 8
        mb = mmap_mod.mmap(-1, nbytes * 3)
        a = (ctypes.c_double * n).from_buffer(memoryview(mb)[0:nbytes])
        b = (ctypes.c_double * n).from_buffer(memoryview(mb)[nbytes : 2 * nbytes])
        c = (ctypes.c_double * n).from_buffer(memoryview(mb)[2 * nbytes : 3 * nbytes])
        for i in range(n):
            b[i] = 1.0
            c[i] = 2.0
            a[i] = 0.0
        ap = ctypes.cast(a, ctypes.POINTER(ctypes.c_double))
        bp = ctypes.cast(b, ctypes.POINTER(ctypes.c_double))
        cp = ctypes.cast(c, ctypes.POINTER(ctypes.c_double))

    bytes_per_call = reps * bytes_moved_per_iter
    fn = (
        kernels.lib.stream_triad_reps
        if kernels.omp
        else kernels.lib.stream_triad_reps_serial
    )

    t_end = time.time() + duration_s
    bucket_idx = 0
    bucket_start = now()
    wall_bucket_start = time.time()
    bytes_in_bucket = 0.0
    calls_in_bucket = 0
    peak_gbs = 0.0
    time_to_throttle_s = None
    first_bucket_gbs = None

    while time.time() < t_end:
        t0 = now()
        fn(ap, bp, cp, float(scalar), int(n), int(reps))
        dt = now() - t0
        bytes_in_bucket += bytes_per_call
        calls_in_bucket += 1
        if (time.time() - wall_bucket_start) >= bucket_s:
            elapsed = now() - bucket_start
            gbs = bytes_in_bucket / elapsed / 1e9 if elapsed > 0 else 0.0
            if first_bucket_gbs is None:
                first_bucket_gbs = gbs
            peak_gbs = max(peak_gbs, gbs)
            therm = sample_thermal_and_freq()
            out["buckets"].append(
                {
                    "bucket": bucket_idx,
                    "elapsed_s": elapsed,
                    "calls": calls_in_bucket,
                    "gbs": gbs,
                    "raw_last_call_s": dt,
                    "temps_C": therm["temps_C"],
                    "freqs_kHz": therm["freqs_kHz"],
                    "thermal_evidence": therm["evidence"],
                }
            )
            if (
                time_to_throttle_s is None
                and first_bucket_gbs
                and gbs < 0.90 * first_bucket_gbs
            ):
                time_to_throttle_s = bucket_idx * bucket_s
            bucket_idx += 1
            bucket_start = now()
            wall_bucket_start = time.time()
            bytes_in_bucket = 0.0
            calls_in_bucket = 0

    if calls_in_bucket > 0:
        elapsed = now() - bucket_start
        gbs = bytes_in_bucket / elapsed / 1e9 if elapsed > 0 else 0.0
        peak_gbs = max(peak_gbs, gbs)
        therm = sample_thermal_and_freq()
        out["buckets"].append(
            {
                "bucket": bucket_idx,
                "elapsed_s": elapsed,
                "calls": calls_in_bucket,
                "gbs": gbs,
                "partial": True,
                "temps_C": therm["temps_C"],
                "freqs_kHz": therm["freqs_kHz"],
                "thermal_evidence": therm["evidence"],
            }
        )

    sustained = _median([b["gbs"] for b in out["buckets"][-3:]]) if out["buckets"] else 0.0
    out["summary"] = {
        "peak_bucket_gbs": peak_gbs,
        "sustained_last3_median_gbs": sustained,
        "sustained_over_peak": (sustained / peak_gbs) if peak_gbs else None,
        "time_to_throttle_s": time_to_throttle_s,
        "throttle_rule": "first bucket with gbs < 90% of first bucket",
        "first_bucket_gbs": first_bucket_gbs,
        "conclusion_skipped": False,
    }
    out["status"] = "OK"
    out["post"] = snapshot("bench3_post")
    return out


def _median(xs: list[float]) -> float:
    import statistics

    return statistics.median(xs) if xs else float("nan")
