"""BENCH 1 — STREAM-triad achievable memory bandwidth (Phase 2.1)."""

from __future__ import annotations

import os
from typing import Any

from .native import NativeKernels, numpy_available
from .util import (
    assert_gbs_reconstructs,
    check_superlinear,
    effective_cores,
    effective_mem,
    ensure_quiet_load,
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
    bytes_per_iter = n * 8 * 3  # a write + b read + c read

    has_np, np_ver = numpy_available()
    backend = (
        f"numpy {np_ver} allocation + OpenMP-C stream_triad_reps"
        if has_np
        else f"mmap/ctypes allocation + OpenMP-C stream_triad_reps ({kernels.lib_path})"
    )

    gate = ensure_quiet_load("bench1_pre")
    results: dict[str, Any] = {
        "name": "BENCH 1 — ACHIEVABLE MEMORY BANDWIDTH (STREAM triad a=b+s*c)",
        "status": gate.status,
        "backend": backend,
        "backend_evidence": (
            f"numpy_available={has_np} ({np_ver}); compile: {kernels.compile_cmd}; "
            f"omp={kernels.omp}; lib={kernels.lib_path}"
        ),
        "working_set_bytes": actual_ws,
        "working_set_policy": ws_evid,
        "l3_suspect_fallback": suspect,
        "n_elements_per_array": n,
        "bytes_moved_per_iter": bytes_per_iter,
        "pre": gate.snapshot,
        "quiet_gate": {
            "status": gate.status,
            "readings": gate.readings,
            "evidence": gate.evidence,
        },
        "thread_sweeps": [],
        "cache_proof": {},
        "calibration": {},
        "prefault": {},
        "saturation": {},
        "validity": {},
        "post": None,
        "notes": [],
        "FAIL": [],
    }

    if not gate.ok:
        results["notes"].append(
            "BLOCKED: loadavg gate failed; bench not run. readings printed in quiet_gate."
        )
        results["post"] = snapshot("bench1_post_blocked")
        return results

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

        a = np.empty(n, dtype=np.float64)
        b = np.empty(n, dtype=np.float64)
        c = np.empty(n, dtype=np.float64)
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
        ap = ctypes.cast(a, ctypes.POINTER(ctypes.c_double))
        bp = ctypes.cast(b, ctypes.POINTER(ctypes.c_double))
        cp = ctypes.cast(c, ctypes.POINTER(ctypes.c_double))

    # Pre-fault: full write pass BEFORE timing (not just sparse touch).
    pf0 = now()
    if has_np:
        import numpy as np

        b[:] = np.linspace(0.0, 1.0, n, dtype=np.float64)
        c[:] = np.linspace(1.0, 2.0, n, dtype=np.float64)
        a[:] = 0.0
    else:
        for i in range(n):
            b[i] = 1.0
            c[i] = 2.0
            a[i] = 0.0
    # Force one triad to commit `a` pages too
    _triad_reps(kernels, ap, bp, cp, scalar, n, 1, threads=1)
    pf_dt = now() - pf0
    results["prefault"] = {
        "wall_time_s": pf_dt,
        "evidence": (
            f"full write of a,b,c ({actual_ws} bytes) + 1 serial triad before timing; "
            f"prefault_wall_s={pf_dt}"
        ),
    }

    all_cpus = list(range(os.cpu_count() or cores))
    pin_to_cpus(all_cpus)

    # Calibration at MAX threads targeting ~1.0s; hold reps constant for all sweeps.
    cal = _calibrate_reps(kernels, ap, bp, cp, scalar, n, cores, bytes_per_iter)
    results["calibration"] = cal
    reps = cal["reps"]
    results["reps"] = reps
    results["notes"].append(
        f"reps held constant across thread sweep: {reps} "
        f"(calibrated at threads={cores} targeting ~1.0s)"
    )

    for t in range(1, cores + 1):
        set_omp_threads(t)
        aff_ev = pin_to_cpus(all_cpus)

        # Discarded warmup (first timed rep)
        t0 = now()
        _triad_reps(kernels, ap, bp, cp, scalar, n, reps, threads=t)
        warm_dt = now() - t0
        warm_gbs = (reps * bytes_per_iter) / warm_dt / 1e9 if warm_dt > 0 else float("nan")

        samples_gbs: list[float] = []
        raw_times: list[float] = []
        recon_checks: list[dict[str, Any]] = []
        for _rep in range(5):
            t0 = now()
            _triad_reps(kernels, ap, bp, cp, scalar, n, reps, threads=t)
            dt = now() - t0
            gbs = (reps * bytes_per_iter) / dt / 1e9 if dt > 0 else float("nan")
            chk = assert_gbs_reconstructs(gbs, reps, bytes_per_iter, dt)
            if not chk["ok"]:
                results["FAIL"].append(chk["message"])
                results["status"] = "FAIL"
            samples_gbs.append(gbs)
            raw_times.append(dt)
            recon_checks.append(chk)

        stats = rep_stats(samples_gbs)
        results["thread_sweeps"].append(
            {
                "threads": t,
                "reps": reps,
                "bytes_moved_per_iter": bytes_per_iter,
                "bytes_moved_per_timed_call": reps * bytes_per_iter,
                "affinity_evidence": aff_ev,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "discarded_warmup": {
                    "label": "DISCARDED",
                    "time_s": warm_dt,
                    "gbs": warm_gbs,
                    "reps": reps,
                },
                "raw_times_s": raw_times,
                "raw_gbs": samples_gbs,
                "reconstruction_checks": recon_checks,
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
    ts = [s["threads"] for s in sweeps]
    validity = check_superlinear(ts, meds)
    results["validity"] = validity

    if not validity["valid"]:
        results["saturation"] = {
            "saturation_thread_count": None,
            "bandwidth_saturated_median_gbs": None,
            "bandwidth_1thread_median_gbs": meds[0] if meds else None,
            "ratio_sat_over_1": None,
            "peak_median_gbs": max(meds) if meds else None,
            "rule": "NOT REPORTED — curve INVALID (see validity.flag)",
            "invalid": True,
            "flag": validity["flag"],
        }
    else:
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
            "invalid": False,
            "flag": "OK",
        }

    if results["FAIL"] and results["status"] != "BLOCKED":
        results["status"] = "FAIL"
    elif results["status"] not in ("BLOCKED", "FAIL"):
        results["status"] = "OK" if validity["valid"] else "INVALID"

    results["post"] = snapshot("bench1_post")
    return results


def _calibrate_reps(
    kernels, ap, bp, cp, scalar, n, max_threads, bytes_per_iter
) -> dict[str, Any]:
    """Pick reps at max threads targeting ~1.0s wall time."""
    set_omp_threads(max_threads)
    reps = 1
    history: list[dict[str, Any]] = []
    target = 1.0
    for _ in range(12):
        t0 = now()
        _triad_reps(kernels, ap, bp, cp, scalar, n, reps, threads=max_threads)
        dt = now() - t0
        history.append({"reps": reps, "time_s": dt})
        if dt >= 0.85:
            break
        # scale toward target
        scale = target / max(dt, 1e-6)
        reps = max(reps + 1, int(reps * scale))
    return {
        "target_s": target,
        "threads": max_threads,
        "reps": reps,
        "history": history,
        "evidence": (
            f"calibrated at OMP_NUM_THREADS={max_threads}; "
            f"final reps={reps}; last_time_s={history[-1]['time_s'] if history else None}; "
            f"bytes_moved_per_timed_call={reps * bytes_per_iter}"
        ),
    }


def _triad_reps(kernels, ap, bp, cp, scalar, n, reps: int, threads: int) -> None:
    set_omp_threads(threads)
    # Always use the OpenMP entry point when available, even for 1 thread, so
    # the 1-thread baseline shares the same compiled path as the multi-thread
    # sweeps (serial vs OpenMP codegen differences can fake SUPERLINEAR).
    if kernels.omp:
        kernels.lib.stream_triad_reps(ap, bp, cp, float(scalar), int(n), int(reps))
    else:
        kernels.lib.stream_triad_reps_serial(ap, bp, cp, float(scalar), int(n), int(reps))
