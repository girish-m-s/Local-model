"""BENCH 1 — STREAM triad bandwidth with verified OpenMP thread counts (Phase 2.4)."""

from __future__ import annotations

import os
from typing import Any, Optional

from .native import NativeKernels, numpy_available
from .subprocess_run import run_worker
from .util import (
    check_superlinear,
    effective_cores,
    effective_mem,
    ensure_quiet_load,
    resolve_working_set_bytes,
    snapshot,
)


# Phase 2.3 DIAG B asymptotic floor on this class of host (GB/s, 1 thread).
# Printed for audit alongside the 8×L3 policy; not a silent default for GB/s.
DIAG_B_FLOOR_GBS_NOTE = (
    "Phase 2.3 DIAG B floor ≈ 14.89 GB/s (1-thread asymptotic); "
    "8×L3 was ~2.5% above floor vs ~5.8% at 4×L3"
)


def run_bench1(profile: Any, kernels: NativeKernels | None = None) -> dict[str, Any]:
    _ = kernels  # parent process must not time OpenMP regions; workers compile fresh
    cores = effective_cores(profile)
    eff_mem = effective_mem(profile)
    max_alloc = eff_mem // 2 if eff_mem else 0

    ws, ws_evid, suspect, l3, ratio = resolve_working_set_bytes(profile)
    if max_alloc and ws > max_alloc:
        ws = max_alloc
        ws_evid += f"; capped to 50% effective_mem={max_alloc}"

    n = max(1, int(ws // (3 * 8)))
    actual_ws = n * 3 * 8
    bytes_per_iter = n * 8 * 3
    has_np, np_ver = numpy_available()

    gate = ensure_quiet_load("bench1_pre")
    results: dict[str, Any] = {
        "name": "BENCH 1 — ACHIEVABLE MEMORY BANDWIDTH (STREAM triad a=b+s*c)",
        "status": gate.status,
        "backend": (
            f"subprocess + omp_set_num_threads + stream_triad_reps_observed; "
            f"numpy_available={has_np} ({np_ver})"
        ),
        "working_set_bytes": actual_ws,
        "working_set_policy": ws_evid,
        "l3_reported_bytes": l3,
        "ratio_ws_over_l3": (actual_ws / l3) if l3 else ratio,
        "diag_b_floor_note": DIAG_B_FLOOR_GBS_NOTE,
        "l3_suspect": suspect,
        "n_elements_per_array": n,
        "bytes_moved_per_iter": bytes_per_iter,
        "pre": gate.snapshot,
        "quiet_gate": {
            "status": gate.status,
            "readings": gate.readings,
            "evidence": gate.evidence,
        },
        "thread_sweeps": [],
        "calibration": {},
        "saturation": {},
        "validity": {},
        "recommendation": {},
        "post": None,
        "notes": [],
        "FAIL": [],
    }

    if not gate.ok:
        results["notes"].append("BLOCKED: loadavg gate failed; bench not run.")
        results["post"] = snapshot("bench1_post_blocked")
        return results

    results["cache_proof"] = {
        "working_set_bytes": actual_ws,
        "l3_bytes": l3,
        "ratio_ws_over_l3": results["ratio_ws_over_l3"],
        "evidence": (
            f"Phase 2.4 policy: working_set = 8 * L3_reported; "
            f"ws={actual_ws} L3={l3} ratio={results['ratio_ws_over_l3']}; "
            f"{DIAG_B_FLOOR_GBS_NOTE}"
        ),
    }

    # Calibrate reps at max threads in a fresh subprocess (~1.0s)
    reps, cal = _calibrate_reps(n, bytes_per_iter, cores)
    results["reps"] = reps
    results["calibration"] = cal

    all_cpus = list(range(os.cpu_count() or cores))
    for t in range(1, cores + 1):
        row = run_worker(
            {
                "mode": "stream_point",
                "n": n,
                "reps": reps,
                "bytes_moved_per_iter": bytes_per_iter,
                "scalar": 3.0,
                "requested_threads": t,
                "affinity_cpus": all_cpus[: max(t, 1)],
                "omp_proc_bind": "close",
                "omp_places": "cores",
            }
        )
        if row.get("status") != "OK" or not row.get("thread_count_applied"):
            results["FAIL"].append(
                f"threads={t}: THREAD COUNT NOT APPLIED "
                f"(req={t} actual={row.get('omp_num_threads_actual')}); INVALID"
            )
            results["thread_sweeps"].append(
                {
                    "threads": t,
                    "status": "INVALID",
                    "thread_flag": row.get("thread_flag"),
                    "pid": row.get("pid"),
                    "requested_threads": t,
                    "omp_num_threads_actual": row.get("omp_num_threads_actual"),
                    "n_distinct_cpus": row.get("n_distinct_cpus"),
                    "cpu_ids": row.get("cpu_ids"),
                    "error": row.get("error"),
                }
            )
            continue

        results["thread_sweeps"].append(
            {
                "threads": t,
                "status": "OK",
                "pid": row["pid"],
                "reps": reps,
                "bytes_moved_per_iter": bytes_per_iter,
                "bytes_moved_per_timed_call": reps * bytes_per_iter,
                "requested_threads": t,
                "env_OMP_NUM_THREADS": row.get("env_OMP_NUM_THREADS"),
                "omp_num_threads_actual": row["omp_num_threads_actual"],
                "omp_max_threads": row.get("omp_max_threads"),
                "n_distinct_cpus": row["n_distinct_cpus"],
                "cpu_ids": row.get("cpu_ids"),
                "thread_flag": row.get("thread_flag"),
                "thread_count_applied": True,
                "discarded_warmup": row.get("discarded_warmup"),
                "raw_times_s": row["raw_times_s"],
                "raw_gbs": row["raw_gbs"],
                "reconstruction_checks": row.get("reconstruction_checks"),
                "median_gbs": row["median_gbs"],
                "min_gbs": row["min_gbs"],
                "max_gbs": row["max_gbs"],
                "mean_gbs": row["mean_gbs"],
                "cv_pct": row["cv_pct"],
                "noisy": row["noisy"],
                "affinity_cpus": all_cpus[: max(t, 1)],
            }
        )
        results["FAIL"].extend(row.get("FAIL") or [])

    ok_sweeps = [s for s in results["thread_sweeps"] if s.get("status") == "OK"]
    if not ok_sweeps:
        results["status"] = "INVALID"
        results["saturation"] = {
            "invalid": True,
            "flag": "no verified thread sweeps",
            "saturation_thread_count": None,
        }
        results["post"] = snapshot("bench1_post")
        return results

    ts = [s["threads"] for s in ok_sweeps]
    meds = [s["median_gbs"] for s in ok_sweeps]
    one = next(s["median_gbs"] for s in ok_sweeps if s["threads"] == 1)
    validity = check_superlinear(ts, meds, baseline_gbs=one, baseline_label="verified_1thread")
    results["validity"] = validity

    # Efficiency vs 1-thread: eff(n) = median(n) / (n * median(1))
    eff_rows = []
    for s in ok_sweeps:
        t = s["threads"]
        eff = (s["median_gbs"] / (t * one)) if (one and t) else None
        eff_rows.append({"threads": t, "median_gbs": s["median_gbs"], "efficiency": eff})
    results["efficiency_curve"] = eff_rows

    max_t = max(ts)
    max_eff = next(e["efficiency"] for e in eff_rows if e["threads"] == max_t)
    peak = max(meds)
    if max_eff is not None and max_eff > 0.90:
        results["saturation"] = {
            "invalid": False,
            "no_saturation_observed": True,
            "flag": "NO SATURATION OBSERVED — USE ALL PHYSICAL CORES",
            "saturation_thread_count": max_t,
            "bandwidth_saturated_median_gbs": next(
                s["median_gbs"] for s in ok_sweeps if s["threads"] == max_t
            ),
            "bandwidth_1thread_median_gbs": one,
            "ratio_sat_over_1": (
                next(s["median_gbs"] for s in ok_sweeps if s["threads"] == max_t) / one
            ),
            "peak_median_gbs": peak,
            "efficiency_at_max_cores": max_eff,
            "rule": (
                "if efficiency at max cores > 90%, do not invent a knee; "
                "recommend all physical/effective cores"
            ),
        }
        rec = max_t
        rec_ev = (
            f"NO SATURATION OBSERVED — USE ALL PHYSICAL CORES "
            f"(eff@{max_t}={max_eff:.3f} > 0.90); recommended_thread_count={rec}"
        )
    else:
        # Lowest thread count achieving >=95% of peak median
        sat_t = ok_sweeps[0]["threads"]
        for s in ok_sweeps:
            if peak > 0 and s["median_gbs"] >= 0.95 * peak:
                sat_t = s["threads"]
                break
        sat_gbs = next(s["median_gbs"] for s in ok_sweeps if s["threads"] == sat_t)
        results["saturation"] = {
            "invalid": bool(validity.get("superlinear")),
            "no_saturation_observed": False,
            "flag": validity.get("flag") if validity.get("superlinear") else "OK",
            "saturation_thread_count": None if validity.get("superlinear") else sat_t,
            "bandwidth_saturated_median_gbs": None if validity.get("superlinear") else sat_gbs,
            "bandwidth_1thread_median_gbs": one,
            "ratio_sat_over_1": (sat_gbs / one) if one and not validity.get("superlinear") else None,
            "peak_median_gbs": peak,
            "efficiency_at_max_cores": max_eff,
            "rule": "lowest thread count with median >= 95% of peak (when efficiency saturates)",
        }
        rec = None if validity.get("superlinear") else sat_t
        rec_ev = (
            f"BENCH 1 bandwidth saturation thread count={rec} "
            f"(eff@max={max_eff}); not from deleted BENCH 2"
        )

    # Max-efficiency thread count: among OK sweeps, maximize efficiency then GB/s
    best = max(
        ok_sweeps,
        key=lambda s: (
            (s["median_gbs"] / (s["threads"] * one)) if one else 0.0,
            s["median_gbs"],
        ),
    )
    # For sustained: prefer recommended (all cores if no sat) else sat
    max_eff_threads = rec if rec is not None else best["threads"]
    results["recommendation"] = {
        "recommended_thread_count": rec,
        "recommended_thread_count_evidence": rec_ev,
        "max_efficiency_thread_count": max_eff_threads,
        "max_efficiency_thread_count_evidence": (
            f"from BENCH 1 verified curve; no_saturation="
            f"{results['saturation'].get('no_saturation_observed')}"
        ),
    }

    if results["FAIL"] or any(s.get("status") == "INVALID" for s in results["thread_sweeps"]):
        results["status"] = "INVALID"
    elif validity.get("superlinear"):
        results["status"] = "INVALID"
    else:
        results["status"] = "OK"

    results["post"] = snapshot("bench1_post")
    return results


def _calibrate_reps(n: int, bpi: int, max_threads: int) -> tuple[int, dict[str, Any]]:
    reps = 1
    hist = []
    for _ in range(10):
        row = run_worker(
            {
                "mode": "stream_point",
                "n": n,
                "reps": reps,
                "bytes_moved_per_iter": bpi,
                "scalar": 3.0,
                "requested_threads": max_threads,
                "affinity_cpus": list(range(max_threads)),
                "omp_proc_bind": "close",
                "omp_places": "cores",
            }
        )
        times = row.get("raw_times_s") or []
        dt = float(sum(times) / len(times)) if times else 0.0
        hist.append(
            {
                "reps": reps,
                "mean_time_s": dt,
                "pid": row.get("pid"),
                "actual_threads": row.get("omp_num_threads_actual"),
            }
        )
        if row.get("status") != "OK":
            break
        if dt >= 0.85:
            break
        reps = max(reps + 1, int(reps * 1.0 / max(dt, 1e-6)))
    return reps, {
        "target_s": 1.0,
        "threads": max_threads,
        "reps": reps,
        "history": hist,
        "evidence": f"calibrated in fresh subprocess at threads={max_threads}; reps={reps}",
    }
