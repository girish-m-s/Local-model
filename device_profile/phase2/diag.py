"""Phase 2.2 diagnostics — DIAG A (primary), DIAG B, DIAG C. No full bench sweep."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Optional

from device_profile.detect import detect_device_profile

from .native import NativeKernels, build_kernels, build_serial_kernels
from .util import (
    GIB,
    assert_gbs_reconstructs,
    effective_cores,
    effective_mem,
    evaluate_curve_validity,
    now,
    pin_to_cpus,
    read_steal_jiffies,
    rep_stats,
    resolve_working_set_bytes,
    set_omp_env,
    set_omp_threads,
)


def run_diag() -> dict[str, Any]:
    profile = detect_device_profile()
    omp_k = build_kernels()
    serial_k = build_serial_kernels()
    cores = effective_cores(profile)
    emem = effective_mem(profile)

    l3 = _l3_bytes(profile)
    # DIAG buffers: use the new floor policy for a DRAM-escaping WS when possible,
    # but keep within 50% effective mem.
    ws, ws_ev, suspect = resolve_working_set_bytes(profile)
    max_alloc = emem // 2 if emem else ws
    if max_alloc and ws > max_alloc:
        ws = max_alloc
        ws_ev += f"; capped to 50% effective_mem={max_alloc}"

    n = max(1, int(ws // (3 * 8)))
    actual_ws = n * 3 * 8
    bytes_per_iter = n * 8 * 3
    scalar = 3.0

    import ctypes
    import numpy as np

    a = np.empty(n, dtype=np.float64)
    b = np.linspace(0.0, 1.0, n, dtype=np.float64)
    c = np.linspace(1.0, 2.0, n, dtype=np.float64)
    a[:] = 0.0
    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Prefault
    pf0 = now()
    a[:] = 0.0
    b[:] = b
    c[:] = c
    serial_k.lib.stream_triad_reps_pure_serial(ap, bp, cp, float(scalar), int(n), 1)
    pf_dt = now() - pf0

    # Calibrate reps on A1 (pure serial) targeting ~1.0s — hold constant for all.
    reps, cal_hist = _calibrate_reps_serial(serial_k, ap, bp, cp, scalar, n)

    diag_a = run_diag_a(
        omp_k=omp_k,
        serial_k=serial_k,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=reps,
        bytes_per_iter=bytes_per_iter,
        cores=cores,
    )

    diag_b = run_diag_b(
        omp_k=omp_k,
        profile=profile,
        l3=l3,
        emem=emem,
        cores=cores,
        fixed_threads=max(1, diag_a.get("recommended_threads_for_sweep") or 2),
    )

    diag_c = run_diag_c(
        thread_medians_omp=diag_a.get("omp_thread_probe") or [],
        a1_median_gbs=diag_a["configs"]["A1"]["median_gbs"],
        cores=cores,
    )

    conclusion = diag_a["conclusion"]

    return {
        "phase": "2.2",
        "host_class": (
            profile.host_class.value
            if profile.host_class.is_detected()
            else str(profile.host_class.value)
        ),
        "host_class_evidence": profile.host_class.evidence,
        "effective_cores": cores,
        "effective_mem_bytes": emem,
        "l3_reported_bytes": l3,
        "working_set_bytes": actual_ws,
        "working_set_policy": ws_ev,
        "l3_suspect": suspect,
        "bytes_moved_per_iter": bytes_per_iter,
        "reps": reps,
        "calibration_history": cal_hist,
        "prefault_wall_s": pf_dt,
        "omp_compile": omp_k.compile_cmd,
        "serial_compile": serial_k.compile_cmd,
        "diag_a": diag_a,
        "diag_b": diag_b,
        "diag_c": diag_c,
        "primary_deliverable": {
            "name": "DIAG A conclusion",
            "conclusion": conclusion,
        },
        "note": (
            "Phase 2.2 is a debugging phase. Full BENCH 1–4 sweep is NOT re-run "
            "until the 1-thread baseline is explained. DIAG A is the deliverable."
        ),
    }


def run_diag_a(
    *,
    omp_k: NativeKernels,
    serial_k: NativeKernels,
    ap,
    bp,
    cp,
    scalar: float,
    n: int,
    reps: int,
    bytes_per_iter: int,
    cores: int,
) -> dict[str, Any]:
    all_cpus = list(range(os.cpu_count() or cores))

    configs: list[tuple[str, str, Callable[[], None]]] = []

    def setup_a1() -> None:
        # Pure serial: clear OMP bind knobs; affinity = all (neutral)
        set_omp_env(threads=None, proc_bind=None, places=None)
        pin_to_cpus(all_cpus)

    def setup_a2() -> None:
        set_omp_env(threads=1, proc_bind=None, places=None)
        pin_to_cpus(all_cpus)

    def setup_a3() -> None:
        set_omp_env(threads=1, proc_bind="close", places="cores")
        pin_to_cpus(all_cpus)

    def setup_a4() -> None:
        set_omp_env(threads=1, proc_bind=None, places=None)
        pin_to_cpus([0])

    def setup_a5() -> None:
        set_omp_env(threads=2, proc_bind="close", places="cores")
        pin_to_cpus([0, 1] if (os.cpu_count() or 0) >= 2 else [0])

    configs = [
        ("A1", "pure serial C, compiled WITHOUT -fopenmp, no pragma", setup_a1),
        ("A2", "-fopenmp, OMP_NUM_THREADS=1, affinity=all cores (current)", setup_a2),
        ("A3", "-fopenmp, OMP_NUM_THREADS=1, OMP_PROC_BIND=close, OMP_PLACES=cores", setup_a3),
        ("A4", "-fopenmp, OMP_NUM_THREADS=1, sched_setaffinity CPU 0 only", setup_a4),
        ("A5", "-fopenmp, OMP_NUM_THREADS=2, affinity=CPUs 0,1, PROC_BIND=close", setup_a5),
    ]

    results: dict[str, Any] = {}
    for key, desc, setup in configs:
        setup()
        aff = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        steal0, steal0_ev = read_steal_jiffies()
        # DISCARDED warmup
        _run_triad(key, omp_k, serial_k, ap, bp, cp, scalar, n, reps)
        warm_t0 = now()
        _run_triad(key, omp_k, serial_k, ap, bp, cp, scalar, n, reps)
        warm_dt = now() - warm_t0
        warm_gbs = (reps * bytes_per_iter) / warm_dt / 1e9

        raw_t: list[float] = []
        raw_gbs: list[float] = []
        checks: list[dict[str, Any]] = []
        for _ in range(5):
            t0 = now()
            _run_triad(key, omp_k, serial_k, ap, bp, cp, scalar, n, reps)
            dt = now() - t0
            gbs = (reps * bytes_per_iter) / dt / 1e9
            chk = assert_gbs_reconstructs(gbs, reps, bytes_per_iter, dt)
            raw_t.append(dt)
            raw_gbs.append(gbs)
            checks.append(chk)
        steal1, steal1_ev = read_steal_jiffies()
        st = rep_stats(raw_gbs)
        results[key] = {
            "description": desc,
            "reps": reps,
            "bytes_moved_per_iter": bytes_per_iter,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
            "affinity": aff,
            "discarded_warmup": {"label": "DISCARDED", "time_s": warm_dt, "gbs": warm_gbs},
            "raw_times_s": raw_t,
            "raw_gbs": raw_gbs,
            "reconstruction_checks": checks,
            "median_gbs": st.median,
            "min_gbs": st.minimum,
            "max_gbs": st.maximum,
            "cv_pct": st.cv_pct,
            "steal_before": steal0,
            "steal_after": steal1,
            "steal_delta": (steal1 - steal0) if (steal0 is not None and steal1 is not None) else None,
            "steal_evidence": [steal0_ev, steal1_ev],
            "FAIL": [c["message"] for c in checks if not c["ok"]],
        }

    a1 = results["A1"]["median_gbs"]
    for key, row in results.items():
        row["ratio_to_A1"] = (row["median_gbs"] / a1) if a1 and a1 > 0 else None

    conclusion = _conclude_diag_a(results)

    # Small OMP thread probe (1..min(4,cores)) using A2-style OpenMP path for DIAG C
    probe = []
    for t in range(1, min(4, cores) + 1):
        set_omp_env(threads=t, proc_bind="close", places="cores")
        pin_to_cpus(list(range(t)))
        samples = []
        _run_triad("A2", omp_k, serial_k, ap, bp, cp, scalar, n, reps)
        for _ in range(3):
            t0 = now()
            _run_triad("A2", omp_k, serial_k, ap, bp, cp, scalar, n, reps)
            dt = now() - t0
            samples.append((reps * bytes_per_iter) / dt / 1e9)
        probe.append({"threads": t, "median_gbs": float(rep_stats(samples).median)})

    # Restore env
    set_omp_env(threads=None, proc_bind=None, places=None)
    pin_to_cpus(all_cpus)

    return {
        "reps": reps,
        "bytes_moved_per_iter": bytes_per_iter,
        "configs": results,
        "conclusion": conclusion,
        "omp_thread_probe": probe,
        "recommended_threads_for_sweep": 2 if cores >= 2 else 1,
    }


def run_diag_b(
    *,
    omp_k: NativeKernels,
    profile: Any,
    l3: Optional[int],
    emem: int,
    cores: int,
    fixed_threads: int,
) -> dict[str, Any]:
    """Working-set sweep over multiples of reported L3 at fixed thread count."""
    if not l3 or l3 <= 0:
        return {
            "status": "SKIPPED",
            "reason": "L3_reported undetected; cannot sweep ws/L3 ratios",
        }

    import ctypes
    import numpy as np

    ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    max_alloc = emem // 2 if emem else int(8 * l3)
    set_omp_env(threads=fixed_threads, proc_bind="close", places="cores")
    pin_to_cpus(list(range(min(fixed_threads, os.cpu_count() or fixed_threads))))

    points: list[dict[str, Any]] = []
    for ratio in ratios:
        ws = int(ratio * l3)
        if ws > max_alloc:
            points.append(
                {
                    "ratio_ws_over_l3": ratio,
                    "working_set_bytes": ws,
                    "status": "SKIPPED",
                    "reason": f"ws={ws} > 50% effective_mem={max_alloc}",
                }
            )
            continue
        # triad needs 3 arrays
        n = max(1, ws // (3 * 8))
        actual = n * 3 * 8
        bytes_per_iter = n * 8 * 3
        a = np.empty(n, dtype=np.float64)
        b = np.linspace(0.0, 1.0, n, dtype=np.float64)
        c = np.linspace(1.0, 2.0, n, dtype=np.float64)
        a[:] = 0.0
        ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # calibrate short reps targeting ~0.4s at this size
        reps = _calibrate_reps_omp(omp_k, ap, bp, cp, 3.0, n, fixed_threads, target=0.4)
        # warmup discarded
        omp_k.lib.stream_triad_reps(ap, bp, cp, 3.0, int(n), int(reps))
        samples = []
        times = []
        for _ in range(5):
            t0 = now()
            omp_k.lib.stream_triad_reps(ap, bp, cp, 3.0, int(n), int(reps))
            dt = now() - t0
            gbs = (reps * bytes_per_iter) / dt / 1e9
            chk = assert_gbs_reconstructs(gbs, reps, bytes_per_iter, dt)
            if not chk["ok"]:
                raise RuntimeError(chk["message"])
            samples.append(gbs)
            times.append(dt)
        st = rep_stats(samples)
        points.append(
            {
                "ratio_ws_over_l3": ratio,
                "working_set_bytes": actual,
                "reps": reps,
                "bytes_moved_per_iter": bytes_per_iter,
                "raw_gbs": samples,
                "raw_times_s": times,
                "median_gbs": st.median,
                "cv_pct": st.cv_pct,
                "status": "OK",
            }
        )

    escape = _find_escape_point(points)
    new_floor = max(4 * l3, GIB)
    justification = (
        f"Phase 2.2 policy: working_set = max(4 * L3_reported, 1 GiB) = "
        f"max(4*{l3}, {GIB}) = {new_floor}. "
        f"Escape point from sweep: {escape}. "
        f"Prior SUSPECT fallback of 512 MiB left ratio_ws_over_l3≈1.6 "
        f"(never escaped L3); 1 GiB floor forces ≥1 GiB even when L3 is SUSPECT."
    )
    return {
        "status": "OK",
        "l3_reported_bytes": l3,
        "fixed_threads": fixed_threads,
        "points": points,
        "escape_point": escape,
        "new_working_set_policy_bytes": new_floor,
        "justification": justification,
    }


def run_diag_c(
    *,
    thread_medians_omp: list[dict[str, Any]],
    a1_median_gbs: float,
    cores: int,
) -> dict[str, Any]:
    ts = [p["threads"] for p in thread_medians_omp]
    meds = [p["median_gbs"] for p in thread_medians_omp]
    validity = evaluate_curve_validity(
        ts, meds, serial_baseline_gbs=a1_median_gbs, cores=cores
    )
    return {
        "omp_probe": thread_medians_omp,
        "serial_baseline_A1_gbs": a1_median_gbs,
        "validity": validity,
        "note": (
            "Default efficiency baseline is A1 (pure serial, no OpenMP). "
            "If A1 is a slow codegen outlier vs OpenMP@1, DIAG C discounts "
            "SUPERLINEAR-vs-A1 and re-checks vs OMP_NUM_THREADS=1. "
            "Absolute per-core bounds 1..60 GB/s are heuristic."
        ),
    }


def _run_triad(key, omp_k, serial_k, ap, bp, cp, scalar, n, reps) -> None:
    if key == "A1":
        serial_k.lib.stream_triad_reps_pure_serial(
            ap, bp, cp, float(scalar), int(n), int(reps)
        )
    else:
        omp_k.lib.stream_triad_reps(ap, bp, cp, float(scalar), int(n), int(reps))


def _calibrate_reps_serial(serial_k, ap, bp, cp, scalar, n) -> tuple[int, list[dict]]:
    reps = 1
    hist = []
    for _ in range(12):
        t0 = now()
        serial_k.lib.stream_triad_reps_pure_serial(
            ap, bp, cp, float(scalar), int(n), int(reps)
        )
        dt = now() - t0
        hist.append({"reps": reps, "time_s": dt})
        if dt >= 0.85:
            break
        reps = max(reps + 1, int(reps * 1.0 / max(dt, 1e-6)))
    return reps, hist


def _calibrate_reps_omp(omp_k, ap, bp, cp, scalar, n, threads, target=0.4) -> int:
    set_omp_threads(threads)
    reps = 1
    for _ in range(10):
        t0 = now()
        omp_k.lib.stream_triad_reps(ap, bp, cp, float(scalar), int(n), int(reps))
        dt = now() - t0
        if dt >= target * 0.85:
            return reps
        reps = max(reps + 1, int(reps * target / max(dt, 1e-6)))
    return reps


def _conclude_diag_a(results: dict[str, Any]) -> str:
    a1 = results["A1"]["median_gbs"]
    a2 = results["A2"]["median_gbs"]
    a3 = results["A3"]["median_gbs"]
    a4 = results["A4"]["median_gbs"]
    a5 = results["A5"]["median_gbs"]
    if not a1 or a1 <= 0:
        return "INCONCLUSIVE: A1 baseline median is non-positive."

    r2 = a2 / a1
    r5 = a5 / a1

    # If A1 ~= A2 (within 15%), OpenMP@1 is exonerated.
    if 0.85 <= r2 <= 1.15:
        if r5 > 1.5:
            return (
                f"A1≈A2 ({a1:.2f}≈{a2:.2f} GB/s): OpenMP path exonerated; "
                f"A5/A1={r5:.2f} shows multi-thread gain — 4x gap cause is elsewhere "
                f"(not OMP_NUM_THREADS=1 alone)."
            )
        return (
            f"A1≈A2 ({a1:.2f}≈{a2:.2f} GB/s): OpenMP path exonerated; "
            f"cause of any prior 4x gap is elsewhere — not forcing a conclusion onto OpenMP."
        )

    # A2 much lower than A1 → OpenMP@1 penalty
    if r2 < 0.5:
        # Check if bind/affinity recover it
        if a3 / a1 >= 0.85 or a4 / a1 >= 0.85:
            which = "A3 (PROC_BIND=close)" if a3 >= a4 else "A4 (affinity CPU0)"
            return (
                f"A2/A1={r2:.2f} shows OpenMP@1 ~{1/r2:.1f}x slow vs pure serial A1; "
                f"{which} recovers toward A1 — affinity/bind explains the 4x gap."
            )
        if a5 / max(a2, 1e-9) > 3:
            return (
                f"A2/A1={r2:.2f} (OpenMP@1 ~{1/r2:.1f}x slow vs A1); "
                f"A5/A2={a5/a2:.2f} recovers at 2 threads — "
                f"OMP_NUM_THREADS=1 OpenMP path (not pure serial) explains the 4x gap."
            )
        return (
            f"A2/A1={r2:.2f}: OpenMP OMP_NUM_THREADS=1 is ~{1/r2:.1f}x slower than "
            f"pure-serial A1; A3/A1={a3/a1:.2f} A4/A1={a4/a1:.2f} A5/A1={r5:.2f} — "
            f"OpenMP@1 runtime overhead/scheduling explains the 4x gap."
        )

    # A2 higher than A1 — serial codegen is the slow outlier
    if r2 > 1.3:
        a3r, a4r = a3 / a1, a4 / a1
        flat = max(a2, a3, a4, a5) / min(a2, a3, a4, a5)
        return (
            f"A2/A1={r2:.2f} (A1={a1:.2f}, A2={a2:.2f} GB/s): pure-serial A1 is the "
            f"~{r2:.1f}x SLOW outlier; A2≈A3≈A4≈A5 (spread={flat:.3f}) so "
            f"OMP_NUM_THREADS=1 / affinity / PROC_BIND do NOT explain the gap — "
            f"OpenMP-compiled triad (likely better auto-vectorization) does. "
            f"Prior BENCH 1-thread ~4x-low signature matches the slow serial path, "
            f"not OMP_NUM_THREADS=1. Use OpenMP@1 (A2) as the operational baseline, "
            f"not A1. A3/A1={a3r:.2f} A4/A1={a4r:.2f} A5/A1={r5:.2f}."
        )

    return (
        f"INCONCLUSIVE within thresholds: A1={a1:.2f} A2={a2:.2f} (r={r2:.2f}) "
        f"A3={a3:.2f} A4={a4:.2f} A5={a5:.2f} (r={r5:.2f}); "
        f"not forcing a conclusion."
    )


def _find_escape_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [p for p in points if p.get("status") == "OK"]
    if len(ok) < 2:
        return {"ratio": None, "note": "insufficient points"}
    # Escape = smallest ratio whose median is within 5% of the asymptotic
    # (largest-size) GB/s. Ignores early plateaus inside L3/L2.
    asym = ok[-1]["median_gbs"]
    peak = max(p["median_gbs"] for p in ok)
    if asym <= 0:
        return {"ratio_ws_over_l3": None, "note": "non-positive asymptotic GB/s"}
    escape = ok[-1]
    for p in ok:
        if p["median_gbs"] <= asym * 1.05:
            escape = p
            break
    return {
        "ratio_ws_over_l3": escape["ratio_ws_over_l3"],
        "median_gbs": escape["median_gbs"],
        "asymptotic_gbs": asym,
        "peak_gbs": peak,
        "note": (
            f"escape = smallest ratio with GB/s ≤ 1.05 * asymptotic "
            f"(largest ratio={ok[-1]['ratio_ws_over_l3']} → {asym:.3f} GB/s); "
            f"peak was {peak:.3f} GB/s at small WS (cache). "
            f"Chosen ratio={escape['ratio_ws_over_l3']} → {escape['median_gbs']:.3f} GB/s"
        ),
    }


def _l3_bytes(profile: Any) -> Optional[int]:
    if profile.cache.lscpu_l3_bytes and isinstance(profile.cache.lscpu_l3_bytes.value, int):
        return profile.cache.lscpu_l3_bytes.value
    if isinstance(profile.cache.l3_bytes.value, int):
        return profile.cache.l3_bytes.value
    return None


def format_diag_report(bundle: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== PHASE 2.2 BASELINE DIAGNOSTIC REPORT ===")
    lines.append(f"HOST CLASS: {bundle.get('host_class')}")
    lines.append(f"  <- {bundle.get('host_class_evidence')}")
    lines.append(f"effective_cores: {bundle.get('effective_cores')}")
    lines.append(f"effective_mem_bytes: {bundle.get('effective_mem_bytes')}")
    lines.append(f"l3_reported_bytes: {bundle.get('l3_reported_bytes')}")
    lines.append(f"working_set_bytes: {bundle.get('working_set_bytes')}")
    lines.append(f"  <- {bundle.get('working_set_policy')}")
    lines.append(f"reps (constant): {bundle.get('reps')}")
    lines.append(f"prefault_wall_s: {bundle.get('prefault_wall_s')}")
    lines.append(f"omp_compile: {bundle.get('omp_compile')}")
    lines.append(f"serial_compile: {bundle.get('serial_compile')}")
    lines.append(f"note: {bundle.get('note')}")
    lines.append("")

    a = bundle["diag_a"]
    lines.append("=== DIAG A — ISOLATE THE 1-THREAD PATH ===")
    lines.append(f"reps={a['reps']} bytes_moved_per_iter={a['bytes_moved_per_iter']}")
    lines.append(
        f"{'cfg':<4} {'median_GB/s':>12} {'ratio_A1':>10} {'CV%':>8} "
        f"{'stealΔ':>8}  description"
    )
    for key in ("A1", "A2", "A3", "A4", "A5"):
        r = a["configs"][key]
        lines.append(
            f"{key:<4} {r['median_gbs']:12.4f} {r['ratio_to_A1']:10.4f} "
            f"{r['cv_pct']:8.2f} {str(r['steal_delta']):>8}  {r['description']}"
        )
        disc = r["discarded_warmup"]
        lines.append(
            f"     DISCARDED warmup gbs={disc['gbs']:.4f} time_s={disc['time_s']}"
        )
        lines.append(f"     raw_gbs: {r['raw_gbs']}")
        lines.append(f"     raw_times_s: {r['raw_times_s']}")
        lines.append(
            f"     env: OMP_NUM_THREADS={r['omp_num_threads']} "
            f"OMP_PROC_BIND={r['omp_proc_bind']} OMP_PLACES={r['omp_places']} "
            f"affinity={r['affinity']}"
        )
        for chk in r["reconstruction_checks"]:
            tag = "OK" if chk["ok"] else "FAIL"
            lines.append(
                f"     reconstruct[{tag}]: reported={chk['reported_gbs']} "
                f"expected={chk['expected_gbs']} rel_err={chk['rel_err']}"
            )
        for ev in r["steal_evidence"]:
            lines.append(f"     <- steal: {ev}")
        for fail in r["FAIL"]:
            lines.append(f"     FAIL: {fail}")
    lines.append("")
    lines.append(f"DIAG A CONCLUSION: {a['conclusion']}")
    lines.append("")

    b = bundle["diag_b"]
    lines.append("=== DIAG B — WORKING SET SIZE SWEEP ===")
    if b.get("status") != "OK":
        lines.append(f"status: {b.get('status')} <- {b.get('reason')}")
    else:
        lines.append(f"fixed_threads: {b['fixed_threads']} L3={b['l3_reported_bytes']}")
        for p in b["points"]:
            if p.get("status") != "OK":
                lines.append(
                    f"  ratio={p['ratio_ws_over_l3']}: SKIPPED ({p.get('reason')})"
                )
            else:
                lines.append(
                    f"  ratio={p['ratio_ws_over_l3']}: ws={p['working_set_bytes']} "
                    f"reps={p['reps']} median={p['median_gbs']:.4f} GB/s "
                    f"CV%={p['cv_pct']:.2f}"
                )
                lines.append(f"    raw_gbs: {p['raw_gbs']}")
        lines.append(f"escape_point: {b['escape_point']}")
        lines.append(f"new_policy_bytes: {b['new_working_set_policy_bytes']}")
        lines.append(f"  <- {b['justification']}")
    lines.append("")

    c = bundle["diag_c"]
    lines.append("=== DIAG C — INDEPENDENT VALIDITY BASELINE ===")
    lines.append(f"serial_baseline_A1_gbs: {c['serial_baseline_A1_gbs']}")
    lines.append(f"omp_probe: {c['omp_probe']}")
    lines.append(f"validity: {c['validity']}")
    lines.append(f"  <- {c['note']}")
    lines.append("")

    lines.append("=== PRIMARY DELIVERABLE ===")
    lines.append(bundle["primary_deliverable"]["conclusion"])
    lines.append("")
    lines.append("=== PHASE 2.2 JSON ===")
    lines.append(json.dumps(bundle, indent=2, default=str))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    bundle = run_diag()
    sys.stdout.write(format_diag_report(bundle))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
