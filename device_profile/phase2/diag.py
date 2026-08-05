"""Phase 2.3 — thread-count verification (DIAG A) + DIAG B escape fix.

Deliverable: Step 1 thread-count table + Step 3 restated conclusion.
Do not re-run BENCH 1–4. Do not adopt Phase 2.2 OpenMP@1 baseline recommendation.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from device_profile.detect import detect_device_profile

from .util import (
    effective_cores,
    effective_mem,
    evaluate_curve_validity,
    rep_stats,
    resolve_working_set_bytes,
)


def run_diag() -> dict[str, Any]:
    profile = detect_device_profile()
    cores = effective_cores(profile)
    emem = effective_mem(profile)
    l3 = _l3_bytes(profile)

    ws, ws_ev, suspect, _l3_pol, _ws_ratio = resolve_working_set_bytes(profile)
    max_alloc = emem // 2 if emem else ws
    if max_alloc and ws > max_alloc:
        ws = max_alloc
        ws_ev += f"; capped to 50% effective_mem={max_alloc}"

    n = max(1, int(ws // (3 * 8)))
    actual_ws = n * 3 * 8
    bytes_per_iter = n * 8 * 3
    scalar = 3.0

    # Calibrate reps in a short serial subprocess targeting ~1.0s
    reps, cal = _calibrate_reps_subprocess(n, bytes_per_iter, scalar)

    diag_a = run_diag_a(
        n=n,
        reps=reps,
        bytes_per_iter=bytes_per_iter,
        scalar=scalar,
        cores=cores,
    )

    codegen = run_step4_codegen(
        omp_lib=diag_a.get("omp_lib_path"),
        serial_lib=diag_a.get("serial_lib_path"),
        ws_bytes=actual_ws,
        l3=l3,
    )

    # DIAG B uses verified thread count from A6/A2 if applied, else 1
    fixed_t = 1
    for key in ("A6", "A2"):
        cfg = diag_a["configs"].get(key) or {}
        if cfg.get("thread_count_applied") and cfg.get("omp_num_threads_actual"):
            # For WS sweep use 1 verified thread (DRAM-bound single stream)
            fixed_t = 1
            break

    diag_b = run_diag_b(
        l3=l3,
        emem=emem,
        fixed_threads=fixed_t,
        scalar=scalar,
    )

    # DIAG C probe from verified subprocess configs only
    probe = []
    for key in ("A6", "A5"):
        c = diag_a["configs"].get(key) or {}
        if c.get("status") == "OK" and c.get("thread_count_applied"):
            probe.append(
                {
                    "threads": c["omp_num_threads_actual"],
                    "median_gbs": c["median_gbs"],
                    "from": key,
                }
            )
    # Also run verified 1..min(4,cores) probe in fresh subprocesses
    probe = _verified_thread_probe(
        n=n, reps=max(1, reps // 4), bytes_per_iter=bytes_per_iter, scalar=scalar, cores=cores
    )

    a1_gbs = (diag_a["configs"].get("A1") or {}).get("median_gbs")
    diag_c = {
        "omp_probe_verified": probe,
        "serial_baseline_A1_gbs": a1_gbs,
        "validity": evaluate_curve_validity(
            [p["threads"] for p in probe],
            [p["median_gbs"] for p in probe],
            serial_baseline_gbs=float(a1_gbs or 0.0),
            cores=cores,
        ),
        "note": (
            "Phase 2.3: serial A1 baseline retained; absolute bound is FLAG; "
            "SUPERLINEAR-vs-A1 is NOT discounted."
        ),
    }

    return {
        "phase": "2.3",
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
        "working_set_policy_pre": ws_ev,
        "l3_suspect": suspect,
        "bytes_moved_per_iter": bytes_per_iter,
        "reps": reps,
        "calibration": cal,
        "diag_a": diag_a,
        "codegen": codegen,
        "diag_b": diag_b,
        "diag_c": diag_c,
        "primary_deliverable": {
            "step1_thread_count_table": diag_a["thread_count_table"],
            "step3_conclusion": diag_a["conclusion"],
        },
        "note": (
            "Phase 2.3 single-bug phase: verify OpenMP thread counts. "
            "BENCH 1–4 not re-run. Phase 2.2 OpenMP@1 baseline NOT adopted."
        ),
    }


def run_diag_a(
    *, n: int, reps: int, bytes_per_iter: int, scalar: float, cores: int
) -> dict[str, Any]:
    all_cpus = list(range(os.cpu_count() or cores))
    configs_spec = [
        {
            "name": "A1",
            "description": "pure serial C, WITHOUT -fopenmp, no pragma",
            "serial": True,
            "requested_threads": None,
            "omp_proc_bind": None,
            "omp_places": None,
            "affinity_cpus": all_cpus,
        },
        {
            "name": "A2",
            "description": "-fopenmp, requested 1 thread, affinity=all (omp_set_num_threads)",
            "serial": False,
            "requested_threads": 1,
            "omp_proc_bind": None,
            "omp_places": None,
            "affinity_cpus": all_cpus,
        },
        {
            "name": "A3",
            "description": "-fopenmp, requested 1, PROC_BIND=close, PLACES=cores",
            "serial": False,
            "requested_threads": 1,
            "omp_proc_bind": "close",
            "omp_places": "cores",
            "affinity_cpus": all_cpus,
        },
        {
            "name": "A4",
            "description": "-fopenmp, requested 1, affinity=CPU0 only",
            "serial": False,
            "requested_threads": 1,
            "omp_proc_bind": None,
            "omp_places": None,
            "affinity_cpus": [0],
        },
        {
            "name": "A5",
            "description": "-fopenmp, requested 2, affinity=0,1, PROC_BIND=close",
            "serial": False,
            "requested_threads": 2,
            "omp_proc_bind": "close",
            "omp_places": "cores",
            "affinity_cpus": [0, 1] if (os.cpu_count() or 0) >= 2 else [0],
        },
        {
            "name": "A6",
            "description": "-fopenmp, verified 1 thread, fresh subprocess (duplicate of A2)",
            "serial": False,
            "requested_threads": 1,
            "omp_proc_bind": None,
            "omp_places": None,
            "affinity_cpus": all_cpus,
        },
    ]

    results: dict[str, Any] = {}
    omp_lib = None
    serial_lib = None
    for spec in configs_spec:
        cfg = {
            **spec,
            "n": n,
            "reps": reps,
            "bytes_per_iter": bytes_per_iter,
            "scalar": scalar,
        }
        row = _run_config_subprocess(cfg)
        results[spec["name"]] = row
        if spec["serial"]:
            serial_lib = row.get("lib_path")
        else:
            omp_lib = row.get("lib_path") or omp_lib

    a1 = results["A1"]["median_gbs"]
    for key, row in results.items():
        row["ratio_to_A1"] = (row["median_gbs"] / a1) if a1 and a1 > 0 else None

    table = []
    for key in ("A1", "A2", "A3", "A4", "A5", "A6"):
        r = results[key]
        table.append(
            {
                "cfg": key,
                "pid": r.get("pid"),
                "env_OMP_NUM_THREADS": r.get("env_OMP_NUM_THREADS"),
                "requested_threads": r.get("requested_threads"),
                "omp_num_threads_actual": r.get("omp_num_threads_actual"),
                "omp_get_max_threads": r.get("omp_max_threads"),
                "n_distinct_cpus": r.get("n_distinct_cpus"),
                "cpu_ids": r.get("cpu_ids"),
                "thread_flag": r.get("thread_flag"),
                "status": r.get("status"),
                "median_gbs": r.get("median_gbs"),
                "ratio_to_A1": r.get("ratio_to_A1"),
            }
        )

    conclusion = _conclude_diag_a_phase23(results)
    return {
        "reps": reps,
        "bytes_moved_per_iter": bytes_per_iter,
        "configs": results,
        "thread_count_table": table,
        "conclusion": conclusion,
        "omp_lib_path": omp_lib,
        "serial_lib_path": serial_lib,
    }


def run_diag_b(
    *, l3: Optional[int], emem: int, fixed_threads: int, scalar: float
) -> dict[str, Any]:
    if not l3 or l3 <= 0:
        return {"status": "SKIPPED", "reason": "L3_reported undetected"}

    ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    max_alloc = emem // 2 if emem else int(8 * l3)

    # Hold total bytes moved constant: choose target from mid-size geometry.
    # target_bytes_moved = reps * bytes_per_iter, same for every point.
    ref_ratio = 4.0
    ref_ws = int(ref_ratio * l3)
    if ref_ws > max_alloc:
        ref_ws = max_alloc
    ref_n = max(1, ref_ws // (3 * 8))
    ref_bpi = ref_n * 8 * 3
    # Aim ~0.8s at verified 1 thread → pick reps for ref size via subprocess calibrate
    ref_reps, _ = _calibrate_reps_subprocess(
        ref_n, ref_bpi, scalar, requested_threads=fixed_threads, target=0.8
    )
    target_bytes_moved = ref_reps * ref_bpi

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
        n = max(1, ws // (3 * 8))
        bpi = n * 8 * 3
        reps = max(1, int(round(target_bytes_moved / bpi)))
        # Exact constant: adjust note if rounding
        actual_moved = reps * bpi
        row = _run_config_subprocess(
            {
                "name": f"B_r{ratio}",
                "description": f"DIAG B ratio={ratio}",
                "serial": False,
                "requested_threads": fixed_threads,
                "omp_proc_bind": "close",
                "omp_places": "cores",
                "affinity_cpus": list(range(fixed_threads)),
                "n": n,
                "reps": reps,
                "bytes_per_iter": bpi,
                "scalar": scalar,
            }
        )
        points.append(
            {
                "ratio_ws_over_l3": ratio,
                "working_set_bytes": n * 3 * 8,
                "reps": reps,
                "bytes_moved_per_iter": bpi,
                "bytes_moved_total": actual_moved,
                "target_bytes_moved": target_bytes_moved,
                "pid": row.get("pid"),
                "omp_num_threads_actual": row.get("omp_num_threads_actual"),
                "thread_flag": row.get("thread_flag"),
                "status": (
                    "INVALID"
                    if row.get("status") == "INVALID"
                    else row.get("status", "OK")
                ),
                "median_gbs": row.get("median_gbs"),
                "cv_pct": row.get("cv_pct"),
                "raw_gbs": row.get("raw_gbs"),
                "raw_times_s": row.get("raw_times_s"),
            }
        )

    escape = _find_escape_two_consecutive(points)
    policy = None
    if escape.get("ratio_ws_over_l3") is not None:
        policy = {
            "working_set_bytes": int(escape["ratio_ws_over_l3"] * l3),
            "rule": f"working_set = escape_ratio * L3_reported = {escape['ratio_ws_over_l3']} * {l3}",
            "from_escape": True,
        }
    else:
        policy = {
            "working_set_bytes": None,
            "rule": "NO ESCAPE OBSERVED — policy not set",
            "from_escape": False,
        }

    return {
        "status": "OK",
        "l3_reported_bytes": l3,
        "fixed_threads_requested": fixed_threads,
        "target_bytes_moved": target_bytes_moved,
        "points": points,
        "escape_point": escape,
        "working_set_policy": policy,
    }


def run_step4_codegen(
    *, omp_lib: Optional[str], serial_lib: Optional[str], ws_bytes: int, l3: Optional[int]
) -> dict[str, Any]:
    src_dir = Path(__file__).resolve().parent
    diff = subprocess.run(
        ["diff", "-u", str(src_dir / "kernels_serial.c"), str(src_dir / "kernels.c")],
        capture_output=True,
        text=True,
        check=False,
    )
    diff_text = (diff.stdout or "") + (diff.stderr or "")

    omp_dis = _disassemble_triad(omp_lib, "stream_triad_reps_observed") if omp_lib else {}
    ser_dis = (
        _disassemble_triad(serial_lib, "stream_triad_reps_pure_serial_observed")
        if serial_lib
        else {}
    )

    ratio = (ws_bytes / l3) if (l3 and l3 > 0) else None
    physical = (
        f"Working set={ws_bytes} vs L3_reported={l3} (ratio={ratio}). "
        f"At ≥4× L3 a triad is DRAM-bound; instruction-level codegen "
        f"(vector vs scalar) cannot plausibly produce a sustained ~3× bandwidth "
        f"difference once both kernels miss in cache. If a 3× gap remains after "
        f"verified thread counts, the cause is unknown — not auto-vectorization."
    )

    return {
        "source_diff_kernels_serial_vs_kernels": diff_text[:20000],
        "source_diff_returncode": diff.returncode,
        "omp_disassembly": omp_dis,
        "serial_disassembly": ser_dis,
        "physical_argument": physical,
        "vectorization_claim": _vectorization_verdict(omp_dis, ser_dis),
    }


def _vectorization_verdict(omp_dis: dict, ser_dis: dict) -> str:
    o = omp_dis.get("vector_class", "unknown")
    s = ser_dis.get("vector_class", "unknown")
    ont = omp_dis.get("nontemporal", False)
    snt = ser_dis.get("nontemporal", False)
    if o == s and not ont and not snt:
        return (
            f"Both triad loops look {o}; nontemporal stores absent in both. "
            f"Auto-vectorization does NOT explain a 3× gap."
        )
    if o != s:
        return (
            f"OpenMP .so triad uses {o}; serial .so uses {s}; "
            f"nontemporal omp={ont} serial={snt}. Codegen differs, but at DRAM-bound "
            f"WS a 3× bandwidth claim from vectorization alone is still not physical — "
            f"if gap persists with verified threads, say DO NOT KNOW."
        )
    return f"omp={o} serial={s} nontemporal omp={ont} serial={snt}"


def _disassemble_triad(so_path: str, symbol: str) -> dict[str, Any]:
    # Find symbol address then disassemble
    nm = subprocess.run(["nm", "-D", so_path], capture_output=True, text=True, check=False)
    addr = None
    for line in (nm.stdout or "").splitlines():
        if symbol in line.split():
            parts = line.split()
            if parts:
                addr = parts[0]
                break
    obj = subprocess.run(
        ["objdump", "-d", so_path],
        capture_output=True,
        text=True,
        check=False,
    )
    text = obj.stdout or ""
    # Extract a window around the symbol name
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if f"<{symbol}>" in line or (addr and line.startswith(addr)):
            start = i
            break
    chunk = lines[start : start + 120] if start is not None else []
    blob = "\n".join(chunk)
    has_zmm = bool(re.search(r"\bzmm\d+", blob))
    has_ymm = bool(re.search(r"\bymm\d+", blob))
    has_xmm = bool(re.search(r"\bxmm\d+", blob))
    has_nt = bool(re.search(r"vmovnt", blob))
    if has_zmm:
        vclass = "vector-zmm"
    elif has_ymm:
        vclass = "vector-ymm"
    elif has_xmm:
        vclass = "vector-xmm"
    else:
        vclass = "scalar-or-unresolved"
    return {
        "so": so_path,
        "symbol": symbol,
        "vector_class": vclass,
        "nontemporal": has_nt,
        "has_zmm": has_zmm,
        "has_ymm": has_ymm,
        "has_xmm": has_xmm,
        "disasm_excerpt": blob[:4000],
        "nm_match": addr,
    }


def _conclude_diag_a_phase23(results: dict[str, Any]) -> str:
    a1 = results["A1"]
    a2 = results["A2"]
    a6 = results["A6"]
    # Require verified thread counts for OpenMP configs
    unverified = [
        k
        for k in ("A2", "A3", "A4", "A5", "A6")
        if results[k].get("thread_flag") == "THREAD COUNT NOT APPLIED"
        or results[k].get("status") == "INVALID"
    ]
    a1_g = a1["median_gbs"]
    a6_g = a6["median_gbs"]
    a2_g = a2["median_gbs"]

    parts = []
    parts.append(
        f"Thread verification: "
        + ", ".join(
            f"{k}=actual:{results[k].get('omp_num_threads_actual')}"
            f"/req:{results[k].get('requested_threads')}"
            f"[{results[k].get('thread_flag')}]"
            for k in ("A1", "A2", "A3", "A4", "A5", "A6")
        )
    )
    if unverified:
        parts.append(
            f"INVALID configs (THREAD COUNT NOT APPLIED): {unverified}. "
            f"Cannot trust their GB/s."
        )

    if a6.get("status") == "OK" and a1_g and a1_g > 0:
        r = a6_g / a1_g
        if 0.7 <= r <= 1.3:
            parts.append(
                f"A6 (verified OpenMP@1)={a6_g:.2f} GB/s ≈ A1={a1_g:.2f} GB/s "
                f"(ratio {r:.2f}). Phase 2.2 conclusion is FALSIFIED — "
                f"A1 is NOT a 3x outlier once OpenMP@1 is verified to run at 1 thread."
            )
        elif r > 1.5:
            parts.append(
                f"A6/A1={r:.2f} with verified thread counts: A1 remains slower than "
                f"verified OpenMP@1. At DRAM-bound WS, codegen cannot physically "
                f"explain ~3x bandwidth — DO NOT KNOW the cause; do not claim "
                f"auto-vectorization. Do NOT adopt OpenMP@1 as baseline replacement."
            )
        else:
            parts.append(
                f"A6/A1={r:.2f} (A6={a6_g:.2f}, A1={a1_g:.2f}). "
                f"A1 does not remain a ~3x outlier under verified threads."
            )
    else:
        parts.append("A6 not OK; cannot restate vs A1.")

    # Flatness check A2 vs A5 with verified counts
    if (
        a2.get("status") == "OK"
        and results["A5"].get("status") == "OK"
        and a2.get("thread_count_applied")
        and results["A5"].get("thread_count_applied")
    ):
        a5_g = results["A5"]["median_gbs"]
        parts.append(
            f"Verified A2@1={a2_g:.2f} vs A5@2={a5_g:.2f} GB/s "
            f"(ratio A5/A2={a5_g/a2_g:.2f} if a2>0)."
        )

    return " ".join(parts)


def _run_config_subprocess(cfg: dict[str, Any]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        cfg_path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "device_profile.phase2.diag_worker", cfg_path],
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        marker = "PHASE23_WORKER_JSON:"
        payload = None
        for line in out.splitlines():
            if line.startswith(marker):
                payload = json.loads(line[len(marker) :])
                break
        if payload is None:
            return {
                "name": cfg.get("name"),
                "status": "INVALID",
                "thread_flag": "THREAD COUNT NOT APPLIED",
                "error": f"worker failed rc={proc.returncode} err={err[:2000]} out={out[:2000]}",
                "median_gbs": float("nan"),
                "pid": None,
                "omp_num_threads_actual": None,
                "requested_threads": cfg.get("requested_threads"),
                "thread_count_applied": False,
            }
        return payload
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def _calibrate_reps_subprocess(
    n: int,
    bytes_per_iter: int,
    scalar: float,
    *,
    requested_threads: Optional[int] = None,
    target: float = 1.0,
) -> tuple[int, dict[str, Any]]:
    # Binary search-ish via serial A1 worker with growing reps
    reps = 1
    hist = []
    serial = requested_threads is None
    for _ in range(10):
        row = _run_config_subprocess(
            {
                "name": "CAL",
                "description": "calibration",
                "serial": serial,
                "requested_threads": requested_threads,
                "omp_proc_bind": None,
                "omp_places": None,
                "affinity_cpus": [0],
                "n": n,
                "reps": reps,
                "bytes_per_iter": bytes_per_iter,
                "scalar": scalar,
            }
        )
        # Use median of raw times * approx — or discarded warmup time
        times = row.get("raw_times_s") or []
        dt = float(rep_stats(times).median) if times else 0.0
        hist.append({"reps": reps, "median_time_s": dt, "pid": row.get("pid")})
        if dt >= target * 0.85:
            break
        scale = target / max(dt, 1e-6)
        reps = max(reps + 1, int(reps * scale))
    return reps, {"history": hist, "target_s": target}


def _verified_thread_probe(
    *, n: int, reps: int, bytes_per_iter: int, scalar: float, cores: int
) -> list[dict[str, Any]]:
    out = []
    for t in range(1, min(4, cores) + 1):
        row = _run_config_subprocess(
            {
                "name": f"PROBE_t{t}",
                "description": f"verified probe threads={t}",
                "serial": False,
                "requested_threads": t,
                "omp_proc_bind": "close",
                "omp_places": "cores",
                "affinity_cpus": list(range(t)),
                "n": n,
                "reps": reps,
                "bytes_per_iter": bytes_per_iter,
                "scalar": scalar,
            }
        )
        out.append(
            {
                "threads_requested": t,
                "threads": row.get("omp_num_threads_actual"),
                "thread_flag": row.get("thread_flag"),
                "status": row.get("status"),
                "median_gbs": row.get("median_gbs"),
                "pid": row.get("pid"),
                "n_distinct_cpus": row.get("n_distinct_cpus"),
            }
        )
    return out


def _find_escape_two_consecutive(points: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Escape = smallest ratio where TWO CONSECUTIVE points are within 5% of each
    other AND both sit near the asymptotic floor (within 10% of the minimum
    OK median). This rejects early in-cache plateaus (e.g. 0.25≈0.5 at ~2×
    the DRAM rate) that would otherwise match the literal consecutive rule.
    """
    ok = [
        p
        for p in points
        if p.get("status") == "OK" and p.get("median_gbs") is not None
    ]
    if len(ok) < 2:
        return {
            "ratio_ws_over_l3": None,
            "flag": "NO ESCAPE OBSERVED",
            "note": "insufficient OK points",
        }
    floor = min(p["median_gbs"] for p in ok)
    if floor <= 0:
        return {
            "ratio_ws_over_l3": None,
            "flag": "NO ESCAPE OBSERVED",
            "note": "non-positive floor",
        }
    candidates = []
    for i in range(len(ok) - 1):
        a, b = ok[i]["median_gbs"], ok[i + 1]["median_gbs"]
        rel = abs(a - b) / max(a, b)
        near_floor = (a <= floor * 1.10) and (b <= floor * 1.10)
        if rel <= 0.05 and near_floor:
            candidates.append((i, rel, a, b))
    if not candidates:
        # Report any consecutive-within-5% pairs that were rejected as early plateaus
        early = []
        for i in range(len(ok) - 1):
            a, b = ok[i]["median_gbs"], ok[i + 1]["median_gbs"]
            rel = abs(a - b) / max(a, b)
            if rel <= 0.05:
                early.append(
                    {
                        "ratios": [
                            ok[i]["ratio_ws_over_l3"],
                            ok[i + 1]["ratio_ws_over_l3"],
                        ],
                        "gbs": [a, b],
                        "rejected": "above asymptotic floor (in-cache plateau)",
                    }
                )
        return {
            "ratio_ws_over_l3": None,
            "flag": "NO ESCAPE OBSERVED",
            "early_plateaus_rejected": early,
            "floor_gbs": floor,
            "note": (
                "no consecutive pair within 5% that is also within 10% of "
                f"min median ({floor:.3f} GB/s); policy not set"
            ),
        }
    i, rel, a, b = candidates[0]
    return {
        "ratio_ws_over_l3": ok[i]["ratio_ws_over_l3"],
        "pair_ratios": [
            ok[i]["ratio_ws_over_l3"],
            ok[i + 1]["ratio_ws_over_l3"],
        ],
        "pair_gbs": [a, b],
        "rel_diff": rel,
        "floor_gbs": floor,
        "flag": "OK",
        "note": (
            f"escape = smallest near-floor consecutive pair within 5%: "
            f"{ok[i]['ratio_ws_over_l3']} & {ok[i+1]['ratio_ws_over_l3']} "
            f"({a:.3f} vs {b:.3f}); floor={floor:.3f} GB/s"
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
    lines.append("=== PHASE 2.3 THREAD COUNT VERIFICATION REPORT ===")
    lines.append(f"HOST CLASS: {bundle.get('host_class')}")
    lines.append(f"  <- {bundle.get('host_class_evidence')}")
    lines.append(f"effective_cores: {bundle.get('effective_cores')}")
    lines.append(f"working_set_bytes: {bundle.get('working_set_bytes')}")
    lines.append(f"reps (constant): {bundle.get('reps')}")
    lines.append(f"note: {bundle.get('note')}")
    lines.append("")

    a = bundle["diag_a"]
    lines.append("=== STEP 1 — INSTRUMENTED THREAD COUNT TABLE (DELIVERABLE) ===")
    lines.append(
        f"{'cfg':<4} {'pid':>8} {'env':>6} {'req':>4} {'actual':>7} {'max':>5} "
        f"{'nCPU':>5} {'flag':<28} {'GB/s':>10} {'/A1':>8}"
    )
    for row in a["thread_count_table"]:
        lines.append(
            f"{row['cfg']:<4} {str(row['pid']):>8} {str(row['env_OMP_NUM_THREADS']):>6} "
            f"{str(row['requested_threads']):>4} {str(row['omp_num_threads_actual']):>7} "
            f"{str(row['omp_get_max_threads']):>5} {str(row['n_distinct_cpus']):>5} "
            f"{str(row['thread_flag']):<28} "
            f"{row['median_gbs'] if row['median_gbs']==row['median_gbs'] else float('nan'):10.4f} "
            f"{row['ratio_to_A1'] if row['ratio_to_A1'] else float('nan'):8.4f}"
        )
        full = a["configs"][row["cfg"]]
        lines.append(f"     cpu_ids={row['cpu_ids']} status={row['status']}")
        lines.append(f"     description: {full.get('description')}")
        lines.append(f"     raw_gbs: {full.get('raw_gbs')}")
        lines.append(f"     raw_times_s: {full.get('raw_times_s')}")
        for chk in full.get("reconstruction_checks") or []:
            tag = "OK" if chk.get("ok") else "FAIL"
            lines.append(
                f"     reconstruct[{tag}]: reported={chk.get('reported_gbs')} "
                f"rel_err={chk.get('rel_err')}"
            )
        if full.get("thread_flag") == "THREAD COUNT NOT APPLIED":
            lines.append("     THREAD COUNT NOT APPLIED — config INVALID")
    lines.append("")

    lines.append("=== STEP 3 — DIAG A RESTATEMENT (DELIVERABLE) ===")
    lines.append(a["conclusion"])
    lines.append("")

    cg = bundle.get("codegen") or {}
    lines.append("=== STEP 4 — CODEGEN / SOURCE CHECK ===")
    lines.append(f"vectorization_claim: {cg.get('vectorization_claim')}")
    lines.append(f"physical_argument: {cg.get('physical_argument')}")
    lines.append(f"omp_disassembly: { {k:v for k,v in (cg.get('omp_disassembly') or {}).items() if k!='disasm_excerpt'} }")
    lines.append(
        f"serial_disassembly: { {k:v for k,v in (cg.get('serial_disassembly') or {}).items() if k!='disasm_excerpt'} }"
    )
    lines.append("--- diff kernels_serial.c vs kernels.c (truncated) ---")
    lines.append((cg.get("source_diff_kernels_serial_vs_kernels") or "")[:6000])
    lines.append("")

    b = bundle.get("diag_b") or {}
    lines.append("=== STEP 5 — DIAG B ESCAPE (corrected) ===")
    lines.append(f"target_bytes_moved (held constant): {b.get('target_bytes_moved')}")
    for p in b.get("points") or []:
        lines.append(
            f"  ratio={p.get('ratio_ws_over_l3')}: status={p.get('status')} "
            f"ws={p.get('working_set_bytes')} reps={p.get('reps')} "
            f"moved={p.get('bytes_moved_total')} "
            f"actual_threads={p.get('omp_num_threads_actual')} "
            f"median={p.get('median_gbs')} CV%={p.get('cv_pct')}"
        )
    lines.append(f"escape_point: {b.get('escape_point')}")
    lines.append(f"working_set_policy: {b.get('working_set_policy')}")
    lines.append("")

    c = bundle.get("diag_c") or {}
    lines.append("=== STEP 6 — VALIDITY GATE (not weakened) ===")
    lines.append(f"probe: {c.get('omp_probe_verified')}")
    lines.append(f"validity: {c.get('validity')}")
    lines.append(f"  <- {c.get('note')}")
    lines.append("")

    lines.append("=== PRIMARY DELIVERABLE ===")
    lines.append("STEP 1 TABLE:")
    for row in bundle["primary_deliverable"]["step1_thread_count_table"]:
        lines.append(f"  {row}")
    lines.append("STEP 3 CONCLUSION:")
    lines.append(bundle["primary_deliverable"]["step3_conclusion"])
    lines.append("")
    lines.append("=== PHASE 2.3 JSON ===")
    lines.append(json.dumps(bundle, indent=2, default=str))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    # Remove accidental bad import usage
    bundle = run_diag()
    sys.stdout.write(format_diag_report(bundle))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
