"""Phase 2.3 subprocess worker — one DIAG A config per process (clean libgomp init)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def run_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    cfg keys: name, description, serial, requested_threads, omp_proc_bind,
              omp_places, affinity_cpus, n, reps, bytes_per_iter, scalar
    Env OMP_* must be set BEFORE importing/building OpenMP kernels.
    """
    # Set bind/places env before libgomp init (dlopen via build_kernels).
    if cfg.get("omp_proc_bind"):
        os.environ["OMP_PROC_BIND"] = str(cfg["omp_proc_bind"])
    else:
        os.environ.pop("OMP_PROC_BIND", None)
    if cfg.get("omp_places"):
        os.environ["OMP_PLACES"] = str(cfg["omp_places"])
    else:
        os.environ.pop("OMP_PLACES", None)
    # Also set OMP_NUM_THREADS for init, but authoritative control is omp_set_num_threads.
    if cfg.get("requested_threads") is not None and not cfg.get("serial"):
        os.environ["OMP_NUM_THREADS"] = str(cfg["requested_threads"])
    else:
        os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["OMP_DYNAMIC"] = "false"

    from device_profile.phase2.native import (
        build_kernels,
        build_serial_kernels,
        run_observed_triad,
    )
    from device_profile.phase2.util import (
        assert_gbs_reconstructs,
        now,
        pin_to_cpus,
        read_steal_jiffies,
        rep_stats,
    )

    import ctypes
    import numpy as np

    name = cfg["name"]
    n = int(cfg["n"])
    reps = int(cfg["reps"])
    bytes_per_iter = int(cfg["bytes_per_iter"])
    scalar = float(cfg["scalar"])
    requested = cfg.get("requested_threads")
    serial = bool(cfg.get("serial"))

    aff_ev = pin_to_cpus(list(cfg["affinity_cpus"]))
    pid = os.getpid()

    if serial:
        kernels = build_serial_kernels()
    else:
        kernels = build_kernels()

    a = np.empty(n, dtype=np.float64)
    b = np.linspace(0.0, 1.0, n, dtype=np.float64)
    c = np.linspace(1.0, 2.0, n, dtype=np.float64)
    a[:] = 0.0
    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Prefault
    run_observed_triad(
        kernels,
        serial=serial,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=1,
        requested_threads=requested,
    )

    steal0, steal0_ev = read_steal_jiffies()

    # DISCARDED warmup
    obs_warm = run_observed_triad(
        kernels,
        serial=serial,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=reps,
        requested_threads=requested,
    )
    t0 = now()
    run_observed_triad(
        kernels,
        serial=serial,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=reps,
        requested_threads=requested,
    )
    warm_dt = now() - t0
    warm_gbs = (reps * bytes_per_iter) / warm_dt / 1e9

    raw_t: list[float] = []
    raw_gbs: list[float] = []
    checks: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for _ in range(5):
        t0 = now()
        obs = run_observed_triad(
            kernels,
            serial=serial,
            ap=ap,
            bp=bp,
            cp=cp,
            scalar=scalar,
            n=n,
            reps=reps,
            requested_threads=requested,
        )
        dt = now() - t0
        gbs = (reps * bytes_per_iter) / dt / 1e9
        chk = assert_gbs_reconstructs(gbs, reps, bytes_per_iter, dt)
        raw_t.append(dt)
        raw_gbs.append(gbs)
        checks.append(chk)
        observations.append(
            {
                "omp_num_threads_actual": obs.omp_num_threads_actual,
                "omp_max_threads": obs.omp_max_threads,
                "n_distinct_cpus": obs.n_distinct_cpus,
                "cpu_ids": obs.cpu_ids,
                "flag": obs.flag,
            }
        )

    steal1, steal1_ev = read_steal_jiffies()
    st = rep_stats(raw_gbs)

    # Authoritative observation = last timed rep (all should match)
    last = observations[-1]
    actual = last["omp_num_threads_actual"]
    if serial:
        applied = True
        flag = "SERIAL_NO_OMP"
        status = "OK"
    elif requested is not None and actual != int(requested):
        applied = False
        flag = "THREAD COUNT NOT APPLIED"
        status = "INVALID"
    else:
        applied = True
        flag = "OK"
        status = "OK"

    return {
        "name": name,
        "description": cfg.get("description"),
        "pid": pid,
        "status": status,
        "thread_flag": flag,
        "reps": reps,
        "bytes_moved_per_iter": bytes_per_iter,
        "requested_threads": requested,
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "omp_num_threads_actual": actual,
        "omp_max_threads": last["omp_max_threads"],
        "n_distinct_cpus": last["n_distinct_cpus"],
        "cpu_ids": last["cpu_ids"],
        "thread_count_applied": applied,
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "omp_places": os.environ.get("OMP_PLACES"),
        "affinity_evidence": aff_ev,
        "lib_path": kernels.lib_path,
        "compile_cmd": kernels.compile_cmd,
        "discarded_warmup": {
            "label": "DISCARDED",
            "time_s": warm_dt,
            "gbs": warm_gbs,
            "observation": {
                "omp_num_threads_actual": obs_warm.omp_num_threads_actual,
                "n_distinct_cpus": obs_warm.n_distinct_cpus,
                "cpu_ids": obs_warm.cpu_ids,
            },
        },
        "raw_times_s": raw_t,
        "raw_gbs": raw_gbs,
        "reconstruction_checks": checks,
        "per_rep_observations": observations,
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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: diag_worker <config.json>", file=sys.stderr)
        return 2
    with open(args[0], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    result = run_config(cfg)
    sys.stdout.write("PHASE23_WORKER_JSON:")
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
