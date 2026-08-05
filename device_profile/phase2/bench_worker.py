"""Phase 2.4 bench subprocess worker — fresh libgomp per timed region."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _prep_omp_env(cfg: dict[str, Any]) -> None:
    if cfg.get("omp_proc_bind"):
        os.environ["OMP_PROC_BIND"] = str(cfg["omp_proc_bind"])
    else:
        os.environ.pop("OMP_PROC_BIND", None)
    if cfg.get("omp_places"):
        os.environ["OMP_PLACES"] = str(cfg["omp_places"])
    else:
        os.environ.pop("OMP_PLACES", None)
    req = cfg.get("requested_threads")
    if req is not None:
        os.environ["OMP_NUM_THREADS"] = str(req)
    else:
        os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["OMP_DYNAMIC"] = "false"


def run_stream_point(cfg: dict[str, Any]) -> dict[str, Any]:
    from device_profile.phase2.native import build_kernels, run_observed_triad
    from device_profile.phase2.util import (
        assert_gbs_reconstructs,
        now,
        pin_to_cpus,
        rep_stats,
    )

    import ctypes
    import numpy as np

    n = int(cfg["n"])
    reps = int(cfg["reps"])
    bpi = int(cfg["bytes_moved_per_iter"])
    scalar = float(cfg.get("scalar", 3.0))
    requested = int(cfg["requested_threads"])
    aff = list(cfg.get("affinity_cpus") or list(range(requested)))

    pin_to_cpus(aff)
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
        serial=False,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=1,
        requested_threads=requested,
    )

    # DISCARDED warmup
    t0 = now()
    obs_w = run_observed_triad(
        kernels,
        serial=False,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=reps,
        requested_threads=requested,
    )
    warm_dt = now() - t0
    warm_gbs = (reps * bpi) / warm_dt / 1e9

    raw_t: list[float] = []
    raw_gbs: list[float] = []
    checks: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for _ in range(5):
        t0 = now()
        obs = run_observed_triad(
            kernels,
            serial=False,
            ap=ap,
            bp=bp,
            cp=cp,
            scalar=scalar,
            n=n,
            reps=reps,
            requested_threads=requested,
        )
        dt = now() - t0
        gbs = (reps * bpi) / dt / 1e9
        chk = assert_gbs_reconstructs(gbs, reps, bpi, dt)
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

    last = observations[-1]
    actual = last["omp_num_threads_actual"]
    applied = actual == requested
    st = rep_stats(raw_gbs)
    return {
        "mode": "stream_point",
        "pid": os.getpid(),
        "status": "OK" if applied else "INVALID",
        "thread_flag": "OK" if applied else "THREAD COUNT NOT APPLIED",
        "requested_threads": requested,
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "omp_num_threads_actual": actual,
        "omp_max_threads": last["omp_max_threads"],
        "n_distinct_cpus": last["n_distinct_cpus"],
        "cpu_ids": last["cpu_ids"],
        "thread_count_applied": applied,
        "reps": reps,
        "bytes_moved_per_iter": bpi,
        "discarded_warmup": {
            "label": "DISCARDED",
            "time_s": warm_dt,
            "gbs": warm_gbs,
            "omp_num_threads_actual": obs_w.omp_num_threads_actual,
            "n_distinct_cpus": obs_w.n_distinct_cpus,
        },
        "raw_times_s": raw_t,
        "raw_gbs": raw_gbs,
        "reconstruction_checks": checks,
        "per_rep_observations": observations,
        "median_gbs": st.median,
        "min_gbs": st.minimum,
        "max_gbs": st.maximum,
        "mean_gbs": st.mean,
        "cv_pct": st.cv_pct,
        "noisy": st.noisy,
        "FAIL": [c["message"] for c in checks if not c["ok"]],
        "lib_path": kernels.lib_path,
        "compile_cmd": kernels.compile_cmd,
    }


def run_stream_sustained(cfg: dict[str, Any]) -> dict[str, Any]:
    from device_profile.phase2.native import build_kernels, run_observed_triad
    from device_profile.phase2.util import now, pin_to_cpus, sample_thermal_and_freq

    import ctypes
    import numpy as np

    n = int(cfg["n"])
    reps = int(cfg["reps"])
    bpi = int(cfg["bytes_moved_per_iter"])
    scalar = float(cfg.get("scalar", 3.0))
    requested = int(cfg["requested_threads"])
    duration_s = float(cfg.get("duration_s", 180.0))
    bucket_s = float(cfg.get("bucket_s", 10.0))
    aff = list(cfg.get("affinity_cpus") or list(range(requested)))

    pin_to_cpus(aff)
    kernels = build_kernels()

    probe = sample_thermal_and_freq()
    if probe.get("thermal_blind"):
        # Still verify thread count once, then exit without claiming sustained/peak
        a = np.empty(min(n, 1024 * 1024), dtype=np.float64)
        b = np.ones_like(a)
        c = np.ones_like(a)
        ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        obs = run_observed_triad(
            kernels,
            serial=False,
            ap=ap,
            bp=bp,
            cp=cp,
            scalar=scalar,
            n=len(a),
            reps=1,
            requested_threads=requested,
        )
        applied = obs.omp_num_threads_actual == requested
        return {
            "mode": "stream_sustained",
            "pid": os.getpid(),
            "status": "THERMAL_BLIND" if applied else "INVALID",
            "thread_flag": "OK" if applied else "THREAD COUNT NOT APPLIED",
            "thermal_blind": True,
            "thermal_probe": probe,
            "requested_threads": requested,
            "omp_num_threads_actual": obs.omp_num_threads_actual,
            "omp_max_threads": obs.omp_max_threads,
            "n_distinct_cpus": obs.n_distinct_cpus,
            "cpu_ids": obs.cpu_ids,
            "thread_count_applied": applied,
            "buckets": [],
            "summary": {
                "flag": "THERMAL BLIND — RESULT NOT MEANINGFUL",
                "explanation": (
                    "zero thermal zones and zero cpufreq scaling_cur_freq readable; "
                    "sustained/peak conclusion SKIPPED"
                ),
                "conclusion_skipped": True,
            },
            "notes": [
                "THERMAL BLIND — RESULT NOT MEANINGFUL; 180s STREAM skipped after "
                "thread-count verify"
            ],
        }

    a = np.empty(n, dtype=np.float64)
    b = np.linspace(0.0, 1.0, n, dtype=np.float64)
    c = np.linspace(1.0, 2.0, n, dtype=np.float64)
    a[:] = 0.0
    ap = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    bp = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cp = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    run_observed_triad(
        kernels,
        serial=False,
        ap=ap,
        bp=bp,
        cp=cp,
        scalar=scalar,
        n=n,
        reps=1,
        requested_threads=requested,
    )

    bytes_per_call = reps * bpi
    t_end = time.time() + duration_s
    bucket_idx = 0
    bucket_start = now()
    wall_bucket_start = time.time()
    bytes_in_bucket = 0.0
    calls_in_bucket = 0
    peak_gbs = 0.0
    time_to_throttle_s = None
    first_bucket_gbs = None
    buckets: list[dict[str, Any]] = []
    last_obs = None

    while time.time() < t_end:
        t0 = now()
        last_obs = run_observed_triad(
            kernels,
            serial=False,
            ap=ap,
            bp=bp,
            cp=cp,
            scalar=scalar,
            n=n,
            reps=reps,
            requested_threads=requested,
        )
        _ = now() - t0
        bytes_in_bucket += bytes_per_call
        calls_in_bucket += 1
        if (time.time() - wall_bucket_start) >= bucket_s:
            elapsed = now() - bucket_start
            gbs = bytes_in_bucket / elapsed / 1e9 if elapsed > 0 else 0.0
            if first_bucket_gbs is None:
                first_bucket_gbs = gbs
            peak_gbs = max(peak_gbs, gbs)
            therm = sample_thermal_and_freq()
            buckets.append(
                {
                    "bucket": bucket_idx,
                    "elapsed_s": elapsed,
                    "calls": calls_in_bucket,
                    "gbs": gbs,
                    "omp_num_threads_actual": last_obs.omp_num_threads_actual,
                    "n_distinct_cpus": last_obs.n_distinct_cpus,
                    "cpu_ids": last_obs.cpu_ids,
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

    if calls_in_bucket > 0 and last_obs is not None:
        elapsed = now() - bucket_start
        gbs = bytes_in_bucket / elapsed / 1e9 if elapsed > 0 else 0.0
        peak_gbs = max(peak_gbs, gbs)
        therm = sample_thermal_and_freq()
        buckets.append(
            {
                "bucket": bucket_idx,
                "elapsed_s": elapsed,
                "calls": calls_in_bucket,
                "gbs": gbs,
                "partial": True,
                "omp_num_threads_actual": last_obs.omp_num_threads_actual,
                "n_distinct_cpus": last_obs.n_distinct_cpus,
                "cpu_ids": last_obs.cpu_ids,
                "temps_C": therm["temps_C"],
                "freqs_kHz": therm["freqs_kHz"],
                "thermal_evidence": therm["evidence"],
            }
        )

    import statistics

    sustained = (
        statistics.median([b["gbs"] for b in buckets[-3:]]) if buckets else float("nan")
    )
    actual = last_obs.omp_num_threads_actual if last_obs else None
    applied = actual == requested if actual is not None else False
    return {
        "mode": "stream_sustained",
        "pid": os.getpid(),
        "status": "OK" if applied else "INVALID",
        "thread_flag": "OK" if applied else "THREAD COUNT NOT APPLIED",
        "thermal_blind": False,
        "requested_threads": requested,
        "omp_num_threads_actual": actual,
        "omp_max_threads": last_obs.omp_max_threads if last_obs else None,
        "n_distinct_cpus": last_obs.n_distinct_cpus if last_obs else None,
        "cpu_ids": last_obs.cpu_ids if last_obs else None,
        "thread_count_applied": applied,
        "duration_s": duration_s,
        "bucket_s": bucket_s,
        "buckets": buckets,
        "summary": {
            "peak_bucket_gbs": peak_gbs,
            "sustained_last3_median_gbs": sustained,
            "sustained_over_peak": (sustained / peak_gbs) if peak_gbs else None,
            "time_to_throttle_s": time_to_throttle_s,
            "throttle_rule": "first bucket with gbs < 90% of first bucket",
            "first_bucket_gbs": first_bucket_gbs,
            "conclusion_skipped": False,
        },
    }


def run_mmap_file(cfg: dict[str, Any]) -> dict[str, Any]:
    """Timed mmap stride passes in a fresh process; OpenMP set to 1 for hygiene."""
    import ctypes
    import mmap

    from device_profile.phase2.native import build_kernels, run_observed_triad
    from device_profile.phase2.util import (
        now,
        pin_to_cpus,
        read_io_read_bytes,
        read_minflt_majflt,
    )

    import numpy as np

    need = int(cfg["file_bytes"])
    path = cfg["path"]
    label = cfg.get("label", "mmap")
    requested = int(cfg.get("requested_threads", 1))
    pin_to_cpus(list(cfg.get("affinity_cpus") or [0]))
    kernels = build_kernels()

    # Prove thread control with a tiny observed triad (not the timed mmap kernel).
    tiny = 1 << 16
    a = np.zeros(tiny, dtype=np.float64)
    b = np.ones(tiny, dtype=np.float64)
    c = np.ones(tiny, dtype=np.float64)
    obs = run_observed_triad(
        kernels,
        serial=False,
        ap=a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bp=b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cp=c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scalar=3.0,
        n=tiny,
        reps=1,
        requested_threads=requested,
    )
    applied = obs.omp_num_threads_actual == requested
    if not applied:
        return {
            "mode": "mmap_file",
            "pid": os.getpid(),
            "status": "INVALID",
            "thread_flag": "THREAD COUNT NOT APPLIED",
            "requested_threads": requested,
            "omp_num_threads_actual": obs.omp_num_threads_actual,
            "n_distinct_cpus": obs.n_distinct_cpus,
            "cpu_ids": obs.cpu_ids,
            "thread_count_applied": False,
            "label": label,
        }

    chunk = 8 * 1024 * 1024
    pattern = bytes((i * 17 + 31) & 0xFF for i in range(65536))
    pattern = (pattern * (chunk // len(pattern) + 1))[:chunk]
    write_t0 = now()
    written = 0
    with open(path, "wb") as fh:
        while written < need:
            to_write = min(chunk, need - written)
            fh.write(pattern[:to_write])
            written += to_write
        fh.flush()
        os.fsync(fh.fileno())
    write_dt = now() - write_t0

    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            fadvise_ev = f"os.posix_fadvise({path!r}, POSIX_FADV_DONTNEED) OK"
        finally:
            os.close(fd)
    except (AttributeError, OSError) as exc:
        fadvise_ev = f"UNDETECTED (reason: posix_fadvise failed: {exc})"

    stride = 4096
    pages = (need + stride - 1) // stride

    def one_pass() -> dict[str, Any]:
        min0, maj0, flt0 = read_minflt_majflt()
        io0, io0e = read_io_read_bytes()
        t0 = now()
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            arr = np.frombuffer(mm, dtype=np.uint8, count=need)
            try:
                acc = kernels.lib.stride_touch_addr(
                    ctypes.c_uint64(arr.ctypes.data),
                    ctypes.c_size_t(need),
                    ctypes.c_size_t(stride),
                )
            finally:
                del arr
                mm.close()
        dt = now() - t0
        min1, maj1, flt1 = read_minflt_majflt()
        io1, io1e = read_io_read_bytes()
        read_delta = (io1 - io0) if (io0 is not None and io1 is not None) else None
        io_basis = (
            (read_delta / dt / 1e9)
            if (read_delta is not None and read_delta > 0 and dt > 0)
            else None
        )
        file_basis = (need / dt / 1e9) if dt > 0 else None
        if read_delta == 0:
            eff, basis, note = file_basis, "file_bytes / time", (
                "FULLY CACHED (io read_bytes delta = 0 is the proof)"
            )
        elif read_delta is not None and read_delta > 0:
            eff, basis, note = io_basis, "/proc/self/io read_bytes delta / time", (
                "block I/O observed via read_bytes delta"
            )
        else:
            eff, basis, note = None, "UNDETECTED", "read_bytes unavailable"
        return {
            "seconds": dt,
            "checksum": int(acc),
            "minflt_before": min0,
            "minflt_after": min1,
            "minflt_delta": min1 - min0,
            "majflt_before": maj0,
            "majflt_after": maj1,
            "majflt_delta": maj1 - maj0,
            "io_read_bytes_before": io0,
            "io_read_bytes_after": io1,
            "io_read_bytes_delta": read_delta,
            "io_basis_gbs": io_basis,
            "file_basis_gbs": file_basis,
            "effective_read_gbs": eff,
            "pages_per_second": (pages / dt) if dt > 0 else None,
            "fault_evidence": [flt0, flt1],
            "io_evidence": [io0e, io1e],
            "bandwidth_basis": basis,
            "bandwidth_note": note,
            # Timed kernel is serial stride_touch; OpenMP verify was separate.
            "timed_kernel": "stride_touch_addr (serial; no OpenMP parallel region)",
            "omp_num_threads_actual": obs.omp_num_threads_actual,
            "n_distinct_cpus": obs.n_distinct_cpus,
        }

    cold = one_pass()
    warm = one_pass()
    try:
        os.remove(path)
    except OSError:
        pass

    return {
        "mode": "mmap_file",
        "pid": os.getpid(),
        "status": "OK",
        "thread_flag": "OK",
        "label": label,
        "path": path,
        "file_bytes": need,
        "requested_threads": requested,
        "omp_num_threads_actual": obs.omp_num_threads_actual,
        "omp_max_threads": obs.omp_max_threads,
        "n_distinct_cpus": obs.n_distinct_cpus,
        "cpu_ids": obs.cpu_ids,
        "thread_count_applied": True,
        "write_seconds": write_dt,
        "write_gbs_file_size_over_time": (need / write_dt / 1e9) if write_dt else None,
        "fadvise_evidence": fadvise_ev,
        "stride": stride,
        "pages": pages,
        "cold": cold,
        "warm": warm,
        "note": (
            "OpenMP verified at requested_threads via tiny triad before mmap; "
            "timed mmap kernel is serial stride_touch."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    from device_profile.phase2.subprocess_run import MARKER

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: bench_worker <config.json>", file=sys.stderr)
        return 2
    with open(args[0], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    _prep_omp_env(cfg)
    mode = cfg.get("mode", "stream_point")
    if mode == "stream_point":
        result = run_stream_point(cfg)
    elif mode == "stream_sustained":
        result = run_stream_sustained(cfg)
    elif mode == "mmap_file":
        result = run_mmap_file(cfg)
    else:
        result = {"status": "INVALID", "error": f"unknown mode {mode}"}
    sys.stdout.write(MARKER)
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
