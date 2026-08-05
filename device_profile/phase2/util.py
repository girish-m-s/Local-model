"""Shared Phase 2 helpers: system snapshot, stats, affinity, L3 policy."""

from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

GIB = 1 << 30  # 2^30 — GGUF sizes are quoted in GiB


@dataclass
class SystemSnapshot:
    when: str
    loadavg_1_5_15: tuple[float, float, float]
    mem_available_bytes: Optional[int]
    mem_free_bytes: Optional[int]
    mem_total_bytes: Optional[int]
    contaminated: bool
    load_threshold: float
    threshold_overridden: bool
    evidence: list[str] = field(default_factory=list)


def get_load_threshold() -> tuple[float, bool]:
    """Single threshold for all snapshots / quiet gates. Returns (threshold, overridden)."""
    if "PHASE2_LOAD_THRESHOLD" in os.environ:
        return float(os.environ["PHASE2_LOAD_THRESHOLD"]), True
    return 0.3, False


@dataclass
class RepStats:
    samples: list[float]
    median: float
    minimum: float
    maximum: float
    mean: float
    cv_pct: float
    noisy: bool


@dataclass
class QuietGateResult:
    ok: bool
    status: str  # OK | BLOCKED
    readings: list[tuple[float, float, float]]
    snapshot: SystemSnapshot
    evidence: list[str]


def read_loadavg() -> tuple[tuple[float, float, float], str]:
    with open("/proc/loadavg", "r", encoding="utf-8") as fh:
        line = fh.read().strip()
    parts = line.split()
    vals = (float(parts[0]), float(parts[1]), float(parts[2]))
    return vals, f"/proc/loadavg: {line}"


def read_meminfo_bytes() -> tuple[dict[str, int], list[str]]:
    out: dict[str, int] = {}
    evid: list[str] = []
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            for key in ("MemTotal", "MemFree", "MemAvailable"):
                if line.startswith(key + ":"):
                    kb = int(line.split()[1])
                    out[key] = kb * 1024
                    evid.append(line.rstrip())
    return out, evid


def snapshot(
    when: str, contaminate_threshold: float | None = None
) -> SystemSnapshot:
    default_thr, default_over = get_load_threshold()
    if contaminate_threshold is None:
        threshold = default_thr
        overridden = default_over
    else:
        threshold = contaminate_threshold
        # Explicit arg that differs from default env/default still marks OVERRIDE
        # when it came from PHASE2_LOAD_THRESHOLD or caller override path.
        overridden = default_over or (abs(threshold - 0.3) > 1e-12)
    load, load_ev = read_loadavg()
    mem, mem_ev = read_meminfo_bytes()
    contaminated = load[0] > threshold
    tag = "OVERRIDE" if overridden else "default"
    thr_ev = (
        f"load_threshold={threshold} ({tag}); contaminated={contaminated} "
        f"(loadavg1={load[0]} > {threshold})"
    )
    return SystemSnapshot(
        when=when,
        loadavg_1_5_15=load,
        mem_available_bytes=mem.get("MemAvailable"),
        mem_free_bytes=mem.get("MemFree"),
        mem_total_bytes=mem.get("MemTotal"),
        contaminated=contaminated,
        load_threshold=threshold,
        threshold_overridden=overridden,
        evidence=[load_ev, thr_ev, *mem_ev],
    )


def ensure_quiet_load(
    when: str,
    *,
    threshold: float | None = None,
    max_attempts: int | None = None,
    sleep_s: float | None = None,
) -> QuietGateResult:
    """
    BUG 5: if loadavg(1m) > threshold, sleep and re-check up to max_attempts.
    If still high, ABORT with BLOCKED — do not run the bench.

    Optional env overrides (must be printed as evidence if used):
      PHASE2_LOAD_THRESHOLD, PHASE2_QUIET_ATTEMPTS, PHASE2_QUIET_SLEEP_S
    """
    override_notes: list[str] = []
    if threshold is None:
        threshold, overridden = get_load_threshold()
        if overridden:
            override_notes.append(f"PHASE2_LOAD_THRESHOLD={threshold} (OVERRIDE)")
    if max_attempts is None:
        if "PHASE2_QUIET_ATTEMPTS" in os.environ:
            max_attempts = int(os.environ["PHASE2_QUIET_ATTEMPTS"])
            override_notes.append(f"PHASE2_QUIET_ATTEMPTS={max_attempts} (OVERRIDE)")
        else:
            max_attempts = 5
    if sleep_s is None:
        if "PHASE2_QUIET_SLEEP_S" in os.environ:
            sleep_s = float(os.environ["PHASE2_QUIET_SLEEP_S"])
            override_notes.append(f"PHASE2_QUIET_SLEEP_S={sleep_s} (OVERRIDE)")
        else:
            sleep_s = 30.0

    readings: list[tuple[float, float, float]] = []
    evid: list[str] = list(override_notes)
    snap: SystemSnapshot | None = None
    for attempt in range(1, max_attempts + 1):
        snap = snapshot(f"{when}_attempt{attempt}", contaminate_threshold=threshold)
        readings.append(snap.loadavg_1_5_15)
        evid.append(
            f"attempt {attempt}/{max_attempts}: loadavg1={snap.loadavg_1_5_15[0]} "
            f"threshold={threshold}; " + "; ".join(snap.evidence)
        )
        if snap.loadavg_1_5_15[0] <= threshold:
            return QuietGateResult(
                ok=True,
                status="OK",
                readings=readings,
                snapshot=snap,
                evidence=evid,
            )
        if attempt < max_attempts:
            evid.append(f"sleeping {sleep_s}s before re-check")
            time.sleep(sleep_s)
    assert snap is not None
    evid.append(
        f"ABORT BLOCKED: loadavg(1m) still > {threshold} after {max_attempts} attempts; "
        f"readings={readings}"
    )
    return QuietGateResult(
        ok=False,
        status="BLOCKED",
        readings=readings,
        snapshot=snap,
        evidence=evid,
    )


def rep_stats(samples: Sequence[float], cv_noisy_pct: float = 10.0) -> RepStats:
    xs = list(samples)
    if not xs:
        return RepStats(
            [], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), True
        )
    med = statistics.median(xs)
    mn = min(xs)
    mx = max(xs)
    mean = statistics.fmean(xs)
    if len(xs) >= 2:
        sd = statistics.stdev(xs)
        cv = (sd / mean * 100.0) if mean != 0 else float("inf")
    else:
        cv = float("nan")
    return RepStats(
        samples=xs,
        median=med,
        minimum=mn,
        maximum=mx,
        mean=mean,
        cv_pct=cv,
        noisy=(cv > cv_noisy_pct) if not math.isnan(cv) else True,
    )


def resolve_working_set_bytes(profile: Any) -> tuple[int, str, bool]:
    """
    Interim buffer sizing for diagnostics/benches before DIAG B escape is known.
    Phase 2.3: the authoritative policy is set FROM the DIAG B escape point
    (see diag.run_diag_b); this helper must not invent a floor that disagrees
    with a measured escape. Uses max(4*L3, 1 GiB) only as a pre-escape default
    matching the observed near-floor consecutive pair at 4×/8× on this host.
    """
    l3 = None
    if profile.cache.lscpu_l3_bytes is not None and isinstance(
        profile.cache.lscpu_l3_bytes.value, int
    ):
        l3 = profile.cache.lscpu_l3_bytes.value
    elif isinstance(profile.cache.l3_bytes.value, int):
        l3 = profile.cache.l3_bytes.value

    logical = (
        profile.logical_cores.value
        if isinstance(profile.logical_cores.value, int)
        else None
    )
    suspect = bool(
        l3 is not None and logical is not None and l3 > 64 * 1024 * 1024 and logical <= 8
    )
    for cc in profile.cross_checks:
        if "plausible" in cc.name.lower() and cc.result == "SUSPECT":
            suspect = True

    if l3 is None:
        ws = GIB
        evid = (
            f"L3 undetected (logical_cores={logical}); "
            f"working_set = 1 GiB pre-escape default; authoritative policy from DIAG B escape"
        )
        return ws, evid, True

    ws = max(4 * l3, GIB)
    evid = (
        f"pre-escape default working_set = max(4 * L3_reported, 1 GiB) = "
        f"max(4*{l3}, {GIB}) = {ws}"
        + (f"; L3_SUSPECT=True (logical_cores={logical})" if suspect else "")
        + "; authoritative policy MUST come from DIAG B escape (Phase 2.3 Step 5)"
    )
    return ws, evid, suspect


def effective_cores(profile: Any) -> int:
    if profile.exec_env and isinstance(profile.exec_env.effective_cores.value, (int, float)):
        return max(1, int(profile.exec_env.effective_cores.value))
    if isinstance(profile.logical_cores.value, int):
        return max(1, profile.logical_cores.value)
    return max(1, os.cpu_count() or 1)


def effective_mem(profile: Any) -> int:
    if profile.exec_env and isinstance(profile.exec_env.effective_mem_bytes.value, int):
        return profile.exec_env.effective_mem_bytes.value
    if isinstance(profile.mem_total_bytes.value, int):
        return profile.mem_total_bytes.value
    return 0


def l2_bytes_per_core(profile: Any) -> int:
    if isinstance(profile.cache.l2_bytes.value, int):
        return profile.cache.l2_bytes.value
    return 256 * 1024  # UNDETECTED fallback stated by caller


def set_omp_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OMP_DYNAMIC"] = "false"


def set_omp_env(
    *,
    threads: int | None,
    proc_bind: str | None,
    places: str | None,
) -> None:
    """Set or clear OpenMP env knobs. None clears the variable."""
    os.environ["OMP_DYNAMIC"] = "false"
    if threads is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    if proc_bind is None:
        os.environ.pop("OMP_PROC_BIND", None)
    else:
        os.environ["OMP_PROC_BIND"] = proc_bind
    if places is None:
        os.environ.pop("OMP_PLACES", None)
    else:
        os.environ["OMP_PLACES"] = places


def read_steal_jiffies() -> tuple[Optional[int], str]:
    """Aggregate steal jiffies from /proc/stat cpu line (field index 8)."""
    path = "/proc/stat"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("cpu "):
                    parts = line.split()
                    # cpu user nice system idle iowait irq softirq steal guest guest_nice
                    if len(parts) < 9:
                        return None, f"UNDETECTED (reason: {path} cpu line short: {line!r})"
                    steal = int(parts[8])
                    return steal, f"{path} cpu steal_jiffies={steal}; raw={line.rstrip()!r}"
        return None, f"UNDETECTED (reason: no cpu line in {path})"
    except OSError as exc:
        return None, f"UNDETECTED (reason: {path} failed: {exc})"


def pin_to_cpus(cpus: list[int]) -> str:
    """sched_setaffinity current process; return evidence string."""
    try:
        os.sched_setaffinity(0, set(cpus))
        got = sorted(os.sched_getaffinity(0))
        return f"os.sched_setaffinity(0, {sorted(cpus)}) → affinity={got}"
    except (AttributeError, OSError) as exc:
        return f"UNDETECTED (reason: sched_setaffinity failed: {exc})"


def read_minflt_majflt() -> tuple[int, int, str]:
    with open("/proc/self/stat", "r", encoding="utf-8") as fh:
        raw = fh.read()
    rparen = raw.rfind(")")
    rest = raw[rparen + 2 :].split()
    # after state: ... minflt(7), cminflt(8), majflt(9), cmajflt(10)
    minflt = int(rest[7])
    majflt = int(rest[9])
    return (
        minflt,
        majflt,
        f"/proc/self/stat minflt={minflt} majflt={majflt}; raw_tail={rest[7:11]}",
    )


def read_io_read_bytes() -> tuple[Optional[int], str]:
    path = "/proc/self/io"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return None, f"UNDETECTED (reason: {path} failed: {exc})"
    for line in text.splitlines():
        if line.startswith("read_bytes:"):
            val = int(line.split(":")[1].strip())
            return val, f"{path} read_bytes={val}; raw_line={line!r}"
    return None, f"UNDETECTED (reason: read_bytes not in {path}; raw={text!r})"


def check_superlinear(
    thread_counts: Sequence[int],
    medians: Sequence[float],
    *,
    baseline_gbs: float | None = None,
    baseline_label: str = "median(1)",
) -> dict[str, Any]:
    """
    If bandwidth(n)/baseline > n for any n, curve is SUPERLINEAR.
    baseline defaults to median(1); DIAG C passes A1 serial instead.
    """
    violations: list[str] = []
    one = baseline_gbs
    if one is None:
        for t, m in zip(thread_counts, medians):
            if t == 1:
                one = m
                break
    if one is None or one <= 0:
        return {
            "valid": False,
            "superlinear": False,
            "violations": [f"no positive baseline ({baseline_label})"],
            "flag": "INVALID — missing baseline",
            "baseline_gbs": one,
            "baseline_label": baseline_label,
        }
    for t, m in zip(thread_counts, medians):
        # Efficiency vs own median(1): skip t==1. Vs external A1: only t>=2
        # (t==1 vs A1 is a caveat, not SUPERLINEAR).
        if t <= 1:
            continue
        ratio = m / one
        if ratio > t + 1e-9:
            violations.append(
                f"threads={t}: median/{baseline_label}={ratio:.6f} > {t} "
                f"(SUPERLINEAR)"
            )
    if violations:
        return {
            "valid": False,
            "superlinear": True,
            "violations": violations,
            "flag": (
                f"SUPERLINEAR vs {baseline_label}: efficiency > 100%; "
                "implies undisclosed work-unit change, cache cliff, or measurement bug"
            ),
            "baseline_gbs": one,
            "baseline_label": baseline_label,
        }
    return {
        "valid": True,
        "superlinear": False,
        "violations": [],
        "flag": "OK",
        "baseline_gbs": one,
        "baseline_label": baseline_label,
    }


def evaluate_curve_validity(
    thread_counts: Sequence[int],
    medians: Sequence[float],
    *,
    serial_baseline_gbs: float,
    cores: int,
) -> dict[str, Any]:
    """
    Phase 2.3: do NOT discount SUPERLINEAR-vs-A1 or adopt OpenMP@1 as baseline.
    Keep serial baseline. Absolute per-core bound is a FLAG, not a footnote.
    """
    flags: list[str] = []
    # Absolute per-core plausibility (HEURISTIC bounds) — FLAG when fired
    abs_lo, abs_hi = 1.0, 60.0
    for t, m in zip(thread_counts, medians):
        per_core = m / max(t, 1)
        if per_core > abs_hi:
            flags.append(
                f"FLAG absolute: threads={t}: per-core={per_core:.3f} GB/s > {abs_hi} "
                f"(HEURISTIC upper bound)"
            )
        if per_core < abs_lo:
            flags.append(
                f"FLAG absolute: threads={t}: per-core={per_core:.3f} GB/s < {abs_lo} "
                f"(HEURISTIC lower bound)"
            )

    # Monotonicity: allow 5% noise
    for i in range(1, len(medians)):
        if medians[i] + 1e-12 < medians[i - 1] * 0.95:
            flags.append(
                f"FLAG non-monotonic: threads {thread_counts[i-1]}→{thread_counts[i]} "
                f"{medians[i-1]:.3f}→{medians[i]:.3f} (>5% drop)"
            )

    sl = check_superlinear(
        thread_counts,
        medians,
        baseline_gbs=serial_baseline_gbs,
        baseline_label="A1_serial",
    )
    if sl["superlinear"]:
        flags.extend([f"FLAG {v}" for v in sl["violations"]])

    if flags:
        status = "INVALID"
    else:
        status = "VALID"

    return {
        "status": status,
        "flags": flags,
        "caveats": [],
        "absolute_bounds_GBps_per_core": {
            "lo": abs_lo,
            "hi": abs_hi,
            "note": "HEURISTIC — FLAG when exceeded, not a soft caveat",
        },
        "superlinear_vs_A1": sl,
        "baseline_policy": (
            "serial A1 baseline retained (Phase 2.3); "
            "OpenMP@1 must NOT replace it; SUPERLINEAR-vs-A1 is NOT discounted"
        ),
        "cores": cores,
    }


def assert_gbs_reconstructs(
    reported_gbs: float,
    reps: int,
    bytes_moved_per_iter: int,
    time_s: float,
    *,
    tol: float = 0.001,
) -> dict[str, Any]:
    expected = (reps * bytes_moved_per_iter) / time_s / 1e9 if time_s > 0 else float("nan")
    if reported_gbs == 0 or math.isnan(reported_gbs) or math.isnan(expected):
        rel = float("inf")
        ok = False
    else:
        rel = abs(reported_gbs - expected) / reported_gbs
        ok = rel < tol
    return {
        "ok": ok,
        "reported_gbs": reported_gbs,
        "expected_gbs": expected,
        "rel_err": rel,
        "reps": reps,
        "bytes_moved_per_iter": bytes_moved_per_iter,
        "time_s": time_s,
        "message": (
            "OK"
            if ok
            else (
                f"FAIL: abs(reported_gbs - reps*bytes/time)/reported_gbs = {rel} "
                f">= {tol} (reported={reported_gbs}, expected={expected}, "
                f"reps={reps}, bytes={bytes_moved_per_iter}, time={time_s})"
            )
        ),
    }


def sample_thermal_and_freq() -> dict[str, Any]:
    out: dict[str, Any] = {"temps_C": {}, "freqs_kHz": {}, "evidence": []}
    evid: list[str] = []
    tz_root = "/sys/class/thermal"
    try:
        for name in sorted(os.listdir(tz_root)):
            if not name.startswith("thermal_zone"):
                continue
            try:
                with open(f"{tz_root}/{name}/type", encoding="utf-8") as fh:
                    typ = fh.read().strip()
                with open(f"{tz_root}/{name}/temp", encoding="utf-8") as fh:
                    raw = fh.read().strip()
                out["temps_C"][f"{name}:{typ}"] = int(raw) / 1000.0
                evid.append(f"{tz_root}/{name}/type={typ}; temp={raw}")
            except (OSError, ValueError):
                continue
    except OSError as exc:
        evid.append(f"thermal list failed: {exc}")

    cf_root = "/sys/devices/system/cpu/cpufreq"
    try:
        for pol in sorted(os.listdir(cf_root)):
            if not pol.startswith("policy"):
                continue
            path = f"{cf_root}/{pol}/scaling_cur_freq"
            try:
                with open(path, encoding="utf-8") as fh:
                    val = fh.read().strip()
                out["freqs_kHz"][pol] = int(val)
                evid.append(f"{path}: {val}")
            except (OSError, ValueError):
                continue
    except OSError as exc:
        evid.append(f"cpufreq list failed: {exc}")

    if not out["temps_C"]:
        evid.append("no thermal_zone* readable")
    if not out["freqs_kHz"]:
        evid.append("no scaling_cur_freq readable")
    out["evidence"] = evid
    out["thermal_blind"] = (not out["temps_C"]) and (not out["freqs_kHz"])
    return out


def find_knee(thread_counts: list[int], medians: list[float], thresh: float = 0.10) -> int:
    """
    Knee = last thread count where scaling vs previous is >= thresh (10%).
    Spec: 'thread count beyond which scaling is < 10%'.
    """
    if not thread_counts:
        return 1
    knee = thread_counts[0]
    for i in range(1, len(thread_counts)):
        prev = medians[i - 1]
        cur = medians[i]
        if prev <= 0:
            knee = thread_counts[i]
            continue
        gain = (cur - prev) / prev
        if gain >= thresh:
            knee = thread_counts[i]
        else:
            return knee
    return knee


def now() -> float:
    return time.perf_counter()
