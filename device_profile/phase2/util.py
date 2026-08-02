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
    evidence: list[str] = field(default_factory=list)


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


def snapshot(when: str, contaminate_threshold: float = 0.3) -> SystemSnapshot:
    load, load_ev = read_loadavg()
    mem, mem_ev = read_meminfo_bytes()
    contaminated = load[0] > contaminate_threshold
    return SystemSnapshot(
        when=when,
        loadavg_1_5_15=load,
        mem_available_bytes=mem.get("MemAvailable"),
        mem_free_bytes=mem.get("MemFree"),
        mem_total_bytes=mem.get("MemTotal"),
        contaminated=contaminated,
        evidence=[load_ev, *mem_ev],
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
        if "PHASE2_LOAD_THRESHOLD" in os.environ:
            threshold = float(os.environ["PHASE2_LOAD_THRESHOLD"])
            override_notes.append(f"PHASE2_LOAD_THRESHOLD={threshold} (OVERRIDE)")
        else:
            threshold = 0.3
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
    BENCH 1 buffer policy: 4x L3, or 512 MiB if L3 SUSPECT.
    Returns (bytes, evidence, used_suspect_fallback).
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

    if suspect or l3 is None:
        ws = 512 * 1024 * 1024
        evid = (
            f"L3 SUSPECT or undetected (l3={l3}, logical_cores={logical}); "
            f"using 512 MiB working set as required by Phase 2 spec"
        )
        return ws, evid, True

    ws = 4 * l3
    evid = f"working_set = 4 * L3 = 4 * {l3} = {ws}"
    return ws, evid, False


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
    thread_counts: Sequence[int], medians: Sequence[float]
) -> dict[str, Any]:
    """
    If bandwidth(n)/bandwidth(1) > n for any n, curve is SUPERLINEAR — INVALID.
    """
    violations: list[str] = []
    one = None
    for t, m in zip(thread_counts, medians):
        if t == 1:
            one = m
            break
    if one is None or one <= 0:
        return {
            "valid": False,
            "superlinear": False,
            "violations": ["no positive 1-thread median"],
            "flag": "INVALID — missing 1-thread baseline",
        }
    for t, m in zip(thread_counts, medians):
        if t <= 1:
            continue
        ratio = m / one
        if ratio > t + 1e-9:
            violations.append(
                f"threads={t}: median/median(1)={ratio:.6f} > {t} "
                f"(SUPERLINEAR — INVALID)"
            )
    if violations:
        return {
            "valid": False,
            "superlinear": True,
            "violations": violations,
            "flag": (
                "SUPERLINEAR — INVALID: efficiency > 100% implies undisclosed "
                "work-unit change, cache cliff, or measurement bug; "
                "do not report saturation from this curve"
            ),
        }
    return {
        "valid": True,
        "superlinear": False,
        "violations": [],
        "flag": "OK",
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
