"""Shared Phase 2 helpers: system snapshot, stats, affinity, L3 policy."""

from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


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


def snapshot(when: str, contaminate_threshold: float = 0.5) -> SystemSnapshot:
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


def rep_stats(samples: Sequence[float], cv_noisy_pct: float = 10.0) -> RepStats:
    xs = list(samples)
    if not xs:
        return RepStats([], float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), True)
    med = statistics.median(xs)
    mn = min(xs)
    mx = max(xs)
    mean = statistics.fmean(xs)
    if len(xs) >= 2:
        sd = statistics.pstdev(xs) if len(xs) < 2 else statistics.stdev(xs)
        # use sample stdev when n>=2
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
        # sysfs per-instance L3 is often the full L3 when shared
        l3 = profile.cache.l3_bytes.value

    logical = (
        profile.logical_cores.value
        if isinstance(profile.logical_cores.value, int)
        else None
    )
    suspect = bool(
        l3 is not None and logical is not None and l3 > 64 * 1024 * 1024 and logical <= 8
    )
    # Also treat as suspect if Phase 1.5 cross-check said SUSPECT
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


def read_majflt() -> tuple[int, str]:
    with open("/proc/self/stat", "r", encoding="utf-8") as fh:
        raw = fh.read()
    # comm can contain spaces/parens; split after last ')'
    rparen = raw.rfind(")")
    rest = raw[rparen + 2 :].split()
    # after state, fields: ppid ... minflt (index 7 in rest? )
    # Full fields after comm: state(0), ppid(1), pgrp(2), session(3), tty_nr(4),
    # tpgid(5), flags(6), minflt(7), cminflt(8), majflt(9), cmajflt(10), ...
    majflt = int(rest[9])
    return majflt, f"/proc/self/stat majflt field (rest[9])={majflt}; raw_tail={rest[7:11]}"


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
    return out


def find_knee(thread_counts: list[int], medians: list[float], thresh: float = 0.10) -> int:
    """
    Knee = last thread count where scaling vs previous is >= thresh (10%),
    or where adding threads gains < 10% relative to previous — return the
    count at/just before diminishing returns.
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
            # beyond previous is < 10% scaling
            return knee
    return knee


def now() -> float:
    return time.perf_counter()
