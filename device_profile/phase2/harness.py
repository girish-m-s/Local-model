"""Phase 2 orchestrator — synthetic microbenchmarks only."""

from __future__ import annotations

import sys
import time
from typing import Any

from device_profile.detect import detect_device_profile

from .bench_compute import run_bench2
from .bench_mmap import run_bench4
from .bench_stream import run_bench1
from .bench_sustained import run_bench3
from .native import build_kernels
from .report_phase2 import format_phase2_report
from .util import effective_cores, effective_mem


def run_phase2() -> dict[str, Any]:
    t_wall0 = time.time()
    profile = detect_device_profile()
    # steal sampling in detect is 5s — already spent

    kernels = build_kernels()
    cores = effective_cores(profile)
    emem = effective_mem(profile)

    b1 = run_bench1(profile, kernels)
    b2 = run_bench2(profile, kernels)

    sat_threads = b2["knee"]["all_cores"]["fp32_knee_threads"]
    # Prefer compute knee; also record bandwidth sat
    bw_sat = b1["saturation"]["saturation_thread_count"]

    # Time budget: keep BENCH3 at 180s; if remaining wall would exceed 10 min
    # after estimated BENCH4, still run 180s as required by spec.
    elapsed = time.time() - t_wall0
    remaining = 600.0 - elapsed
    duration = 180.0
    if remaining < 200:
        # Still run required 180s; may slightly exceed 10 min — note it.
        duration = 180.0

    b3 = run_bench3(
        profile,
        kernels,
        sat_threads=sat_threads,
        n_fp32=b2["n_fp32"],
        reps_fp32=b2["reps_fp32"],
        duration_s=duration,
        bucket_s=10.0,
    )

    b4 = run_bench4(profile, kernels)

    measured_bw = b1["saturation"]["bandwidth_saturated_median_gbs"]
    predictions = []
    for gb in (0.5, 1, 2, 4, 8, 16):
        model_bytes = int(gb * 1e9)  # decimal GB as labeled
        upper = measured_bw * 1e9 / model_bytes
        predictions.append(
            {
                "model_gb": gb,
                "model_bytes": model_bytes,
                "upper_tok_s": upper,
                "realistic_0_55x_tok_s": upper * 0.55,
            }
        )

    # Sustained thread recommendation: if throttled, try max(1, knee-1)
    throttle_t = b3["summary"]["time_to_throttle_s"]
    sust_ratio = b3["summary"]["sustained_over_peak"]
    if sust_ratio is not None and sust_ratio < 0.90:
        rec_sust = max(1, sat_threads - 1)
        sust_ev = (
            f"sustained/peak={sust_ratio:.3f} < 0.90 "
            f"(time_to_throttle_s={throttle_t}); "
            f"recommend knee-1 = {rec_sust}"
        )
    else:
        rec_sust = sat_threads
        sust_ev = (
            f"sustained/peak={sust_ratio}; no >10% drop observed → "
            f"same as compute knee ({sat_threads})"
        )

    total_runtime = time.time() - t_wall0
    contaminated_flags = []
    for name, b in (("bench1", b1), ("bench2", b2), ("bench3", b3), ("bench4", b4)):
        pre = b.get("pre")
        if pre is not None and getattr(pre, "contaminated", False):
            contaminated_flags.append(f"{name} start CONTAMINATED (loadavg>0.5)")

    host_class = (
        profile.host_class.value
        if profile.host_class.is_detected()
        else str(profile.host_class.value)
    )

    bundle: dict[str, Any] = {
        "host_class": host_class,
        "host_class_evidence": profile.host_class.evidence,
        "effective_cores": cores,
        "effective_mem_bytes": emem,
        "usable_tier": (
            profile.isa.usable_tier.value
            if profile.isa.usable_tier.is_detected()
            else profile.isa.usable_tier.value
        ),
        "kernel_compile_cmd": kernels.compile_cmd,
        "kernel_compile_log": kernels.compile_log,
        "total_runtime_s": total_runtime,
        "runtime_budget_s": 600,
        "runtime_over_budget": total_runtime > 600,
        "bench1": b1,
        "bench2": b2,
        "bench3": b3,
        "bench4": b4,
        "derived": {
            "measured_bandwidth_GBps": measured_bw,
            "bandwidth_saturation_threads": bw_sat,
            "predictions": predictions,
            "recommended_thread_count": sat_threads,
            "recommended_thread_count_evidence": (
                f"BENCH 2 fp32 knee on all_cores curve = {sat_threads} "
                f"(not raw core count {cores}); bw_sat_threads={bw_sat}"
            ),
            "recommended_thread_count_sustained": rec_sust,
            "recommended_thread_count_sustained_evidence": sust_ev,
        },
        "self_critique": {
            "measurements likely contaminated by other processes": contaminated_flags
            or [
                "loadavg threshold 0.5 is strict on multi-tenant cloud; "
                "check CONTAMINATED tags above"
            ],
            "likely wrong under virtualization": [
                "BENCH 1 GB/s reflects guest/cgroup memory path, not host DRAM controllers",
                "BENCH 3 thermal/cpufreq sysfs often absent under KVM — throttle detection blind",
                "BENCH 4 overlay FS + virtio block ≠ bare-metal NVMe latency",
                "steal time / noisy neighbors can inflate CV (see NOISY tags)",
            ],
            "where a benchmark may have been served from cache rather than RAM": [
                f"BENCH 1 working_set={b1['working_set_bytes']} vs L3="
                f"{b1['cache_proof']['l3_bytes']} "
                f"(suspect_fallback={b1['l3_suspect_fallback']})",
                "BENCH 2 intentionally L2-resident — compute-bound by design, not DRAM",
                "BENCH 4 uses posix_fadvise(DONTNEED) before cold pass; without root "
                "drop_caches, eviction is best-effort and majflt may under-count",
            ],
            "ISA tier verification": [
                b2["isa_verification"],
                f"Phase 1.5 usable_tier={profile.isa.usable_tier.value} was NOT "
                "exercised by these scalar C/+OpenMP kernels",
            ],
            "other": [
                f"total_runtime_s={total_runtime:.1f} budget=600 "
                f"over={total_runtime > 600}",
                "0.55x decode band is an unvalidated assumption pending Phase 4",
                "BENCH 4 file size uses decimal 2*effective_mem; disk check is hard fail/SKIP",
            ],
        },
    }
    return bundle


def main(argv: list[str] | None = None) -> int:
    _ = argv
    bundle = run_phase2()
    sys.stdout.write(format_phase2_report(bundle))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
