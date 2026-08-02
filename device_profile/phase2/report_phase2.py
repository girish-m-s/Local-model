"""Format Phase 2 measurement report."""

from __future__ import annotations

import json
from typing import Any


def _snap_lines(label: str, snap: dict[str, Any] | Any) -> list[str]:
    if snap is None:
        return [f"{label}: UNDETECTED"]
    if not isinstance(snap, dict):
        # SystemSnapshot dataclass
        d = {
            "when": snap.when,
            "loadavg": snap.loadavg_1_5_15,
            "mem_available_bytes": snap.mem_available_bytes,
            "mem_free_bytes": snap.mem_free_bytes,
            "contaminated": snap.contaminated,
            "evidence": snap.evidence,
        }
    else:
        d = snap
    lines = [
        f"{label}: loadavg={d.get('loadavg') or d.get('loadavg_1_5_15')} "
        f"MemAvailable={d.get('mem_available_bytes')} "
        f"MemFree={d.get('mem_free_bytes')} "
        f"{'CONTAMINATED' if d.get('contaminated') else 'OK'}"
    ]
    for ev in d.get("evidence") or []:
        lines.append(f"  <- {ev}")
    return lines


def format_phase2_report(bundle: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== PHASE 2 MEASUREMENT HARNESS REPORT ===")
    lines.append(f"HOST CLASS: {bundle.get('host_class')}")
    lines.append(f"  <- {bundle.get('host_class_evidence')}")
    if bundle.get("host_class") != "bare-metal":
        lines.append(
            "*** NON-BARE-METAL HOST — bandwidth/thermal numbers may be "
            "unrepresentative of an edge target ***"
        )
    lines.append(f"profile_effective_cores: {bundle.get('effective_cores')}")
    lines.append(f"profile_effective_mem_bytes: {bundle.get('effective_mem_bytes')}")
    lines.append(f"usable_tier (from Phase 1.5): {bundle.get('usable_tier')}")
    lines.append(f"total_runtime_s: {bundle.get('total_runtime_s'):.2f}")
    lines.append(f"kernel_compile: {bundle.get('kernel_compile_cmd')}")
    lines.append(f"  <- log: {bundle.get('kernel_compile_log')!r}")
    lines.append("")

    # BENCH 1
    b1 = bundle["bench1"]
    lines.append(f"=== {b1['name']} ===")
    lines.extend(_snap_lines("pre", b1["pre"]))
    lines.append(f"backend: {b1['backend']}")
    lines.append(f"  <- {b1['backend_evidence']}")
    lines.append(f"working_set_bytes: {b1['working_set_bytes']}")
    lines.append(f"  <- policy: {b1['working_set_policy']}")
    lines.append(f"l3_suspect_fallback: {b1['l3_suspect_fallback']}")
    cp = b1["cache_proof"]
    lines.append(
        f"cache_proof: ws={cp['working_set_bytes']} L3={cp['l3_bytes']} "
        f"ratio={cp['ratio_ws_over_l3']}"
    )
    lines.append(f"  <- {cp['evidence']}")
    for s in b1["thread_sweeps"]:
        noisy = " NOISY" if s["noisy"] else ""
        lines.append(
            f"threads={s['threads']}: median={s['median_gbs']:.4f} "
            f"min={s['min_gbs']:.4f} max={s['max_gbs']:.4f} "
            f"mean={s['mean_gbs']:.4f} CV%={s['cv_pct']:.2f}{noisy} GB/s"
        )
        lines.append(f"  raw_gbs: {s['raw_gbs']}")
        lines.append(f"  raw_times_s: {s['raw_times_s']}")
        lines.append(f"  <- OMP_NUM_THREADS={s['omp_num_threads']}; {s['affinity_evidence']}")
    sat = b1["saturation"]
    lines.append(
        f"saturation_thread_count: {sat['saturation_thread_count']} "
        f"(rule: {sat['rule']})"
    )
    lines.append(
        f"bandwidth_saturated / bandwidth_1thread: "
        f"{sat['bandwidth_saturated_median_gbs']:.4f} / "
        f"{sat['bandwidth_1thread_median_gbs']:.4f} = "
        f"{sat['ratio_sat_over_1']}"
    )
    lines.extend(_snap_lines("post", b1["post"]))
    lines.append("")

    # BENCH 2
    b2 = bundle["bench2"]
    lines.append(f"=== {b2['name']} ===")
    lines.extend(_snap_lines("pre", b2["pre"]))
    lines.append(f"L2/fit: {b2['fit_evidence']}")
    lines.append(f"isa_verification: {b2['isa_verification']}")
    if b2.get("hybrid_note"):
        lines.append(b2["hybrid_note"])
    for label, curve in b2["curves"].items():
        lines.append(f"-- curve: {label} --")
        lines.append(f"  <- affinity: {curve['affinity']}")
        for r in curve["fp32"]:
            noisy = " NOISY" if r["noisy"] else ""
            lines.append(
                f"  fp32 threads={r['threads']}: median={r['median_gflops']:.4f} "
                f"min={r['min_gflops']:.4f} max={r['max_gflops']:.4f} "
                f"CV%={r['cv_pct']:.2f}{noisy} GFLOPS"
            )
            lines.append(f"    raw_gflops: {r['raw_gflops']}")
            lines.append(f"    raw_times_s: {r['raw_times_s']}")
        for r in curve["int8"]:
            noisy = " NOISY" if r["noisy"] else ""
            lines.append(
                f"  int8 threads={r['threads']}: median={r['median_gops']:.4f} "
                f"min={r['min_gops']:.4f} max={r['max_gops']:.4f} "
                f"CV%={r['cv_pct']:.2f}{noisy} GOPS"
            )
            lines.append(f"    raw_gops: {r['raw_gops']}")
            lines.append(f"    raw_times_s: {r['raw_times_s']}")
        knee = b2["knee"][label]
        lines.append(
            f"  knee(fp32): {knee['fp32_knee_threads']}  <- {knee['rule']}"
        )
    if b2.get("hybrid_conclusion"):
        lines.append(f"hybrid_conclusion: {b2['hybrid_conclusion']}")
    lines.extend(_snap_lines("post", b2["post"]))
    lines.append("")

    # BENCH 3
    b3 = bundle["bench3"]
    lines.append(f"=== {b3['name']} ===")
    lines.extend(_snap_lines("pre", b3["pre"]))
    lines.append(f"sat_threads: {b3['sat_threads']}")
    lines.append(f"duration_s: {b3['duration_s']}; bucket_s: {b3['bucket_s']}")
    lines.append(f"battery_dual_run: {b3['battery_dual_run']}")
    for b in b3["buckets"]:
        lines.append(
            f"bucket[{b['bucket']}]: gflops={b['gflops']:.4f} "
            f"calls={b['calls']} elapsed_s={b['elapsed_s']:.3f} "
            f"temps={b.get('temps_C')} freqs_kHz={b.get('freqs_kHz')}"
        )
        for ev in b.get("thermal_evidence") or []:
            lines.append(f"  <- {ev}")
    sm = b3["summary"]
    lines.append(
        f"peak_bucket_gflops: {sm['peak_bucket_gflops']:.4f}; "
        f"sustained_last3_median: {sm['sustained_last3_median_gflops']:.4f}; "
        f"sustained/peak: {sm['sustained_over_peak']}"
    )
    lines.append(
        f"time_to_throttle_s: {sm['time_to_throttle_s']}  <- {sm['throttle_rule']}"
    )
    lines.extend(_snap_lines("post", b3["post"]))
    lines.append("")

    # BENCH 4
    b4 = bundle["bench4"]
    lines.append(f"=== {b4['name']} ===")
    lines.extend(_snap_lines("pre", b4["pre"]))
    lines.append(
        f"required_bytes: {b4['required_bytes']} "
        f"(2x effective_mem={b4['effective_mem_bytes']})"
    )
    lines.append(
        f"disk free/total on {b4['cache_dir']}: {b4['disk_free']} / {b4['disk_total']}"
    )
    lines.append(f"status: {b4['status']}")
    if b4["status"] == "SKIPPED":
        lines.append(f"  <- SKIPPED: {b4.get('skip_reason')}")
    else:
        lines.append(
            f"write: {b4['write_seconds']:.3f}s ({b4['write_gbs']:.4f} GB/s)"
        )
        lines.append(
            f"cold-read: {b4['cold_seconds']:.3f}s ({b4['cold_gbs']:.4f} GB/s); "
            f"warm: {b4['warm_seconds']:.3f}s ({b4['warm_gbs']:.4f} GB/s)"
        )
        if b4.get("cold_touch_gbs") is not None:
            lines.append(
                f"  <- touch_gbs (1B/page): cold={b4['cold_touch_gbs']:.6f} "
                f"warm={b4['warm_touch_gbs']:.6f}; "
                f"pages={b4.get('pages_touched_per_pass')}"
            )
        lines.append(
            f"majflt cold_delta={b4['majflt_cold_delta']} "
            f"warm_delta={b4['majflt_warm_delta']}"
        )
        for ev in b4.get("majflt_evidence") or []:
            lines.append(f"  <- {ev}")
        lines.append(f"  <- {b4.get('fadvise_evidence')}")
        lines.append(f"note: {b4.get('note')}")
    lines.extend(_snap_lines("post", b4["post"]))
    lines.append("")

    # DERIVED
    d = bundle["derived"]
    lines.append("=== SECTION D — DERIVED OUTPUT ===")
    lines.append(
        f"measured_bandwidth_GBps (BENCH1 saturated median): "
        f"{d['measured_bandwidth_GBps']}"
    )
    lines.append(
        "predicted_decode_tok_s(model_bytes) = measured_bandwidth_GBps * 1e9 / model_bytes"
    )
    lines.append(
        "UPPER BOUND assuming perfect bandwidth utilization. Real engines typically "
        "hit 40-70% of it."
    )
    lines.append(
        f"{'model_GB':>10}  {'upper_tok_s':>14}  {'realistic_0.55x_tok_s':>22}"
    )
    for row in d["predictions"]:
        lines.append(
            f"{row['model_gb']:10.1f}  {row['upper_tok_s']:14.2f}  "
            f"{row['realistic_0_55x_tok_s']:22.2f}"
        )
    lines.append(
        "0.55x realistic band: UNVALIDATED ASSUMPTION — to be checked in Phase 4 "
        "against a real engine."
    )
    lines.append(
        f"recommended_thread_count: {d['recommended_thread_count']}  "
        f"<- {d['recommended_thread_count_evidence']}"
    )
    lines.append(
        f"recommended_thread_count_sustained: {d['recommended_thread_count_sustained']}  "
        f"<- {d['recommended_thread_count_sustained_evidence']}"
    )
    lines.append("")

    lines.append("=== SELF-CRITIQUE ===")
    for key, items in bundle["self_critique"].items():
        lines.append(f"- {key}:")
        for item in items:
            lines.append(f"  - {item}")
    lines.append("")

    lines.append("=== PHASE 2 JSON ===")
    lines.append(json.dumps(bundle, indent=2, default=str))
    return "\n".join(lines)
