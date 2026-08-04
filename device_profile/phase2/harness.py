"""Phase 2.1 orchestrator — synthetic microbenchmarks only (bugfix)."""

from __future__ import annotations

import os
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
from .util import GIB, effective_cores, effective_mem


# Representative dense 7B-class decoder configs. Stated explicitly — not from a GGUF.
# Primary = GQA (matches shipping GGUF-class models); secondary = full MHA.
REP_7B_GQA = {
    "name": "representative_dense_7B_GQA_primary",
    "attention": "GQA",
    "n_layers": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "bytes_per_element": 2,  # fp16 KV
    "model_weight_GiB": 7.0,
    "note": (
        "PRIMARY: GQA (n_kv_heads=8). Dense decode: every weight every token + "
        "KV up to n_ctx. NOT a real GGUF measurement."
    ),
}
REP_7B_MHA = {
    "name": "representative_dense_7B_MHA_secondary",
    "attention": "MHA",
    "n_layers": 32,
    "n_kv_heads": 32,
    "head_dim": 128,
    "bytes_per_element": 2,
    "model_weight_GiB": 7.0,
    "note": (
        "SECONDARY: full MHA (n_kv_heads=32). Overstates KV ~4x vs GQA; kept for "
        "comparison only."
    ),
}


def run_phase2() -> dict[str, Any]:
    t_wall0 = time.time()
    profile = detect_device_profile()

    kernels = build_kernels()
    cores = effective_cores(profile)
    emem = effective_mem(profile)

    b1 = run_bench1(profile, kernels)
    b2 = run_bench2(profile, kernels)

    bw_sat = b1.get("saturation", {}).get("saturation_thread_count")
    knee = None
    if b2.get("knee", {}).get("all_cores") and not b2["knee"]["all_cores"].get("invalid"):
        knee = b2["knee"]["all_cores"].get("fp32_knee_threads")

    duration = 180.0
    if "PHASE2_BENCH3_DURATION_S" in os.environ:
        duration = float(os.environ["PHASE2_BENCH3_DURATION_S"])

    b3 = run_bench3(
        profile,
        kernels,
        sat_threads=bw_sat,
        n_elements=int(b1.get("n_elements_per_array") or 0),
        reps=int(b1.get("reps") or 0),
        bytes_moved_per_iter=int(b1.get("bytes_moved_per_iter") or 0),
        duration_s=duration,
        bucket_s=10.0,
    )

    b4 = run_bench4(profile, kernels)

    measured_bw = b1.get("saturation", {}).get("bandwidth_saturated_median_gbs")
    predictions = []
    if measured_bw is not None:
        for gib in (0.5, 1, 2, 4, 8, 16):
            model_bytes = int(gib * GIB)
            # BW is decimal GB/s (1e9); convert to bytes/s then / model_bytes
            upper = measured_bw * 1e9 / model_bytes
            predictions.append(
                {
                    "model_GiB": gib,
                    "model_bytes": model_bytes,
                    "model_bytes_basis": "GiB = 2^30",
                    "upper_tok_s": upper,
                    "realistic_0_55x_tok_s": upper * 0.55,
                    "decimal_GB_alias": gib,  # same numeric label, different byte count
                    "decimal_GB_bytes_1e9": int(gib * 1e9),
                    "upper_tok_s_if_decimal_GB": measured_bw * 1e9 / (gib * 1e9),
                }
            )

    kv_preds = _kv_cache_predictions(measured_bw)

    # Sustained recommendation
    rec_threads = knee
    rec_sust = None
    sust_ev = "UNAVAILABLE"
    if b3.get("thermal_blind") or b3.get("status") == "THERMAL_BLIND":
        sust_ev = (
            "THERMAL BLIND — RESULT NOT MEANINGFUL; "
            "recommended_thread_count_sustained NOT concluded from BENCH 3"
        )
        rec_sust = None
    elif b3.get("status") == "BLOCKED":
        sust_ev = "BENCH 3 BLOCKED by loadavg gate"
    elif b3.get("status") == "SKIPPED":
        sust_ev = f"BENCH 3 SKIPPED: {b3.get('skip_reason')}"
    elif b3.get("summary") and not b3["summary"].get("conclusion_skipped"):
        sust_ratio = b3["summary"].get("sustained_over_peak")
        throttle_t = b3["summary"].get("time_to_throttle_s")
        if knee is None:
            sust_ev = "compute knee unavailable; cannot derive sustained recommendation"
        elif sust_ratio is not None and sust_ratio < 0.90:
            rec_sust = max(1, knee - 1)
            sust_ev = (
                f"sustained/peak={sust_ratio:.3f} < 0.90 "
                f"(time_to_throttle_s={throttle_t}); recommend knee-1 = {rec_sust}"
            )
        else:
            rec_sust = knee
            sust_ev = (
                f"sustained/peak={sust_ratio}; no >10% drop observed → "
                f"same as compute knee ({knee})"
            )

    total_runtime = time.time() - t_wall0
    blocked = []
    for name, b in (("bench1", b1), ("bench2", b2), ("bench3", b3), ("bench4", b4)):
        if b.get("status") == "BLOCKED":
            blocked.append(
                f"{name} BLOCKED by loadavg gate; readings={b.get('quiet_gate', {}).get('readings')}"
            )

    host_class = (
        profile.host_class.value
        if profile.host_class.is_detected()
        else str(profile.host_class.value)
    )

    bundle: dict[str, Any] = {
        "phase": "2.1",
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
            "prediction_units_note": (
                "model_GiB uses 2^30 bytes (GGUF convention). "
                "measured_bandwidth_GBps uses decimal 1e9 bytes/s (STREAM convention). "
                "upper_tok_s = BW_bytes_per_s / model_bytes."
            ),
            "kv_cache_predictions": kv_preds,
            "moe_note": (
                "WRONG FOR MoE: this model assumes every weight is read every token. "
                "Mixture-of-Experts models only activate a subset of experts per token, "
                "so weight bytes/token are much smaller and MoE will be systematically "
                "UNDERRATED (predicted tok/s too low) by BW/model_bytes. "
                "FLAG FOR PHASE 3 selection policy."
            ),
            "prefill_note": (
                "OPEN GAP: prefill / time-to-first-token is compute-bound "
                "(or mixed), not bandwidth-bound. Nothing in this harness predicts TTFT."
            ),
            "recommended_thread_count": rec_threads,
            "recommended_thread_count_evidence": (
                f"BENCH 2 fp32 knee on all_cores = {knee} "
                f"(not raw core count {cores}); bw_sat_threads={bw_sat}; "
                f"bench2_status={b2.get('status')}"
            ),
            "recommended_thread_count_sustained": rec_sust,
            "recommended_thread_count_sustained_evidence": sust_ev,
        },
        "self_critique": {
            "measurements blocked or contaminated": blocked
            or [
                "no bench BLOCKED by loadavg>0.3 after retries "
                "(or gate not exercised)"
            ],
            "likely wrong under virtualization": [
                "BENCH 1 GB/s reflects guest/cgroup memory path, not host DRAM controllers",
                "BENCH 3 thermal/cpufreq often absent under KVM — THERMAL BLIND path",
                "BENCH 4 overlay FS + virtio block ≠ bare-metal NVMe latency",
                "steal time / noisy neighbors can still inflate CV (see NOISY tags)",
            ],
            "where a benchmark may have been served from cache rather than RAM": [
                f"BENCH 1 working_set={b1.get('working_set_bytes')} vs L3="
                f"{(b1.get('cache_proof') or {}).get('l3_bytes')} "
                f"(suspect_fallback={b1.get('l3_suspect_fallback')}); "
                f"prefault_s={(b1.get('prefault') or {}).get('wall_time_s')}",
                "BENCH 2 per-thread 0.5*L2 — compute-bound by design, not DRAM",
                "BENCH 4 uses posix_fadvise(DONTNEED); without root drop_caches, "
                "cold pass may still hit page cache — trust io read_bytes delta",
            ],
            "ISA tier verification": [
                b2.get("isa_verification"),
                f"Phase 1.5 usable_tier={profile.isa.usable_tier.value} was NOT "
                "exercised by these scalar C/+OpenMP kernels",
            ],
            "other": [
                f"total_runtime_s={total_runtime:.1f} budget=600 "
                f"over={total_runtime > 600}",
                "0.55x decode band is an unvalidated assumption pending Phase 4",
                "Phase 2.1: constant reps + reconstruction assert; SUPERLINEAR gate; "
                "CONTAMINATED aborts; STREAM thermals; GiB prediction table",
            ],
        },
    }
    return bundle


def _kv_cache_predictions(measured_bw: float | None) -> dict[str, Any]:
    def _rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
        n_layers = cfg["n_layers"]
        n_kv_heads = cfg["n_kv_heads"]
        head_dim = cfg["head_dim"]
        bpe = cfg["bytes_per_element"]
        model_bytes = int(cfg["model_weight_GiB"] * GIB)
        rows = []
        for n_ctx in (512, 4096, 32768):
            kv_bytes = 2 * n_layers * n_kv_heads * head_dim * n_ctx * bpe
            total = model_bytes + kv_bytes
            kv_share = kv_bytes / total if total else None
            if measured_bw is None:
                upper = None
                upper_weights_only = None
            else:
                bps = measured_bw * 1e9
                upper = bps / total
                upper_weights_only = bps / model_bytes
            rows.append(
                {
                    "attention": cfg["attention"],
                    "n_ctx": n_ctx,
                    "n_kv_heads": n_kv_heads,
                    "kv_bytes": kv_bytes,
                    "model_weight_bytes": model_bytes,
                    "bytes_per_token_total": total,
                    "kv_share_of_total": kv_share,
                    "upper_tok_s_weights_plus_kv": upper,
                    "upper_tok_s_weights_only": upper_weights_only,
                    "formula": (
                        "kv_bytes(n_ctx)=2*n_layers*n_kv_heads*head_dim*n_ctx*"
                        "bytes_per_element; tok_s = BW / (model_bytes + kv_bytes)"
                    ),
                }
            )
        return rows

    return {
        "primary_config": dict(REP_7B_GQA),
        "secondary_config": dict(REP_7B_MHA),
        "primary_GQA_rows": _rows(REP_7B_GQA),
        "secondary_MHA_rows": _rows(REP_7B_MHA),
        # Back-compat alias: primary rows
        "rows": _rows(REP_7B_GQA),
        "note": (
            "PRIMARY=GQA (n_kv_heads=8); SECONDARY=MHA (n_kv_heads=32, ~4x KV). "
            "UPPER BOUND with perfect bandwidth utilization."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    _ = argv
    bundle = run_phase2()
    sys.stdout.write(format_phase2_report(bundle))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
