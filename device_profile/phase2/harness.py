"""Phase 2.4 orchestrator — verified-thread STREAM / thermal / mmap (no BENCH 2)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

from device_profile.detect import detect_device_profile

from .bench_mmap import run_bench4
from .bench_stream import run_bench1
from .bench_sustained import run_bench3
from .native import build_kernels
from .report_phase2 import format_phase2_report
from .subprocess_run import assert_no_phase2_overrides
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


def run_phase2(*, smoke: bool = False) -> dict[str, Any]:
    override_notes = assert_no_phase2_overrides(allow_smoke=smoke)
    t_wall0 = time.time()
    profile = detect_device_profile()

    kernels = build_kernels()
    cores = effective_cores(profile)
    emem = effective_mem(profile)

    b1 = run_bench1(profile, kernels)

    rec = (b1.get("recommendation") or {}).get("recommended_thread_count")
    max_eff_t = (b1.get("recommendation") or {}).get("max_efficiency_thread_count")
    bench3_threads = max_eff_t if max_eff_t is not None else rec
    bw_sat = b1.get("saturation", {}).get("saturation_thread_count")

    duration = 180.0
    if smoke and "PHASE2_BENCH3_DURATION_S" in os.environ:
        duration = float(os.environ["PHASE2_BENCH3_DURATION_S"])
    elif (not smoke) and "PHASE2_BENCH3_DURATION_S" in os.environ:
        # Should have been refused by assert_no_phase2_overrides; belt-and-suspenders.
        raise RuntimeError("PHASE2_BENCH3_DURATION_S set on non-smoke deliverable run")

    b3 = run_bench3(
        profile,
        kernels,
        sat_threads=bench3_threads,
        n_elements=int(b1.get("n_elements_per_array") or 0),
        reps=int(b1.get("reps") or 0),
        bytes_moved_per_iter=int(b1.get("bytes_moved_per_iter") or 0),
        duration_s=duration,
        bucket_s=10.0,
    )

    b4 = run_bench4(profile, kernels)

    measured_bw = b1.get("saturation", {}).get("bandwidth_saturated_median_gbs")
    if measured_bw is None and not b1.get("saturation", {}).get("invalid"):
        # No-sat path still has a usable max-core median.
        measured_bw = b1.get("saturation", {}).get("bandwidth_saturated_median_gbs")
    predictions = []
    if measured_bw is not None:
        for gib in (0.5, 1, 2, 4, 8, 16):
            model_bytes = int(gib * GIB)
            upper = measured_bw * 1e9 / model_bytes
            predictions.append(
                {
                    "model_GiB": gib,
                    "model_bytes": model_bytes,
                    "model_bytes_basis": "GiB = 2^30",
                    "upper_tok_s": upper,
                    "realistic_0_55x_tok_s": upper * 0.55,
                    "decimal_GB_alias": gib,
                    "decimal_GB_bytes_1e9": int(gib * 1e9),
                    "upper_tok_s_if_decimal_GB": measured_bw * 1e9 / (gib * 1e9),
                }
            )

    kv_preds = _kv_cache_predictions(measured_bw)

    # Sustained recommendation from BENCH 3 relative to BENCH 1 rec (not deleted BENCH 2).
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
    elif b3.get("status") == "INVALID":
        sust_ev = "BENCH 3 INVALID (thread count not applied)"
    elif b3.get("summary") and not b3["summary"].get("conclusion_skipped"):
        sust_ratio = b3["summary"].get("sustained_over_peak")
        throttle_t = b3["summary"].get("time_to_throttle_s")
        if rec is None:
            sust_ev = (
                "BENCH 1 recommended_thread_count unavailable; "
                "cannot derive sustained recommendation"
            )
        elif sust_ratio is not None and sust_ratio < 0.90:
            rec_sust = max(1, int(rec) - 1)
            sust_ev = (
                f"sustained/peak={sust_ratio:.3f} < 0.90 "
                f"(time_to_throttle_s={throttle_t}); recommend rec-1 = {rec_sust}"
            )
        else:
            rec_sust = rec
            sust_ev = (
                f"sustained/peak={sust_ratio}; no >10% drop observed → "
                f"same as BENCH 1 recommended_thread_count ({rec})"
            )

    total_runtime = time.time() - t_wall0
    blocked = []
    for name, b in (("bench1", b1), ("bench3", b3), ("bench4", b4)):
        if b.get("status") == "BLOCKED":
            blocked.append(
                f"{name} BLOCKED by loadavg gate; readings={b.get('quiet_gate', {}).get('readings')}"
            )

    host_class = (
        profile.host_class.value
        if profile.host_class.is_detected()
        else str(profile.host_class.value)
    )

    sat = b1.get("saturation") or {}
    rec_ev = (b1.get("recommendation") or {}).get("recommended_thread_count_evidence")
    if sat.get("no_saturation_observed"):
        rec_ev = sat.get("flag") + f"; {rec_ev}"

    deleted_bench2 = {
        "name": "BENCH 2 — DELETED (Phase 2.4)",
        "status": "DELETED",
        "reason": (
            "Kernels compile to SSE2 (vector-xmm; no ymm/zmm; no -march). "
            "They cannot exercise the detected AVX512_VNNI tier. A hand-rolled "
            "dot product also does not predict llama.cpp VNNI GEMM. Compute "
            "throughput deferred to Phase 4 against a real engine. "
            "recommended_thread_count now comes from BENCH 1 bandwidth curve."
        ),
    }

    bundle: dict[str, Any] = {
        "phase": "2.4",
        "smoke_not_deliverable": bool(smoke),
        "override_notes": override_notes,
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
        "bench2": deleted_bench2,
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
            "recommended_thread_count": rec,
            "recommended_thread_count_evidence": rec_ev
            or "from BENCH 1 verified bandwidth curve (BENCH 2 deleted)",
            "recommended_thread_count_sustained": rec_sust,
            "recommended_thread_count_sustained_evidence": sust_ev,
            "bench3_thread_source": (
                f"BENCH 3 used max_efficiency_thread_count={bench3_threads} "
                f"from BENCH 1 (not deleted BENCH 2 knee)"
            ),
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
                f"{b1.get('l3_reported_bytes')} ratio={b1.get('ratio_ws_over_l3')}; "
                f"{b1.get('diag_b_floor_note')}",
                "BENCH 2 DELETED — no compute microbench cache analysis",
                "BENCH 4: io_read_bytes_delta==0 → FULLY CACHED (file_bytes/time basis)",
            ],
            "ISA tier verification": [
                "BENCH 2 deleted; scalar STREAM/mmap kernels do not exercise "
                f"Phase 1.5 usable_tier={profile.isa.usable_tier.value}",
                "Compute throughput deferred to Phase 4 real-engine measurement",
            ],
            "other": [
                f"total_runtime_s={total_runtime:.1f} budget=600 "
                f"over={total_runtime > 600}",
                "0.55x decode band is an unvalidated assumption pending Phase 4",
                "Phase 2.4: omp_set_num_threads in fresh subprocess; "
                "actual!=requested → INVALID; WS=8×L3; BENCH 2 deleted",
                *(override_notes or []),
                *(
                    ["SMOKE_NOT_DELIVERABLE: container/smoke path"]
                    if smoke
                    else []
                ),
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
        "rows": _rows(REP_7B_GQA),
        "note": (
            "PRIMARY=GQA (n_kv_heads=8); SECONDARY=MHA (n_kv_heads=32, ~4x KV). "
            "UPPER BOUND with perfect bandwidth utilization. GiB = 2^30."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2.4 measurement harness")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "SMOKE_NOT_DELIVERABLE: allow PHASE2_* overrides with "
            "PHASE2_ALLOW_OVERRIDES=1; container path only"
        ),
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        bundle = run_phase2(smoke=bool(args.smoke))
    except RuntimeError as exc:
        sys.stderr.write(f"ABORT: {exc}\n")
        return 2
    sys.stdout.write(format_phase2_report(bundle))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
