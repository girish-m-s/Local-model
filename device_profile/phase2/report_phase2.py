"""Format Phase 2.1 measurement report."""

from __future__ import annotations

import json
from typing import Any


def _snap_lines(label: str, snap: dict[str, Any] | Any) -> list[str]:
    if snap is None:
        return [f"{label}: UNDETECTED"]
    if not isinstance(snap, dict):
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


def _quiet_gate_lines(qg: dict[str, Any] | None) -> list[str]:
    if not qg:
        return []
    lines = [f"quiet_gate: status={qg.get('status')} readings={qg.get('readings')}"]
    for ev in qg.get("evidence") or []:
        lines.append(f"  <- {ev}")
    return lines


def format_phase2_report(bundle: dict[str, Any]) -> str:
    lines: list[str] = []
    phase = bundle.get("phase", "2")
    lines.append(f"=== PHASE {phase} MEASUREMENT HARNESS REPORT ===")
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
    lines.append(f"status: {b1.get('status')}")
    lines.extend(_snap_lines("pre", b1.get("pre")))
    lines.extend(_quiet_gate_lines(b1.get("quiet_gate")))
    if b1.get("status") == "BLOCKED":
        lines.append("BLOCKED: bench not run (loadavg gate).")
        lines.extend(_snap_lines("post", b1.get("post")))
        lines.append("")
    else:
        lines.append(f"backend: {b1['backend']}")
        lines.append(f"  <- {b1['backend_evidence']}")
        lines.append(f"working_set_bytes: {b1['working_set_bytes']}")
        lines.append(f"  <- policy: {b1['working_set_policy']}")
        lines.append(f"l3_suspect_fallback: {b1['l3_suspect_fallback']}")
        lines.append(f"bytes_moved_per_iter: {b1['bytes_moved_per_iter']}")
        lines.append(f"reps (constant across sweep): {b1.get('reps')}")
        cal = b1.get("calibration") or {}
        lines.append(f"calibration: {cal.get('evidence')}")
        for h in cal.get("history") or []:
            lines.append(f"  <- cal_step reps={h['reps']} time_s={h['time_s']}")
        pf = b1.get("prefault") or {}
        lines.append(f"prefault_wall_s: {pf.get('wall_time_s')}")
        lines.append(f"  <- {pf.get('evidence')}")
        cp = b1.get("cache_proof") or {}
        lines.append(
            f"cache_proof: ws={cp.get('working_set_bytes')} L3={cp.get('l3_bytes')} "
            f"ratio={cp.get('ratio_ws_over_l3')}"
        )
        lines.append(f"  <- {cp.get('evidence')}")
        for s in b1.get("thread_sweeps") or []:
            noisy = " NOISY" if s.get("noisy") else ""
            lines.append(
                f"threads={s['threads']}: reps={s.get('reps')} "
                f"median={s['median_gbs']:.4f} min={s['min_gbs']:.4f} "
                f"max={s['max_gbs']:.4f} mean={s['mean_gbs']:.4f} "
                f"CV%={s['cv_pct']:.2f}{noisy} GB/s"
            )
            disc = s.get("discarded_warmup") or {}
            lines.append(
                f"  DISCARDED warmup: time_s={disc.get('time_s')} "
                f"gbs={disc.get('gbs')} reps={disc.get('reps')}"
            )
            lines.append(f"  raw_gbs: {s['raw_gbs']}")
            lines.append(f"  raw_times_s: {s['raw_times_s']}")
            lines.append(
                f"  bytes_moved_per_timed_call: {s.get('bytes_moved_per_timed_call')} "
                f"(= reps * bytes_moved_per_iter)"
            )
            for chk in s.get("reconstruction_checks") or []:
                tag = "OK" if chk.get("ok") else "FAIL"
                lines.append(
                    f"  reconstruct[{tag}]: reported={chk.get('reported_gbs')} "
                    f"expected={chk.get('expected_gbs')} rel_err={chk.get('rel_err')}"
                )
            lines.append(
                f"  <- OMP_NUM_THREADS={s['omp_num_threads']}; {s['affinity_evidence']}"
            )
        for fail in b1.get("FAIL") or []:
            lines.append(f"FAIL: {fail}")
        val = b1.get("validity") or {}
        lines.append(f"validity: {val.get('flag')}")
        for v in val.get("violations") or []:
            lines.append(f"  <- {v}")
        sat = b1.get("saturation") or {}
        if sat.get("invalid"):
            lines.append(
                f"saturation_thread_count: NOT REPORTED  <- {sat.get('flag')}"
            )
        else:
            lines.append(
                f"saturation_thread_count: {sat.get('saturation_thread_count')} "
                f"(rule: {sat.get('rule')})"
            )
            lines.append(
                f"bandwidth_saturated / bandwidth_1thread: "
                f"{sat.get('bandwidth_saturated_median_gbs')} / "
                f"{sat.get('bandwidth_1thread_median_gbs')} = "
                f"{sat.get('ratio_sat_over_1')}"
            )
        lines.extend(_snap_lines("post", b1.get("post")))
        lines.append("")

    # BENCH 2
    b2 = bundle["bench2"]
    lines.append(f"=== {b2['name']} ===")
    lines.append(f"status: {b2.get('status')}")
    lines.extend(_snap_lines("pre", b2.get("pre")))
    lines.extend(_quiet_gate_lines(b2.get("quiet_gate")))
    if b2.get("status") == "BLOCKED":
        lines.append("BLOCKED: bench not run (loadavg gate).")
        lines.extend(_snap_lines("post", b2.get("post")))
        lines.append("")
    else:
        lines.append(f"sizing_rule: {b2.get('sizing_rule')}")
        lines.append(f"L2/fit: {b2.get('fit_evidence')}")
        lines.append(f"isa_verification: {b2.get('isa_verification')}")
        cal = b2.get("calibration") or {}
        lines.append(f"calibration: {cal.get('evidence')}")
        if b2.get("hybrid_note"):
            lines.append(b2["hybrid_note"])
        for label, curve in (b2.get("curves") or {}).items():
            lines.append(f"-- curve: {label} --")
            lines.append(f"  <- affinity: {curve['affinity']}")
            for r in curve["fp32"]:
                noisy = " NOISY" if r.get("noisy") else ""
                lines.append(
                    f"  fp32 threads={r['threads']}: "
                    f"bytes_per_thread={r.get('bytes_per_thread'):.1f} "
                    f"ws={r.get('working_set_bytes')} reps={r.get('reps')} "
                    f"median={r['median_gflops']:.4f} "
                    f"min={r['min_gflops']:.4f} max={r['max_gflops']:.4f} "
                    f"CV%={r['cv_pct']:.2f}{noisy} GFLOPS"
                )
                disc = r.get("discarded_warmup") or {}
                lines.append(
                    f"    DISCARDED warmup: time_s={disc.get('time_s')} "
                    f"gflops={disc.get('gflops')}"
                )
                lines.append(f"    raw_gflops: {r['raw_gflops']}")
                lines.append(f"    raw_times_s: {r['raw_times_s']}")
            for r in curve["int8"]:
                noisy = " NOISY" if r.get("noisy") else ""
                lines.append(
                    f"  int8 threads={r['threads']}: "
                    f"bytes_per_thread={r.get('bytes_per_thread'):.1f} "
                    f"ws={r.get('working_set_bytes')} reps={r.get('reps')} "
                    f"median={r['median_gops']:.4f} "
                    f"min={r['min_gops']:.4f} max={r['max_gops']:.4f} "
                    f"CV%={r['cv_pct']:.2f}{noisy} GOPS"
                )
                disc = r.get("discarded_warmup") or {}
                lines.append(
                    f"    DISCARDED warmup: time_s={disc.get('time_s')} "
                    f"gops={disc.get('gops')}"
                )
                lines.append(f"    raw_gops: {r['raw_gops']}")
                lines.append(f"    raw_times_s: {r['raw_times_s']}")
            knee = (b2.get("knee") or {}).get(label) or {}
            val = (b2.get("validity") or {}).get(label) or {}
            lines.append(f"  validity: {val.get('flag')}")
            for v in val.get("violations") or []:
                lines.append(f"    <- {v}")
            if knee.get("invalid"):
                lines.append(
                    f"  knee(fp32): NOT REPORTED  <- {knee.get('rule')}"
                )
            else:
                lines.append(
                    f"  knee(fp32): {knee.get('fp32_knee_threads')}  "
                    f"<- {knee.get('rule')}"
                )
        if b2.get("hybrid_conclusion"):
            lines.append(f"hybrid_conclusion: {b2['hybrid_conclusion']}")
        lines.extend(_snap_lines("post", b2.get("post")))
        lines.append("")

    # BENCH 3
    b3 = bundle["bench3"]
    lines.append(f"=== {b3['name']} ===")
    lines.append(f"status: {b3.get('status')}")
    lines.extend(_snap_lines("pre", b3.get("pre")))
    lines.extend(_quiet_gate_lines(b3.get("quiet_gate")))
    lines.append(f"kernel: {b3.get('kernel')}")
    lines.append(f"sat_threads (from BENCH 1 BW saturation): {b3.get('sat_threads')}")
    lines.append(f"duration_s: {b3.get('duration_s')}; bucket_s: {b3.get('bucket_s')}")
    lines.append(f"battery_dual_run: {b3.get('battery_dual_run')}")
    if b3.get("status") == "BLOCKED":
        lines.append("BLOCKED: bench not run (loadavg gate).")
    elif b3.get("status") == "SKIPPED":
        lines.append(f"SKIPPED: {b3.get('skip_reason')}")
    elif b3.get("thermal_blind") or b3.get("status") == "THERMAL_BLIND":
        sm = b3.get("summary") or {}
        lines.append(f"THERMAL BLIND — RESULT NOT MEANINGFUL")
        lines.append(f"  <- {sm.get('explanation') or sm.get('flag')}")
        for n in b3.get("notes") or []:
            lines.append(f"  <- {n}")
        for ev in (b3.get("thermal_probe") or {}).get("evidence") or []:
            lines.append(f"  <- {ev}")
    else:
        for b in b3.get("buckets") or []:
            lines.append(
                f"bucket[{b['bucket']}]: gbs={b['gbs']:.4f} "
                f"calls={b['calls']} elapsed_s={b['elapsed_s']:.3f} "
                f"temps={b.get('temps_C')} freqs_kHz={b.get('freqs_kHz')}"
            )
            for ev in b.get("thermal_evidence") or []:
                lines.append(f"  <- {ev}")
        sm = b3.get("summary") or {}
        if sm.get("conclusion_skipped"):
            lines.append("sustained/peak: NOT CONCLUDED (see thermal blind / skip)")
        else:
            lines.append(
                f"peak_bucket_gbs: {sm.get('peak_bucket_gbs')}; "
                f"sustained_last3_median: {sm.get('sustained_last3_median_gbs')}; "
                f"sustained/peak: {sm.get('sustained_over_peak')}"
            )
            lines.append(
                f"time_to_throttle_s: {sm.get('time_to_throttle_s')}  "
                f"<- {sm.get('throttle_rule')}"
            )
    lines.extend(_snap_lines("post", b3.get("post")))
    lines.append("")

    # BENCH 4
    b4 = bundle["bench4"]
    lines.append(f"=== {b4['name']} ===")
    lines.append(f"status: {b4.get('status')}")
    lines.extend(_snap_lines("pre", b4.get("pre")))
    lines.extend(_quiet_gate_lines(b4.get("quiet_gate")))
    lines.append(
        f"cached_file_bytes: {b4.get('cached_file_bytes')} "
        f"(0.4x effective_mem={b4.get('effective_mem_bytes')})"
    )
    lines.append(
        f"oversized_file_bytes: {b4.get('oversized_file_bytes')} "
        f"(1.5x effective_mem)"
    )
    lines.append(
        f"disk free/total on {b4.get('cache_dir')}: "
        f"{b4.get('disk_free')} / {b4.get('disk_total')}"
    )
    if b4.get("status") == "BLOCKED":
        lines.append("BLOCKED: bench not run (loadavg gate).")
    elif b4.get("status") == "SKIPPED":
        lines.append(f"  <- SKIPPED: {b4.get('skip_reason')}")
    else:
        for key in ("cached", "oversized"):
            f = (b4.get("files") or {}).get(key) or {}
            lines.append(f"-- file: {f.get('label')} path={f.get('path')} --")
            lines.append(
                f"  write: {f.get('write_seconds')}s "
                f"({f.get('write_gbs_file_size_over_time')} GB/s file_size/time)"
            )
            lines.append(f"  <- {f.get('fadvise_evidence')}")
            for pass_name in ("cold", "warm"):
                p = f.get(pass_name) or {}
                lines.append(
                    f"  {pass_name}: {p.get('seconds')}s "
                    f"effective_read_gbs={p.get('effective_read_gbs')} "
                    f"pages_per_second={p.get('pages_per_second')}"
                )
                lines.append(
                    f"    minflt_delta={p.get('minflt_delta')} "
                    f"(before={p.get('minflt_before')} after={p.get('minflt_after')}) "
                    f"majflt_delta={p.get('majflt_delta')} "
                    f"(before={p.get('majflt_before')} after={p.get('majflt_after')})"
                )
                lines.append(
                    f"    io_read_bytes_delta={p.get('io_read_bytes_delta')} "
                    f"(before={p.get('io_read_bytes_before')} "
                    f"after={p.get('io_read_bytes_after')})"
                )
                for ev in (p.get("fault_evidence") or []) + (p.get("io_evidence") or []):
                    lines.append(f"    <- {ev}")
                lines.append(f"    <- bandwidth_basis: {p.get('bandwidth_basis')}")
        lines.append(f"delta (cached - oversized): {b4.get('delta')}")
        lines.append(f"note: {b4.get('note')}")
    lines.extend(_snap_lines("post", b4.get("post")))
    lines.append("")

    # DERIVED
    d = bundle["derived"]
    lines.append("=== SECTION D — DERIVED OUTPUT ===")
    lines.append(
        f"measured_bandwidth_GBps (BENCH1 saturated median): "
        f"{d.get('measured_bandwidth_GBps')}"
    )
    lines.append(d.get("prediction_units_note", ""))
    lines.append(
        "predicted_decode_tok_s(model_bytes) = measured_bandwidth_GBps * 1e9 / model_bytes"
    )
    lines.append(
        "UPPER BOUND assuming perfect bandwidth utilization. Real engines typically "
        "hit 40-70% of it."
    )
    lines.append(
        f"{'model_GiB':>10}  {'bytes(2^30)':>14}  {'upper_tok_s':>14}  "
        f"{'realistic_0.55x':>14}  {'upper_if_1e9':>12}"
    )
    for row in d.get("predictions") or []:
        lines.append(
            f"{row['model_GiB']:10.1f}  {row['model_bytes']:14d}  "
            f"{row['upper_tok_s']:14.2f}  {row['realistic_0_55x_tok_s']:14.2f}  "
            f"{row['upper_tok_s_if_decimal_GB']:12.2f}"
        )
    lines.append(
        "0.55x realistic band: UNVALIDATED ASSUMPTION — to be checked in Phase 4 "
        "against a real engine."
    )
    lines.append("")
    lines.append("-- KV-cache extension (weights-only model is incomplete) --")
    kv = d.get("kv_cache_predictions") or {}
    cfg = kv.get("config") or {}
    lines.append(f"config: {cfg}")
    lines.append(kv.get("note", ""))
    for row in kv.get("rows") or []:
        lines.append(
            f"  n_ctx={row['n_ctx']}: kv_bytes={row['kv_bytes']} "
            f"total_bytes/token={row['bytes_per_token_total']} "
            f"kv_share={row['kv_share_of_total']} "
            f"upper_tok_s={row['upper_tok_s_weights_plus_kv']} "
            f"(weights_only={row['upper_tok_s_weights_only']})"
        )
    lines.append("")
    lines.append(f"MoE: {d.get('moe_note')}")
    lines.append(f"Prefill/TTFT: {d.get('prefill_note')}")
    lines.append("")
    lines.append(
        f"recommended_thread_count: {d.get('recommended_thread_count')}  "
        f"<- {d.get('recommended_thread_count_evidence')}"
    )
    lines.append(
        f"recommended_thread_count_sustained: "
        f"{d.get('recommended_thread_count_sustained')}  "
        f"<- {d.get('recommended_thread_count_sustained_evidence')}"
    )
    lines.append("")

    lines.append("=== SELF-CRITIQUE ===")
    for key, items in (bundle.get("self_critique") or {}).items():
        lines.append(f"- {key}:")
        for item in items:
            lines.append(f"  - {item}")
    lines.append("")

    lines.append(f"=== PHASE {phase} JSON ===")
    lines.append(json.dumps(bundle, indent=2, default=str))
    return "\n".join(lines)
