"""Format Phase 2.4 measurement report."""

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
            "load_threshold": getattr(snap, "load_threshold", None),
            "threshold_overridden": getattr(snap, "threshold_overridden", False),
            "evidence": snap.evidence,
        }
    else:
        d = snap
    thr = d.get("load_threshold")
    over = d.get("threshold_overridden")
    thr_tag = ""
    if thr is not None:
        thr_tag = f" threshold={thr}" + (" OVERRIDE" if over else "")
    lines = [
        f"{label}: loadavg={d.get('loadavg') or d.get('loadavg_1_5_15')} "
        f"MemAvailable={d.get('mem_available_bytes')} "
        f"MemFree={d.get('mem_free_bytes')} "
        f"{'CONTAMINATED' if d.get('contaminated') else 'OK'}{thr_tag}"
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


def _thread_verify_line(s: dict[str, Any]) -> str:
    return (
        f"  thread_verify: requested={s.get('requested_threads', s.get('threads'))} "
        f"omp_get_num_threads_actual={s.get('omp_num_threads_actual')} "
        f"n_distinct_cpus={s.get('n_distinct_cpus')} "
        f"flag={s.get('thread_flag', 'OK' if s.get('thread_count_applied') else 'INVALID')}"
    )


def format_phase2_report(bundle: dict[str, Any]) -> str:
    lines: list[str] = []
    phase = bundle.get("phase", "2.4")
    lines.append(f"=== PHASE {phase} MEASUREMENT HARNESS REPORT ===")
    if bundle.get("smoke_not_deliverable"):
        lines.append("*** SMOKE_NOT_DELIVERABLE — container/smoke path; not a device deliverable ***")
    for note in bundle.get("override_notes") or []:
        lines.append(f"  <- {note}")
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
        lines.append(f"backend: {b1.get('backend')}")
        lines.append(f"working_set_bytes: {b1.get('working_set_bytes')}")
        lines.append(f"  <- policy: {b1.get('working_set_policy')}")
        lines.append(
            f"l3_reported_bytes: {b1.get('l3_reported_bytes')} "
            f"ratio_ws_over_l3: {b1.get('ratio_ws_over_l3')}"
        )
        lines.append(f"  <- {b1.get('diag_b_floor_note')}")
        lines.append(f"l3_suspect: {b1.get('l3_suspect')}")
        lines.append(f"bytes_moved_per_iter: {b1.get('bytes_moved_per_iter')}")
        lines.append(f"reps (constant across sweep): {b1.get('reps')}")
        cal = b1.get("calibration") or {}
        lines.append(f"calibration: {cal.get('evidence')}")
        for h in cal.get("history") or []:
            lines.append(
                f"  <- cal_step reps={h.get('reps')} mean_time_s={h.get('mean_time_s')} "
                f"actual_threads={h.get('actual_threads')} pid={h.get('pid')}"
            )
        cp = b1.get("cache_proof") or {}
        lines.append(
            f"cache_proof: ws={cp.get('working_set_bytes')} L3={cp.get('l3_bytes')} "
            f"ratio={cp.get('ratio_ws_over_l3')}"
        )
        lines.append(f"  <- {cp.get('evidence')}")
        for s in b1.get("thread_sweeps") or []:
            if s.get("status") == "INVALID":
                lines.append(
                    f"threads={s.get('threads')}: INVALID "
                    f"req={s.get('requested_threads')} "
                    f"actual={s.get('omp_num_threads_actual')} "
                    f"n_distinct_cpus={s.get('n_distinct_cpus')}"
                )
                lines.append(_thread_verify_line(s))
                continue
            noisy = " NOISY" if s.get("noisy") else ""
            lines.append(
                f"threads={s['threads']}: reps={s.get('reps')} "
                f"median={s['median_gbs']:.4f} min={s['min_gbs']:.4f} "
                f"max={s['max_gbs']:.4f} mean={s['mean_gbs']:.4f} "
                f"CV%={s['cv_pct']:.2f}{noisy} GB/s"
            )
            lines.append(_thread_verify_line(s))
            disc = s.get("discarded_warmup") or {}
            lines.append(
                f"  DISCARDED warmup: time_s={disc.get('time_s')} "
                f"gbs={disc.get('gbs')} actual={disc.get('omp_num_threads_actual')} "
                f"n_cpus={disc.get('n_distinct_cpus')}"
            )
            lines.append(f"  raw_gbs: {s.get('raw_gbs')}")
            lines.append(f"  raw_times_s: {s.get('raw_times_s')}")
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
                f"  <- env_OMP_NUM_THREADS={s.get('env_OMP_NUM_THREADS')}; "
                f"affinity={s.get('affinity_cpus')}"
            )
        for fail in b1.get("FAIL") or []:
            lines.append(f"FAIL: {fail}")
        val = b1.get("validity") or {}
        lines.append(f"validity: {val.get('flag')}")
        for v in val.get("violations") or []:
            lines.append(f"  <- {v}")
        for e in b1.get("efficiency_curve") or []:
            lines.append(
                f"efficiency threads={e.get('threads')}: "
                f"median_gbs={e.get('median_gbs')} eff={e.get('efficiency')}"
            )
        sat = b1.get("saturation") or {}
        if sat.get("no_saturation_observed"):
            lines.append(f"saturation: {sat.get('flag')}")
            lines.append(
                f"  efficiency_at_max_cores={sat.get('efficiency_at_max_cores')} "
                f"peak_median_gbs={sat.get('peak_median_gbs')}"
            )
        elif sat.get("invalid"):
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
        rec = b1.get("recommendation") or {}
        lines.append(
            f"recommended_thread_count (from BENCH 1 BW curve): "
            f"{rec.get('recommended_thread_count')}  <- {rec.get('recommended_thread_count_evidence')}"
        )
        lines.append(
            f"max_efficiency_thread_count (BENCH 3 source): "
            f"{rec.get('max_efficiency_thread_count')}  "
            f"<- {rec.get('max_efficiency_thread_count_evidence')}"
        )
        lines.extend(_snap_lines("post", b1.get("post")))
        lines.append("")

    # BENCH 2 deleted
    b2 = bundle.get("bench2") or {}
    lines.append(f"=== {b2.get('name', 'BENCH 2 — DELETED')} ===")
    lines.append(f"status: {b2.get('status', 'DELETED')}")
    lines.append(f"  <- {b2.get('reason')}")
    lines.append("")

    # BENCH 3
    b3 = bundle["bench3"]
    lines.append(f"=== {b3['name']} ===")
    lines.append(f"status: {b3.get('status')}")
    lines.extend(_snap_lines("pre", b3.get("pre")))
    lines.extend(_quiet_gate_lines(b3.get("quiet_gate")))
    lines.append(f"kernel: {b3.get('kernel')}")
    lines.append(
        f"threads (BENCH 1 max-efficiency / recommended): {b3.get('sat_threads')}"
    )
    lines.append(
        f"thread_verify: requested={b3.get('requested_threads')} "
        f"actual={b3.get('omp_num_threads_actual')} "
        f"n_distinct_cpus={b3.get('n_distinct_cpus')} "
        f"applied={b3.get('thread_count_applied')} flag={b3.get('thread_flag')}"
    )
    lines.append(f"duration_s: {b3.get('duration_s')}; bucket_s: {b3.get('bucket_s')}")
    lines.append(f"battery_dual_run: {b3.get('battery_dual_run')}")
    if b3.get("status") == "BLOCKED":
        lines.append("BLOCKED: bench not run (loadavg gate).")
    elif b3.get("status") == "SKIPPED":
        lines.append(f"SKIPPED: {b3.get('skip_reason')}")
    elif b3.get("status") == "INVALID":
        for f in b3.get("FAIL") or []:
            lines.append(f"FAIL: {f}")
    elif b3.get("thermal_blind") or b3.get("status") == "THERMAL_BLIND":
        sm = b3.get("summary") or {}
        lines.append("THERMAL BLIND — RESULT NOT MEANINGFUL")
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
                f"actual_threads={b.get('omp_num_threads_actual')} "
                f"n_cpus={b.get('n_distinct_cpus')} "
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
    lines.append(f"thread_control: {b4.get('thread_control')}")
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
                f"  thread_verify: requested={f.get('requested_threads')} "
                f"actual={f.get('omp_num_threads_actual')} "
                f"n_distinct_cpus={f.get('n_distinct_cpus')} "
                f"flag={f.get('thread_flag')}"
            )
            if f.get("status") == "INVALID":
                lines.append(f"  INVALID: {f.get('error')}")
                continue
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
                    f"    io_basis_gbs={p.get('io_basis_gbs')} "
                    f"file_basis_gbs={p.get('file_basis_gbs')} "
                    f"(both printed for audit)"
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
                lines.append(
                    f"    <- bandwidth_basis: {p.get('bandwidth_basis')}; "
                    f"{p.get('bandwidth_note')}"
                )
        delta = b4.get("delta") or {}
        lines.append(f"delta (cached - oversized): {delta}")
        if delta.get("warm_cliff_ratio_cached_over_oversized") is not None:
            lines.append(
                f"warm_cliff_ratio (cached/oversized): "
                f"{delta.get('warm_cliff_ratio_cached_over_oversized')}"
            )
        lines.append(f"note: {b4.get('note')}")
        for fail in b4.get("FAIL") or []:
            lines.append(f"FAIL: {fail}")
    lines.extend(_snap_lines("post", b4.get("post")))
    lines.append("")

    # DERIVED
    d = bundle["derived"]
    lines.append("=== SECTION D — DERIVED OUTPUT ===")
    lines.append(
        f"measured_bandwidth_GBps (BENCH1 saturated / max-core median): "
        f"{d.get('measured_bandwidth_GBps')}"
    )
    lines.append(d.get("prediction_units_note", ""))
    lines.append(
        "predicted_decode_tok_s(model_bytes) = measured_bandwidth_GBps * 1e9 / model_bytes"
    )
    lines.append(
        "UPPER BOUND assuming perfect bandwidth utilization. Real engines typically "
        "hit 40-70% of it. GiB = 2^30."
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
    lines.append(kv.get("note", ""))
    lines.append(f"PRIMARY GQA config: {kv.get('primary_config')}")
    for row in kv.get("primary_GQA_rows") or kv.get("rows") or []:
        lines.append(
            f"  [GQA] n_ctx={row['n_ctx']} n_kv_heads={row.get('n_kv_heads')}: "
            f"kv_bytes={row['kv_bytes']} total_bytes/token={row['bytes_per_token_total']} "
            f"kv_share={row['kv_share_of_total']} "
            f"upper_tok_s={row['upper_tok_s_weights_plus_kv']} "
            f"(weights_only={row['upper_tok_s_weights_only']})"
        )
    lines.append(f"SECONDARY MHA config: {kv.get('secondary_config')}")
    for row in kv.get("secondary_MHA_rows") or []:
        lines.append(
            f"  [MHA] n_ctx={row['n_ctx']} n_kv_heads={row.get('n_kv_heads')}: "
            f"kv_bytes={row['kv_bytes']} total_bytes/token={row['bytes_per_token_total']} "
            f"kv_share={row['kv_share_of_total']} "
            f"upper_tok_s={row['upper_tok_s_weights_plus_kv']} "
            f"(weights_only={row['upper_tok_s_weights_only']})"
        )
    lines.append("")
    lines.append(f"MoE: {d.get('moe_note')}")
    lines.append(f"Prefill/TTFT: {d.get('prefill_note')}")
    lines.append("")
    lines.append(f"bench3_thread_source: {d.get('bench3_thread_source')}")
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
