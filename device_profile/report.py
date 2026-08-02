"""Stdout report formatter for Phase 1 / 1.5b DeviceProfile detection."""

from __future__ import annotations

import json
import platform
import sys

from .models import DeviceProfile, EvidenceField


def _fmt_val(ef: EvidenceField) -> str:
    return str(ef.value)


def _evid(ef: EvidenceField) -> str:
    return f"  <- {ef.evidence}"


def format_report(profile: DeviceProfile) -> str:
    lines: list[str] = []
    p = profile

    # HOST CLASS first
    lines.append(f"HOST CLASS: {_fmt_val(p.host_class)}")
    lines.append(_evid(p.host_class))
    if p.host_class_banner or (
        p.host_class.is_detected() and p.host_class.value != "bare-metal"
    ):
        lines.append("=" * 72)
        lines.append(
            p.host_class_banner
            or (
                f"*** HOST CLASS = {p.host_class.value} — PROFILE NOT "
                "REPRESENTATIVE OF AN EDGE TARGET ***"
            )
        )
        lines.append("=" * 72)
    if p.devices_reached_note:
        lines.append(p.devices_reached_note)
    lines.append("")

    lines.append("=== SECTION 0: RAW SOURCE DUMP ===")
    lines.append(p.raw_source_dump)
    lines.append("")

    lines.append("=== SECTION 1: IDENTITY ===")
    lines.append(f"model_name: {_fmt_val(p.model_name)}")
    lines.append(_evid(p.model_name))
    lines.append(
        f"vendor / family / model / stepping: "
        f"{_fmt_val(p.vendor)} / {_fmt_val(p.cpu_family)} / "
        f"{_fmt_val(p.model)} / {_fmt_val(p.stepping)}"
    )
    lines.append(f"  <- vendor: {p.vendor.evidence}")
    lines.append(f"  <- family: {p.cpu_family.evidence}")
    lines.append(f"  <- model: {p.model.evidence}")
    lines.append(f"  <- stepping: {p.stepping.evidence}")
    lines.append(f"arch (uname -m): {_fmt_val(p.arch)}")
    lines.append(_evid(p.arch))
    lines.append(
        f"base_mhz / max_mhz: {_fmt_val(p.base_mhz)} / {_fmt_val(p.max_mhz)}"
    )
    lines.append(f"  <- base_mhz: {p.base_mhz.evidence}")
    lines.append(f"  <- max_mhz: {p.max_mhz.evidence}")
    lines.append(
        f"uarch (HOST-LEVEL INFORMATION ONLY): {_fmt_val(p.uarch_host_level)}"
    )
    lines.append(_evid(p.uarch_host_level))
    lines.append("")

    lines.append("=== SECTION 2: TOPOLOGY ===")
    lines.append(
        f"logical_cores / physical_cores / threads_per_core / sockets: "
        f"{_fmt_val(p.logical_cores)} / {_fmt_val(p.physical_cores)} / "
        f"{_fmt_val(p.threads_per_core)} / {_fmt_val(p.sockets)}"
    )
    lines.append(f"  <- logical_cores: {p.logical_cores.evidence}")
    lines.append(f"  <- physical_cores: {p.physical_cores.evidence}")
    lines.append(f"  <- threads_per_core: {p.threads_per_core.evidence}")
    lines.append(f"  <- sockets: {p.sockets.evidence}")
    lines.append(f"smt_enabled: {_fmt_val(p.smt_enabled)}")
    lines.append(_evid(p.smt_enabled))
    h = p.hybrid
    lines.append(f"hybrid: {_fmt_val(h.is_hybrid)}")
    lines.append(_evid(h.is_hybrid))
    lines.append(f"numa_nodes + cpu-to-node map: {_fmt_val(p.numa_nodes)}")
    lines.append(_evid(p.numa_nodes))
    lines.append(f"  <- cpu-to-node map: {p.cpu_to_node_map.evidence}")
    c = p.cache
    lines.append(
        f"cache L1d / L1i / L2 / L3 (bytes, sysfs per-instance): "
        f"{_fmt_val(c.l1d_bytes)} / {_fmt_val(c.l1i_bytes)} / "
        f"{_fmt_val(c.l2_bytes)} / {_fmt_val(c.l3_bytes)}; "
        f"L2 sharing: {_fmt_val(c.l2_sharing)}"
    )
    lines.append(f"  <- L1d: {c.l1d_bytes.evidence}")
    lines.append(f"  <- L1i: {c.l1i_bytes.evidence}")
    lines.append(f"  <- L2: {c.l2_bytes.evidence}")
    lines.append(f"  <- L2 sharing: {c.l2_sharing.evidence}")
    lines.append(f"  <- L3: {c.l3_bytes.evidence}")
    lines.append("")

    lines.append("=== SECTION 2b: EXECUTION ENVIRONMENT / CONTAINMENT ===")
    ee = p.exec_env
    if ee is None:
        lines.append("UNDETECTED (reason: exec_env not populated)")
    else:
        lines.append(
            f"virtualization hypervisor vendor (lscpu): "
            f"{_fmt_val(ee.hypervisor_vendor_lscpu)}"
        )
        lines.append(_evid(ee.hypervisor_vendor_lscpu))
        lines.append(f"/sys/hypervisor: {_fmt_val(ee.hypervisor_sysfs)}")
        lines.append(_evid(ee.hypervisor_sysfs))
        lines.append(f"systemd-detect-virt: {_fmt_val(ee.systemd_detect_virt)}")
        lines.append(_evid(ee.systemd_detect_virt))
        lines.append(f"/.dockerenv: {_fmt_val(ee.dockerenv)}")
        lines.append(_evid(ee.dockerenv))
        lines.append(f"/run/.containerenv: {_fmt_val(ee.containerenv)}")
        lines.append(_evid(ee.containerenv))
        lines.append(f"/proc/1/cgroup: {_fmt_val(ee.proc1_cgroup)}")
        lines.append(_evid(ee.proc1_cgroup))
        lines.append(f"cgroup version: {_fmt_val(ee.cgroup_version)}")
        lines.append(_evid(ee.cgroup_version))
        lines.append(f"cgroup memory.max: {_fmt_val(ee.memory_max)}")
        lines.append(_evid(ee.memory_max))
        lines.append(f"cgroup memory.high: {_fmt_val(ee.memory_high)}")
        lines.append(_evid(ee.memory_high))
        lines.append(f"cgroup memory.current: {_fmt_val(ee.memory_current)}")
        lines.append(_evid(ee.memory_current))
        lines.append(f"cgroup cpu.max: {_fmt_val(ee.cpu_max)}")
        lines.append(_evid(ee.cpu_max))
        lines.append(
            f"cgroup cpu.max → effective CPU quota (cores): "
            f"{_fmt_val(ee.cpu_max_quota_period)}"
        )
        lines.append(_evid(ee.cpu_max_quota_period))
        lines.append(
            f"cpuset.cpus.effective: {_fmt_val(ee.cpuset_cpus_effective)} "
            f"(size={_fmt_val(ee.cpuset_size)})"
        )
        lines.append(_evid(ee.cpuset_cpus_effective))
        lines.append(
            f"len(os.sched_getaffinity(0)): {_fmt_val(ee.sched_affinity_len)}  |  "
            f"os.cpu_count(): {_fmt_val(ee.os_cpu_count)}  |  "
            f"set={_fmt_val(ee.sched_affinity_set)}"
        )
        lines.append(_evid(ee.sched_affinity_len))
        lines.append(_evid(ee.os_cpu_count))
        lines.append(
            f"effective_cores = min(affinity, cpu.max quota, logical_cores): "
            f"{_fmt_val(ee.effective_cores)}"
        )
        lines.append(f"  inputs/winner: {ee.effective_cores_inputs}")
        lines.append(_evid(ee.effective_cores))
        lines.append(
            f"effective_mem_bytes = min(MemTotal, cgroup memory.max): "
            f"{_fmt_val(ee.effective_mem_bytes)}"
        )
        lines.append(f"  inputs/winner: {ee.effective_mem_inputs}")
        lines.append(_evid(ee.effective_mem_bytes))
        lines.append(
            "  note: a container/VM MUST NOT treat host totals as effective; "
            "downstream defaults use effective_* only."
        )
        lines.append(
            f"steal time delta % of elapsed jiffies: {_fmt_val(ee.steal_percent)}"
        )
        lines.append(_evid(ee.steal_percent))
    lines.append("")

    lines.append("=== SECTION 3: ISA FLAGS ===")
    lines.append(f"flag_match_mode: {p.isa.flag_match_mode}")
    for flag, ef in p.isa.flags.items():
        lines.append(f"{flag}: {ef.value}  <- {ef.evidence}")
    if p.isa.absent_branch_note:
        lines.append(p.isa.absent_branch_note)
    if p.isa.macos_sysctl_feats:
        lines.append("macOS sysctl FEAT_* probes:")
        for k, ef in p.isa.macos_sysctl_feats.items():
            lines.append(f"  {k}: {_fmt_val(ef)}")
            lines.append(f"    <- {ef.evidence}")
    lines.append(f"cpuid_tier: {_fmt_val(p.isa.cpuid_tier)}")
    lines.append(_evid(p.isa.cpuid_tier))
    lines.append(f"usable_tier: {_fmt_val(p.isa.usable_tier)}")
    lines.append(_evid(p.isa.usable_tier))
    lines.append(f"tier_runtime_verified: {p.isa.tier_runtime_verified}")
    lines.append(f"  note: {p.isa.tier_runtime_verified_note}")
    lines.append(f"  reasoning: {p.isa.quant_kernel_reasoning}")
    lines.append("")

    lines.append("=== SECTION 3b: PARSER NEGATIVE CONTROL ===")
    pnc = p.parser_negative_control
    if pnc is None:
        lines.append("UNDETECTED (reason: negative control not run)")
    else:
        lines.append(f"match_mode: {pnc.match_mode}")
        for probe in pnc.probes:
            lines.append(
                f"{probe['flag']}: {probe['result']}  <- {probe['evidence']}  "
                f"[{probe['status']}; expected {probe['expectation']}]"
            )
        lines.append(f"overall: {pnc.overall}")
    lines.append("")

    lines.append("=== SECTION 3c: RUNTIME CAPABILITY PROBE (AMX) ===")
    amx = p.amx_runtime_probe
    if amx is None:
        lines.append("UNDETECTED (reason: AMX probe not run / non-x86)")
    else:
        lines.append(
            f"prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA=18) rc: {_fmt_val(amx.prctl_rc)}"
        )
        lines.append(_evid(amx.prctl_rc))
        lines.append(f"prctl errno: {_fmt_val(amx.prctl_errno)}")
        lines.append(_evid(amx.prctl_errno))
        lines.append(f"note: {amx.note}")
    lines.append("")

    lines.append("=== SECTION 4: MEMORY ===")
    lines.append(
        f"mem_total_bytes / mem_available_bytes / mem_free_bytes: "
        f"{_fmt_val(p.mem_total_bytes)} / {_fmt_val(p.mem_available_bytes)} / "
        f"{_fmt_val(p.mem_free_bytes)}"
    )
    lines.append(f"  <- MemTotal: {p.mem_total_bytes.evidence}")
    lines.append(f"  <- MemAvailable: {p.mem_available_bytes.evidence}")
    lines.append(f"  <- MemFree: {p.mem_free_bytes.evidence}")
    if p.exec_env:
        lines.append(
            f"effective_mem_bytes (USED DOWNSTREAM): "
            f"{_fmt_val(p.exec_env.effective_mem_bytes)} "
            f"(winner: {p.exec_env.effective_memory_winner})"
        )
        lines.append(f"  inputs: {p.exec_env.effective_mem_inputs}")
    lines.append(
        f"swap_total / swap_used / swappiness: "
        f"{_fmt_val(p.swap_total_bytes)} / {_fmt_val(p.swap_used_bytes)} / "
        f"{_fmt_val(p.swappiness)}"
    )
    lines.append(f"  <- swap_total: {p.swap_total_bytes.evidence}")
    lines.append(f"  <- swappiness: {p.swappiness.evidence}")
    lines.append(
        f"page_size; hugepages (count, size): {_fmt_val(p.page_size_bytes)}; "
        f"count={_fmt_val(p.hugepages_count)}, "
        f"size_bytes={_fmt_val(p.hugepages_size_bytes)}"
    )
    lines.append(f"  <- page_size: {p.page_size_bytes.evidence}")
    lines.append(
        f"memory channels / DIMM count / DIMM speed: "
        f"{_fmt_val(p.memory_channels)} / {_fmt_val(p.dimm_count)} / "
        f"{_fmt_val(p.dimm_speed_mts)}"
    )
    lines.append(f"  <- channels: {p.memory_channels.evidence}")
    lines.append(
        f"measured_bandwidth_GBps: {_fmt_val(p.measured_bandwidth_GBps)}"
    )
    lines.append(_evid(p.measured_bandwidth_GBps))
    lines.append("")

    lines.append("=== SECTION 4b: STORAGE ===")
    st = p.storage
    if st is None:
        lines.append("UNDETECTED (reason: storage not populated)")
    else:
        lines.append(f"model cache dir: {_fmt_val(st.cache_dir)}")
        lines.append(_evid(st.cache_dir))
        lines.append(
            f"total_bytes / free_bytes: {_fmt_val(st.total_bytes)} / "
            f"{_fmt_val(st.free_bytes)}"
        )
        lines.append(_evid(st.total_bytes))
        lines.append(_evid(st.free_bytes))
        lines.append(f"filesystem type: {_fmt_val(st.filesystem_type)}")
        lines.append(_evid(st.filesystem_type))
        lines.append(f"rotational: {_fmt_val(st.rotational)}")
        lines.append(_evid(st.rotational))
        lines.append(f"sparse/mmap note: {_fmt_val(st.sparse_mmap_note)}")
        lines.append(_evid(st.sparse_mmap_note))
    lines.append("")

    lines.append("=== SECTION 4c: THERMAL & POWER ===")
    tp = p.thermal_power
    if tp is None:
        lines.append("UNDETECTED (reason: thermal/power not populated)")
    else:
        lines.append(f"on_ac_power: {_fmt_val(tp.on_ac_power)}")
        lines.append(_evid(tp.on_ac_power))
        lines.append(f"battery_present: {_fmt_val(tp.battery_present)}")
        lines.append(_evid(tp.battery_present))
        lines.append(f"battery_capacity: {_fmt_val(tp.battery_capacity)}")
        lines.append(_evid(tp.battery_capacity))
        lines.append(f"thermal_zones: {_fmt_val(tp.thermal_zones)}")
        lines.append(_evid(tp.thermal_zones))
        lines.append(f"cpufreq governors: {_fmt_val(tp.cpufreq_governors)}")
        lines.append(_evid(tp.cpufreq_governors))
        lines.append(f"thermal throttling indicators: {_fmt_val(tp.thermal_throttling)}")
        lines.append(_evid(tp.thermal_throttling))
        if tp.platform_notes:
            lines.append(f"platform notes: {tp.platform_notes}")
    lines.append("")

    lines.append("=== SECTION 5: CROSS-CHECKS ===")
    for cc in p.cross_checks:
        lines.append(f"{cc.name}: {cc.result}")
        lines.append(f"  left:  {cc.left}")
        lines.append(f"  right: {cc.right}")
    lines.append("core_count_sources (all):")
    for k, v in p.core_count_sources.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("=== SECTION 6: DERIVED DEFAULTS (PROVISIONAL BUDGET) ===")
    lines.append(f"suggested_thread_count: {_fmt_val(p.suggested_thread_count)}")
    lines.append(f"  formula: {p.suggested_thread_formula}")
    lines.append(
        "  note: provisional; uses effective_cores; to be REPLACED by "
        "measured saturation in Phase 2"
    )
    mb = p.memory_budget
    if mb is None:
        lines.append("memory budget: UNDETECTED")
    else:
        lines.append(f"PROVISIONAL memory budget (effective_mem={mb.effective_mem_bytes})")
        lines.append(f"os_reserve_bytes: {_fmt_val(mb.os_reserve_bytes)}")
        lines.append(_evid(mb.os_reserve_bytes))
        lines.append(
            f"runtime_overhead_bytes: {_fmt_val(mb.runtime_overhead_bytes)}"
        )
        lines.append(_evid(mb.runtime_overhead_bytes))
        lines.append(f"kv_formula: {mb.kv_formula}")
        lines.append(
            "weight_budget_bytes = effective_mem - os_reserve - "
            "runtime_overhead - kv_budget(n_ctx)"
        )
        lines.append(
            f"{'n_ctx':>8}  {'os_reserve':>12}  {'runtime':>12}  "
            f"{'kv_budget':>12}  {'weight_budget':>14}"
        )
        for row in mb.rows:
            lines.append(
                f"{row.n_ctx:8d}  {row.os_reserve_bytes:12d}  "
                f"{row.runtime_overhead_bytes:12d}  {row.kv_budget_bytes:12d}  "
                f"{row.weight_budget_bytes:14d}"
            )
        lines.append(f"note: {mb.provisional_note}")
        for row in mb.rows:
            lines.append(f"  assumptions n_ctx={row.n_ctx}: {row.assumptions}")
    lines.append("")

    lines.append("=== SECTION 7: SELF-CRITIQUE ===")
    for key, items in p.self_critique.items():
        lines.append(f"- {key}:")
        for item in items:
            lines.append(f"  - {item}")
    lines.append("")

    lines.append("=== SECTION 8: PROFILE JSON ===")
    lines.append(json.dumps(p.to_json_dict(), indent=2, default=str))
    lines.append("")

    lines.append("=== SECTION 9: TARGET RELIABILITY STATEMENT ===")
    lines.append(p.target_reliability_statement or "UNDETECTED")
    lines.append("")

    lines.append("=== META: COMMANDS & VERSIONS ===")
    lines.append(f"Python: {sys.version}")
    lines.append(f"platform.python_version(): {platform.python_version()}")
    lines.append(f"platform.platform(): {platform.platform()}")
    for lib in p.detection_libraries:
        lines.append(f"  - {lib}")
    lines.append("Commands / reads (capture-once, sha256 shown):")
    for cmd in p.commands_run:
        lines.append(f"  $ {cmd}")

    return "\n".join(lines)
