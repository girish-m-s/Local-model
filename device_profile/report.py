"""Exact stdout report formatter for Phase 1 / 1.5 DeviceProfile detection."""

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

    if p.is_development_proxy and p.proxy_banner:
        lines.append("=" * 72)
        lines.append(p.proxy_banner)
        lines.append("=" * 72)
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
    lines.append(
        "  note: any host memory-channel or L3 figure this uarch implies "
        "does NOT describe this guest's available share."
    )
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
    if h.is_hybrid.value is True:
        lines.append(
            f"hybrid: true; P-cores={_fmt_val(h.p_core_count) if h.p_core_count else '?'} "
            f"E-cores={_fmt_val(h.e_core_count) if h.e_core_count else '?'}; "
            f"P logical IDs={_fmt_val(h.p_core_logical_ids) if h.p_core_logical_ids else '?'}; "
            f"E logical IDs={_fmt_val(h.e_core_logical_ids) if h.e_core_logical_ids else '?'}"
        )
    else:
        lines.append(f"hybrid: {_fmt_val(h.is_hybrid)}")
    lines.append(_evid(h.is_hybrid))
    if h.p_core_count:
        lines.append(f"  <- P-cores: {h.p_core_count.evidence}")
    if h.e_core_count:
        lines.append(f"  <- E-cores: {h.e_core_count.evidence}")

    lines.append(f"numa_nodes + cpu-to-node map: {_fmt_val(p.numa_nodes)}")
    lines.append(_evid(p.numa_nodes))
    lines.append(f"  <- cpu-to-node map: {p.cpu_to_node_map.evidence}")
    if not (
        isinstance(p.cpu_to_node_map.value, str)
        and str(p.cpu_to_node_map.value).startswith("UNDETECTED")
    ):
        lines.append(f"  map value: {p.cpu_to_node_map.value}")

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
    if c.lscpu_l1d_bytes:
        lines.append(
            f"  <- lscpu L1d aggregate: {c.lscpu_l1d_bytes.evidence} "
            f"(instances={_fmt_val(c.lscpu_l1d_instances) if c.lscpu_l1d_instances else '?'})"
        )
    if c.lscpu_l1i_bytes:
        lines.append(
            f"  <- lscpu L1i aggregate: {c.lscpu_l1i_bytes.evidence} "
            f"(instances={_fmt_val(c.lscpu_l1i_instances) if c.lscpu_l1i_instances else '?'})"
        )
    if c.lscpu_l2_bytes:
        lines.append(
            f"  <- lscpu L2 aggregate: {c.lscpu_l2_bytes.evidence} "
            f"(instances={_fmt_val(c.lscpu_l2_instances) if c.lscpu_l2_instances else '?'})"
        )
    if c.lscpu_l3_bytes:
        lines.append(
            f"  <- lscpu L3 aggregate: {c.lscpu_l3_bytes.evidence} "
            f"(instances={_fmt_val(c.lscpu_l3_instances) if c.lscpu_l3_instances else '?'})"
        )
    lines.append("")

    # SECTION 2b
    lines.append("=== SECTION 2b: EXECUTION ENVIRONMENT ===")
    ee = p.exec_env
    if ee is None:
        lines.append("UNDETECTED (reason: exec_env not populated)")
    else:
        lines.append(
            f"virtualization hypervisor vendor (lscpu): "
            f"{_fmt_val(ee.hypervisor_vendor_lscpu)}"
        )
        lines.append(_evid(ee.hypervisor_vendor_lscpu))
        lines.append(
            f"virtualization (lscpu Virtualization / type): "
            f"{_fmt_val(ee.virtualization_lscpu)}"
        )
        lines.append(_evid(ee.virtualization_lscpu))
        lines.append(f"/sys/hypervisor: {_fmt_val(ee.hypervisor_sysfs)}")
        lines.append(_evid(ee.hypervisor_sysfs))
        lines.append(
            f"systemd-detect-virt: {_fmt_val(ee.systemd_detect_virt)}"
        )
        lines.append(_evid(ee.systemd_detect_virt))
        lines.append(f"/.dockerenv: {_fmt_val(ee.dockerenv)}")
        lines.append(_evid(ee.dockerenv))
        lines.append(f"/run/.containerenv: {_fmt_val(ee.containerenv)}")
        lines.append(_evid(ee.containerenv))
        lines.append(f"/proc/1/cgroup (verbatim): {_fmt_val(ee.proc1_cgroup)}")
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
            f"cgroup cpu.max quota/period: {_fmt_val(ee.cpu_max_quota_period)}"
        )
        lines.append(_evid(ee.cpu_max_quota_period))
        lines.append(
            f"cpuset.cpus.effective: {_fmt_val(ee.cpuset_cpus_effective)} "
            f"(size={_fmt_val(ee.cpuset_size)})"
        )
        lines.append(_evid(ee.cpuset_cpus_effective))
        lines.append(_evid(ee.cpuset_size))
        lines.append(
            f"os.cpu_count(): {_fmt_val(ee.os_cpu_count)}  |  "
            f"len(os.sched_getaffinity(0)): {_fmt_val(ee.sched_affinity_len)}  |  "
            f"affinity set: {_fmt_val(ee.sched_affinity_set)}"
        )
        lines.append(_evid(ee.os_cpu_count))
        lines.append(_evid(ee.sched_affinity_len))
        lines.append(_evid(ee.sched_affinity_set))
        lines.append(f"steal sample t0: {_fmt_val(ee.steal_sample_t0)}")
        lines.append(_evid(ee.steal_sample_t0))
        lines.append(f"steal sample t1 (+5s): {_fmt_val(ee.steal_sample_t1)}")
        lines.append(_evid(ee.steal_sample_t1))
        lines.append(
            f"steal time delta % of elapsed jiffies: {_fmt_val(ee.steal_percent)}"
        )
        lines.append(_evid(ee.steal_percent))
        lines.append(
            f"effective_memory_limit: {_fmt_val(ee.effective_memory_limit)} "
            f"(winner: {ee.effective_memory_winner})"
        )
        lines.append(_evid(ee.effective_memory_limit))
        lines.append(
            f"  raw MemTotal for comparison: {_fmt_val(p.mem_total_bytes)}"
        )
        lines.append(
            f"effective_cpu_count: {_fmt_val(ee.effective_cpu_count)} "
            f"(winner: {ee.effective_cpu_winner})"
        )
        lines.append(_evid(ee.effective_cpu_count))
        lines.append(
            "  note: every downstream thread/RAM default uses effective_* "
            "not raw /proc or lscpu alone."
        )
    lines.append("")

    lines.append("=== SECTION 3: ISA FLAGS ===")
    lines.append(f"flag_match_mode: {p.isa.flag_match_mode}")
    for flag, ef in p.isa.flags.items():
        lines.append(f"{flag}: {ef.value}  <- {ef.evidence}")
    lines.append(f"cpuid_tier: {_fmt_val(p.isa.cpuid_tier)}")
    lines.append(_evid(p.isa.cpuid_tier))
    lines.append(f"usable_tier: {_fmt_val(p.isa.usable_tier)}")
    lines.append(_evid(p.isa.usable_tier))
    lines.append(f"  reasoning: {p.isa.quant_kernel_reasoning}")
    lines.append("")

    # SECTION 3b
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
        if pnc.overall == "FAIL":
            lines.append(
                "PARSER BROKEN: a negative-control probe returned PRESENT."
            )
    lines.append("")

    # SECTION 3c
    lines.append("=== SECTION 3c: RUNTIME CAPABILITY PROBE (AMX) ===")
    amx = p.amx_runtime_probe
    if amx is None:
        lines.append("UNDETECTED (reason: AMX probe not run)")
    else:
        lines.append(f"prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA=18) rc: {_fmt_val(amx.prctl_rc)}")
        lines.append(_evid(amx.prctl_rc))
        lines.append(f"prctl errno: {_fmt_val(amx.prctl_errno)}")
        lines.append(_evid(amx.prctl_errno))
        lines.append(f"ARCH_GET_XCOMP_SUPP: {_fmt_val(amx.xcomp_supp)}")
        lines.append(_evid(amx.xcomp_supp))
        lines.append(f"ARCH_GET_XCOMP_PERM: {_fmt_val(amx.xcomp_perm)}")
        lines.append(_evid(amx.xcomp_perm))
        lines.append(f"/proc/self/status XCOMP: {_fmt_val(amx.status_xcomp)}")
        lines.append(_evid(amx.status_xcomp))
        lines.append(f"note: {amx.note}")
        lines.append(
            f"quant_kernel_tier split → cpuid_tier={_fmt_val(p.isa.cpuid_tier)} "
            f"/ usable_tier={_fmt_val(p.isa.usable_tier)}"
        )
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
    lines.append(f"  <- note: {p.mem_available_source_note}")
    if p.exec_env:
        lines.append(
            f"effective_memory_limit (USED DOWNSTREAM): "
            f"{_fmt_val(p.exec_env.effective_memory_limit)} "
            f"(winner: {p.exec_env.effective_memory_winner})"
        )
        lines.append(_evid(p.exec_env.effective_memory_limit))
    lines.append(
        f"swap_total / swap_used / swappiness: "
        f"{_fmt_val(p.swap_total_bytes)} / {_fmt_val(p.swap_used_bytes)} / "
        f"{_fmt_val(p.swappiness)}"
    )
    lines.append(f"  <- swap_total: {p.swap_total_bytes.evidence}")
    lines.append(f"  <- swap_used: {p.swap_used_bytes.evidence}")
    lines.append(f"  <- swappiness: {p.swappiness.evidence}")
    lines.append(
        f"page_size; hugepages configured (count, size): "
        f"{_fmt_val(p.page_size_bytes)}; "
        f"count={_fmt_val(p.hugepages_count)}, "
        f"size_bytes={_fmt_val(p.hugepages_size_bytes)}"
    )
    lines.append(f"  <- page_size: {p.page_size_bytes.evidence}")
    lines.append(f"  <- hugepages_count: {p.hugepages_count.evidence}")
    lines.append(f"  <- hugepages_size: {p.hugepages_size_bytes.evidence}")
    lines.append(
        f"memory channels / DIMM count / DIMM speed: "
        f"{_fmt_val(p.memory_channels)} / {_fmt_val(p.dimm_count)} / "
        f"{_fmt_val(p.dimm_speed_mts)}"
    )
    lines.append(f"  <- channels: {p.memory_channels.evidence}")
    lines.append(f"  <- DIMM count: {p.dimm_count.evidence}")
    lines.append(f"  <- DIMM speed: {p.dimm_speed_mts.evidence}")
    lines.append(
        f"theoretical_peak_bandwidth_GBps: "
        f"{_fmt_val(p.theoretical_peak_bandwidth_GBps)}"
    )
    lines.append(f"  formula: {p.bandwidth_formula}")
    lines.append(f"  confidence: {p.bandwidth_confidence}")
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

    lines.append("=== SECTION 6: DERIVED DEFAULTS (provisional) ===")
    lines.append(f"suggested_thread_count: {_fmt_val(p.suggested_thread_count)}")
    lines.append(f"  formula: {p.suggested_thread_formula}")
    lines.append(
        "  note: provisional; to be REPLACED by the measured saturation point "
        "in Phase 2; uses effective_cpu_count"
    )
    lines.append(f"headroom_factor_applied: {p.headroom_factor}")
    lines.append(
        f"usable_ram_for_model_bytes (from effective_memory_limit): "
        f"{_fmt_val(p.usable_ram_for_model_bytes)}"
    )
    lines.append(f"  formula: {p.usable_ram_formula}")
    lines.append(
        f"usable_ram_from_raw_MemAvailable (NOT USED DOWNSTREAM): "
        f"{_fmt_val(p.usable_ram_from_memavailable_bytes)}"
    )
    lines.append(_evid(p.usable_ram_from_memavailable_bytes))
    if p.usable_ram_warning:
        lines.append(f"  {p.usable_ram_warning}")
    lines.append(
        "  policy: headroom 0.70 (not 0.85) — leaves ~30% for OS, runtime, "
        "activations, and fragmentation; additional -5% when SwapTotal==0."
    )
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
    lines.append("Detection libraries:")
    for lib in p.detection_libraries:
        lines.append(f"  - {lib}")
    lines.append(
        "Why no third-party lib: raw /proc, sysfs, and lscpu were available; "
        "stdlib subprocess/os/hashlib/ctypes suffice and keep evidence verbatim. "
        "ctypes used only for prctl AMX XCOMP probe (no raw file equivalent)."
    )
    lines.append(
        "Capture-once: each command/file below was executed/read exactly once; "
        "sha256 of the stored blob is shown."
    )
    lines.append("Commands / reads (in order):")
    for cmd in p.commands_run:
        lines.append(f"  $ {cmd}")

    return "\n".join(lines)
