"""Exact stdout report formatter for Phase 1 DeviceProfile detection."""

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

    lines.append("=== SECTION 0: RAW SOURCE DUMP ===")
    lines.append(p.raw_source_dump)
    lines.append("")

    lines.append("=== SECTION 1: IDENTITY ===")
    lines.append(f"model_name: { _fmt_val(p.model_name) }")
    lines.append(_evid(p.model_name))
    lines.append(
        f"vendor / family / model / stepping: "
        f"{_fmt_val(p.vendor)} / {_fmt_val(p.cpu_family)} / "
        f"{_fmt_val(p.model)} / {_fmt_val(p.stepping)}"
    )
    lines.append(
        f"  <- vendor: {p.vendor.evidence}"
    )
    lines.append(
        f"  <- family: {p.cpu_family.evidence}"
    )
    lines.append(
        f"  <- model: {p.model.evidence}"
    )
    lines.append(
        f"  <- stepping: {p.stepping.evidence}"
    )
    lines.append(f"arch (uname -m): {_fmt_val(p.arch)}")
    lines.append(_evid(p.arch))
    lines.append(
        f"base_mhz / max_mhz: {_fmt_val(p.base_mhz)} / {_fmt_val(p.max_mhz)}"
    )
    lines.append(f"  <- base_mhz: {p.base_mhz.evidence}")
    lines.append(f"  <- max_mhz: {p.max_mhz.evidence}")
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
        f"cache L1d / L1i / L2 / L3 (bytes): "
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

    lines.append("=== SECTION 3: ISA FLAGS ===")
    for flag, ef in p.isa.flags.items():
        lines.append(f"{flag}: {ef.value}  <- {ef.evidence}")
    lines.append(f"quant_kernel_tier: {_fmt_val(p.isa.quant_kernel_tier)}")
    lines.append(f"  reasoning: {p.isa.quant_kernel_reasoning}")
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
        "in Phase 2"
    )
    lines.append(
        f"usable_ram_for_model_bytes: {_fmt_val(p.usable_ram_for_model_bytes)}"
    )
    lines.append(f"  formula: {p.usable_ram_formula}")
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

    lines.append("=== META: COMMANDS & VERSIONS ===")
    lines.append(f"Python: {sys.version}")
    lines.append(f"platform.python_version(): {platform.python_version()}")
    lines.append(f"platform.platform(): {platform.platform()}")
    lines.append("Detection libraries:")
    for lib in p.detection_libraries:
        lines.append(f"  - {lib}")
    lines.append(
        "Why no third-party lib: raw /proc, sysfs, and lscpu were available; "
        "stdlib subprocess/os suffice and keep evidence verbatim."
    )
    lines.append("Commands shelled out to (in order, including re-reads):")
    for cmd in p.commands_run:
        lines.append(f"  $ {cmd}")

    return "\n".join(lines)
