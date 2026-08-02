"""DeviceProfile data model for Phase 1 CPU/memory detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class QuantKernelTier(str, Enum):
    AMX = "AMX"
    AVX512_VNNI = "AVX512_VNNI"
    AVX512 = "AVX512"
    AVX2 = "AVX2"
    SSE_ONLY = "SSE_ONLY"
    ARM_I8MM = "ARM_I8MM"
    ARM_DOTPROD = "ARM_DOTPROD"
    ARM_BASELINE = "ARM_BASELINE"


@dataclass
class EvidenceField:
    """A detected value paired with the verbatim source it was parsed from."""

    value: Any
    evidence: str  # verbatim raw line / library attr, or UNDETECTED reason

    def is_detected(self) -> bool:
        return not (
            isinstance(self.value, str) and self.value.startswith("UNDETECTED")
        ) and self.value is not None and not (
            isinstance(self.evidence, str) and self.evidence.startswith("UNDETECTED")
        )


def undetected(reason: str) -> EvidenceField:
    return EvidenceField(
        value=f"UNDETECTED (reason: {reason})",
        evidence=f"UNDETECTED (reason: {reason})",
    )


@dataclass
class CacheInfo:
    l1d_bytes: EvidenceField
    l1i_bytes: EvidenceField
    l2_bytes: EvidenceField
    l2_sharing: EvidenceField  # "per-core" | "shared" | UNDETECTED
    l3_bytes: EvidenceField


@dataclass
class HybridInfo:
    is_hybrid: EvidenceField
    p_core_count: Optional[EvidenceField] = None
    e_core_count: Optional[EvidenceField] = None
    p_core_logical_ids: Optional[EvidenceField] = None
    e_core_logical_ids: Optional[EvidenceField] = None


@dataclass
class IsaFlags:
    """Platform-specific ISA presence map: flag -> (PRESENT|ABSENT, evidence)."""

    arch_family: str  # "x86" | "arm" | "other"
    flags: dict[str, EvidenceField] = field(default_factory=dict)
    quant_kernel_tier: EvidenceField = field(
        default_factory=lambda: undetected("tier not computed")
    )
    quant_kernel_reasoning: str = ""


@dataclass
class CrossCheck:
    name: str
    left: str
    right: str
    result: str  # PASS | FAIL | SKIP


@dataclass
class DeviceProfile:
    # Meta
    platform: str
    python_version: str
    detection_libraries: list[str]
    commands_run: list[str]
    raw_source_dump: str

    # Identity
    model_name: EvidenceField
    vendor: EvidenceField
    cpu_family: EvidenceField
    model: EvidenceField
    stepping: EvidenceField
    arch: EvidenceField
    base_mhz: EvidenceField
    max_mhz: EvidenceField

    # Topology
    logical_cores: EvidenceField
    physical_cores: EvidenceField
    threads_per_core: EvidenceField
    sockets: EvidenceField
    smt_enabled: EvidenceField
    hybrid: HybridInfo
    numa_nodes: EvidenceField
    cpu_to_node_map: EvidenceField
    cache: CacheInfo

    # ISA
    isa: IsaFlags

    # Memory
    mem_total_bytes: EvidenceField
    mem_available_bytes: EvidenceField
    mem_free_bytes: EvidenceField
    mem_available_source_note: str
    swap_total_bytes: EvidenceField
    swap_used_bytes: EvidenceField
    swappiness: EvidenceField
    page_size_bytes: EvidenceField
    hugepages_count: EvidenceField
    hugepages_size_bytes: EvidenceField
    memory_channels: EvidenceField
    dimm_count: EvidenceField
    dimm_speed_mts: EvidenceField
    theoretical_peak_bandwidth_GBps: EvidenceField
    bandwidth_formula: str
    bandwidth_confidence: str

    # Derived (provisional)
    suggested_thread_count: EvidenceField
    suggested_thread_formula: str
    usable_ram_for_model_bytes: EvidenceField
    usable_ram_formula: str

    # Cross-checks & critique
    cross_checks: list[CrossCheck] = field(default_factory=list)
    core_count_sources: dict[str, Any] = field(default_factory=dict)
    self_critique: dict[str, list[str]] = field(default_factory=dict)

    # Independent core-type counts for hybrid cross-check
    per_core_type_counts: dict[str, int] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for SECTION 8; EvidenceField becomes {value, evidence}."""

        def convert(obj: Any) -> Any:
            if isinstance(obj, EvidenceField):
                return {"value": obj.value, "evidence": obj.evidence}
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, CrossCheck):
                return asdict(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj

        return convert(self)
