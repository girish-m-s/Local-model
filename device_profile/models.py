"""DeviceProfile data model for Phase 1 / 1.5 CPU/memory detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class QuantKernelTier(str, Enum):
    AMX = "AMX"
    AVX512_VNNI = "AVX512_VNNI"
    AVX512 = "AVX512"
    AVX_VNNI = "AVX_VNNI"
    AVX2 = "AVX2"
    SSE_ONLY = "SSE_ONLY"
    ARM_SME = "ARM_SME"
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
class CaptureBlob:
    """One raw source captured exactly once."""

    key: str
    command: str
    content: str
    sha256: str
    skipped_reason: Optional[str] = None


@dataclass
class CacheInfo:
    l1d_bytes: EvidenceField  # per-core/instance from sysfs when available
    l1i_bytes: EvidenceField
    l2_bytes: EvidenceField
    l2_sharing: EvidenceField
    l3_bytes: EvidenceField
    # Aggregates from lscpu (for cross-check)
    lscpu_l1d_bytes: Optional[EvidenceField] = None
    lscpu_l1i_bytes: Optional[EvidenceField] = None
    lscpu_l2_bytes: Optional[EvidenceField] = None
    lscpu_l3_bytes: Optional[EvidenceField] = None
    lscpu_l1d_instances: Optional[EvidenceField] = None
    lscpu_l1i_instances: Optional[EvidenceField] = None
    lscpu_l2_instances: Optional[EvidenceField] = None
    lscpu_l3_instances: Optional[EvidenceField] = None


@dataclass
class HybridInfo:
    is_hybrid: EvidenceField
    p_core_count: Optional[EvidenceField] = None
    e_core_count: Optional[EvidenceField] = None
    p_core_logical_ids: Optional[EvidenceField] = None
    e_core_logical_ids: Optional[EvidenceField] = None


@dataclass
class IsaFlags:
    arch_family: str  # "x86" | "arm" | "other"
    flags: dict[str, EvidenceField] = field(default_factory=dict)
    flag_match_mode: str = "whitespace-tokenized"
    absent_branch_note: str = ""
    cpuid_tier: EvidenceField = field(
        default_factory=lambda: undetected("tier not computed")
    )
    usable_tier: EvidenceField = field(
        default_factory=lambda: undetected("tier not computed")
    )
    tier_runtime_verified: bool = False
    tier_runtime_verified_note: str = (
        "tier_runtime_verified=false: CPUID presence does NOT imply the "
        "inference runtime has kernels for that tier; confirm by probe later."
    )
    quant_kernel_reasoning: str = ""
    quant_kernel_tier: EvidenceField = field(
        default_factory=lambda: undetected("tier not computed")
    )
    macos_sysctl_feats: dict[str, EvidenceField] = field(default_factory=dict)


@dataclass
class CrossCheck:
    name: str
    left: str
    right: str
    result: str  # PASS | FAIL | SKIP


@dataclass
class ParserNegativeControl:
    match_mode: str
    probes: list[dict[str, str]] = field(default_factory=list)
    overall: str = "PASS"


@dataclass
class AmxRuntimeProbe:
    prctl_rc: EvidenceField
    prctl_errno: EvidenceField
    xcomp_supp: EvidenceField
    xcomp_perm: EvidenceField
    status_xcomp: EvidenceField
    note: str = ""


@dataclass
class ExecutionEnvironment:
    hypervisor_vendor_lscpu: EvidenceField
    virtualization_lscpu: EvidenceField
    hypervisor_sysfs: EvidenceField
    systemd_detect_virt: EvidenceField
    dockerenv: EvidenceField
    containerenv: EvidenceField
    proc1_cgroup: EvidenceField
    cgroup_version: EvidenceField
    memory_max: EvidenceField
    memory_high: EvidenceField
    memory_current: EvidenceField
    cpu_max: EvidenceField
    cpu_max_quota_period: EvidenceField
    cpuset_cpus_effective: EvidenceField
    cpuset_size: EvidenceField
    os_cpu_count: EvidenceField
    sched_affinity_len: EvidenceField
    sched_affinity_set: EvidenceField
    steal_sample_t0: EvidenceField
    steal_sample_t1: EvidenceField
    steal_percent: EvidenceField
    effective_memory_limit: EvidenceField
    effective_memory_winner: str
    effective_mem_inputs: str = ""
    effective_cpu_count: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    effective_cpu_winner: str = ""
    # Spec aliases
    effective_cores: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    effective_cores_inputs: str = ""
    effective_mem_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )


@dataclass
class DeviceProfile:
    # Meta
    platform: str
    python_version: str
    detection_libraries: list[str]
    commands_run: list[str]
    raw_source_dump: str
    captures: list[CaptureBlob] = field(default_factory=list)
    is_development_proxy: bool = False
    proxy_banner: str = ""
    target_reliability_statement: str = ""

    # Identity
    model_name: EvidenceField = field(default_factory=lambda: undetected("unset"))
    vendor: EvidenceField = field(default_factory=lambda: undetected("unset"))
    cpu_family: EvidenceField = field(default_factory=lambda: undetected("unset"))
    model: EvidenceField = field(default_factory=lambda: undetected("unset"))
    stepping: EvidenceField = field(default_factory=lambda: undetected("unset"))
    arch: EvidenceField = field(default_factory=lambda: undetected("unset"))
    base_mhz: EvidenceField = field(default_factory=lambda: undetected("unset"))
    max_mhz: EvidenceField = field(default_factory=lambda: undetected("unset"))
    uarch_host_level: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )

    # Topology
    logical_cores: EvidenceField = field(default_factory=lambda: undetected("unset"))
    physical_cores: EvidenceField = field(default_factory=lambda: undetected("unset"))
    threads_per_core: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    sockets: EvidenceField = field(default_factory=lambda: undetected("unset"))
    smt_enabled: EvidenceField = field(default_factory=lambda: undetected("unset"))
    hybrid: HybridInfo = field(
        default_factory=lambda: HybridInfo(is_hybrid=undetected("unset"))
    )
    numa_nodes: EvidenceField = field(default_factory=lambda: undetected("unset"))
    cpu_to_node_map: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    cache: CacheInfo = field(
        default_factory=lambda: CacheInfo(
            l1d_bytes=undetected("unset"),
            l1i_bytes=undetected("unset"),
            l2_bytes=undetected("unset"),
            l2_sharing=undetected("unset"),
            l3_bytes=undetected("unset"),
        )
    )

    # Host class / execution environment
    host_class: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    host_class_banner: str = ""
    exec_env: Optional[ExecutionEnvironment] = None
    storage: Any = None  # extras.StorageInfo
    thermal_power: Any = None  # extras.ThermalPowerInfo
    memory_budget: Any = None  # extras.MemoryBudget

    # ISA
    isa: IsaFlags = field(default_factory=lambda: IsaFlags(arch_family="other"))
    parser_negative_control: Optional[ParserNegativeControl] = None
    amx_runtime_probe: Optional[AmxRuntimeProbe] = None

    # Memory
    mem_total_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    mem_available_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    mem_free_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    mem_available_source_note: str = ""
    swap_total_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    swap_used_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    swappiness: EvidenceField = field(default_factory=lambda: undetected("unset"))
    page_size_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    hugepages_count: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    hugepages_size_bytes: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    memory_channels: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    dimm_count: EvidenceField = field(default_factory=lambda: undetected("unset"))
    dimm_speed_mts: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    # Bandwidth is measured, not computed from DMI.
    measured_bandwidth_GBps: EvidenceField = field(
        default_factory=lambda: EvidenceField(
            value="PENDING_PHASE_2",
            evidence=(
                "placeholder: theoretical DMI bandwidth deleted; "
                "bandwidth will be measured in Phase 2, not computed"
            ),
        )
    )

    # Derived (provisional) — MUST use effective_* limits
    suggested_thread_count: EvidenceField = field(
        default_factory=lambda: undetected("unset")
    )
    suggested_thread_formula: str = ""

    # Cross-checks & critique
    cross_checks: list[CrossCheck] = field(default_factory=list)
    core_count_sources: dict[str, Any] = field(default_factory=dict)
    self_critique: dict[str, list[str]] = field(default_factory=dict)
    per_core_type_counts: dict[str, int] = field(default_factory=dict)
    lscpu_unique_cores: Optional[int] = None
    devices_reached_note: str = ""

    def to_json_dict(self) -> dict[str, Any]:
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
