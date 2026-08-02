"""Cross-platform entry point for DeviceProfile detection."""

from __future__ import annotations

import platform
import sys

from .models import DeviceProfile, EvidenceField, HybridInfo, IsaFlags, CacheInfo, undetected


def detect_device_profile() -> DeviceProfile:
    system = platform.system().lower()
    if system == "linux":
        from .linux import detect_linux

        return detect_linux()
    if system == "darwin":
        return _unsupported_stub("macos", "macOS detector not implemented in Phase 1 scaffold")
    if system == "windows":
        return _unsupported_stub("windows", "Windows detector not implemented in Phase 1 scaffold")
    return _unsupported_stub(system, f"unsupported platform: {system}")


def _unsupported_stub(platform_name: str, reason: str) -> DeviceProfile:
    """Return a profile full of UNDETECTED rather than fabricating values."""
    u = undetected(reason)
    return DeviceProfile(
        platform=platform_name,
        python_version=platform.python_version(),
        detection_libraries=[f"Python {platform.python_version()} stdlib only"],
        commands_run=[],
        raw_source_dump=f"UNDETECTED (reason: {reason})",
        model_name=u,
        vendor=u,
        cpu_family=u,
        model=u,
        stepping=u,
        arch=EvidenceField(
            value=platform.machine(),
            evidence=f"python platform.machine() (Python {platform.python_version()})",
        ),
        base_mhz=u,
        max_mhz=u,
        logical_cores=u,
        physical_cores=u,
        threads_per_core=u,
        sockets=u,
        smt_enabled=u,
        hybrid=HybridInfo(is_hybrid=u),
        numa_nodes=u,
        cpu_to_node_map=u,
        cache=CacheInfo(
            l1d_bytes=u,
            l1i_bytes=u,
            l2_bytes=u,
            l2_sharing=u,
            l3_bytes=u,
        ),
        isa=IsaFlags(arch_family="other"),
        mem_total_bytes=u,
        mem_available_bytes=u,
        mem_free_bytes=u,
        mem_available_source_note=reason,
        swap_total_bytes=u,
        swap_used_bytes=u,
        swappiness=u,
        page_size_bytes=u,
        hugepages_count=u,
        hugepages_size_bytes=u,
        memory_channels=u,
        dimm_count=u,
        dimm_speed_mts=u,
        theoretical_peak_bandwidth_GBps=u,
        bandwidth_formula="UNDETECTED",
        bandwidth_confidence="guess",
        suggested_thread_count=u,
        suggested_thread_formula="UNDETECTED",
        usable_ram_for_model_bytes=u,
        usable_ram_formula="UNDETECTED",
        cross_checks=[],
        self_critique={
            "fields that are heuristic, not directly read": [],
            "platform code paths not exercised on this machine": [
                "linux lscpu//proc path",
                "macos sysctl path",
                "windows CIM path",
            ],
            "places a default could have been silently substituted": [
                "entire profile left UNDETECTED rather than guessed",
            ],
            "known parsing fragility": [reason],
        },
    )


def main(argv: list[str] | None = None) -> int:
    from .report import format_report

    _ = argv  # reserved
    profile = detect_device_profile()
    sys.stdout.write(format_report(profile))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
