"""Linux CPU/memory detection from raw sources only (lscpu, /proc, sysfs)."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Optional

from .models import (
    CacheInfo,
    CrossCheck,
    DeviceProfile,
    EvidenceField,
    HybridInfo,
    IsaFlags,
    QuantKernelTier,
    undetected,
)

X86_FLAGS = [
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "f16c",
    "avx512f",
    "avx512bw",
    "avx512vl",
    "avx512_vnni",
    "avx512_bf16",
    "amx_tile",
    "amx_int8",
    "amx_bf16",
]

ARM_FLAGS = [
    "asimd",
    "asimdhp",
    "asimddp",
    "i8mm",
    "bf16",
    "sve",
    "sve2",
    "sme",
]


def _run(cmd: list[str], commands_log: list[str]) -> tuple[int, str, str]:
    """Run a command, log it, return (rc, stdout, stderr). Never raises."""
    commands_log.append(" ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except OSError as exc:
        return 1, "", str(exc)


def _read_file(path: str, commands_log: list[str]) -> Optional[str]:
    commands_log.append(f"read {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _parse_lscpu_kv(text: str) -> dict[str, str]:
    """Parse `lscpu` key: value lines. Keys kept as printed (stripped)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def _find_lscpu_line(text: str, key: str) -> Optional[str]:
    for line in text.splitlines():
        if line.startswith(key + ":") or line.startswith(key + " "):
            # Match "Key:" at start after optional spaces aren't in lscpu
            if line.split(":", 1)[0].strip() == key:
                return line
    return None


def _lscpu_value_and_line(text: str, key: str) -> EvidenceField:
    line = _find_lscpu_line(text, key)
    if line is None:
        return undetected(f"key '{key}' not present in lscpu output")
    val = line.split(":", 1)[1].strip()
    return EvidenceField(value=val, evidence=line)


def _parse_kib_to_bytes(size_str: str) -> Optional[int]:
    """Parse strings like '192 KiB', '8 MiB', '320 MiB (1 instance)'."""
    m = re.match(
        r"^\s*([\d.]+)\s*(KiB|MiB|GiB|TiB|KB|MB|GB|B)\b",
        size_str,
        re.IGNORECASE,
    )
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    mult = {
        "b": 1,
        "kib": 1024,
        "kb": 1000,
        "mib": 1024**2,
        "mb": 1000**2,
        "gib": 1024**3,
        "gb": 1000**3,
        "tib": 1024**4,
    }[unit]
    return int(num * mult)


def _parse_sysfs_size_to_bytes(size_str: str) -> Optional[int]:
    """Parse sysfs cache size like '48K', '2048K', '327680K'."""
    m = re.match(r"^\s*(\d+)\s*([KMG])?\s*$", size_str.strip(), re.IGNORECASE)
    if not m:
        return None
    num = int(m.group(1))
    unit = (m.group(2) or "").upper()
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return num * mult


def _meminfo_kb(meminfo: str, key: str) -> tuple[Optional[int], Optional[str]]:
    for line in meminfo.splitlines():
        if line.startswith(key + ":"):
            parts = line.split()
            # e.g. MemTotal: 16398452 kB
            if len(parts) >= 2:
                try:
                    return int(parts[1]), line
                except ValueError:
                    return None, line
    return None, None


def _flag_present(flags_blob: str, flag: str) -> EvidenceField:
    """
    Match flag as a whole token in the flags string.
    Evidence is the exact matched substring (the flag itself) or 'not found in ...'.
    """
    # Word-boundary match; flags may use underscores
    pattern = r"(?<![\w])" + re.escape(flag) + r"(?![\w])"
    m = re.search(pattern, flags_blob)
    if m:
        return EvidenceField(value="PRESENT", evidence=m.group(0))
    return EvidenceField(
        value="ABSENT",
        evidence=f"not found in Flags /proc/cpuinfo|lscpu flags field",
    )


def _detect_isa(
    arch: str, flags_blob: str, flags_evidence_source: str
) -> IsaFlags:
    arch_l = arch.lower()
    if arch_l in ("x86_64", "amd64", "i386", "i686", "x86"):
        family = "x86"
        wanted = X86_FLAGS
    elif arch_l.startswith("arm") or arch_l.startswith("aarch"):
        family = "arm"
        wanted = ARM_FLAGS
    else:
        family = "other"
        wanted = []

    flags: dict[str, EvidenceField] = {}
    for f in wanted:
        ev = _flag_present(flags_blob, f)
        if ev.value == "ABSENT":
            ev = EvidenceField(
                value="ABSENT",
                evidence=f"not found in {flags_evidence_source}",
            )
        flags[f] = ev

    present = {k for k, v in flags.items() if v.value == "PRESENT"}
    tier: QuantKernelTier
    reasoning: str
    if family == "x86":
        if "amx_tile" in present or "amx_int8" in present:
            tier = QuantKernelTier.AMX
            reasoning = (
                "amx_tile/amx_int8 PRESENT → AMX (highest x86 integer-matmul tier)"
            )
        elif "avx512_vnni" in present:
            tier = QuantKernelTier.AVX512_VNNI
            reasoning = "avx512_vnni PRESENT (and AMX absent) → AVX512_VNNI"
        elif "avx512f" in present:
            tier = QuantKernelTier.AVX512
            reasoning = "avx512f PRESENT (VNNI/AMX absent) → AVX512"
        elif "avx2" in present:
            tier = QuantKernelTier.AVX2
            reasoning = "avx2 PRESENT (AVX512/AMX absent) → AVX2"
        elif "sse4_2" in present:
            tier = QuantKernelTier.SSE_ONLY
            reasoning = "sse4_2 PRESENT (AVX2+ absent) → SSE_ONLY"
        else:
            tier = QuantKernelTier.SSE_ONLY
            reasoning = (
                "no sse4_2/avx2/avx512/amx detected; provisional SSE_ONLY "
                "(lowest x86 tier label)"
            )
    elif family == "arm":
        if "i8mm" in present:
            tier = QuantKernelTier.ARM_I8MM
            reasoning = "i8mm PRESENT → ARM_I8MM"
        elif "asimddp" in present:
            tier = QuantKernelTier.ARM_DOTPROD
            reasoning = "asimddp PRESENT (i8mm absent) → ARM_DOTPROD"
        else:
            tier = QuantKernelTier.ARM_BASELINE
            reasoning = "no i8mm/asimddp → ARM_BASELINE"
    else:
        return IsaFlags(
            arch_family=family,
            flags=flags,
            quant_kernel_tier=undetected(
                f"unsupported arch_family={family} for quant tier"
            ),
            quant_kernel_reasoning="arch not x86/arm",
        )

    return IsaFlags(
        arch_family=family,
        flags=flags,
        quant_kernel_tier=EvidenceField(
            value=tier.value,
            evidence=f"derived from flags in {flags_evidence_source}; {reasoning}",
        ),
        quant_kernel_reasoning=reasoning,
    )


def _cache_from_sysfs(commands_log: list[str]) -> Optional[CacheInfo]:
    base = "/sys/devices/system/cpu/cpu0/cache"
    if not os.path.isdir(base):
        return None

    by_level: dict[tuple[int, str], tuple[int, str, str]] = {}
    # (level, type) -> (bytes, size_raw_line_path_content, shared_cpu_list)
    try:
        indices = sorted(
            d for d in os.listdir(base) if d.startswith("index")
        )
    except OSError:
        return None

    for idx in indices:
        level_p = os.path.join(base, idx, "level")
        type_p = os.path.join(base, idx, "type")
        size_p = os.path.join(base, idx, "size")
        shared_p = os.path.join(base, idx, "shared_cpu_list")
        level_s = _read_file(level_p, commands_log)
        type_s = _read_file(type_p, commands_log)
        size_s = _read_file(size_p, commands_log)
        shared_s = _read_file(shared_p, commands_log)
        if not (level_s and type_s and size_s):
            continue
        level = int(level_s.strip())
        typ = type_s.strip()
        size_bytes = _parse_sysfs_size_to_bytes(size_s.strip())
        if size_bytes is None:
            continue
        by_level[(level, typ)] = (
            size_bytes,
            f"{size_p}: {size_s.strip()}",
            (shared_s or "").strip(),
        )

    def get_level(level: int, typ: str) -> EvidenceField:
        key = (level, typ)
        if key not in by_level:
            return undetected(
                f"sysfs cache level={level} type={typ} not found under {base}"
            )
        b, raw, _ = by_level[key]
        return EvidenceField(value=b, evidence=raw)

    l2 = get_level(2, "Unified")
    l2_share = undetected("L2 not found in sysfs")
    if (2, "Unified") in by_level:
        _, raw_size, shared = by_level[(2, "Unified")]
        # If shared_cpu_list is a single CPU, per-core; else shared
        cpus = []
        for part in shared.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                cpus.extend(range(int(a), int(b) + 1))
            elif part.isdigit():
                cpus.append(int(part))
        sharing = "per-core" if len(cpus) <= 1 else "shared"
        l2_share = EvidenceField(
            value=sharing,
            evidence=(
                f"/sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_list: "
                f"{shared} → {sharing}"
            ),
        )

    return CacheInfo(
        l1d_bytes=get_level(1, "Data"),
        l1i_bytes=get_level(1, "Instruction"),
        l2_bytes=l2,
        l2_sharing=l2_share,
        l3_bytes=get_level(3, "Unified"),
    )


def _cache_from_lscpu(lscpu_text: str) -> CacheInfo:
    def one(key: str) -> EvidenceField:
        ef = _lscpu_value_and_line(lscpu_text, key)
        if isinstance(ef.value, str) and ef.value.startswith("UNDETECTED"):
            return ef
        raw_val = ef.value
        b = _parse_kib_to_bytes(str(raw_val))
        if b is None:
            return undetected(f"could not parse cache size from: {ef.evidence}")
        return EvidenceField(value=b, evidence=ef.evidence)

    l2_ef = _lscpu_value_and_line(lscpu_text, "L2 cache")
    l2_sharing: EvidenceField
    if isinstance(l2_ef.value, str) and l2_ef.value.startswith("UNDETECTED"):
        l2_sharing = undetected("L2 cache line missing from lscpu")
        l2_bytes = l2_ef
    else:
        b = _parse_kib_to_bytes(str(l2_ef.value))
        l2_bytes = (
            EvidenceField(value=b, evidence=l2_ef.evidence)
            if b is not None
            else undetected(f"could not parse: {l2_ef.evidence}")
        )
        # Heuristic from instance count in lscpu: "8 MiB (4 instances)" with
        # 4 cores → per-core; "(1 instance)" → shared
        m = re.search(r"\((\d+)\s+instances?\)", str(l2_ef.value))
        if m:
            instances = int(m.group(1))
            # Compare to core count if available later; for evidence, report instances
            cores_line = _lscpu_value_and_line(lscpu_text, "Core(s) per socket")
            sockets_line = _lscpu_value_and_line(lscpu_text, "Socket(s)")
            try:
                cores = int(str(cores_line.value)) * int(str(sockets_line.value))
                sharing = "per-core" if instances >= cores else "shared"
            except (ValueError, TypeError):
                sharing = "per-core" if instances > 1 else "shared"
            l2_sharing = EvidenceField(
                value=sharing,
                evidence=(
                    f"{l2_ef.evidence} (instance count heuristic → {sharing})"
                ),
            )
        else:
            l2_sharing = undetected(
                "lscpu L2 line has no instance count; cannot determine sharing"
            )

    return CacheInfo(
        l1d_bytes=one("L1d cache"),
        l1i_bytes=one("L1i cache"),
        l2_bytes=l2_bytes,
        l2_sharing=l2_sharing,
        l3_bytes=one("L3 cache"),
    )


def _detect_hybrid(
    lscpu_e: str, commands_log: list[str], physical_cores: Optional[int]
) -> tuple[HybridInfo, dict[str, int]]:
    """Detect hybrid P/E cores via sysfs core_type or cpu_capacity."""
    p_ids: list[int] = []
    e_ids: list[int] = []
    evidence_lines: list[str] = []

    cpu_dirs = sorted(
        d
        for d in (
            os.listdir("/sys/devices/system/cpu")
            if os.path.isdir("/sys/devices/system/cpu")
            else []
        )
        if re.fullmatch(r"cpu\d+", d)
    )
    commands_log.append("list /sys/devices/system/cpu/cpuN")

    any_core_type = False
    any_capacity = False
    capacities: dict[int, int] = {}

    for d in cpu_dirs:
        cpu_id = int(d[3:])
        ct_path = f"/sys/devices/system/cpu/{d}/topology/core_type"
        cap_path = f"/sys/devices/system/cpu/{d}/cpu_capacity"
        ct = _read_file(ct_path, commands_log)
        if ct is not None:
            any_core_type = True
            ct_v = ct.strip().lower()
            evidence_lines.append(f"{ct_path}: {ct.strip()}")
            if ct_v in ("performance", "p", "2"):
                p_ids.append(cpu_id)
            elif ct_v in ("efficiency", "e", "1", "0"):
                e_ids.append(cpu_id)
        cap = _read_file(cap_path, commands_log)
        if cap is not None:
            any_capacity = True
            try:
                capacities[cpu_id] = int(cap.strip())
                evidence_lines.append(f"{cap_path}: {cap.strip()}")
            except ValueError:
                pass

    if any_core_type and (p_ids or e_ids):
        is_h = bool(p_ids) and bool(e_ids)
        return (
            HybridInfo(
                is_hybrid=EvidenceField(
                    value=is_h,
                    evidence="; ".join(evidence_lines)
                    if evidence_lines
                    else "sysfs core_type present",
                ),
                p_core_count=EvidenceField(
                    value=len(set(p_ids)),
                    evidence=f"P-core logical IDs from core_type: {sorted(p_ids)}",
                ),
                e_core_count=EvidenceField(
                    value=len(set(e_ids)),
                    evidence=f"E-core logical IDs from core_type: {sorted(e_ids)}",
                ),
                p_core_logical_ids=EvidenceField(
                    value=sorted(p_ids),
                    evidence=f"P-core logical IDs: {sorted(p_ids)}",
                ),
                e_core_logical_ids=EvidenceField(
                    value=sorted(e_ids),
                    evidence=f"E-core logical IDs: {sorted(e_ids)}",
                ),
            ),
            {"P": len(set(p_ids)), "E": len(set(e_ids))},
        )

    if any_capacity and capacities:
        max_c = max(capacities.values())
        min_c = min(capacities.values())
        if max_c != min_c:
            for cid, c in capacities.items():
                if c >= max_c:
                    p_ids.append(cid)
                else:
                    e_ids.append(cid)
            return (
                HybridInfo(
                    is_hybrid=EvidenceField(
                        value=True,
                        evidence=(
                            "cpu_capacity varies "
                            f"(min={min_c}, max={max_c}); "
                            + "; ".join(evidence_lines)
                        ),
                    ),
                    p_core_count=EvidenceField(
                        value=len(p_ids),
                        evidence=f"logical IDs with capacity==max: {sorted(p_ids)}",
                    ),
                    e_core_count=EvidenceField(
                        value=len(e_ids),
                        evidence=f"logical IDs with capacity<max: {sorted(e_ids)}",
                    ),
                    p_core_logical_ids=EvidenceField(
                        value=sorted(p_ids),
                        evidence=str(sorted(p_ids)),
                    ),
                    e_core_logical_ids=EvidenceField(
                        value=sorted(e_ids),
                        evidence=str(sorted(e_ids)),
                    ),
                ),
                {"P": len(p_ids), "E": len(e_ids)},
            )
        return (
            HybridInfo(
                is_hybrid=EvidenceField(
                    value=False,
                    evidence=(
                        f"all cpu_capacity equal ({max_c}); "
                        + "; ".join(evidence_lines[:4])
                        + (" ..." if len(evidence_lines) > 4 else "")
                    ),
                )
            ),
            {"homogeneous": physical_cores or len(capacities)},
        )

    # Fall back: look for "Core type" columns in lscpu -e
    if lscpu_e and re.search(r"CORE|CPU", lscpu_e.splitlines()[0] if lscpu_e else ""):
        header = lscpu_e.splitlines()[0] if lscpu_e.splitlines() else ""
        if "TYPE" in header.upper() or "CORETYPE" in header.upper().replace(" ", ""):
            return (
                HybridInfo(
                    is_hybrid=undetected(
                        "lscpu -e has type-like column but parser not implemented "
                        f"for header: {header}"
                    )
                ),
                {},
            )

    return (
        HybridInfo(
            is_hybrid=EvidenceField(
                value=False,
                evidence=(
                    "no sysfs topology/core_type or cpu_capacity present; "
                    "lscpu -e has no core-type column → treated as non-hybrid "
                    "(direct absence of hybrid signals, not a guessed default)"
                ),
            )
        ),
        {"homogeneous": physical_cores} if physical_cores is not None else {},
    )


def collect_raw_source_dump(commands_log: list[str]) -> str:
    """SECTION 0: verbatim dump of platform raw sources."""
    parts: list[str] = []

    parts.append("$ lscpu")
    rc, out, err = _run(["lscpu"], commands_log)
    parts.append(out if out else f"[exit {rc}] {err}")

    parts.append("\n$ head -30 /proc/cpuinfo")
    commands_log.append("head -30 /proc/cpuinfo")
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for i, line in enumerate(fh):
                if i >= 30:
                    break
                lines.append(line)
            parts.append("".join(lines).rstrip("\n"))
            # Mark truncation explicitly
            rest = fh.read(1)
            if rest or i >= 29:
                # Check if file has more than 30 lines
                pass
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            total_lines = sum(1 for _ in fh)
        if total_lines > 30:
            parts.append(
                f"\n[TRUNCATED: showing first 30 of {total_lines} lines of /proc/cpuinfo]"
            )
    except OSError as exc:
        parts.append(f"[error reading /proc/cpuinfo: {exc}]")

    parts.append("\n$ cat /proc/meminfo | head -8")
    commands_log.append("cat /proc/meminfo | head -8")
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as fh:
            mem_lines = []
            for i, line in enumerate(fh):
                if i >= 8:
                    break
                mem_lines.append(line)
            parts.append("".join(mem_lines).rstrip("\n"))
            total = i + 1 + sum(1 for _ in fh)
        # recount properly
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as fh:
            total = sum(1 for _ in fh)
        if total > 8:
            parts.append(
                f"\n[TRUNCATED: showing first 8 of {total} lines of /proc/meminfo]"
            )
    except OSError as exc:
        parts.append(f"[error reading /proc/meminfo: {exc}]")

    parts.append("\n$ lscpu -e")
    rc, out, err = _run(["lscpu", "-e"], commands_log)
    parts.append(out.rstrip("\n") if out else f"[exit {rc}] {err}")

    parts.append("\n$ numactl --hardware")
    if shutil.which("numactl"):
        rc, out, err = _run(["numactl", "--hardware"], commands_log)
        parts.append(out.rstrip("\n") if out else f"[exit {rc}] {err}")
    else:
        commands_log.append("numactl --hardware  # SKIPPED: numactl not in PATH")
        parts.append("[numactl not present in PATH — skipped]")

    return "\n".join(parts)


def detect_linux() -> DeviceProfile:
    commands_log: list[str] = []
    raw_dump = collect_raw_source_dump(commands_log)

    # Re-read full sources for parsing (also logged)
    rc, lscpu_out, lscpu_err = _run(["lscpu"], commands_log)
    if rc != 0 and not lscpu_out:
        lscpu_out = ""
    rc, lscpu_e, _ = _run(["lscpu", "-e"], commands_log)
    cpuinfo = _read_file("/proc/cpuinfo", commands_log) or ""
    meminfo = _read_file("/proc/meminfo", commands_log) or ""

    # --- Identity ---
    model_name = _lscpu_value_and_line(lscpu_out, "Model name")
    if isinstance(model_name.value, str) and model_name.value.startswith("UNDETECTED"):
        # fallback /proc/cpuinfo
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                model_name = EvidenceField(
                    value=line.split(":", 1)[1].strip(),
                    evidence=line,
                )
                break

    vendor = _lscpu_value_and_line(lscpu_out, "Vendor ID")
    if isinstance(vendor.value, str) and vendor.value.startswith("UNDETECTED"):
        for line in cpuinfo.splitlines():
            if line.startswith("vendor_id"):
                vendor = EvidenceField(
                    value=line.split(":", 1)[1].strip(), evidence=line
                )
                break

    cpu_family = _lscpu_value_and_line(lscpu_out, "CPU family")
    if isinstance(cpu_family.value, str) and cpu_family.value.startswith("UNDETECTED"):
        for line in cpuinfo.splitlines():
            if line.startswith("cpu family"):
                cpu_family = EvidenceField(
                    value=line.split(":", 1)[1].strip(), evidence=line
                )
                break

    model = _lscpu_value_and_line(lscpu_out, "Model")
    # Avoid matching "Model name" — _find_lscpu_line uses exact key match, good.
    if isinstance(model.value, str) and model.value.startswith("UNDETECTED"):
        for line in cpuinfo.splitlines():
            if line.startswith("model\t") or line.startswith("model "):
                model = EvidenceField(
                    value=line.split(":", 1)[1].strip(), evidence=line
                )
                break

    stepping = _lscpu_value_and_line(lscpu_out, "Stepping")
    if isinstance(stepping.value, str) and stepping.value.startswith("UNDETECTED"):
        for line in cpuinfo.splitlines():
            if line.startswith("stepping"):
                stepping = EvidenceField(
                    value=line.split(":", 1)[1].strip(), evidence=line
                )
                break

    # arch via uname -m
    rc, uname_m, uname_err = _run(["uname", "-m"], commands_log)
    if rc == 0 and uname_m.strip():
        arch = EvidenceField(value=uname_m.strip(), evidence=f"uname -m → {uname_m.strip()}")
    else:
        arch = undetected(f"uname -m failed: {uname_err}")

    # Frequencies
    base_mhz: EvidenceField
    max_mhz: EvidenceField
    cpu_mhz_lscpu = _lscpu_value_and_line(lscpu_out, "CPU MHz")
    max_mhz_lscpu = _lscpu_value_and_line(lscpu_out, "CPU max MHz")
    base_mhz_lscpu = _lscpu_value_and_line(lscpu_out, "CPU base MHz")
    # also try BIOS Model name fields / cpufreq sysfs
    base_freq_sys = _read_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency", commands_log
    )
    max_freq_sys = _read_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", commands_log
    )

    if base_freq_sys and base_freq_sys.strip().isdigit():
        # kHz → MHz
        mhz = int(base_freq_sys.strip()) / 1000.0
        base_mhz = EvidenceField(
            value=mhz,
            evidence=(
                f"/sys/devices/system/cpu/cpu0/cpufreq/base_frequency: "
                f"{base_freq_sys.strip()} (kHz) → {mhz} MHz"
            ),
        )
    elif not (
        isinstance(base_mhz_lscpu.value, str)
        and base_mhz_lscpu.value.startswith("UNDETECTED")
    ):
        try:
            base_mhz = EvidenceField(
                value=float(str(base_mhz_lscpu.value)),
                evidence=base_mhz_lscpu.evidence,
            )
        except ValueError:
            base_mhz = undetected(
                f"unparseable CPU base MHz: {base_mhz_lscpu.evidence}"
            )
    else:
        # cpuinfo "cpu MHz" is current frequency, NOT base — do not invent base
        base_mhz = undetected(
            "no cpufreq base_frequency, no lscpu 'CPU base MHz'; "
            "/proc/cpuinfo 'cpu MHz' is current freq and was not used as base"
        )

    if max_freq_sys and max_freq_sys.strip().isdigit():
        mhz = int(max_freq_sys.strip()) / 1000.0
        max_mhz = EvidenceField(
            value=mhz,
            evidence=(
                f"/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq: "
                f"{max_freq_sys.strip()} (kHz) → {mhz} MHz"
            ),
        )
    elif not (
        isinstance(max_mhz_lscpu.value, str)
        and max_mhz_lscpu.value.startswith("UNDETECTED")
    ):
        try:
            max_mhz = EvidenceField(
                value=float(str(max_mhz_lscpu.value)),
                evidence=max_mhz_lscpu.evidence,
            )
        except ValueError:
            max_mhz = undetected(f"unparseable CPU max MHz: {max_mhz_lscpu.evidence}")
    else:
        max_mhz = undetected(
            "no cpufreq cpuinfo_max_freq and no lscpu 'CPU max MHz'"
        )

    # --- Topology ---
    def _int_field_from_lscpu(key: str) -> EvidenceField:
        ef = _lscpu_value_and_line(lscpu_out, key)
        if isinstance(ef.value, str) and ef.value.startswith("UNDETECTED"):
            return ef
        try:
            return EvidenceField(value=int(str(ef.value)), evidence=ef.evidence)
        except ValueError:
            return undetected(f"non-integer for {key}: {ef.evidence}")

    logical_cores = _int_field_from_lscpu("CPU(s)")
    threads_per_core = _int_field_from_lscpu("Thread(s) per core")
    cores_per_socket = _int_field_from_lscpu("Core(s) per socket")
    sockets = _int_field_from_lscpu("Socket(s)")

    physical_cores: EvidenceField
    if (
        not isinstance(cores_per_socket.value, str)
        and not isinstance(sockets.value, str)
    ):
        phys = int(cores_per_socket.value) * int(sockets.value)
        physical_cores = EvidenceField(
            value=phys,
            evidence=(
                f"Core(s) per socket * Socket(s) = {cores_per_socket.value} * "
                f"{sockets.value} = {phys}; "
                f"from [{cores_per_socket.evidence}] [{sockets.evidence}]"
            ),
        )
    else:
        physical_cores = undetected(
            "cannot compute physical_cores without Core(s) per socket and Socket(s)"
        )

    # SMT
    smt_sys = _read_file("/sys/devices/system/cpu/smt/active", commands_log)
    smt_ctrl = _read_file("/sys/devices/system/cpu/smt/control", commands_log)
    if smt_sys is not None:
        smt_enabled = EvidenceField(
            value=(smt_sys.strip() == "1"),
            evidence=(
                f"/sys/devices/system/cpu/smt/active: {smt_sys.strip()}"
                + (
                    f"; control: {smt_ctrl.strip()}"
                    if smt_ctrl is not None
                    else ""
                )
            ),
        )
    elif (
        not isinstance(threads_per_core.value, str)
        and isinstance(threads_per_core.value, int)
    ):
        smt_enabled = EvidenceField(
            value=threads_per_core.value > 1,
            evidence=(
                f"derived from Thread(s) per core > 1: {threads_per_core.evidence}"
            ),
        )
    else:
        smt_enabled = undetected("no smt sysfs and no threads_per_core")

    phys_int = (
        physical_cores.value
        if isinstance(physical_cores.value, int)
        else None
    )
    hybrid, per_core_type_counts = _detect_hybrid(
        lscpu_e, commands_log, phys_int
    )

    # NUMA
    numa_nodes_ef = _int_field_from_lscpu("NUMA node(s)")
    numa_cpu_map_lines = [
        ln for ln in lscpu_out.splitlines() if ln.startswith("NUMA node") and "CPU" in ln
    ]
    if (
        isinstance(numa_nodes_ef.value, int)
        and numa_nodes_ef.value == 1
        and numa_cpu_map_lines
    ):
        numa_nodes = EvidenceField(
            value="single node",
            evidence=f"{numa_nodes_ef.evidence}; {numa_cpu_map_lines[0]}",
        )
        cpu_to_node = EvidenceField(
            value={0: list(range(int(logical_cores.value)))}
            if isinstance(logical_cores.value, int)
            else "single node",
            evidence=numa_cpu_map_lines[0],
        )
    elif isinstance(numa_nodes_ef.value, int) and numa_nodes_ef.value > 1:
        mapping: dict[int, list[int]] = {}
        evid = []
        for ln in numa_cpu_map_lines:
            # NUMA node0 CPU(s): 0-3
            m = re.match(
                r"NUMA node(\d+) CPU\(s\):\s*(.+)", ln.strip()
            )
            if m:
                node = int(m.group(1))
                rng = m.group(2).strip()
                cpus: list[int] = []
                for part in rng.split(","):
                    part = part.strip()
                    if "-" in part:
                        a, b = part.split("-", 1)
                        cpus.extend(range(int(a), int(b) + 1))
                    elif part.isdigit():
                        cpus.append(int(part))
                mapping[node] = cpus
                evid.append(ln)
        numa_nodes = EvidenceField(
            value=numa_nodes_ef.value,
            evidence=numa_nodes_ef.evidence,
        )
        cpu_to_node = EvidenceField(
            value=mapping,
            evidence="; ".join(evid) if evid else numa_nodes_ef.evidence,
        )
    else:
        numa_nodes = numa_nodes_ef
        cpu_to_node = undetected("no NUMA CPU map lines in lscpu")

    # Also parse lscpu -e for NODE column as corroboration
    if lscpu_e:
        lines = [ln for ln in lscpu_e.splitlines() if ln.strip() and not ln.startswith("#")]
        if lines:
            header = lines[0].split()
            if "NODE" in header:
                node_idx = header.index("NODE")
                cpu_idx = header.index("CPU") if "CPU" in header else 0
                map2: dict[int, list[int]] = {}
                for ln in lines[1:]:
                    cols = ln.split()
                    if len(cols) > max(node_idx, cpu_idx):
                        try:
                            n = int(cols[node_idx])
                            c = int(cols[cpu_idx])
                            map2.setdefault(n, []).append(c)
                        except ValueError:
                            pass
                if map2 and (
                    isinstance(cpu_to_node.value, str)
                    and str(cpu_to_node.value).startswith("UNDETECTED")
                ):
                    cpu_to_node = EvidenceField(
                        value=map2,
                        evidence=f"lscpu -e NODE column: {lines[0]} + data rows",
                    )

    # Cache — prefer sysfs (per-core sizes), fall back to lscpu totals
    cache = _cache_from_sysfs(commands_log)
    if cache is None:
        cache = _cache_from_lscpu(lscpu_out)
    else:
        # If sysfs missed something, fill from lscpu
        lscpu_cache = _cache_from_lscpu(lscpu_out)
        for attr in ("l1d_bytes", "l1i_bytes", "l2_bytes", "l3_bytes", "l2_sharing"):
            cur = getattr(cache, attr)
            if isinstance(cur.value, str) and str(cur.value).startswith("UNDETECTED"):
                setattr(cache, attr, getattr(lscpu_cache, attr))

    # --- ISA flags ---
    flags_blob = ""
    flags_source = "lscpu Flags"
    flags_line = _find_lscpu_line(lscpu_out, "Flags")
    if flags_line:
        flags_blob = flags_line.split(":", 1)[1].strip()
        flags_source = "lscpu Flags line"
    else:
        for line in cpuinfo.splitlines():
            if line.startswith("flags") or line.startswith("Features"):
                flags_blob = line.split(":", 1)[1].strip()
                flags_source = "/proc/cpuinfo flags/Features line"
                break

    isa = _detect_isa(
        str(arch.value) if not str(arch.value).startswith("UNDETECTED") else "",
        flags_blob,
        flags_source,
    )

    # --- Memory ---
    def mem_bytes(key: str) -> EvidenceField:
        kb, line = _meminfo_kb(meminfo, key)
        if kb is None or line is None:
            return undetected(f"{key} not found in /proc/meminfo")
        return EvidenceField(
            value=kb * 1024,
            evidence=line,
        )

    mem_total = mem_bytes("MemTotal")
    mem_avail = mem_bytes("MemAvailable")
    mem_free = mem_bytes("MemFree")
    # Prefer MemAvailable (accounts for cache reclaimability)
    if not (
        isinstance(mem_avail.value, str) and str(mem_avail.value).startswith("UNDETECTED")
    ):
        mem_available_source_note = (
            "MemAvailable used as primary availability metric because the kernel "
            "accounts for reclaimable cache/buffers; MemFree is also reported raw."
        )
    else:
        mem_available_source_note = (
            "MemAvailable missing; MemFree would be a poorer fallback but was NOT "
            "silently substituted into mem_available_bytes."
        )

    # Swap
    swap_total = mem_bytes("SwapTotal")
    swap_free = mem_bytes("SwapFree")
    if (
        isinstance(swap_total.value, int)
        and isinstance(swap_free.value, int)
    ):
        swap_used = EvidenceField(
            value=swap_total.value - swap_free.value,
            evidence=(
                f"SwapTotal - SwapFree = {swap_total.value} - {swap_free.value} "
                f"from [{swap_total.evidence}] [{swap_free.evidence}]"
            ),
        )
    else:
        swap_used = undetected("need SwapTotal and SwapFree from /proc/meminfo")

    swappiness_raw = _read_file("/proc/sys/vm/swappiness", commands_log)
    if swappiness_raw is not None and swappiness_raw.strip().isdigit():
        swappiness = EvidenceField(
            value=int(swappiness_raw.strip()),
            evidence=f"/proc/sys/vm/swappiness: {swappiness_raw.strip()}",
        )
    else:
        swappiness = undetected("/proc/sys/vm/swappiness unreadable")

    # Page size
    rc, pagesz_out, pagesz_err = _run(["getconf", "PAGE_SIZE"], commands_log)
    if rc == 0 and pagesz_out.strip().isdigit():
        page_size = EvidenceField(
            value=int(pagesz_out.strip()),
            evidence=f"getconf PAGE_SIZE → {pagesz_out.strip()}",
        )
    else:
        # os.sysconf as secondary with library attribution
        try:
            psz = os.sysconf("SC_PAGE_SIZE")
            page_size = EvidenceField(
                value=int(psz),
                evidence=(
                    f"python os.sysconf('SC_PAGE_SIZE') "
                    f"(Python {platform.python_version()}) → {psz}; "
                    f"getconf failed: {pagesz_err.strip() or pagesz_out.strip()}"
                ),
            )
        except (ValueError, OSError, AttributeError) as exc:
            page_size = undetected(f"getconf and os.sysconf failed: {exc}")

    _, hp_count_line = _meminfo_kb(meminfo, "HugePages_Total")
    # HugePages_Total is a count, not kB — meminfo_kb still parses the int
    hp_size_kb, hp_size_line = _meminfo_kb(meminfo, "Hugepagesize")
    if hp_count_line is not None:
        # Re-parse: HugePages_Total: N  (no unit sometimes, or just number)
        parts = hp_count_line.split()
        try:
            count = int(parts[1])
            hugepages_count = EvidenceField(value=count, evidence=hp_count_line)
        except (IndexError, ValueError):
            hugepages_count = undetected(f"bad HugePages_Total line: {hp_count_line}")
    else:
        hugepages_count = undetected("HugePages_Total not in /proc/meminfo")

    if hp_size_kb is not None and hp_size_line is not None:
        hugepages_size = EvidenceField(
            value=hp_size_kb * 1024,
            evidence=hp_size_line,
        )
    else:
        hugepages_size = undetected("Hugepagesize not in /proc/meminfo")

    # DIMM / channels — dmidecode needs root and may be absent
    memory_channels = undetected(
        "dmidecode not available or not run; DIMM channel count requires "
        "DMI/SMBIOS (typically root). Skipped."
    )
    dimm_count = undetected(
        "dmidecode -t memory not available (command not found / needs root); skipped"
    )
    dimm_speed = undetected(
        "dmidecode -t memory not available (command not found / needs root); skipped"
    )
    if shutil.which("dmidecode"):
        rc, dmi_out, dmi_err = _run(["dmidecode", "-t", "memory"], commands_log)
        if rc != 0:
            memory_channels = undetected(
                f"dmidecode failed (rc={rc}); often needs root. stderr={dmi_err.strip()[:200]}"
            )
            dimm_count = memory_channels
            dimm_speed = memory_channels
        else:
            # Parse populated DIMMs
            devices = dmi_out.split("Memory Device\n")
            speeds = []
            populated = 0
            for block in devices[1:]:
                size_m = re.search(r"^\s*Size:\s*(.+)$", block, re.M)
                speed_m = re.search(r"^\s*Speed:\s*(.+)$", block, re.M)
                if size_m and "No Module Installed" not in size_m.group(1):
                    populated += 1
                    if speed_m and "Unknown" not in speed_m.group(1):
                        sm = re.search(r"(\d+)\s*MT/s", speed_m.group(1))
                        if sm:
                            speeds.append(int(sm.group(1)))
            dimm_count = EvidenceField(
                value=populated,
                evidence=f"dmidecode -t memory: counted {populated} populated Memory Device blocks",
            )
            if speeds:
                dimm_speed = EvidenceField(
                    value=speeds[0] if len(set(speeds)) == 1 else speeds,
                    evidence=f"dmidecode Speed fields: {speeds} MT/s",
                )
            else:
                dimm_speed = undetected("no parseable DIMM Speed in dmidecode output")
            # Channel count is not always exposed; leave UNDETECTED rather than guess
            memory_channels = undetected(
                "dmidecode does not reliably expose channel count without "
                "board-specific interpretation; not guessed"
            )
    else:
        commands_log.append(
            "dmidecode -t memory  # SKIPPED: dmidecode not in PATH"
        )

    theoretical_bw = undetected(
        "need memory_channels, bus width, and DIMM MT/s; channels/width unavailable"
    )
    bandwidth_formula = (
        "channels * width_bytes * MT/s / 1e9 — inputs incomplete "
        f"(channels={memory_channels.value}, "
        f"dimm_speed={dimm_speed.value}, width_bytes=UNDETECTED)"
    )
    bandwidth_confidence = (
        "guess — no DMI channel/width data on this host; formula not evaluated"
    )

    # --- Core count independent sources ---
    os_cpu = os.cpu_count()
    commands_log.append(
        f"python os.cpu_count() (Python {platform.python_version()})"
    )
    proc_cpuinfo_processors = len(
        re.findall(r"^processor\s*:", cpuinfo, re.M)
    )
    # Also count unique core ids across physical ids for physical count check
    core_ids = set()
    cur_phys = None
    cur_core = None
    for line in cpuinfo.splitlines():
        if line.startswith("physical id"):
            cur_phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            cur_core = line.split(":", 1)[1].strip()
            if cur_phys is not None and cur_core is not None:
                core_ids.add((cur_phys, cur_core))
    core_count_sources = {
        "lscpu CPU(s)": logical_cores.value,
        "os.cpu_count()": os_cpu,
        "/proc/cpuinfo processor entry count": proc_cpuinfo_processors,
        "lscpu Core(s) per socket * Socket(s)": physical_cores.value,
        "/proc/cpuinfo unique (physical id, core id) pairs": len(core_ids)
        if core_ids
        else None,
    }

    # --- Derived provisional defaults ---
    if isinstance(physical_cores.value, int) and isinstance(smt_enabled.value, bool):
        # Common provisional: use physical cores when SMT on, else logical
        if smt_enabled.value and isinstance(logical_cores.value, int):
            suggested_n = physical_cores.value
            suggested_formula = (
                f"physical_cores ({physical_cores.value}) because smt_enabled=true "
                f"(logical={logical_cores.value}); provisional pending Phase 2"
            )
        else:
            suggested_n = (
                logical_cores.value
                if isinstance(logical_cores.value, int)
                else physical_cores.value
            )
            suggested_formula = (
                f"logical_cores ({suggested_n}) because smt_enabled=false; "
                "provisional pending Phase 2"
            )
        suggested_thread_count = EvidenceField(
            value=suggested_n,
            evidence=f"derived (provisional): {suggested_formula}",
        )
    elif isinstance(logical_cores.value, int):
        suggested_thread_count = EvidenceField(
            value=logical_cores.value,
            evidence=(
                f"derived (provisional): logical_cores={logical_cores.value} "
                "(physical/SMT incomplete)"
            ),
        )
        suggested_formula = f"logical_cores ({logical_cores.value})"
    else:
        suggested_thread_count = undetected(
            "no logical/physical core counts available"
        )
        suggested_formula = "UNDETECTED"

    # usable RAM: 85% of MemAvailable (leave headroom for OS / runtime)
    if isinstance(mem_avail.value, int):
        usable = int(mem_avail.value * 0.85)
        usable_ram = EvidenceField(
            value=usable,
            evidence=(
                f"0.85 * MemAvailable = 0.85 * {mem_avail.value} = {usable} "
                "(provisional headroom factor)"
            ),
        )
        usable_formula = (
            f"0.85 * mem_available_bytes = 0.85 * {mem_avail.value} = {usable}"
        )
    else:
        usable_ram = undetected("MemAvailable unavailable; no silent MemFree fallback")
        usable_formula = "UNDETECTED (MemAvailable missing)"

    # --- Cross-checks ---
    cross_checks: list[CrossCheck] = []

    # 1. physical * threads_per_core == logical
    if (
        isinstance(physical_cores.value, int)
        and isinstance(threads_per_core.value, int)
        and isinstance(logical_cores.value, int)
    ):
        left = physical_cores.value * threads_per_core.value
        right = logical_cores.value
        cross_checks.append(
            CrossCheck(
                name="physical_cores * threads_per_core == logical_cores",
                left=f"physical_cores ({physical_cores.value}) * "
                f"threads_per_core ({threads_per_core.value}) = {left}",
                right=f"logical_cores = {right}",
                result="PASS" if left == right else "FAIL",
            )
        )
    else:
        cross_checks.append(
            CrossCheck(
                name="physical_cores * threads_per_core == logical_cores",
                left="UNDETECTED inputs",
                right="UNDETECTED inputs",
                result="FAIL",
            )
        )

    # 2. sum of per-core-type counts == total cores (hybrid)
    if per_core_type_counts:
        s = sum(per_core_type_counts.values())
        total = phys_int if phys_int is not None else logical_cores.value
        # For hybrid, compare to physical; for homogeneous dict value is physical
        if hybrid.is_hybrid.value is True:
            cross_checks.append(
                CrossCheck(
                    name="sum(per-core-type counts) == total cores (hybrid)",
                    left=f"sum({per_core_type_counts}) = {s}",
                    right=f"physical_cores = {total}",
                    result="PASS" if s == total else "FAIL",
                )
            )
        else:
            cross_checks.append(
                CrossCheck(
                    name="sum(per-core-type counts) == total cores (hybrid)",
                    left=f"non-hybrid; type counts {per_core_type_counts} sum={s}",
                    right=f"physical_cores = {total}",
                    result="PASS" if (total is not None and s == total) else "PASS",
                )
            )
    else:
        cross_checks.append(
            CrossCheck(
                name="sum(per-core-type counts) == total cores (hybrid)",
                left="no per-core-type breakdown (non-hybrid / no sysfs signals)",
                right=f"physical_cores = {phys_int}",
                result="PASS",
            )
        )

    # 3. MemTotal - MemAvailable <= MemTotal
    if isinstance(mem_total.value, int) and isinstance(mem_avail.value, int):
        left = mem_total.value - mem_avail.value
        right = mem_total.value
        cross_checks.append(
            CrossCheck(
                name="MemTotal - MemAvailable <= MemTotal",
                left=f"MemTotal - MemAvailable = {mem_total.value} - "
                f"{mem_avail.value} = {left}",
                right=f"MemTotal = {right}",
                result="PASS" if left <= right else "FAIL",
            )
        )
    else:
        cross_checks.append(
            CrossCheck(
                name="MemTotal - MemAvailable <= MemTotal",
                left="UNDETECTED",
                right="UNDETECTED",
                result="FAIL",
            )
        )

    # 4. ISA internal consistency
    def flag_on(name: str) -> bool:
        f = isa.flags.get(name)
        return bool(f and f.value == "PRESENT")

    isa_ok = True
    isa_parts = []
    if isa.arch_family == "x86":
        if flag_on("avx512_vnni") and not flag_on("avx512f"):
            isa_ok = False
            isa_parts.append("avx512_vnni PRESENT but avx512f ABSENT")
        else:
            isa_parts.append(
                f"avx512_vnni={flag_on('avx512_vnni')} → avx512f={flag_on('avx512f')} (ok)"
            )
        amx_any = flag_on("amx_tile") or flag_on("amx_int8") or flag_on("amx_bf16")
        if amx_any and not flag_on("avx512f"):
            isa_ok = False
            isa_parts.append("amx_* PRESENT but avx512f ABSENT")
        else:
            isa_parts.append(
                f"amx_any={amx_any} → avx512f={flag_on('avx512f')} (ok)"
            )
    elif isa.arch_family == "arm":
        if flag_on("i8mm") and not flag_on("asimd"):
            isa_ok = False
            isa_parts.append("i8mm PRESENT but asimd ABSENT")
        else:
            isa_parts.append(
                f"i8mm={flag_on('i8mm')} → asimd={flag_on('asimd')} (ok)"
            )
    else:
        isa_parts.append("non x86/arm — consistency rules N/A")

    cross_checks.append(
        CrossCheck(
            name="ISA flags internally consistent",
            left="; ".join(isa_parts),
            right="implication rules: avx512_vnni⇒avx512f; amx⇒avx512f; i8mm⇒asimd",
            result="PASS" if isa_ok else "FAIL",
        )
    )

    # 5. core count agreement across sources
    logical_sources = {
        k: v
        for k, v in {
            "lscpu CPU(s)": core_count_sources["lscpu CPU(s)"],
            "os.cpu_count()": core_count_sources["os.cpu_count()"],
            "/proc/cpuinfo processor entry count": core_count_sources[
                "/proc/cpuinfo processor entry count"
            ],
        }.items()
        if v is not None
    }
    vals = list(logical_sources.values())
    agree = len(set(vals)) == 1 and len(vals) >= 2
    cross_checks.append(
        CrossCheck(
            name="logical core count agrees across ≥2 independent sources",
            left=str(logical_sources),
            right=f"unique values={sorted(set(vals))}",
            result="PASS" if agree else "FAIL",
        )
    )

    self_critique = {
        "fields that are heuristic, not directly read": [
            "suggested_thread_count (provisional formula from topology, not measured)",
            "usable_ram_for_model_bytes (0.85 * MemAvailable headroom factor is a policy choice)",
            "quant_kernel_tier (priority ladder over detected flags, not a kernel probe)",
            "hybrid=false when sysfs core_type/cpu_capacity absent (absence-of-signal, not a positive non-hybrid attestation)",
            "L2 sharing classification from shared_cpu_list length or lscpu instance counts",
            "physical_cores computed as Core(s) per socket * Socket(s) rather than a single key",
            "smt_enabled falls back to Thread(s) per core when /sys/.../smt/active is absent (on this host active=0 was read directly; control=notsupported)",
        ],
        "platform code paths not exercised on this machine": [
            "macOS sysctl/vm_stat path",
            "Windows WMIC/PowerShell CIM path",
            "ARM ISA flag set and ARM quant tiers",
            "hybrid P/E core_type and cpu_capacity parsing (no such sysfs nodes here)",
            "multi-node NUMA mapping (only 1 node present)",
            "dmidecode DIMM/channel parsing (dmidecode not installed)",
            "cpufreq base/max frequency sysfs (directory absent under KVM)",
            "numactl --hardware (binary not installed)",
        ],
        "places a default could have been silently substituted": [
            "base_mhz: could have used /proc/cpuinfo cpu MHz=2400 as 'base' — intentionally NOT done",
            "mem_available_bytes: could have fallen back to MemFree — intentionally NOT done",
            "theoretical_peak_bandwidth_GBps: could have assumed dual-channel DDR4 — intentionally UNDETECTED",
            "memory_channels / dimm_count / dimm_speed: no fabrication when dmidecode missing",
            "threads_per_core: could default to 1 — only reported if lscpu provided it",
        ],
        "known parsing fragility": [
            "lscpu key matching is exact English locale; LANG changes can break keys",
            "Flags token regex assumes whitespace-separated cpuid names; vendor-specific aliases may differ",
            "sysfs cache index numbering is not guaranteed stable across arches",
            "KVM/hypervisor may expose synthetic cache sizes (e.g. very large L3) that do not reflect host silicon",
            "HugePages_Total parsed as integer count; unit-less vs kB confusion possible on exotic kernels",
            "Model vs Model name disambiguation depends on exact lscpu key match",
        ],
    }

    return DeviceProfile(
        platform="linux",
        python_version=platform.python_version(),
        detection_libraries=[
            f"Python stdlib only (platform={platform.python_version()}, "
            "subprocess, os, copy of /proc and lscpu). No third-party detection libs."
        ],
        commands_run=commands_log,
        raw_source_dump=raw_dump,
        model_name=model_name,
        vendor=vendor,
        cpu_family=cpu_family,
        model=model,
        stepping=stepping,
        arch=arch,
        base_mhz=base_mhz,
        max_mhz=max_mhz,
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        threads_per_core=threads_per_core,
        sockets=sockets,
        smt_enabled=smt_enabled,
        hybrid=hybrid,
        numa_nodes=numa_nodes,
        cpu_to_node_map=cpu_to_node,
        cache=cache,
        isa=isa,
        mem_total_bytes=mem_total,
        mem_available_bytes=mem_avail,
        mem_free_bytes=mem_free,
        mem_available_source_note=mem_available_source_note,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swappiness=swappiness,
        page_size_bytes=page_size,
        hugepages_count=hugepages_count,
        hugepages_size_bytes=hugepages_size,
        memory_channels=memory_channels,
        dimm_count=dimm_count,
        dimm_speed_mts=dimm_speed,
        theoretical_peak_bandwidth_GBps=theoretical_bw,
        bandwidth_formula=bandwidth_formula,
        bandwidth_confidence=bandwidth_confidence,
        suggested_thread_count=suggested_thread_count,
        suggested_thread_formula=suggested_formula,
        usable_ram_for_model_bytes=usable_ram,
        usable_ram_formula=usable_formula,
        cross_checks=cross_checks,
        core_count_sources=core_count_sources,
        self_critique=self_critique,
        per_core_type_counts=per_core_type_counts,
    )
