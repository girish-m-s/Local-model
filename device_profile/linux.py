"""Linux CPU/memory detection — Phase 1.5 (capture-once, sandbox-aware)."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import time
from typing import Optional

from .models import (
    AmxRuntimeProbe,
    CacheInfo,
    CaptureBlob,
    CrossCheck,
    DeviceProfile,
    EvidenceField,
    ExecutionEnvironment,
    HybridInfo,
    IsaFlags,
    ParserNegativeControl,
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

# HOST-LEVEL ONLY. (vendor_id, family, model) → uarch name.
# Model is compared as int (CPUID display model).
UARCH_TABLE: dict[tuple[str, int, int], str] = {
    ("GenuineIntel", 6, 0xCF): "Emerald Rapids (Intel Xeon; CPUID model 0xCF)",
    ("GenuineIntel", 6, 0x8F): "Sapphire Rapids",
    ("GenuineIntel", 6, 0xAD): "Granite Rapids",
    ("GenuineIntel", 6, 0xAF): "Sierra Forest",
    ("GenuineIntel", 6, 0xA7): "Rocket Lake",
    ("GenuineIntel", 6, 0x97): "Alder Lake (desktop S)",
    ("GenuineIntel", 6, 0x9A): "Alder Lake (mobile H/P/U)",
    ("GenuineIntel", 6, 0xB7): "Raptor Lake",
    ("GenuineIntel", 6, 0xBA): "Raptor Lake (refresh/mobile)",
    ("GenuineIntel", 6, 0xC6): "Arrow Lake / related",
    ("GenuineIntel", 6, 0xAA): "Meteor Lake",
    ("GenuineIntel", 6, 0x3A): "Ivy Bridge",
    ("GenuineIntel", 6, 0x3C): "Haswell",
    ("GenuineIntel", 6, 0x3D): "Broadwell",
    ("GenuineIntel", 6, 0x4E): "Skylake (mobile)",
    ("GenuineIntel", 6, 0x5E): "Skylake (desktop)",
    ("GenuineIntel", 6, 0x55): "Skylake-SP / Cascade Lake / Cooper Lake (model 0x55)",
    ("GenuineIntel", 6, 0x6A): "Ice Lake-SP",
    ("GenuineIntel", 6, 0x6C): "Ice Lake-SP (alt)",
    ("GenuineIntel", 6, 0x8A): "Lakefield",
    ("AuthenticAMD", 25, 1): "Zen 3 (family 19h model 1)",
    ("AuthenticAMD", 25, 8): "Zen 3 (Chagall/etc)",
    ("AuthenticAMD", 25, 33): "Zen 3 (Vermeer)",
    ("AuthenticAMD", 25, 80): "Zen 3 (Cezanne)",
    ("AuthenticAMD", 26, 1): "Zen 4 (family 1Ah)",
}


class CaptureStore:
    """Capture each raw source exactly once; parse only stored text."""

    def __init__(self) -> None:
        self.blobs: dict[str, CaptureBlob] = {}
        self.order: list[str] = []

    def _add(
        self,
        key: str,
        command: str,
        content: str,
        skipped_reason: Optional[str] = None,
    ) -> CaptureBlob:
        if key in self.blobs:
            raise RuntimeError(
                f"capture-once violation: '{key}' already captured"
            )
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        blob = CaptureBlob(
            key=key,
            command=command,
            content=content,
            sha256=digest,
            skipped_reason=skipped_reason,
        )
        self.blobs[key] = blob
        self.order.append(key)
        return blob

    def run(self, key: str, cmd: list[str]) -> CaptureBlob:
        command = " ".join(cmd)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if proc.returncode != 0 and not proc.stdout:
                content = (
                    f"[exit {proc.returncode}] {proc.stderr}"
                )
            else:
                content = proc.stdout
            return self._add(key, command, content)
        except FileNotFoundError:
            return self._add(
                key,
                command,
                "",
                skipped_reason=f"command not found: {cmd[0]}",
            )
        except OSError as exc:
            return self._add(
                key, command, "", skipped_reason=str(exc)
            )

    def read(self, key: str, path: str) -> CaptureBlob:
        command = f"read {path}"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            return self._add(key, command, content)
        except OSError as exc:
            return self._add(
                key, command, "", skipped_reason=f"unreadable: {exc}"
            )

    def exists(self, key: str, path: str) -> CaptureBlob:
        command = f"test -e {path}"
        exists = os.path.exists(path)
        content = "EXISTS" if exists else "ABSENT"
        return self._add(key, command, content)

    def skip(self, key: str, command: str, reason: str) -> CaptureBlob:
        return self._add(key, command, "", skipped_reason=reason)

    def get(self, key: str) -> CaptureBlob:
        return self.blobs[key]

    def text(self, key: str) -> str:
        return self.blobs[key].content

    def sha(self, key: str) -> str:
        return self.blobs[key].sha256

    def commands_log(self) -> list[str]:
        out = []
        for key in self.order:
            b = self.blobs[key]
            if b.skipped_reason:
                out.append(f"{b.command}  # SKIPPED: {b.skipped_reason} sha256={b.sha256}")
            else:
                out.append(f"{b.command}  # sha256={b.sha256}")
        return out


def _find_lscpu_line(text: str, key: str) -> Optional[str]:
    for line in text.splitlines():
        if ":" not in line:
            continue
        if line.split(":", 1)[0].strip() == key:
            return line
    return None


def _lscpu_value_and_line(text: str, key: str) -> EvidenceField:
    line = _find_lscpu_line(text, key)
    if line is None:
        return undetected(f"key '{key}' not present in lscpu output")
    val = line.split(":", 1)[1].strip()
    return EvidenceField(value=val, evidence=line)


def _parse_size_to_bytes(size_str: str) -> Optional[int]:
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
            if len(parts) >= 2:
                try:
                    return int(parts[1]), line
                except ValueError:
                    return None, line
    return None, None


def _flag_tokens(flags_blob: str) -> set[str]:
    """Whitespace-tokenized flag set (NOT substring matching)."""
    return set(flags_blob.split())


def _flag_present_tokenized(
    flags_blob: str, flag: str, source_label: str
) -> EvidenceField:
    tokens = _flag_tokens(flags_blob)
    if flag in tokens:
        return EvidenceField(value="PRESENT", evidence=flag)
    return EvidenceField(
        value="ABSENT",
        evidence=f"not found in {source_label} (whitespace-tokenized)",
    )


def _tier_from_flags(family: str, present: set[str]) -> tuple[QuantKernelTier, str]:
    if family == "x86":
        if "amx_tile" in present or "amx_int8" in present:
            return (
                QuantKernelTier.AMX,
                "amx_tile/amx_int8 PRESENT → AMX",
            )
        if "avx512_vnni" in present:
            return (
                QuantKernelTier.AVX512_VNNI,
                "avx512_vnni PRESENT (AMX absent) → AVX512_VNNI",
            )
        if "avx512f" in present:
            return (
                QuantKernelTier.AVX512,
                "avx512f PRESENT (VNNI/AMX absent) → AVX512",
            )
        if "avx2" in present:
            return QuantKernelTier.AVX2, "avx2 PRESENT → AVX2"
        return QuantKernelTier.SSE_ONLY, "sse4_2/baseline → SSE_ONLY"
    if family == "arm":
        if "i8mm" in present:
            return QuantKernelTier.ARM_I8MM, "i8mm PRESENT → ARM_I8MM"
        if "asimddp" in present:
            return QuantKernelTier.ARM_DOTPROD, "asimddp PRESENT → ARM_DOTPROD"
        return QuantKernelTier.ARM_BASELINE, "no i8mm/asimddp → ARM_BASELINE"
    return QuantKernelTier.SSE_ONLY, f"unsupported family {family}"


def _parse_cpuset_list(spec: str) -> list[int]:
    cpus: list[int] = []
    spec = spec.strip()
    if not spec or spec == "":
        return cpus
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            cpus.append(int(part))
    return cpus


def _decode_uarch(
    vendor: str, family: Any, model: Any, stepping: Any
) -> EvidenceField:
    try:
        fam_i = int(str(family))
        mod_i = int(str(model))
    except (TypeError, ValueError):
        return undetected(
            f"non-integer family/model for uarch decode: {family}/{model}"
        )
    key = (vendor, fam_i, mod_i)
    if key in UARCH_TABLE:
        name = UARCH_TABLE[key]
        return EvidenceField(
            value=name,
            evidence=(
                f"HOST-LEVEL INFORMATION ONLY: lookup({vendor!r}, "
                f"family={fam_i}, model={mod_i}/0x{mod_i:X}, "
                f"stepping={stepping}) → {name}. "
                "Any host memory-channel or L3 figure this uarch implies "
                "does NOT describe this guest's available share."
            ),
        )
    return EvidenceField(
        value=f"UNKNOWN_UARCH (vendor={vendor}, family={fam_i}, model=0x{mod_i:X})",
        evidence=(
            f"HOST-LEVEL INFORMATION ONLY: no table entry for "
            f"({vendor!r}, {fam_i}, 0x{mod_i:X}); stepping={stepping}. "
            "Do not infer guest memory channels/L3 from this."
        ),
    )


def _probe_amx_runtime(store: CaptureStore) -> AmxRuntimeProbe:
    """prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA=18) via ctypes."""
    import ctypes
    import ctypes.util

    status = store.text("proc_self_status")
    xcomp_lines = [
        ln
        for ln in status.splitlines()
        if "XCOMP" in ln.upper() or "xcomp" in ln
    ]
    if xcomp_lines:
        status_xcomp = EvidenceField(
            value="; ".join(xcomp_lines),
            evidence=(
                f"/proc/self/status XCOMP lines "
                f"(sha256={store.sha('proc_self_status')}): "
                + "; ".join(xcomp_lines)
            ),
        )
    else:
        status_xcomp = EvidenceField(
            value="UNDETECTED (reason: no XCOMP* keys in /proc/self/status)",
            evidence=(
                f"/proc/self/status has no XCOMP keys "
                f"(sha256={store.sha('proc_self_status')})"
            ),
        )

    libname = ctypes.util.find_library("c")
    try:
        libc = ctypes.CDLL(libname, use_errno=True)
    except (OSError, TypeError) as exc:
        return AmxRuntimeProbe(
            prctl_rc=undetected(f"ctypes CDLL failed: {exc}"),
            prctl_errno=undetected("no prctl"),
            xcomp_supp=undetected("no prctl"),
            xcomp_perm=undetected("no prctl"),
            status_xcomp=status_xcomp,
            note="ctypes libc load failed",
        )

    ARCH_GET_XCOMP_SUPP = 0x1021
    ARCH_GET_XCOMP_PERM = 0x1022
    ARCH_REQ_XCOMP_PERM = 0x1023
    XFEATURE_XTILEDATA = 18

    features_supp = ctypes.c_uint64(0)
    ctypes.set_errno(0)
    rc_supp = libc.prctl(
        ARCH_GET_XCOMP_SUPP, ctypes.byref(features_supp), 0, 0, 0
    )
    errno_supp = ctypes.get_errno()

    features_perm = ctypes.c_uint64(0)
    ctypes.set_errno(0)
    rc_perm = libc.prctl(
        ARCH_GET_XCOMP_PERM, ctypes.byref(features_perm), 0, 0, 0
    )
    errno_perm = ctypes.get_errno()

    ctypes.set_errno(0)
    rc_req = libc.prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA, 0, 0, 0)
    errno_req = ctypes.get_errno()

    lib_ev = (
        f"ctypes.CDLL({libname!r}) / libc.prctl; "
        f"Python {platform.python_version()}"
    )
    return AmxRuntimeProbe(
        prctl_rc=EvidenceField(
            value=rc_req,
            evidence=(
                f"{lib_ev}: prctl(ARCH_REQ_XCOMP_PERM=0x1023, "
                f"XFEATURE_XTILEDATA=18) → return code {rc_req}"
            ),
        ),
        prctl_errno=EvidenceField(
            value=errno_req,
            evidence=f"ctypes.get_errno() after ARCH_REQ_XCOMP_PERM → {errno_req}",
        ),
        xcomp_supp=EvidenceField(
            value={
                "rc": rc_supp,
                "errno": errno_supp,
                "features": hex(features_supp.value),
                "bit18_xtiledata": bool(features_supp.value & (1 << 18)),
            },
            evidence=(
                f"{lib_ev}: prctl(ARCH_GET_XCOMP_SUPP=0x1021) → rc={rc_supp} "
                f"errno={errno_supp} features={hex(features_supp.value)}"
            ),
        ),
        xcomp_perm=EvidenceField(
            value={
                "rc": rc_perm,
                "errno": errno_perm,
                "features": hex(features_perm.value),
                "bit18_xtiledata": bool(features_perm.value & (1 << 18)),
            },
            evidence=(
                f"{lib_ev}: prctl(ARCH_GET_XCOMP_PERM=0x1022) → rc={rc_perm} "
                f"errno={errno_perm} features={hex(features_perm.value)}"
            ),
        ),
        status_xcomp=status_xcomp,
        note=(
            "llama.cpp AMX support is build-flag gated and covers a limited "
            "quant set; usable_tier will be re-confirmed against the actual "
            "binary in Phase 5."
        ),
    )


def _build_section0(store: CaptureStore) -> str:
    parts: list[str] = []
    parts.append(
        "CAPTURE-ONCE DISCIPLINE: each source below was read exactly once; "
        "all '<-' evidence lines parse the same stored blob (sha256 shown)."
    )

    def dump_cmd(key: str, header: str, body_transform=None) -> None:
        b = store.get(key)
        parts.append(f"\n$ {b.command}")
        parts.append(f"sha256: {b.sha256}")
        if b.skipped_reason:
            parts.append(f"[{b.skipped_reason}]")
            return
        body = b.content
        if body_transform:
            body = body_transform(body, b)
        parts.append(body.rstrip("\n") if body else "[empty]")

    dump_cmd("lscpu", "lscpu")

    def head_n(n: int, label: str):
        def _xform(content: str, b: CaptureBlob) -> str:
            lines = content.splitlines(keepends=True)
            shown = "".join(lines[:n]).rstrip("\n")
            if len(lines) > n:
                shown += (
                    f"\n\n[TRUNCATED: showing first {n} of {len(lines)} "
                    f"lines of {label}; full blob sha256={b.sha256}]"
                )
            return shown

        return _xform

    dump_cmd(
        "proc_cpuinfo",
        "head -30 /proc/cpuinfo",
        head_n(30, "/proc/cpuinfo"),
    )
    # Override displayed command for clarity while keeping capture key
    # (command string in blob is "read /proc/cpuinfo" — annotate)
    parts[-3] = "$ head -30 /proc/cpuinfo  # derived from single capture: read /proc/cpuinfo"

    dump_cmd(
        "proc_meminfo",
        "head -8 /proc/meminfo",
        head_n(8, "/proc/meminfo"),
    )
    parts[-3] = "$ cat /proc/meminfo | head -8  # derived from single capture: read /proc/meminfo"

    dump_cmd("lscpu_e", "lscpu -e")
    dump_cmd("numactl", "numactl --hardware")

    # Also dump env-related captures for independent verification
    parts.append("\n--- additional captures for SECTION 2b (same store) ---")
    for key in (
        "systemd_detect_virt",
        "sys_hypervisor_type",
        "dockerenv",
        "containerenv",
        "proc1_cgroup",
        "proc_self_cgroup",
        "cgroup_memory_max",
        "cgroup_memory_high",
        "cgroup_memory_current",
        "cgroup_cpu_max",
        "cgroup_cpuset_cpus_effective",
        "proc_stat_t0",
        "proc_stat_t1",
        "proc_self_status",
        "uname_m",
    ):
        if key in store.blobs:
            b = store.get(key)
            parts.append(f"\n$ {b.command}")
            parts.append(f"sha256: {b.sha256}")
            if b.skipped_reason:
                parts.append(f"[{b.skipped_reason}]")
            else:
                # Truncate very long blobs in section 0 addendum only if needed
                text = b.content.rstrip("\n")
                if len(text) > 4000:
                    parts.append(text[:4000])
                    parts.append(
                        f"\n[TRUNCATED display to 4000 chars; "
                        f"full blob sha256={b.sha256}]"
                    )
                else:
                    parts.append(text if text else "[empty]")

    return "\n".join(parts).lstrip("\n")


def _capture_all() -> CaptureStore:
    store = CaptureStore()

    # Core topology/identity (once each)
    store.run("lscpu", ["lscpu"])
    store.read("proc_cpuinfo", "/proc/cpuinfo")
    store.read("proc_meminfo", "/proc/meminfo")
    store.run("lscpu_e", ["lscpu", "-e"])
    if shutil.which("numactl"):
        store.run("numactl", ["numactl", "--hardware"])
    else:
        store.skip("numactl", "numactl --hardware", "numactl not in PATH")

    store.run("uname_m", ["uname", "-m"])

    # Freq / SMT
    store.read(
        "cpufreq_base",
        "/sys/devices/system/cpu/cpu0/cpufreq/base_frequency",
    )
    store.read(
        "cpufreq_max",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
    )
    store.read("smt_active", "/sys/devices/system/cpu/smt/active")
    store.read("smt_control", "/sys/devices/system/cpu/smt/control")

    # Hybrid probes (per cpu)
    cpu_root = "/sys/devices/system/cpu"
    try:
        cpu_dirs = sorted(
            d for d in os.listdir(cpu_root) if re.fullmatch(r"cpu\d+", d)
        )
    except OSError:
        cpu_dirs = []
    store._add(
        "cpu_dirs_list",
        f"list {cpu_root}/cpuN",
        "\n".join(cpu_dirs),
    )
    for d in cpu_dirs:
        store.read(
            f"{d}_core_type",
            f"{cpu_root}/{d}/topology/core_type",
        )
        store.read(
            f"{d}_cpu_capacity",
            f"{cpu_root}/{d}/cpu_capacity",
        )

    # Cache sysfs for cpu0
    for idx in ("index0", "index1", "index2", "index3"):
        base = f"/sys/devices/system/cpu/cpu0/cache/{idx}"
        for leaf in ("level", "type", "size", "shared_cpu_list"):
            store.read(f"cache_{idx}_{leaf}", f"{base}/{leaf}")

    # Memory extras
    store.read("swappiness", "/proc/sys/vm/swappiness")
    store.run("getconf_page_size", ["getconf", "PAGE_SIZE"])
    if shutil.which("dmidecode"):
        store.run("dmidecode_memory", ["dmidecode", "-t", "memory"])
    else:
        store.skip(
            "dmidecode_memory",
            "dmidecode -t memory",
            "dmidecode not in PATH",
        )

    # Virtualization / container
    if os.path.isdir("/sys/hypervisor"):
        store.read("sys_hypervisor_type", "/sys/hypervisor/type")
        store.read("sys_hypervisor_uuid", "/sys/hypervisor/uuid")
    else:
        store.skip(
            "sys_hypervisor_type",
            "read /sys/hypervisor/type",
            "/sys/hypervisor absent",
        )
    if shutil.which("systemd-detect-virt"):
        store.run("systemd_detect_virt", ["systemd-detect-virt"])
    else:
        store.skip(
            "systemd_detect_virt",
            "systemd-detect-virt",
            "systemd-detect-virt not in PATH",
        )
    store.exists("dockerenv", "/.dockerenv")
    store.exists("containerenv", "/run/.containerenv")
    store.read("proc1_cgroup", "/proc/1/cgroup")
    store.read("proc_self_cgroup", "/proc/self/cgroup")
    store.read("proc_self_status", "/proc/self/status")

    # cgroup v2 (or note v1)
    # Resolve cgroup path from /proc/self/cgroup
    self_cg = store.text("proc_self_cgroup").strip()
    cg_rel = "/"
    if self_cg.startswith("0::"):
        cg_rel = self_cg[3:] or "/"
    cg_base = os.path.normpath("/sys/fs/cgroup" + (cg_rel if cg_rel != "/" else ""))
    if cg_rel == "/":
        cg_base = "/sys/fs/cgroup"

    store._add("cgroup_base_resolved", "resolve cgroup base", cg_base)

    v2_memory_max = os.path.join(cg_base, "memory.max")
    if os.path.exists(v2_memory_max):
        store._add("cgroup_version_probe", "test cgroup v2 memory.max", "v2")
        for name, fname in (
            ("cgroup_memory_max", "memory.max"),
            ("cgroup_memory_high", "memory.high"),
            ("cgroup_memory_current", "memory.current"),
            ("cgroup_cpu_max", "cpu.max"),
            ("cgroup_cpuset_cpus_effective", "cpuset.cpus.effective"),
        ):
            store.read(name, os.path.join(cg_base, fname))
    else:
        store._add(
            "cgroup_version_probe",
            "test cgroup v2 memory.max",
            "v2 absent; trying v1",
        )
        # cgroup v1 fallbacks
        for name, path in (
            ("cgroup_memory_max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            ("cgroup_memory_high", "/sys/fs/cgroup/memory/memory.soft_limit_in_bytes"),
            ("cgroup_memory_current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ("cgroup_cpu_max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            ("cgroup_cpu_period_v1", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            ("cgroup_cpuset_cpus_effective", "/sys/fs/cgroup/cpuset/cpuset.cpus"),
        ):
            store.read(name, path)

    # Steal time: two samples 5s apart (intentionally two captures)
    store.read("proc_stat_t0", "/proc/stat")
    time.sleep(5.0)
    store.read("proc_stat_t1", "/proc/stat")

    return store


def detect_linux() -> DeviceProfile:
    store = _capture_all()
    raw_dump = _build_section0(store)

    lscpu_out = store.text("lscpu")
    lscpu_e = store.text("lscpu_e")
    cpuinfo = store.text("proc_cpuinfo")
    meminfo = store.text("proc_meminfo")
    lscpu_sha = store.sha("lscpu")
    cpuinfo_sha = store.sha("proc_cpuinfo")
    meminfo_sha = store.sha("proc_meminfo")

    def evid_line(line: str, blob_key: str) -> str:
        return f"{line}  [sha256={store.sha(blob_key)}]"

    # --- Identity ---
    model_name = _lscpu_value_and_line(lscpu_out, "Model name")
    if model_name.is_detected():
        model_name = EvidenceField(
            value=model_name.value,
            evidence=evid_line(model_name.evidence, "lscpu"),
        )
    vendor = _lscpu_value_and_line(lscpu_out, "Vendor ID")
    if vendor.is_detected():
        vendor = EvidenceField(
            value=vendor.value, evidence=evid_line(vendor.evidence, "lscpu")
        )
    cpu_family = _lscpu_value_and_line(lscpu_out, "CPU family")
    if cpu_family.is_detected():
        cpu_family = EvidenceField(
            value=cpu_family.value,
            evidence=evid_line(cpu_family.evidence, "lscpu"),
        )
    model = _lscpu_value_and_line(lscpu_out, "Model")
    if model.is_detected():
        model = EvidenceField(
            value=model.value, evidence=evid_line(model.evidence, "lscpu")
        )
    stepping = _lscpu_value_and_line(lscpu_out, "Stepping")
    if stepping.is_detected():
        stepping = EvidenceField(
            value=stepping.value,
            evidence=evid_line(stepping.evidence, "lscpu"),
        )

    uname_blob = store.get("uname_m")
    if not uname_blob.skipped_reason and uname_blob.content.strip():
        arch = EvidenceField(
            value=uname_blob.content.strip(),
            evidence=(
                f"uname -m → {uname_blob.content.strip()} "
                f"[sha256={uname_blob.sha256}]"
            ),
        )
    else:
        arch = undetected(
            f"uname -m failed: {uname_blob.skipped_reason}"
        )

    # Frequencies — never invent from cpuinfo current MHz
    base_blob = store.get("cpufreq_base")
    max_blob = store.get("cpufreq_max")
    if not base_blob.skipped_reason and base_blob.content.strip().isdigit():
        mhz = int(base_blob.content.strip()) / 1000.0
        base_mhz = EvidenceField(
            value=mhz,
            evidence=(
                f"/sys/.../base_frequency: {base_blob.content.strip()} kHz "
                f"→ {mhz} MHz [sha256={base_blob.sha256}]"
            ),
        )
    else:
        base_mhz_lscpu = _lscpu_value_and_line(lscpu_out, "CPU base MHz")
        if base_mhz_lscpu.is_detected():
            try:
                base_mhz = EvidenceField(
                    value=float(str(base_mhz_lscpu.value)),
                    evidence=evid_line(base_mhz_lscpu.evidence, "lscpu"),
                )
            except ValueError:
                base_mhz = undetected(
                    f"unparseable CPU base MHz: {base_mhz_lscpu.evidence}"
                )
        else:
            base_mhz = undetected(
                "no cpufreq base_frequency, no lscpu 'CPU base MHz'; "
                "/proc/cpuinfo 'cpu MHz' is current freq and was not used as base"
            )

    if not max_blob.skipped_reason and max_blob.content.strip().isdigit():
        mhz = int(max_blob.content.strip()) / 1000.0
        max_mhz = EvidenceField(
            value=mhz,
            evidence=(
                f"/sys/.../cpuinfo_max_freq: {max_blob.content.strip()} kHz "
                f"→ {mhz} MHz [sha256={max_blob.sha256}]"
            ),
        )
    else:
        max_mhz_lscpu = _lscpu_value_and_line(lscpu_out, "CPU max MHz")
        if max_mhz_lscpu.is_detected():
            try:
                max_mhz = EvidenceField(
                    value=float(str(max_mhz_lscpu.value)),
                    evidence=evid_line(max_mhz_lscpu.evidence, "lscpu"),
                )
            except ValueError:
                max_mhz = undetected(
                    f"unparseable CPU max MHz: {max_mhz_lscpu.evidence}"
                )
        else:
            max_mhz = undetected(
                "no cpufreq cpuinfo_max_freq and no lscpu 'CPU max MHz'"
            )

    uarch = _decode_uarch(
        str(vendor.value) if vendor.is_detected() else "",
        cpu_family.value if cpu_family.is_detected() else None,
        model.value if model.is_detected() else None,
        stepping.value if stepping.is_detected() else None,
    )

    # --- Topology ---
    def int_lscpu(key: str) -> EvidenceField:
        ef = _lscpu_value_and_line(lscpu_out, key)
        if not ef.is_detected():
            return ef
        try:
            return EvidenceField(
                value=int(str(ef.value)),
                evidence=evid_line(ef.evidence, "lscpu"),
            )
        except ValueError:
            return undetected(f"non-integer for {key}: {ef.evidence}")

    logical_cores = int_lscpu("CPU(s)")
    threads_per_core = int_lscpu("Thread(s) per core")
    cores_per_socket = int_lscpu("Core(s) per socket")
    sockets = int_lscpu("Socket(s)")

    if cores_per_socket.is_detected() and sockets.is_detected():
        phys = int(cores_per_socket.value) * int(sockets.value)
        physical_cores = EvidenceField(
            value=phys,
            evidence=(
                f"Core(s) per socket * Socket(s) = {cores_per_socket.value} * "
                f"{sockets.value} = {phys}; from "
                f"[{cores_per_socket.evidence}] [{sockets.evidence}]"
            ),
        )
    else:
        physical_cores = undetected(
            "cannot compute physical_cores without Core(s) per socket and Socket(s)"
        )

    smt_active = store.get("smt_active")
    smt_ctrl = store.get("smt_control")
    if not smt_active.skipped_reason:
        smt_enabled = EvidenceField(
            value=(smt_active.content.strip() == "1"),
            evidence=(
                f"/sys/devices/system/cpu/smt/active: "
                f"{smt_active.content.strip()} [sha256={smt_active.sha256}]"
                + (
                    f"; control: {smt_ctrl.content.strip()} "
                    f"[sha256={smt_ctrl.sha256}]"
                    if not smt_ctrl.skipped_reason
                    else ""
                )
            ),
        )
    elif threads_per_core.is_detected():
        smt_enabled = EvidenceField(
            value=int(threads_per_core.value) > 1,
            evidence=f"derived from Thread(s) per core: {threads_per_core.evidence}",
        )
    else:
        smt_enabled = undetected("no smt sysfs and no threads_per_core")

    # Hybrid
    p_ids: list[int] = []
    e_ids: list[int] = []
    evidence_lines: list[str] = []
    any_core_type = False
    capacities: dict[int, int] = {}
    for d in store.text("cpu_dirs_list").splitlines():
        if not d:
            continue
        cpu_id = int(d[3:])
        ct_b = store.get(f"{d}_core_type")
        cap_b = store.get(f"{d}_cpu_capacity")
        if not ct_b.skipped_reason and ct_b.content.strip():
            any_core_type = True
            ct_v = ct_b.content.strip().lower()
            evidence_lines.append(
                f"{ct_b.command}: {ct_b.content.strip()} [sha256={ct_b.sha256}]"
            )
            if ct_v in ("performance", "p", "2"):
                p_ids.append(cpu_id)
            elif ct_v in ("efficiency", "e", "1", "0"):
                e_ids.append(cpu_id)
        if not cap_b.skipped_reason and cap_b.content.strip():
            try:
                capacities[cpu_id] = int(cap_b.content.strip())
                evidence_lines.append(
                    f"{cap_b.command}: {cap_b.content.strip()} "
                    f"[sha256={cap_b.sha256}]"
                )
            except ValueError:
                pass

    phys_int = physical_cores.value if isinstance(physical_cores.value, int) else None
    if any_core_type and (p_ids or e_ids):
        is_h = bool(p_ids) and bool(e_ids)
        hybrid = HybridInfo(
            is_hybrid=EvidenceField(
                value=is_h, evidence="; ".join(evidence_lines) or "core_type"
            ),
            p_core_count=EvidenceField(
                value=len(set(p_ids)), evidence=str(sorted(p_ids))
            ),
            e_core_count=EvidenceField(
                value=len(set(e_ids)), evidence=str(sorted(e_ids))
            ),
            p_core_logical_ids=EvidenceField(
                value=sorted(p_ids), evidence=str(sorted(p_ids))
            ),
            e_core_logical_ids=EvidenceField(
                value=sorted(e_ids), evidence=str(sorted(e_ids))
            ),
        )
        per_core_type_counts = {"P": len(set(p_ids)), "E": len(set(e_ids))}
    elif capacities and max(capacities.values()) != min(capacities.values()):
        max_c = max(capacities.values())
        for cid, c in capacities.items():
            (p_ids if c >= max_c else e_ids).append(cid)
        hybrid = HybridInfo(
            is_hybrid=EvidenceField(
                value=True,
                evidence="; ".join(evidence_lines),
            ),
            p_core_count=EvidenceField(value=len(p_ids), evidence=str(sorted(p_ids))),
            e_core_count=EvidenceField(value=len(e_ids), evidence=str(sorted(e_ids))),
            p_core_logical_ids=EvidenceField(
                value=sorted(p_ids), evidence=str(sorted(p_ids))
            ),
            e_core_logical_ids=EvidenceField(
                value=sorted(e_ids), evidence=str(sorted(e_ids))
            ),
        )
        per_core_type_counts = {"P": len(p_ids), "E": len(e_ids)}
    else:
        hybrid = HybridInfo(
            is_hybrid=EvidenceField(
                value=False,
                evidence=(
                    "no sysfs topology/core_type or varying cpu_capacity; "
                    "lscpu -e has no core-type column → treated as non-hybrid "
                    "(absence of hybrid signals) "
                    f"[lscpu -e sha256={store.sha('lscpu_e')}]"
                ),
            )
        )
        per_core_type_counts = (
            {"homogeneous": phys_int} if phys_int is not None else {}
        )

    # NUMA
    numa_nodes_ef = int_lscpu("NUMA node(s)")
    numa_cpu_map_lines = [
        ln
        for ln in lscpu_out.splitlines()
        if ln.startswith("NUMA node") and "CPU" in ln
    ]
    if (
        isinstance(numa_nodes_ef.value, int)
        and numa_nodes_ef.value == 1
        and numa_cpu_map_lines
    ):
        numa_nodes = EvidenceField(
            value="single node",
            evidence=(
                f"{_find_lscpu_line(lscpu_out, 'NUMA node(s)')}; "
                f"{numa_cpu_map_lines[0]}  [sha256={lscpu_sha}]"
            ),
        )
        cpu_to_node = EvidenceField(
            value={
                0: list(range(int(logical_cores.value)))
                if isinstance(logical_cores.value, int)
                else []
            },
            evidence=f"{numa_cpu_map_lines[0]}  [sha256={lscpu_sha}]",
        )
    elif isinstance(numa_nodes_ef.value, int) and numa_nodes_ef.value > 1:
        mapping: dict[int, list[int]] = {}
        evid = []
        for ln in numa_cpu_map_lines:
            m = re.match(r"NUMA node(\d+) CPU\(s\):\s*(.+)", ln.strip())
            if m:
                mapping[int(m.group(1))] = _parse_cpuset_list(m.group(2))
                evid.append(ln)
        numa_nodes = numa_nodes_ef
        cpu_to_node = EvidenceField(
            value=mapping,
            evidence="; ".join(evid) + f"  [sha256={lscpu_sha}]",
        )
    else:
        numa_nodes = numa_nodes_ef
        cpu_to_node = undetected("no NUMA CPU map lines in lscpu")

    # Cache from sysfs (per-instance) + lscpu aggregates
    def sysfs_cache(level: int, typ: str) -> tuple[EvidenceField, EvidenceField]:
        """Return (size_bytes_ef, sharing_ef)."""
        for idx in ("index0", "index1", "index2", "index3"):
            lvl_b = store.get(f"cache_{idx}_level")
            typ_b = store.get(f"cache_{idx}_type")
            sz_b = store.get(f"cache_{idx}_size")
            sh_b = store.get(f"cache_{idx}_shared_cpu_list")
            if lvl_b.skipped_reason or typ_b.skipped_reason or sz_b.skipped_reason:
                continue
            if lvl_b.content.strip() != str(level):
                continue
            if typ_b.content.strip() != typ:
                continue
            b = _parse_sysfs_size_to_bytes(sz_b.content.strip())
            if b is None:
                return (
                    undetected(f"bad size {sz_b.content!r}"),
                    undetected("no size"),
                )
            size_ef = EvidenceField(
                value=b,
                evidence=(
                    f"{sz_b.command}: {sz_b.content.strip()} "
                    f"[sha256={sz_b.sha256}]"
                ),
            )
            shared = sh_b.content.strip() if not sh_b.skipped_reason else ""
            cpus = _parse_cpuset_list(shared)
            sharing = "per-core" if len(cpus) <= 1 else "shared"
            share_ef = EvidenceField(
                value=sharing,
                evidence=(
                    f"{sh_b.command}: {shared} → {sharing} "
                    f"[sha256={sh_b.sha256}]"
                ),
            )
            return size_ef, share_ef
        return (
            undetected(f"sysfs cache level={level} type={typ} not found"),
            undetected("not found"),
        )

    l1d, _ = sysfs_cache(1, "Data")
    l1i, _ = sysfs_cache(1, "Instruction")
    l2, l2_share = sysfs_cache(2, "Unified")
    l3, _ = sysfs_cache(3, "Unified")

    def lscpu_cache(key: str) -> tuple[EvidenceField, EvidenceField]:
        ef = _lscpu_value_and_line(lscpu_out, key)
        if not ef.is_detected():
            return ef, undetected(f"no instances for {key}")
        b = _parse_size_to_bytes(str(ef.value))
        if b is None:
            return (
                undetected(f"could not parse: {ef.evidence}"),
                undetected("no parse"),
            )
        size_ef = EvidenceField(
            value=b, evidence=evid_line(ef.evidence, "lscpu")
        )
        m = re.search(r"\((\d+)\s+instances?\)", str(ef.value))
        if m:
            inst_ef = EvidenceField(
                value=int(m.group(1)),
                evidence=evid_line(ef.evidence, "lscpu"),
            )
        else:
            inst_ef = undetected(f"no instance count in {ef.evidence}")
        return size_ef, inst_ef

    lscpu_l1d, lscpu_l1d_inst = lscpu_cache("L1d cache")
    lscpu_l1i, lscpu_l1i_inst = lscpu_cache("L1i cache")
    lscpu_l2, lscpu_l2_inst = lscpu_cache("L2 cache")
    lscpu_l3, lscpu_l3_inst = lscpu_cache("L3 cache")

    cache = CacheInfo(
        l1d_bytes=l1d,
        l1i_bytes=l1i,
        l2_bytes=l2,
        l2_sharing=l2_share,
        l3_bytes=l3,
        lscpu_l1d_bytes=lscpu_l1d,
        lscpu_l1i_bytes=lscpu_l1i,
        lscpu_l2_bytes=lscpu_l2,
        lscpu_l3_bytes=lscpu_l3,
        lscpu_l1d_instances=lscpu_l1d_inst,
        lscpu_l1i_instances=lscpu_l1i_inst,
        lscpu_l2_instances=lscpu_l2_inst,
        lscpu_l3_instances=lscpu_l3_inst,
    )

    # --- ISA (whitespace-tokenized) ---
    flags_line = _find_lscpu_line(lscpu_out, "Flags")
    if flags_line:
        flags_blob = flags_line.split(":", 1)[1].strip()
        flags_source = f"lscpu Flags [sha256={lscpu_sha}]"
    else:
        flags_blob = ""
        flags_source = "lscpu Flags (missing)"
        for line in cpuinfo.splitlines():
            if line.startswith("flags") or line.startswith("Features"):
                flags_blob = line.split(":", 1)[1].strip()
                flags_source = f"/proc/cpuinfo flags [sha256={cpuinfo_sha}]"
                break

    arch_s = str(arch.value) if arch.is_detected() else ""
    if arch_s in ("x86_64", "amd64", "i386", "i686", "x86"):
        family = "x86"
        wanted = X86_FLAGS
    elif arch_s.startswith("arm") or arch_s.startswith("aarch"):
        family = "arm"
        wanted = ARM_FLAGS
    else:
        family = "other"
        wanted = []

    flags: dict[str, EvidenceField] = {}
    for f in wanted:
        flags[f] = _flag_present_tokenized(flags_blob, f, flags_source)
    present = {k for k, v in flags.items() if v.value == "PRESENT"}
    cpuid_tier, cpuid_reason = _tier_from_flags(family, present)

    # Parser negative control
    neg_probes = []
    neg_fail = False
    if family == "x86":
        neg_flags = [
            ("i8mm", "ARM flag; must be ABSENT on x86"),
            ("asimddp", "ARM flag; must be ABSENT on x86"),
            ("sve", "ARM flag; must be ABSENT on x86"),
            ("zzz_nonexistent_flag", "synthetic; exists nowhere"),
            (
                "avx512",
                "strict PREFIX of avx512f — NOT a real flag; must be ABSENT",
            ),
        ]
    else:
        neg_flags = [
            ("zzz_nonexistent_flag", "synthetic; exists nowhere"),
            ("amx_tile", "x86 flag; must be ABSENT on ARM"),
            ("avx512f", "x86 flag; must be ABSENT on ARM"),
        ]
    for fname, why in neg_flags:
        ef = _flag_present_tokenized(flags_blob, fname, flags_source)
        ok = ef.value == "ABSENT"
        if not ok:
            neg_fail = True
        neg_probes.append(
            {
                "flag": fname,
                "result": str(ef.value),
                "evidence": ef.evidence,
                "expectation": f"ABSENT ({why})",
                "status": "PASS" if ok else "FAIL",
            }
        )
    parser_nc = ParserNegativeControl(
        match_mode=(
            "whitespace-tokenized (explicit split on whitespace; "
            "NOT substring / NOT regex-prefix). "
            "Previous Phase 1 used word-boundary regex which is equivalent "
            "for these tokens; Phase 1.5 states and uses tokenized matching."
        ),
        probes=neg_probes,
        overall="FAIL" if neg_fail else "PASS",
    )
    if neg_fail:
        raise RuntimeError(
            "SECTION 3b PARSER NEGATIVE CONTROL FAILED: flag parser returned "
            "PRESENT for a probe that must be ABSENT. Refusing to continue."
        )

    # AMX runtime probe → usable_tier
    amx_probe = _probe_amx_runtime(store)
    amx_usable = (
        isinstance(amx_probe.prctl_rc.value, int)
        and amx_probe.prctl_rc.value == 0
    )
    if cpuid_tier == QuantKernelTier.AMX and amx_usable:
        usable_tier = QuantKernelTier.AMX
        usable_reason = (
            "CPUID AMX + prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA)=0 → AMX usable"
        )
    elif cpuid_tier == QuantKernelTier.AMX and not amx_usable:
        # Drop to AVX512_VNNI per spec (even if VNNI somehow absent, still drop)
        if "avx512_vnni" in present:
            usable_tier = QuantKernelTier.AVX512_VNNI
            usable_reason = (
                f"CPUID claims AMX but prctl ARCH_REQ_XCOMP_PERM rc="
                f"{amx_probe.prctl_rc.value} errno={amx_probe.prctl_errno.value}; "
                "usable_tier drops to AVX512_VNNI (do not claim AMX on CPUID alone)"
            )
        elif "avx512f" in present:
            usable_tier = QuantKernelTier.AVX512
            usable_reason = (
                f"AMX probe failed (rc={amx_probe.prctl_rc.value}); "
                "avx512_vnni absent → drop to AVX512"
            )
        else:
            usable_tier = QuantKernelTier.AVX512_VNNI
            usable_reason = (
                f"AMX probe failed (rc={amx_probe.prctl_rc.value}); "
                "spec says drop to AVX512_VNNI"
            )
    else:
        usable_tier = cpuid_tier
        usable_reason = f"cpuid_tier={cpuid_tier.value}; AMX not claimed so usable=cpuid"

    isa = IsaFlags(
        arch_family=family,
        flags=flags,
        flag_match_mode="whitespace-tokenized",
        cpuid_tier=EvidenceField(
            value=cpuid_tier.value,
            evidence=f"from flags in {flags_source}; {cpuid_reason}",
        ),
        usable_tier=EvidenceField(
            value=usable_tier.value,
            evidence=usable_reason,
        ),
        quant_kernel_tier=EvidenceField(
            value=usable_tier.value,
            evidence=f"usable_tier (not cpuid alone): {usable_reason}",
        ),
        quant_kernel_reasoning=(
            f"cpuid_tier={cpuid_tier.value} ({cpuid_reason}); "
            f"usable_tier={usable_tier.value} ({usable_reason}). "
            f"{amx_probe.note}"
        ),
    )

    # --- Memory ---
    def mem_bytes(key: str) -> EvidenceField:
        kb, line = _meminfo_kb(meminfo, key)
        if kb is None or line is None:
            return undetected(f"{key} not found in /proc/meminfo")
        return EvidenceField(
            value=kb * 1024,
            evidence=f"{line}  [sha256={meminfo_sha}]",
        )

    mem_total = mem_bytes("MemTotal")
    mem_avail = mem_bytes("MemAvailable")
    mem_free = mem_bytes("MemFree")
    mem_available_source_note = (
        "MemAvailable used as primary /proc availability metric (reclaimable "
        "cache/buffers); downstream usable_ram uses effective_memory_limit, "
        "not raw MemAvailable."
    )

    swap_total = mem_bytes("SwapTotal")
    swap_free = mem_bytes("SwapFree")
    if isinstance(swap_total.value, int) and isinstance(swap_free.value, int):
        swap_used = EvidenceField(
            value=swap_total.value - swap_free.value,
            evidence=(
                f"SwapTotal - SwapFree = {swap_total.value} - {swap_free.value} "
                f"from [{swap_total.evidence}] [{swap_free.evidence}]"
            ),
        )
    else:
        swap_used = undetected("need SwapTotal and SwapFree")

    sw_b = store.get("swappiness")
    if not sw_b.skipped_reason and sw_b.content.strip().isdigit():
        swappiness = EvidenceField(
            value=int(sw_b.content.strip()),
            evidence=(
                f"/proc/sys/vm/swappiness: {sw_b.content.strip()} "
                f"[sha256={sw_b.sha256}]"
            ),
        )
    else:
        swappiness = undetected("swappiness unreadable")

    pg_b = store.get("getconf_page_size")
    if not pg_b.skipped_reason and pg_b.content.strip().isdigit():
        page_size = EvidenceField(
            value=int(pg_b.content.strip()),
            evidence=(
                f"getconf PAGE_SIZE → {pg_b.content.strip()} "
                f"[sha256={pg_b.sha256}]"
            ),
        )
    else:
        page_size = undetected("getconf PAGE_SIZE failed")

    _, hp_count_line = _meminfo_kb(meminfo, "HugePages_Total")
    hp_size_kb, hp_size_line = _meminfo_kb(meminfo, "Hugepagesize")
    if hp_count_line:
        hugepages_count = EvidenceField(
            value=int(hp_count_line.split()[1]),
            evidence=f"{hp_count_line}  [sha256={meminfo_sha}]",
        )
    else:
        hugepages_count = undetected("HugePages_Total missing")
    if hp_size_kb is not None and hp_size_line:
        hugepages_size = EvidenceField(
            value=hp_size_kb * 1024,
            evidence=f"{hp_size_line}  [sha256={meminfo_sha}]",
        )
    else:
        hugepages_size = undetected("Hugepagesize missing")

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
    dmi = store.get("dmidecode_memory")
    if not dmi.skipped_reason and dmi.content.strip():
        # parse populated DIMMs from the single capture
        devices = dmi.content.split("Memory Device\n")
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
            evidence=(
                f"dmidecode -t memory: {populated} populated devices "
                f"[sha256={dmi.sha256}]"
            ),
        )
        if speeds:
            dimm_speed = EvidenceField(
                value=speeds[0] if len(set(speeds)) == 1 else speeds,
                evidence=f"dmidecode Speed fields: {speeds} [sha256={dmi.sha256}]",
            )
        memory_channels = undetected(
            "dmidecode does not reliably expose channel count without "
            "board-specific interpretation; not guessed"
        )

    theoretical_bw = undetected(
        "need memory_channels, bus width, and DIMM MT/s; channels/width unavailable"
    )
    bandwidth_formula = (
        "channels * width_bytes * MT/s / 1e9 — inputs incomplete "
        f"(channels={memory_channels.value}, dimm_speed={dimm_speed.value}, "
        "width_bytes=UNDETECTED)"
    )
    bandwidth_confidence = (
        "guess — no DMI channel/width data on this host; formula not evaluated"
    )

    # --- SECTION 2b: Execution environment ---
    hyp_vendor = _lscpu_value_and_line(lscpu_out, "Hypervisor vendor")
    if hyp_vendor.is_detected():
        hyp_vendor = EvidenceField(
            value=hyp_vendor.value,
            evidence=evid_line(hyp_vendor.evidence, "lscpu"),
        )
    virt_type = _lscpu_value_and_line(lscpu_out, "Virtualization type")
    virt_flag = _lscpu_value_and_line(lscpu_out, "Virtualization")
    virt_combined = EvidenceField(
        value={
            "Virtualization": virt_flag.value if virt_flag.is_detected() else None,
            "Virtualization type": virt_type.value if virt_type.is_detected() else None,
        },
        evidence=(
            f"{virt_flag.evidence if virt_flag.is_detected() else 'Virtualization: missing'}; "
            f"{virt_type.evidence if virt_type.is_detected() else 'Virtualization type: missing'} "
            f"[sha256={lscpu_sha}]"
        ),
    )
    hyp_sys = store.get("sys_hypervisor_type")
    if hyp_sys.skipped_reason:
        hypervisor_sysfs = undetected(hyp_sys.skipped_reason)
    else:
        hypervisor_sysfs = EvidenceField(
            value=hyp_sys.content.strip(),
            evidence=(
                f"/sys/hypervisor/type: {hyp_sys.content.strip()} "
                f"[sha256={hyp_sys.sha256}]"
            ),
        )
    sdv = store.get("systemd_detect_virt")
    if sdv.skipped_reason:
        systemd_virt = undetected(sdv.skipped_reason)
    else:
        systemd_virt = EvidenceField(
            value=sdv.content.strip(),
            evidence=(
                f"systemd-detect-virt → {sdv.content.strip()} "
                f"[sha256={sdv.sha256}]"
            ),
        )

    dockerenv_b = store.get("dockerenv")
    dockerenv = EvidenceField(
        value=dockerenv_b.content.strip(),
        evidence=f"test -e /.dockerenv → {dockerenv_b.content.strip()} [sha256={dockerenv_b.sha256}]",
    )
    cen_b = store.get("containerenv")
    containerenv = EvidenceField(
        value=cen_b.content.strip(),
        evidence=(
            f"test -e /run/.containerenv → {cen_b.content.strip()} "
            f"[sha256={cen_b.sha256}]"
        ),
    )
    p1cg = store.get("proc1_cgroup")
    proc1_cgroup = EvidenceField(
        value=p1cg.content if not p1cg.skipped_reason else p1cg.skipped_reason,
        evidence=(
            f"/proc/1/cgroup verbatim ({len(p1cg.content)} bytes) "
            f"[sha256={p1cg.sha256}]: {p1cg.content!r}"
        ),
    )

    cg_ver_b = store.get("cgroup_version_probe")
    cgroup_version = EvidenceField(
        value=cg_ver_b.content.strip(),
        evidence=f"{cg_ver_b.command} → {cg_ver_b.content.strip()} [sha256={cg_ver_b.sha256}]",
    )

    def cg_file(key: str) -> EvidenceField:
        b = store.get(key)
        if b.skipped_reason:
            return undetected(b.skipped_reason)
        return EvidenceField(
            value=b.content.strip(),
            evidence=f"{b.command}: {b.content.strip()} [sha256={b.sha256}]",
        )

    memory_max_ef = cg_file("cgroup_memory_max")
    memory_high_ef = cg_file("cgroup_memory_high")
    memory_current_ef = cg_file("cgroup_memory_current")
    cpu_max_ef = cg_file("cgroup_cpu_max")
    cpuset_ef = cg_file("cgroup_cpuset_cpus_effective")

    # cpu.max quota/period
    cpu_quota_period: EvidenceField
    cgroup_cpu_limit: Optional[float] = None
    if cpu_max_ef.is_detected():
        raw = str(cpu_max_ef.value)
        if raw == "max" or raw.startswith("max"):
            cpu_quota_period = EvidenceField(
                value="unlimited",
                evidence=f"cpu.max={raw} → unlimited [from {cpu_max_ef.evidence}]",
            )
        else:
            parts = raw.split()
            # v2: "quota period"; v1 fallback may be just quota
            if len(parts) >= 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if period > 0 and quota > 0:
                    cgroup_cpu_limit = quota / period
                    cpu_quota_period = EvidenceField(
                        value=cgroup_cpu_limit,
                        evidence=(
                            f"cpu.max quota/period = {quota}/{period} = "
                            f"{cgroup_cpu_limit} [from {cpu_max_ef.evidence}]"
                        ),
                    )
                elif quota < 0:
                    cpu_quota_period = EvidenceField(
                        value="unlimited",
                        evidence=f"quota={quota} → unlimited",
                    )
                else:
                    cpu_quota_period = undetected(f"unusable cpu.max: {raw}")
            else:
                # v1: only quota captured; need period
                if "cgroup_cpu_period_v1" in store.blobs:
                    per_b = store.get("cgroup_cpu_period_v1")
                    try:
                        quota = int(raw)
                        period = int(per_b.content.strip())
                        if quota < 0:
                            cpu_quota_period = EvidenceField(
                                value="unlimited",
                                evidence=f"v1 cfs_quota_us={quota}",
                            )
                        else:
                            cgroup_cpu_limit = quota / period
                            cpu_quota_period = EvidenceField(
                                value=cgroup_cpu_limit,
                                evidence=(
                                    f"v1 quota/period = {quota}/{period} = "
                                    f"{cgroup_cpu_limit}"
                                ),
                            )
                    except ValueError:
                        cpu_quota_period = undetected("v1 cpu quota parse fail")
                else:
                    cpu_quota_period = undetected(f"cannot parse cpu.max: {raw}")
    else:
        cpu_quota_period = undetected("cpu.max unavailable")

    cpuset_list = (
        _parse_cpuset_list(str(cpuset_ef.value))
        if cpuset_ef.is_detected()
        else []
    )
    cpuset_size = EvidenceField(
        value=len(cpuset_list) if cpuset_ef.is_detected() else "UNDETECTED (reason: no cpuset)",
        evidence=(
            f"cpuset.cpus.effective={cpuset_ef.value!s} → {len(cpuset_list)} CPUs; "
            f"{cpuset_ef.evidence}"
            if cpuset_ef.is_detected()
            else cpuset_ef.evidence
        ),
    )

    # affinity + cpu_count (library — mandatory)
    affinity_set = sorted(os.sched_getaffinity(0))
    affinity_len = len(affinity_set)
    os_cc = os.cpu_count()
    os_cpu_count_ef = EvidenceField(
        value=os_cc,
        evidence=(
            f"python os.cpu_count() (Python {platform.python_version()}) → {os_cc}"
        ),
    )
    sched_affinity_len_ef = EvidenceField(
        value=affinity_len,
        evidence=(
            f"python len(os.sched_getaffinity(0)) "
            f"(Python {platform.python_version()}) → {affinity_len}"
        ),
    )
    sched_affinity_set_ef = EvidenceField(
        value=affinity_set,
        evidence=(
            f"python os.sched_getaffinity(0) "
            f"(Python {platform.python_version()}) → {affinity_set}"
        ),
    )

    # Steal time
    def parse_steal(stat_text: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
        for line in stat_text.splitlines():
            if line.startswith("cpu "):
                parts = line.split()
                # parts[0]=cpu; numeric fields 1..; steal is 8th numeric = parts[8]
                # user nice system idle iowait irq softirq steal guest guest_nice
                # idx:  1    2    3      4    5      6   7       8     9     10
                if len(parts) < 9:
                    return None, None, line
                steal = int(parts[8])
                total = sum(int(x) for x in parts[1:])
                return steal, total, line
        return None, None, None

    t0 = store.get("proc_stat_t0")
    t1 = store.get("proc_stat_t1")
    s0, tot0, line0 = parse_steal(t0.content)
    s1, tot1, line1 = parse_steal(t1.content)
    steal_t0 = EvidenceField(
        value={"steal": s0, "total_jiffies": tot0},
        evidence=(
            f"/proc/stat t0 cpu line field 8 (steal, 1-indexed among numeric "
            f"fields) = {s0}; line={line0!r} [sha256={t0.sha256}]"
        ),
    )
    steal_t1 = EvidenceField(
        value={"steal": s1, "total_jiffies": tot1},
        evidence=(
            f"/proc/stat t1 (t0+5s) steal={s1}; line={line1!r} "
            f"[sha256={t1.sha256}]"
        ),
    )
    if s0 is not None and s1 is not None and tot0 is not None and tot1 is not None:
        d_steal = s1 - s0
        d_tot = tot1 - tot0
        if d_tot > 0:
            pct = 100.0 * d_steal / d_tot
            steal_pct = EvidenceField(
                value=pct,
                evidence=(
                    f"delta_steal={d_steal}, delta_jiffies={d_tot}, "
                    f"steal%={pct:.6f} (sampled 5s apart)"
                ),
            )
        else:
            steal_pct = undetected("elapsed jiffies delta was 0")
    else:
        steal_pct = undetected("could not parse /proc/stat cpu steal field")

    # effective_memory_limit = min(MemTotal, cgroup memory.max)
    cgroup_mem_bytes: Optional[int] = None
    if memory_max_ef.is_detected():
        raw = str(memory_max_ef.value)
        if raw != "max":
            try:
                cgroup_mem_bytes = int(raw)
            except ValueError:
                cgroup_mem_bytes = None

    mem_total_int = mem_total.value if isinstance(mem_total.value, int) else None
    if mem_total_int is not None and cgroup_mem_bytes is not None:
        if cgroup_mem_bytes <= mem_total_int:
            eff_mem = cgroup_mem_bytes
            eff_mem_winner = "cgroup memory.max"
        else:
            eff_mem = mem_total_int
            eff_mem_winner = "MemTotal (/proc)"
        effective_memory_limit = EvidenceField(
            value=eff_mem,
            evidence=(
                f"min(MemTotal={mem_total_int}, cgroup memory.max="
                f"{cgroup_mem_bytes}) = {eff_mem}; winner={eff_mem_winner}"
            ),
        )
    elif mem_total_int is not None:
        eff_mem = mem_total_int
        eff_mem_winner = "MemTotal (/proc; cgroup max unavailable or 'max')"
        effective_memory_limit = EvidenceField(
            value=eff_mem,
            evidence=(
                f"cgroup memory.max={memory_max_ef.value}; using MemTotal="
                f"{mem_total_int}"
            ),
        )
    else:
        eff_mem = None
        eff_mem_winner = "UNDETECTED"
        effective_memory_limit = undetected("MemTotal and cgroup memory.max unavailable")

    # effective_cpu_count = min(affinity, cgroup cpu.max quota/period)
    aff_f = float(affinity_len)
    if cgroup_cpu_limit is not None:
        if aff_f <= cgroup_cpu_limit:
            eff_cpu_val = aff_f
            eff_cpu_winner = "affinity"
        else:
            eff_cpu_val = cgroup_cpu_limit
            eff_cpu_winner = "cgroup cpu.max quota/period"
    else:
        eff_cpu_val = aff_f
        eff_cpu_winner = "affinity (cgroup cpu.max unlimited/unavailable)"
    # Prefer int when whole
    if float(eff_cpu_val) == int(eff_cpu_val):
        eff_cpu_out: Any = int(eff_cpu_val)
    else:
        eff_cpu_out = eff_cpu_val
    effective_cpu_count = EvidenceField(
        value=eff_cpu_out,
        evidence=(
            f"min(affinity={affinity_len}, cgroup_cpu_limit="
            f"{cgroup_cpu_limit if cgroup_cpu_limit is not None else 'unavailable'}) "
            f"= {eff_cpu_out}; winner={eff_cpu_winner}; "
            f"cpuset_size={cpuset_size.value}"
        ),
    )

    exec_env = ExecutionEnvironment(
        hypervisor_vendor_lscpu=hyp_vendor,
        virtualization_lscpu=virt_combined,
        hypervisor_sysfs=hypervisor_sysfs,
        systemd_detect_virt=systemd_virt,
        dockerenv=dockerenv,
        containerenv=containerenv,
        proc1_cgroup=proc1_cgroup,
        cgroup_version=cgroup_version,
        memory_max=memory_max_ef,
        memory_high=memory_high_ef,
        memory_current=memory_current_ef,
        cpu_max=cpu_max_ef,
        cpu_max_quota_period=cpu_quota_period,
        cpuset_cpus_effective=cpuset_ef,
        cpuset_size=cpuset_size,
        os_cpu_count=os_cpu_count_ef,
        sched_affinity_len=sched_affinity_len_ef,
        sched_affinity_set=sched_affinity_set_ef,
        steal_sample_t0=steal_t0,
        steal_sample_t1=steal_t1,
        steal_percent=steal_pct,
        effective_memory_limit=effective_memory_limit,
        effective_memory_winner=eff_mem_winner,
        effective_cpu_count=effective_cpu_count,
        effective_cpu_winner=eff_cpu_winner,
    )

    # Proxy detection
    is_proxy = False
    proxy_reasons = []
    if hyp_vendor.is_detected():
        is_proxy = True
        proxy_reasons.append(f"Hypervisor vendor={hyp_vendor.value}")
    if systemd_virt.is_detected() and str(systemd_virt.value) not in (
        "none",
        "",
    ):
        is_proxy = True
        proxy_reasons.append(f"systemd-detect-virt={systemd_virt.value}")
    if dockerenv.value == "EXISTS":
        is_proxy = True
        proxy_reasons.append("/.dockerenv EXISTS")
    proxy_banner = ""
    if is_proxy:
        proxy_banner = (
            "*** DEVELOPMENT PROXY / SANDBOX — NOT A BARE-METAL DEPLOYMENT TARGET *** "
            + "; ".join(proxy_reasons)
            + ". Values below describe this guest/container share, not host silicon capacity."
        )

    target_reliability = (
        "This machine is a development proxy (KVM guest inside a Docker/cgroup "
        "sandbox), not the intended bare-metal deployment target. The following "
        "detected values would NOT transfer to different hardware: hypervisor-"
        "synthesized cache sizes (especially the 320 MiB L3), CPU model string "
        f"('{(model_name.value if model_name.is_detected() else '?')}'), "
        "CPUID flags that the guest advertises but cannot enable (AMX/"
        "XTILEDATA prctl failed here), cgroup memory.max/cpu.max quotas, "
        "steal time, sched affinity, DIMM/channel topology (undetected in-"
        "guest), and any HOST-LEVEL uarch decode implications about memory "
        "channels or package L3. Only re-run detection on the real target."
    )

    # --- Derived defaults from EFFECTIVE limits ---
    headroom = 0.70
    usable_warning = ""
    no_swap = isinstance(swap_total.value, int) and swap_total.value == 0
    # Use centi-precision ints to avoid 0.70-0.05 → 0.649999... display junk.
    factor_cents = 70
    if no_swap:
        factor_cents -= 5  # additional 5%
        usable_warning = (
            "WARNING: SwapTotal == 0; subtracted an additional 5% headroom "
            f"(factor 0.70 → {factor_cents / 100:.2f}). Failure mode is "
            "OOM-kill rather than swap-backed degradation."
        )
    effective_factor = factor_cents / 100.0

    if isinstance(effective_memory_limit.value, int):
        usable = int(effective_memory_limit.value * effective_factor)
        usable_ram = EvidenceField(
            value=usable,
            evidence=(
                f"{effective_factor} * effective_memory_limit = "
                f"{effective_factor} * {effective_memory_limit.value} = {usable} "
                f"(winner was {eff_mem_winner})"
            ),
        )
        usable_formula = (
            f"{effective_factor} * effective_memory_limit "
            f"({effective_memory_limit.value}) = {usable}"
            + (" [no-swap -5% applied]" if no_swap else "")
        )
    else:
        usable_ram = undetected("effective_memory_limit unavailable")
        usable_formula = "UNDETECTED"

    # Side-by-side from raw MemAvailable (NOT used downstream)
    if isinstance(mem_avail.value, int):
        alt = int(mem_avail.value * effective_factor)
        usable_from_avail = EvidenceField(
            value=alt,
            evidence=(
                f"NOT USED DOWNSTREAM: {effective_factor} * MemAvailable "
                f"= {effective_factor} * {mem_avail.value} = {alt}"
            ),
        )
    else:
        usable_from_avail = undetected("MemAvailable unavailable")

    # suggested threads from effective_cpu_count
    if isinstance(effective_cpu_count.value, (int, float)):
        # Use floor for threads
        n_threads = int(effective_cpu_count.value)
        if n_threads < 1:
            n_threads = 1
        suggested_thread_count = EvidenceField(
            value=n_threads,
            evidence=(
                f"effective_cpu_count={effective_cpu_count.value} "
                f"(winner={eff_cpu_winner}); provisional pending Phase 2"
            ),
        )
        suggested_formula = (
            f"int(effective_cpu_count) = int({effective_cpu_count.value}) "
            f"= {n_threads} [NOT raw os.cpu_count()/lscpu]"
        )
    else:
        suggested_thread_count = undetected("effective_cpu_count unavailable")
        suggested_formula = "UNDETECTED"

    # --- Cross-checks (non-tautological) ---
    cross_checks: list[CrossCheck] = []

    # 1. physical * threads == logical
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
                left=f"{physical_cores.value} * {threads_per_core.value} = {left}",
                right=f"logical_cores = {right}",
                result="PASS" if left == right else "FAIL",
            )
        )
    else:
        cross_checks.append(
            CrossCheck(
                name="physical_cores * threads_per_core == logical_cores",
                left="UNDETECTED",
                right="UNDETECTED",
                result="FAIL",
            )
        )

    # 2. MemFree <= MemAvailable <= MemTotal
    if (
        isinstance(mem_free.value, int)
        and isinstance(mem_avail.value, int)
        and isinstance(mem_total.value, int)
    ):
        ok = mem_free.value <= mem_avail.value <= mem_total.value
        cross_checks.append(
            CrossCheck(
                name="MemFree <= MemAvailable <= MemTotal",
                left=f"MemFree={mem_free.value}, MemAvailable={mem_avail.value}",
                right=f"MemTotal={mem_total.value}",
                result="PASS" if ok else "FAIL",
            )
        )
    else:
        cross_checks.append(
            CrossCheck(
                name="MemFree <= MemAvailable <= MemTotal",
                left="UNDETECTED",
                right="UNDETECTED",
                result="FAIL",
            )
        )

    # 3. effective_memory_limit vs MemTotal — FAIL loudly if cgroup < /proc
    if (
        isinstance(effective_memory_limit.value, int)
        and isinstance(mem_total.value, int)
    ):
        if cgroup_mem_bytes is not None and cgroup_mem_bytes < mem_total.value:
            cross_checks.append(
                CrossCheck(
                    name="effective_memory_limit <= MemTotal (cgroup binding)",
                    left=(
                        f"CGROUP BINDS BELOW /proc: cgroup memory.max="
                        f"{cgroup_mem_bytes} < MemTotal={mem_total.value}; "
                        f"effective={effective_memory_limit.value}"
                    ),
                    right=f"MemTotal={mem_total.value}",
                    result="FAIL",
                )
            )
        else:
            ok = effective_memory_limit.value <= mem_total.value
            cross_checks.append(
                CrossCheck(
                    name="effective_memory_limit <= MemTotal (cgroup binding)",
                    left=(
                        f"effective={effective_memory_limit.value} "
                        f"(winner={eff_mem_winner}); "
                        f"cgroup memory.max={cgroup_mem_bytes}"
                    ),
                    right=f"MemTotal={mem_total.value}",
                    result="PASS" if ok else "FAIL",
                )
            )
    else:
        cross_checks.append(
            CrossCheck(
                name="effective_memory_limit <= MemTotal (cgroup binding)",
                left="UNDETECTED",
                right="UNDETECTED",
                result="FAIL",
            )
        )

    # 4. os.cpu_count() == len(affinity) == cpuset size
    cc_vals = {
        "os.cpu_count()": os_cc,
        "len(os.sched_getaffinity(0))": affinity_len,
        "cgroup cpuset.cpus.effective size": (
            cpuset_size.value if isinstance(cpuset_size.value, int) else None
        ),
    }
    defined = {k: v for k, v in cc_vals.items() if v is not None}
    agree = len(set(defined.values())) == 1 and len(defined) >= 2
    # Mandatory: if cpu_count != affinity, FAIL
    if os_cc != affinity_len:
        agree = False
    cross_checks.append(
        CrossCheck(
            name="os.cpu_count() == len(os.sched_getaffinity(0)) == cgroup cpuset size",
            left=str(defined),
            right=f"unique={sorted(set(defined.values()))}",
            result="PASS" if agree else "FAIL",
        )
    )

    # 5. sum of lscpu -e unique CORE values == physical_cores
    unique_cores: set[int] = set()
    lscpu_e_lines = [
        ln for ln in lscpu_e.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    if lscpu_e_lines:
        header = lscpu_e_lines[0].split()
        if "CORE" in header:
            core_idx = header.index("CORE")
            for ln in lscpu_e_lines[1:]:
                cols = ln.split()
                if len(cols) > core_idx:
                    try:
                        unique_cores.add(int(cols[core_idx]))
                    except ValueError:
                        pass
    lscpu_unique_core_count = len(unique_cores)
    if phys_int is not None and unique_cores:
        cross_checks.append(
            CrossCheck(
                name="sum of lscpu -e unique CORE values == physical_cores",
                left=(
                    f"unique CORE ids={sorted(unique_cores)} "
                    f"count={lscpu_unique_core_count} "
                    f"[sha256={store.sha('lscpu_e')}]"
                ),
                right=f"physical_cores={phys_int}",
                result="PASS" if lscpu_unique_core_count == phys_int else "FAIL",
            )
        )
    else:
        cross_checks.append(
            CrossCheck(
                name="sum of lscpu -e unique CORE values == physical_cores",
                left=f"unique_cores={sorted(unique_cores) if unique_cores else 'UNDETECTED'}",
                right=f"physical_cores={phys_int}",
                result="FAIL",
            )
        )

    # 6. Cache: per-core * instances == lscpu aggregate (every level)
    def cache_check(
        name: str,
        per_core: EvidenceField,
        instances: Optional[EvidenceField],
        aggregate: Optional[EvidenceField],
    ) -> None:
        if (
            isinstance(per_core.value, int)
            and instances is not None
            and isinstance(instances.value, int)
            and aggregate is not None
            and isinstance(aggregate.value, int)
        ):
            left = per_core.value * instances.value
            right = aggregate.value
            cross_checks.append(
                CrossCheck(
                    name=name,
                    left=f"per_instance({per_core.value}) * instances({instances.value}) = {left}",
                    right=f"lscpu aggregate = {right}",
                    result="PASS" if left == right else "FAIL",
                )
            )
        else:
            cross_checks.append(
                CrossCheck(
                    name=name,
                    left=f"per={per_core.value}, inst={instances.value if instances else None}",
                    right=f"agg={aggregate.value if aggregate else None}",
                    result="FAIL",
                )
            )

    cache_check(
        "L1d: sysfs_per_core * lscpu_instances == lscpu_aggregate",
        l1d,
        lscpu_l1d_inst,
        lscpu_l1d,
    )
    cache_check(
        "L1i: sysfs_per_core * lscpu_instances == lscpu_aggregate",
        l1i,
        lscpu_l1i_inst,
        lscpu_l1i,
    )
    cache_check(
        "L2: sysfs_per_core * lscpu_instances == lscpu_aggregate",
        l2,
        lscpu_l2_inst,
        lscpu_l2,
    )
    cache_check(
        "L3: sysfs_per_core * lscpu_instances == lscpu_aggregate",
        l3,
        lscpu_l3_inst,
        lscpu_l3,
    )

    # 7. ISA consistency (kept; can fail)
    def flag_on(name: str) -> bool:
        f = isa.flags.get(name)
        return bool(f and f.value == "PRESENT")

    isa_ok = True
    isa_parts = []
    if family == "x86":
        if flag_on("avx512_vnni") and not flag_on("avx512f"):
            isa_ok = False
            isa_parts.append("avx512_vnni PRESENT but avx512f ABSENT")
        else:
            isa_parts.append(
                f"avx512_vnni={flag_on('avx512_vnni')} → avx512f={flag_on('avx512f')}"
            )
        amx_any = flag_on("amx_tile") or flag_on("amx_int8") or flag_on("amx_bf16")
        if amx_any and not flag_on("avx512f"):
            isa_ok = False
            isa_parts.append("amx_* PRESENT but avx512f ABSENT")
        else:
            isa_parts.append(f"amx_any={amx_any} → avx512f={flag_on('avx512f')}")
    cross_checks.append(
        CrossCheck(
            name="ISA flags internally consistent",
            left="; ".join(isa_parts) if isa_parts else "n/a",
            right="avx512_vnni⇒avx512f; amx⇒avx512f; i8mm⇒asimd",
            result="PASS" if isa_ok else "FAIL",
        )
    )

    # Parser negative control as cross-check too
    cross_checks.append(
        CrossCheck(
            name="SECTION 3b parser negative control",
            left=str([(p["flag"], p["result"]) for p in parser_nc.probes]),
            right="all probes ABSENT",
            result=parser_nc.overall,
        )
    )

    core_count_sources = {
        "lscpu CPU(s)": logical_cores.value,
        "os.cpu_count()": os_cc,
        "len(os.sched_getaffinity(0))": affinity_len,
        "/proc/cpuinfo processor entry count": len(
            re.findall(r"^processor\s*:", cpuinfo, re.M)
        ),
        "cgroup cpuset.cpus.effective size": cpuset_size.value,
        "cgroup cpu.max quota/period": cgroup_cpu_limit,
        "effective_cpu_count": effective_cpu_count.value,
        "lscpu -e unique CORE count": lscpu_unique_core_count,
        "lscpu Core(s) per socket * Socket(s)": physical_cores.value,
    }

    self_critique = {
        "fields that are heuristic, not directly read": [
            "suggested_thread_count from effective_cpu_count (provisional)",
            f"usable_ram_for_model_bytes ({effective_factor} * effective_memory_limit; policy)",
            "cpuid_tier priority ladder over flags",
            "usable_tier drops AMX→AVX512_VNNI on failed prctl (policy from Phase 1.5 spec)",
            "hybrid=false from absence of sysfs hybrid signals",
            "uarch decode from static lookup table (HOST-LEVEL ONLY)",
            "effective_cpu_count / effective_memory_limit min() selection",
        ],
        "platform code paths not exercised on this machine": [
            "macOS sysctl/vm_stat path",
            "Windows WMIC/PowerShell CIM path",
            "ARM ISA flag set and ARM quant tiers",
            "hybrid P/E core_type and cpu_capacity parsing",
            "multi-node NUMA mapping",
            "dmidecode DIMM/channel parsing (not installed)",
            "cpufreq base/max frequency sysfs (absent under KVM)",
            "numactl --hardware (not installed)",
            "cgroup v1 fallback path (this host is cgroup v2)",
            "successful AMX prctl path (rc=0) — only failure path exercised",
            "/sys/hypervisor present path (absent here)",
        ],
        "places a default could have been silently substituted": [
            "base_mhz: could have used /proc/cpuinfo cpu MHz — intentionally NOT done",
            "usable_tier: could have kept AMX from CPUID — intentionally dropped after failed prctl",
            "theoretical_peak_bandwidth / DIMM fields: left UNDETECTED",
            "cgroup memory.max='max' would fall through to MemTotal — reported explicitly",
            "headroom 0.70 (not 0.85); no-swap extra -5% applied and warned",
        ],
        "known parsing fragility": [
            "lscpu English locale keys",
            "whitespace tokenization assumes cpuid names are single tokens",
            "KVM may expose synthetic cache sizes",
            "steal field index assumes standard /proc/stat column order",
            "cgroup path resolution from 0::/ assumes v2 unified hierarchy at /sys/fs/cgroup",
            "uarch table incomplete; UNKNOWN_UARCH is not an error but limits insight",
        ],
    }

    return DeviceProfile(
        platform="linux",
        python_version=platform.python_version(),
        detection_libraries=[
            f"Python stdlib only (platform={platform.python_version()}, "
            "subprocess, os, hashlib, ctypes for prctl). No third-party detection libs."
        ],
        commands_run=store.commands_log(),
        raw_source_dump=raw_dump,
        captures=[store.blobs[k] for k in store.order],
        is_development_proxy=is_proxy,
        proxy_banner=proxy_banner,
        target_reliability_statement=target_reliability,
        model_name=model_name,
        vendor=vendor,
        cpu_family=cpu_family,
        model=model,
        stepping=stepping,
        arch=arch,
        base_mhz=base_mhz,
        max_mhz=max_mhz,
        uarch_host_level=uarch,
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        threads_per_core=threads_per_core,
        sockets=sockets,
        smt_enabled=smt_enabled,
        hybrid=hybrid,
        numa_nodes=numa_nodes,
        cpu_to_node_map=cpu_to_node,
        cache=cache,
        exec_env=exec_env,
        isa=isa,
        parser_negative_control=parser_nc,
        amx_runtime_probe=amx_probe,
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
        usable_ram_from_memavailable_bytes=usable_from_avail,
        usable_ram_warning=usable_warning,
        headroom_factor=effective_factor,
        cross_checks=cross_checks,
        core_count_sources=core_count_sources,
        self_critique=self_critique,
        per_core_type_counts=per_core_type_counts,
        lscpu_unique_cores=lscpu_unique_core_count,
    )
