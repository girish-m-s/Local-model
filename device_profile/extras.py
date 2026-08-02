"""Phase 1.5b extras: host class, storage, thermal/power, memory budget."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import EvidenceField, undetected


@dataclass
class HostClassInfo:
    host_class: EvidenceField  # bare-metal | VM | container
    banner: str
    evidence_lines: list[str] = field(default_factory=list)


@dataclass
class StorageInfo:
    cache_dir: EvidenceField
    total_bytes: EvidenceField
    free_bytes: EvidenceField
    filesystem_type: EvidenceField
    rotational: EvidenceField
    sparse_mmap_note: EvidenceField


@dataclass
class ThermalPowerInfo:
    on_ac_power: EvidenceField
    battery_present: EvidenceField
    battery_capacity: EvidenceField
    thermal_zones: EvidenceField
    cpufreq_governors: EvidenceField
    thermal_throttling: EvidenceField
    platform_notes: str = ""


@dataclass
class BudgetRow:
    n_ctx: int
    os_reserve_bytes: int
    runtime_overhead_bytes: int
    kv_budget_bytes: int
    weight_budget_bytes: int
    assumptions: str


@dataclass
class MemoryBudget:
    os_reserve_bytes: EvidenceField
    runtime_overhead_bytes: EvidenceField
    kv_formula: str
    rows: list[BudgetRow]
    provisional_note: str
    effective_mem_bytes: int


NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb",
    "smb3",
    "fuse.sshfs",
    "fuse.rclone",
    "glusterfs",
    "ceph",
    "afs",
}


def classify_host(
    *,
    dockerenv_exists: bool,
    containerenv_exists: bool,
    systemd_virt: Optional[str],
    hypervisor_vendor: Optional[str],
    proc1_cgroup: str,
    dmi_sys_vendor: Optional[str],
    dmi_product_name: Optional[str],
) -> HostClassInfo:
    evidence: list[str] = []
    evidence.append(
        f"/.dockerenv: {'EXISTS' if dockerenv_exists else 'ABSENT'}"
    )
    evidence.append(
        f"/run/.containerenv: {'EXISTS' if containerenv_exists else 'ABSENT'}"
    )
    evidence.append(f"/proc/1/cgroup: {proc1_cgroup!r}")
    evidence.append(
        f"systemd-detect-virt: {systemd_virt if systemd_virt is not None else 'UNDETECTED'}"
    )
    evidence.append(
        f"lscpu Hypervisor vendor: {hypervisor_vendor if hypervisor_vendor else 'UNDETECTED'}"
    )
    evidence.append(
        f"DMI sys_vendor: {dmi_sys_vendor if dmi_sys_vendor else 'UNDETECTED'}"
    )
    evidence.append(
        f"DMI product_name: {dmi_product_name if dmi_product_name else 'UNDETECTED'}"
    )

    virt = (systemd_virt or "").strip().lower()
    container_virts = {
        "docker",
        "podman",
        "container",
        "lxc",
        "lxc-libvirt",
        "systemd-nspawn",
        "rkt",
        "wsl",
    }
    vm_virts = {
        "kvm",
        "qemu",
        "vmware",
        "microsoft",
        "oracle",
        "xen",
        "bochs",
        "uml",
        "parallels",
        "bhyve",
        "qnx",
        "acrn",
        "powerkvm",
    }

    is_container = (
        dockerenv_exists
        or containerenv_exists
        or virt in container_virts
        or ("docker" in proc1_cgroup.lower())
        or ("kubepods" in proc1_cgroup.lower())
        or ("containerd" in proc1_cgroup.lower())
    )
    is_vm = (
        (hypervisor_vendor is not None and hypervisor_vendor.strip() != "")
        or virt in vm_virts
        or (
            dmi_sys_vendor is not None
            and any(
                x in dmi_sys_vendor.lower()
                for x in ("qemu", "vmware", "virtualbox", "xen", "microsoft corporation")
            )
        )
    )

    if is_container:
        host_class = "container"
        # Still note underlying VM if present
        detail = "container"
        if is_vm or virt in vm_virts or hypervisor_vendor:
            detail = f"container (on VM/hypervisor={hypervisor_vendor or virt or 'unknown'})"
        banner = (
            f"*** HOST CLASS = {host_class.upper()} — PROFILE NOT REPRESENTATIVE "
            f"OF AN EDGE TARGET *** ({detail}). Do not use host totals as "
            "effective capacity; cgroup/affinity limits apply."
        )
    elif is_vm:
        host_class = "VM"
        banner = (
            f"*** HOST CLASS = VM — PROFILE NOT REPRESENTATIVE OF AN EDGE "
            f"TARGET *** (hypervisor={hypervisor_vendor or virt or 'unknown'}). "
            "Guest-visible CPUID/cache/DIMM figures may be synthetic."
        )
    else:
        host_class = "bare-metal"
        banner = ""

    return HostClassInfo(
        host_class=EvidenceField(
            value=host_class,
            evidence="; ".join(evidence),
        ),
        banner=banner,
        evidence_lines=evidence,
    )


def detect_storage(
    store_read,
    store_text,
    store_sha,
    store_get,
    cache_dir: Optional[str] = None,
) -> StorageInfo:
    """
    store_read(key, path) / store helpers from CaptureStore-like API.
    We accept callables to stay capture-once friendly when wired from linux.py.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/localmodel")

    # Ensure parent exists for statvfs of the intended location's mount;
    # do NOT create the cache dir itself if missing — stat the parent.
    probe_path = cache_dir
    created_note = ""
    if not os.path.exists(probe_path):
        parent = os.path.dirname(probe_path) or os.path.expanduser("~")
        if not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
                created_note = f" (created parent {parent} for probe)"
            except OSError as exc:
                return StorageInfo(
                    cache_dir=EvidenceField(
                        value=cache_dir,
                        evidence=f"intended path {cache_dir}",
                    ),
                    total_bytes=undetected(f"cannot create/probe parent: {exc}"),
                    free_bytes=undetected(f"cannot create/probe parent: {exc}"),
                    filesystem_type=undetected("path unreachable"),
                    rotational=undetected("path unreachable"),
                    sparse_mmap_note=undetected("path unreachable"),
                )
        probe_path = parent if not os.path.exists(cache_dir) else cache_dir

    cache_ef = EvidenceField(
        value=cache_dir,
        evidence=(
            f"intended model cache dir default ~/.cache/localmodel → {cache_dir}"
            f"{created_note}; disk_usage probed via {probe_path}"
        ),
    )

    try:
        usage = shutil.disk_usage(probe_path)
        total_ef = EvidenceField(
            value=usage.total,
            evidence=(
                f"shutil.disk_usage({probe_path!r}).total "
                f"(Python {platform.python_version()}) → {usage.total}"
            ),
        )
        free_ef = EvidenceField(
            value=usage.free,
            evidence=(
                f"shutil.disk_usage({probe_path!r}).free "
                f"(Python {platform.python_version()}) → {usage.free}"
            ),
        )
    except OSError as exc:
        total_ef = undetected(str(exc))
        free_ef = undetected(str(exc))

    # Filesystem type from /proc/mounts — longest matching mountpoint
    try:
        mounts = store_text("proc_mounts")
    except Exception:
        mounts = ""
    fstype = undetected("/proc/mounts unavailable")
    mount_src = None
    if mounts:
        best_mp = ""
        best_type = None
        best_line = None
        best_src = None
        abs_probe = os.path.abspath(probe_path)
        for line in mounts.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            src, mp, ft = parts[0], parts[1], parts[2]
            if abs_probe == mp or abs_probe.startswith(
                mp.rstrip("/") + "/" if mp != "/" else "/"
            ):
                if len(mp) >= len(best_mp):
                    best_mp, best_type, best_line, best_src = mp, ft, line, src
        if best_type:
            fstype = EvidenceField(
                value=best_type,
                evidence=(
                    f"/proc/mounts match mountpoint={best_mp}: {best_line} "
                    f"[sha256={store_sha('proc_mounts')}]"
                ),
            )
            mount_src = best_src
        else:
            fstype = undetected(
                f"no /proc/mounts entry covering {abs_probe}"
            )

    # Rotational: resolve block device
    rotational = undetected(
        "could not map mount source to /sys/block/*/queue/rotational"
    )
    if mount_src:
        # e.g. /dev/vda1, /dev/mapper/..., overlay → no block device
        dev = mount_src
        if dev.startswith("/dev/"):
            base = os.path.basename(dev)
            # strip partition digits: vda1 -> vda, nvme0n1p2 -> nvme0n1
            candidates = [base]
            stripped = base.rstrip("0123456789")
            if stripped != base:
                candidates.append(stripped)
            if "nvme" in base and "p" in base:
                candidates.append(base.rsplit("p", 1)[0])
            for cand in candidates:
                path = f"/sys/block/{cand}/queue/rotational"
                # use store_read if key not yet present — caller should pre-capture
                # Fall back to direct read with evidence note if needed
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        val = fh.read().strip()
                    rotational = EvidenceField(
                        value=(val == "1"),
                        evidence=(
                            f"{path}: {val} (1=HDD/rotational, 0=non-rotational); "
                            f"mount source={mount_src}"
                        ),
                    )
                    break
                except OSError:
                    continue
        elif fstype.is_detected() and str(fstype.value) == "overlay":
            rotational = EvidenceField(
                value="UNDETECTED (reason: overlay mount; no single block device)",
                evidence=(
                    f"mount source={mount_src} fstype=overlay — "
                    "rotational attribute not attributable to one device"
                ),
            )

    # Sparse / mmap notes
    ft_val = str(fstype.value) if fstype.is_detected() else ""
    if ft_val.lower() in NETWORK_FS or ft_val.lower().startswith("fuse"):
        sparse_note = EvidenceField(
            value="NETWORK_OR_FUSE_FS — sparse/mmap may be unreliable",
            evidence=(
                f"fstype={ft_val} classified as network/fuse; "
                "sparse files and mmap may not behave like local disk"
            ),
        )
    elif ft_val in ("overlay", "overlayfs"):
        sparse_note = EvidenceField(
            value="OVERLAY — sparse usually OK via upperdir; mmap OK; not a network FS",
            evidence=f"fstype={ft_val} from /proc/mounts",
        )
    elif fstype.is_detected():
        sparse_note = EvidenceField(
            value="LOCAL_FS_ASSUMED — sparse+mmap typically sane on ext4/xfs/btrfs/apfs",
            evidence=f"fstype={ft_val}; not in network-FS set",
        )
    else:
        sparse_note = undetected("filesystem type unknown")

    return StorageInfo(
        cache_dir=cache_ef,
        total_bytes=total_ef,
        free_bytes=free_ef,
        filesystem_type=fstype,
        rotational=rotational,
        sparse_mmap_note=sparse_note,
    )


def detect_thermal_power(store_text_fn, list_dir_fn=os.listdir) -> ThermalPowerInfo:
    """Linux thermal/power via sysfs. Notes for macOS/Android when not Linux."""
    system = platform.system().lower()
    if system == "darwin":
        return ThermalPowerInfo(
            on_ac_power=undetected(
                "macOS path: run `pmset -g batt` (not executed on this Linux host)"
            ),
            battery_present=undetected("macOS: pmset -g batt not run here"),
            battery_capacity=undetected("macOS: pmset -g batt not run here"),
            thermal_zones=undetected(
                "macOS: powermetrics often needs sudo; availability not probed here"
            ),
            cpufreq_governors=undetected(
                "macOS has no sysfs cpufreq; use powermetrics/sysctl (not probed)"
            ),
            thermal_throttling=undetected("macOS thermal indicators not probed"),
            platform_notes=(
                "macOS: `pmset -g batt` for AC/battery; `powermetrics` for thermal "
                "often requires sudo. Android/Termux: /sys/class/power_supply and "
                "thermal_zone may be readable unprivileged but are OEM-dependent; "
                "cpufreq governors often inaccessible without root."
            ),
        )

    # power_supply
    ps_root = "/sys/class/power_supply"
    on_ac = undetected(f"{ps_root} absent or empty")
    bat_present = EvidenceField(
        value=False,
        evidence=f"no BAT* under {ps_root}",
    )
    bat_cap = undetected("no BAT* capacity")
    try:
        entries = sorted(list_dir_fn(ps_root))
    except OSError as exc:
        entries = []
        on_ac = undetected(f"cannot list {ps_root}: {exc}")

    ac_evid = []
    bat_names = []
    for name in entries:
        base = os.path.join(ps_root, name)
        typ_p = os.path.join(base, "type")
        online_p = os.path.join(base, "online")
        typ = None
        try:
            with open(typ_p, "r", encoding="utf-8") as fh:
                typ = fh.read().strip()
        except OSError:
            pass
        if typ in ("Mains", "UPS", "USB") or name.startswith(
            ("AC", "ADP", "ACAD", "Mains")
        ):
            try:
                with open(online_p, "r", encoding="utf-8") as fh:
                    online = fh.read().strip()
                ac_evid.append(f"{online_p}: {online} (type={typ})")
            except OSError as exc:
                ac_evid.append(f"{online_p}: unreadable ({exc})")
        if name.startswith("BAT") or typ == "Battery":
            bat_names.append(name)
            cap_p = os.path.join(base, "capacity")
            try:
                with open(cap_p, "r", encoding="utf-8") as fh:
                    cap = fh.read().strip()
                bat_present = EvidenceField(
                    value=True,
                    evidence=f"{base} present type={typ}",
                )
                bat_cap = EvidenceField(
                    value=cap,
                    evidence=f"{cap_p}: {cap}",
                )
            except OSError as exc:
                bat_present = EvidenceField(
                    value=True,
                    evidence=f"{base} present but capacity unreadable: {exc}",
                )

    if ac_evid:
        # online==1 means on AC for Mains
        on = any(e.endswith(": 1") or ": 1 " in e or e.endswith(": 1 (type=Mains)") or ": 1 (type=" in e for e in ac_evid)
        # parse more carefully
        on = any(
            line.split(": ", 1)[-1].startswith("1") for line in ac_evid if ":" in line
        )
        on_ac = EvidenceField(
            value=on,
            evidence="; ".join(ac_evid),
        )
    elif entries == [] and isinstance(on_ac.value, str) and on_ac.value.startswith("UNDETECTED"):
        pass
    elif entries:
        on_ac = EvidenceField(
            value="UNDETECTED (reason: no Mains/AC* online node found)",
            evidence=f"{ps_root} entries={entries}; no AC online readable",
        )

    # thermal zones
    tz_root = "/sys/class/thermal"
    zones = []
    tz_evid = []
    try:
        tz_entries = sorted(
            e for e in list_dir_fn(tz_root) if e.startswith("thermal_zone")
        )
    except OSError as exc:
        tz_entries = []
        tz_evid.append(f"cannot list {tz_root}: {exc}")
    for z in tz_entries:
        type_p = os.path.join(tz_root, z, "type")
        temp_p = os.path.join(tz_root, z, "temp")
        try:
            with open(type_p, "r", encoding="utf-8") as fh:
                ztype = fh.read().strip()
            with open(temp_p, "r", encoding="utf-8") as fh:
                temp_raw = fh.read().strip()
            # temp is millidegree C on Linux
            try:
                temp_c = int(temp_raw) / 1000.0
            except ValueError:
                temp_c = temp_raw
            zones.append({"zone": z, "type": ztype, "temp_C": temp_c})
            tz_evid.append(f"{type_p}={ztype}; {temp_p}={temp_raw}")
        except OSError as exc:
            tz_evid.append(f"{z}: unreadable ({exc})")
    if zones:
        thermal_zones = EvidenceField(value=zones, evidence="; ".join(tz_evid))
    else:
        thermal_zones = EvidenceField(
            value="UNDETECTED (reason: no readable thermal_zone*)",
            evidence="; ".join(tz_evid) if tz_evid else f"{tz_root} empty/absent",
        )

    # cpufreq governors
    gov_root = "/sys/devices/system/cpu/cpufreq"
    govs = {}
    gov_evid = []
    try:
        policies = sorted(
            e for e in list_dir_fn(gov_root) if e.startswith("policy")
        )
    except OSError as exc:
        policies = []
        gov_evid.append(f"cannot list {gov_root}: {exc}")
    for pol in policies:
        gpath = os.path.join(gov_root, pol, "scaling_governor")
        try:
            with open(gpath, "r", encoding="utf-8") as fh:
                g = fh.read().strip()
            govs[pol] = g
            gov_evid.append(f"{gpath}: {g}")
        except OSError as exc:
            gov_evid.append(f"{gpath}: unreadable ({exc})")
    if govs:
        governors = EvidenceField(value=govs, evidence="; ".join(gov_evid))
    else:
        governors = EvidenceField(
            value="UNDETECTED (reason: no cpufreq policy*/scaling_governor)",
            evidence="; ".join(gov_evid)
            if gov_evid
            else f"{gov_root} absent (common under KVM/container)",
        )

    # thermal throttling indicators
    throttle_bits = []
    # package thermal / intel_powerclamp / cpufreq throttle count
    for cpu in range(0, 8):
        path = (
            f"/sys/devices/system/cpu/cpu{cpu}/thermal_throttle/"
            f"core_throttle_count"
        )
        try:
            with open(path, "r", encoding="utf-8") as fh:
                val = fh.read().strip()
            throttle_bits.append(f"{path}: {val}")
        except OSError:
            break
    if throttle_bits:
        throttling = EvidenceField(
            value=throttle_bits,
            evidence="; ".join(throttle_bits),
        )
    else:
        throttling = EvidenceField(
            value="UNDETECTED (reason: no thermal_throttle sysfs nodes)",
            evidence=(
                "/sys/devices/system/cpu/cpu*/thermal_throttle/"
                "core_throttle_count absent"
            ),
        )

    return ThermalPowerInfo(
        on_ac_power=on_ac,
        battery_present=bat_present,
        battery_capacity=bat_cap,
        thermal_zones=thermal_zones,
        cpufreq_governors=governors,
        thermal_throttling=throttling,
        platform_notes=(
            "Linux sysfs path exercised. macOS: pmset -g batt; powermetrics "
            "(often sudo). Android/Termux: power_supply + thermal_zone often "
            "readable unprivileged; cpufreq governors frequently blocked."
        ),
    )


def build_memory_budget(effective_mem_bytes: int) -> MemoryBudget:
    """
    PROVISIONAL budget table. Assumptions stated explicitly.

    KV cache budget for a stand-in 7B-class GQA model:
      n_layers=32, n_kv_heads=8, head_dim=128, elem_bytes=2 (fp16/bf16),
      K+V → factor 2
      kv_budget(n_ctx) = 2 * 32 * 8 * 128 * 2 * n_ctx
                      = 131072 * n_ctx bytes
    """
    # OS reserve: 1 GiB absolute — covers kernel page cache pressure, sshd,
    # agent runtime, and avoids sitting on the OOM knife-edge in a container.
    os_reserve = 1 * 1024**3
    # Runtime overhead: 512 MiB — process RSS for loader, Python/native runtime,
    # tokenizer, scratch buffers before model weights.
    runtime_overhead = 512 * 1024**2

    kv_per_ctx = 2 * 32 * 8 * 128 * 2  # = 131072
    kv_formula = (
        "kv_budget(n_ctx) = 2 * n_layers * n_kv_heads * head_dim * elem_bytes * n_ctx "
        f"= 2*32*8*128*2 * n_ctx = {kv_per_ctx} * n_ctx  "
        "(PROVISIONAL stand-in for a 7B-class GQA model; not measured)"
    )

    rows = []
    for n_ctx in (2048, 4096, 8192, 32768):
        kv = kv_per_ctx * n_ctx
        weight = effective_mem_bytes - os_reserve - runtime_overhead - kv
        rows.append(
            BudgetRow(
                n_ctx=n_ctx,
                os_reserve_bytes=os_reserve,
                runtime_overhead_bytes=runtime_overhead,
                kv_budget_bytes=kv,
                weight_budget_bytes=weight,
                assumptions=(
                    f"effective_mem={effective_mem_bytes}; "
                    f"os_reserve={os_reserve} (1 GiB absolute); "
                    f"runtime_overhead={runtime_overhead} (512 MiB); "
                    f"kv={kv} via {kv_per_ctx}*n_ctx"
                ),
            )
        )

    return MemoryBudget(
        os_reserve_bytes=EvidenceField(
            value=os_reserve,
            evidence=(
                "PROVISIONAL policy: 1 GiB absolute OS reserve — justified as "
                "headroom for kernel/page-cache/agent so model load does not "
                "sit at cgroup OOM edge. NOT measured."
            ),
        ),
        runtime_overhead_bytes=EvidenceField(
            value=runtime_overhead,
            evidence=(
                "PROVISIONAL policy: 512 MiB runtime overhead — loader, "
                "tokenizer, native allocator arenas, scratch. NOT measured."
            ),
        ),
        kv_formula=kv_formula,
        rows=rows,
        provisional_note=(
            "PROVISIONAL — entire budget table will be REPLACED by measured "
            "values in a later phase. CPUID/topology do not validate these numbers."
        ),
        effective_mem_bytes=effective_mem_bytes,
    )
