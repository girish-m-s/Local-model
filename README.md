# Local-model

Phase 1 / 1.5: **DeviceProfile** CPU / memory / sandbox detection only.

No benchmarks, model selection, download, or optimization code.

## Run

```bash
python3 -m device_profile
```

Prints a verifiable report (sections 0–9) to stdout: capture-once raw
dump with sha256, identity/topology/ISA/memory with `<-` evidence,
execution environment (cgroup/affinity/steal), parser negative control,
AMX runtime probe (`cpuid_tier` vs `usable_tier`), cross-checks,
effective_*–based provisional defaults, self-critique, JSON, and target
reliability statement.

## Design rules

- Every field is either a value actually read from a raw source, or
  `UNDETECTED (reason: ...)`. No silent defaults.
- Each raw command/file is captured exactly once; Section 0 and evidence
  share the same blob (sha256 printed).
- Downstream defaults use `effective_memory_limit` and
  `effective_cpu_count`, not raw `/proc` alone.
- Prefer `/proc`, sysfs, and `lscpu` over third-party libraries
  (`ctypes` only for AMX `prctl` probe).
- macOS / Windows paths are stubbed as UNDETECTED in this phase.
