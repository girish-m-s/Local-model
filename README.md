# Local-model

Phase 1: **DeviceProfile** CPU / memory detection only.

No benchmarks, model selection, download, or optimization code.

## Run

```bash
python3 -m device_profile
```

Prints a verifiable report (sections 0–8) to stdout: raw source dump,
parsed identity/topology/ISA/memory fields with `<-` evidence lines,
cross-checks, provisional derived defaults, self-critique, and JSON.

## Design rules

- Every field is either a value actually read from a raw source, or
  `UNDETECTED (reason: ...)`. No silent defaults.
- Prefer `/proc`, sysfs, and `lscpu` over third-party libraries.
- macOS / Windows paths are stubbed as UNDETECTED in this phase.
