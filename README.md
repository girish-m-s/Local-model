# Local-model

Phase 1 / 1.5b: **DeviceProfile** CPU / memory / sandbox / storage /
thermal detection.

Phase 2: synthetic measurement harness (STREAM / compute / thermal /
mmap). No model selection, download, or inference engine.

## Run

```bash
# Phase 1 / 1.5 — detection
python3 -m device_profile

# Phase 2 — synthetic measurement harness (no model downloads)
python3 -m device_profile.phase2
```

Report opens with `HOST CLASS: bare-metal|VM|container` and evidence.
If not bare-metal, a non-representative banner is printed.
Phase 2 measures STREAM bandwidth, compute thread knee, 180s sustained
thermal, and mmap fault cost; total runtime budget 10 minutes.

## Design rules

- Every field is a raw read or `UNDETECTED (reason: ...)`.
- Capture-once with sha256; Section 0 and `<-` evidence share blobs.
- Downstream defaults use `effective_cores` / `effective_mem_bytes`.
- `measured_bandwidth_GBps` is `PENDING_PHASE_2` until the Phase 2
  harness fills it (not computed from DMI).
- Memory budget is a PROVISIONAL table over `n_ctx`, not a single scalar.
- `tier_runtime_verified=false` until a later runtime probe.
