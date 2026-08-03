# Local-model

Phase 1 / 1.5b: **DeviceProfile** CPU / memory / sandbox / storage /
thermal detection.

Phase 2 / 2.1: synthetic measurement harness (STREAM / compute / thermal /
mmap). No model selection, download, or inference engine.

## Run

```bash
# Phase 1 / 1.5 — detection
python3 -m device_profile

# Phase 2.1 — synthetic measurement harness (no model downloads)
python3 -m device_profile.phase2
```

Report opens with `HOST CLASS: bare-metal|VM|container` and evidence.
If not bare-metal, a non-representative banner is printed.
Phase 2.1 measures STREAM bandwidth, compute thread knee, 180s sustained
DRAM-bound thermal, and mmap fault cost; total runtime budget 10 minutes.

**Phase 2.1 deliverable requires bare-metal x86 (AC + battery) and one ARM
host.** A container-only run is not an acceptable calibration target.

## Design rules

- Every field is a raw read or `UNDETECTED (reason: ...)`.
- Capture-once with sha256; Section 0 and `<-` evidence share blobs.
- Downstream defaults use `effective_cores` / `effective_mem_bytes`.
- Prediction table uses `model_GiB` with `2^30` bytes.
- Memory budget is a PROVISIONAL table over `n_ctx`, not a single scalar.
- `tier_runtime_verified=false` until a later runtime probe.
- If `loadavg(1m) > 0.3` at bench start: sleep/recheck up to 5×, then
  `BLOCKED` (do not run and label).
