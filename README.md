# Local-model

Phase 1 / 1.5b: **DeviceProfile** CPU / memory / sandbox / storage /
thermal detection only.

No benchmarks, model selection, download, or optimization code.

## Run

```bash
python3 -m device_profile
```

Report opens with `HOST CLASS: bare-metal|VM|container` and evidence.
If not bare-metal, a non-representative banner is printed.

## Design rules

- Every field is a raw read or `UNDETECTED (reason: ...)`.
- Capture-once with sha256; Section 0 and `<-` evidence share blobs.
- Downstream defaults use `effective_cores` / `effective_mem_bytes`.
- `measured_bandwidth_GBps` is `PENDING_PHASE_2` (not computed from DMI).
- Memory budget is a PROVISIONAL table over `n_ctx`, not a single scalar.
- `tier_runtime_verified=false` until a later runtime probe.
