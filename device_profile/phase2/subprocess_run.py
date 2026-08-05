"""Spawn Phase 2.4 bench/diag workers in a fresh process (clean libgomp init)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any


MARKER = "PHASE24_WORKER_JSON:"


def run_worker(
    cfg: dict[str, Any],
    *,
    module: str = "device_profile.phase2.bench_worker",
    timeout_s: float | None = None,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        cfg_path = fh.name
    try:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module, cfg_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "INVALID",
                "thread_flag": "THREAD COUNT NOT APPLIED",
                "error": f"worker timeout after {timeout_s}s: {exc}",
                "pid": None,
                "omp_num_threads_actual": None,
                "requested_threads": cfg.get("requested_threads"),
                "thread_count_applied": False,
            }
        out = proc.stdout or ""
        err = proc.stderr or ""
        payload = None
        for line in out.splitlines():
            if line.startswith(MARKER):
                payload = json.loads(line[len(MARKER) :])
                break
        if payload is None:
            return {
                "status": "INVALID",
                "thread_flag": "THREAD COUNT NOT APPLIED",
                "error": (
                    f"worker failed rc={proc.returncode} "
                    f"err={err[:2000]} out={out[:2000]}"
                ),
                "pid": None,
                "omp_num_threads_actual": None,
                "requested_threads": cfg.get("requested_threads"),
                "thread_count_applied": False,
            }
        return payload
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def assert_no_phase2_overrides(*, allow_smoke: bool) -> list[str]:
    """Deliverable runs must not set PHASE2_* overrides."""
    keys = [
        k
        for k in os.environ
        if k.startswith("PHASE2_") and k != "PHASE2_ALLOW_OVERRIDES"
    ]
    if not keys:
        return []
    if allow_smoke and os.environ.get("PHASE2_ALLOW_OVERRIDES") == "1":
        return [
            f"SMOKE_NOT_DELIVERABLE: PHASE2 overrides present: {keys} "
            f"(PHASE2_ALLOW_OVERRIDES=1)"
        ]
    raise RuntimeError(
        "Deliverable run refuses PHASE2_* overrides: "
        f"{keys}. Unset them, or use --smoke with PHASE2_ALLOW_OVERRIDES=1 "
        "(not a deliverable)."
    )
