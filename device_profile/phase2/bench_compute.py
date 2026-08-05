"""BENCH 2 — DELETED in Phase 2.4.

Kernels compile to SSE2 (vector-xmm; no ymm/zmm; no -march), so they cannot
exercise the detected AVX512_VNNI tier. A hand-rolled dot product also does not
predict llama.cpp's tuned VNNI GEMM. Compute throughput moves to Phase 4 against
a real engine. recommended_thread_count is derived from BENCH 1's bandwidth curve.
"""

from __future__ import annotations

from typing import Any

from .native import NativeKernels


def run_bench2(profile: Any, kernels: NativeKernels | None = None) -> dict[str, Any]:
    _ = (profile, kernels)
    return {
        "name": "BENCH 2 — DELETED (Phase 2.4)",
        "status": "DELETED",
        "reason": (
            "SSE2-only kernels cannot exercise AVX512_VNNI; hand-rolled dot product "
            "does not predict engine GEMM. See Phase 4."
        ),
    }
