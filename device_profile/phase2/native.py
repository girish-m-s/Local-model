"""Compile and load Phase 2 C kernels (OpenMP + pure-serial) at runtime."""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class NativeKernels:
    lib: ctypes.CDLL
    lib_path: str
    compile_cmd: str
    compile_log: str
    omp: bool


@dataclass
class ThreadObservation:
    omp_num_threads_actual: int
    omp_max_threads: int
    n_distinct_cpus: int
    cpu_ids: list[int]
    requested_threads: Optional[int]
    env_omp_num_threads: Optional[str]
    thread_count_applied: bool
    flag: str  # OK | THREAD COUNT NOT APPLIED | SERIAL_NO_OMP


def build_kernels() -> NativeKernels:
    src = Path(__file__).with_name("kernels.c")
    tmp = Path(tempfile.mkdtemp(prefix="phase2_kernels_"))
    so = tmp / "kernels.so"
    cmd = [
        "gcc",
        "-O3",
        "-fPIC",
        "-shared",
        "-fopenmp",
        str(src),
        "-o",
        str(so),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or not so.exists():
        cmd2 = ["gcc", "-O3", "-fPIC", "-shared", str(src), "-o", str(so)]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=False)
        log += "\n[retry without -fopenmp]\n" + (proc2.stdout or "") + (proc2.stderr or "")
        if proc2.returncode != 0 or not so.exists():
            raise RuntimeError(f"kernel compile failed: {log}")
        omp = False
        cmd = cmd2
    else:
        omp = True

    lib = ctypes.CDLL(str(so))
    _bind_omp_symbols(lib)
    return NativeKernels(
        lib=lib,
        lib_path=str(so),
        compile_cmd=" ".join(cmd),
        compile_log=log.strip() or "(empty)",
        omp=omp,
    )


def build_serial_kernels() -> NativeKernels:
    """Pure-serial lib: compiled WITHOUT -fopenmp (DIAG A ground truth)."""
    src = Path(__file__).with_name("kernels_serial.c")
    tmp = Path(tempfile.mkdtemp(prefix="phase2_serial_kernels_"))
    so = tmp / "kernels_serial.so"
    cmd = ["gcc", "-O3", "-fPIC", "-shared", str(src), "-o", str(so)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or not so.exists():
        raise RuntimeError(f"serial kernel compile failed: {log}")
    lib = ctypes.CDLL(str(so))
    for name in (
        "stream_triad_reps_pure_serial",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        fn.restype = None
    _bind_observed(lib, "stream_triad_reps_pure_serial_observed")
    return NativeKernels(
        lib=lib,
        lib_path=str(so),
        compile_cmd=" ".join(cmd),
        compile_log=log.strip() or "(empty)",
        omp=False,
    )


def omp_set_num_threads(kernels: NativeKernels, n: int) -> str:
    """Call omp_set_num_threads via the loaded OpenMP-linked .so (not os.environ)."""
    if not hasattr(kernels.lib, "phase2_omp_set_num_threads"):
        return "UNDETECTED (reason: phase2_omp_set_num_threads missing)"
    kernels.lib.phase2_omp_set_num_threads(int(n))
    return f"phase2_omp_set_num_threads({n}) via {kernels.lib_path}"


def run_observed_triad(
    kernels: NativeKernels,
    *,
    serial: bool,
    ap,
    bp,
    cp,
    scalar: float,
    n: int,
    reps: int,
    requested_threads: Optional[int],
) -> ThreadObservation:
    import os

    out_num = ctypes.c_int(-1)
    out_max = ctypes.c_int(-1)
    out_nd = ctypes.c_int(0)
    cpu_cap = 128
    cpu_arr = (ctypes.c_int * cpu_cap)()

    if serial:
        kernels.lib.stream_triad_reps_pure_serial_observed(
            ap,
            bp,
            cp,
            float(scalar),
            int(n),
            int(reps),
            ctypes.byref(out_num),
            ctypes.byref(out_max),
            ctypes.byref(out_nd),
            cpu_arr,
            cpu_cap,
        )
        actual = int(out_num.value)
        applied = True
        flag = "SERIAL_NO_OMP"
    else:
        if requested_threads is not None:
            omp_set_num_threads(kernels, requested_threads)
        kernels.lib.stream_triad_reps_observed(
            ap,
            bp,
            cp,
            float(scalar),
            int(n),
            int(reps),
            ctypes.byref(out_num),
            ctypes.byref(out_max),
            ctypes.byref(out_nd),
            cpu_arr,
            cpu_cap,
        )
        actual = int(out_num.value)
        if requested_threads is None:
            applied = True
            flag = "OK"
        elif actual == int(requested_threads):
            applied = True
            flag = "OK"
        else:
            applied = False
            flag = "THREAD COUNT NOT APPLIED"

    nd = int(out_nd.value)
    cpus = [int(cpu_arr[i]) for i in range(min(nd, cpu_cap))]
    return ThreadObservation(
        omp_num_threads_actual=actual,
        omp_max_threads=int(out_max.value),
        n_distinct_cpus=nd,
        cpu_ids=cpus,
        requested_threads=requested_threads,
        env_omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        thread_count_applied=applied,
        flag=flag,
    )


def _bind_observed(lib: ctypes.CDLL, name: str) -> None:
    fn = getattr(lib, name)
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    fn.restype = None


def _bind_omp_symbols(lib: ctypes.CDLL) -> None:
    if hasattr(lib, "phase2_omp_set_num_threads"):
        lib.phase2_omp_set_num_threads.argtypes = [ctypes.c_int]
        lib.phase2_omp_set_num_threads.restype = None
    if hasattr(lib, "phase2_omp_get_max_threads"):
        lib.phase2_omp_get_max_threads.argtypes = []
        lib.phase2_omp_get_max_threads.restype = ctypes.c_int

    for name in (
        "stream_triad",
        "stream_triad_serial",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_size_t,
        ]
        fn.restype = None

    for name in (
        "stream_triad_reps",
        "stream_triad_reps_serial",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        fn.restype = None

    _bind_observed(lib, "stream_triad_reps_observed")

    for name in ("dot_fp32_reps", "dot_fp32_reps_serial"):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        fn.restype = ctypes.c_double

    for name in ("dot_i8_reps", "dot_i8_reps_serial"):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        fn.restype = ctypes.c_int64

    lib.stride_touch.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.stride_touch.restype = ctypes.c_size_t

    lib.stride_touch_addr.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.stride_touch_addr.restype = ctypes.c_size_t


def numpy_available() -> tuple[bool, str]:
    try:
        import numpy as np

        return True, f"numpy {np.__version__}"
    except ImportError:
        return False, "numpy not installed"
