"""Compile and load Phase 2 C kernels (OpenMP + pure-serial) at runtime."""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NativeKernels:
    lib: ctypes.CDLL
    lib_path: str
    compile_cmd: str
    compile_log: str
    omp: bool


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
    fn = lib.stream_triad_reps_pure_serial
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    fn.restype = None
    return NativeKernels(
        lib=lib,
        lib_path=str(so),
        compile_cmd=" ".join(cmd),
        compile_log=log.strip() or "(empty)",
        omp=False,
    )


def _bind_omp_symbols(lib: ctypes.CDLL) -> None:
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
