"""Jittor custom-op binding for the source CUDA ``distCUDA2`` operator.

The numerical kernel is the framework-independent implementation shipped in
``submodules/simple-knn/simple_knn.cu``.  Jittor compiles and invokes it as a
custom op; the Torch wrapper files in that directory are never imported or
compiled by the production pipeline.
"""

from __future__ import annotations

from pathlib import Path
import os

import jittor as jt


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "submodules" / "simple-knn" / "simple_knn.cu"
CUDA_DECLARATION = ROOT / "submodules" / "simple-knn" / "simple_knn.h"
OP_ABI_VERSION = 2
CUSTOM_OP_NAME = "simple_knn_v2"
OP_HEADER = Path(__file__).resolve().with_name("simple_knn_v2_op.h")
OP_SOURCE = Path(__file__).resolve().with_name("simple_knn_v2_op.cc")
CUDA_INCLUDE_COMPAT = Path(__file__).resolve().with_name("cuda_include_compat")
_CUSTOM_OPS = None


def _custom_ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is None:
        cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
        cuda_include = cuda_home / "include"
        if not cuda_include.is_dir():
            raise FileNotFoundError(f"CUDA include directory is missing: {cuda_include}")
        # Jittor 1.3.11 resolves quoted includes before invoking nvcc.  Its
        # scanner therefore needs the two CUDA subdirectories that contain
        # quote-included implementation headers.  Do not enumerate every CUDA
        # subdirectory: adding nested cuda/std directories directly changes
        # nvcc's normal include precedence and breaks the toolkit headers.
        include_dirs = [
            CUDA_SOURCE.parent.resolve(),
            CUDA_INCLUDE_COMPAT.resolve(),
            cuda_include.resolve(),
            (cuda_include / "nv").resolve(),
            (cuda_include / "crt").resolve(),
        ]
        include_flag = " ".join(
            f'-I"{path.as_posix()}"' for path in dict.fromkeys(include_dirs))
        # The upstream Torch extension receives FLT_MAX transitively from the
        # framework headers.  The framework-independent source uses it
        # directly; map it to the compiler's standard floating-point limit
        # builtin without injecting a framework header into the CUDA source.
        compile_flags = f'{include_flag} -DFLT_MAX=__FLT_MAX__'
        _CUSTOM_OPS = jt.compile_custom_ops(
            [
                str(OP_HEADER),
                str(OP_SOURCE),
                str(CUDA_DECLARATION),
                str(CUDA_SOURCE),
            ],
            extra_flags=compile_flags,
            gen_name_="gen_ops_gang_simple_knn_v2",
        )
    return _CUSTOM_OPS


def distCUDA2(points: jt.Var) -> jt.Var:
    """Return the source operator's mean squared distance to three neighbours."""
    if not isinstance(points, jt.Var):
        raise TypeError("distCUDA2 expects a Jittor Var")
    if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
        raise ValueError(f"distCUDA2 expects [N,3], got {tuple(points.shape)}")
    if points.dtype != jt.float32:
        raise ValueError(f"distCUDA2 expects float32, got {points.dtype}")
    count = int(points.shape[0])
    if count < 4:
        raise ValueError("distCUDA2 requires at least four points")

    return _custom_ops().simple_knn_v2(points.contiguous())


__all__ = [
    "CUDA_DECLARATION", "CUDA_INCLUDE_COMPAT", "CUDA_SOURCE",
    "CUSTOM_OP_NAME", "OP_ABI_VERSION", "OP_HEADER", "OP_SOURCE", "distCUDA2",
]
