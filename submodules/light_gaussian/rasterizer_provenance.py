"""Build identity contract for the native Gaussian rasterizer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ABI_VERSION = 1
RESCUE_CHANNELS = ("opacity", "color", "depth", "normal", "extra")
RESCUE_VARIANTS = tuple(f"fp32_rescue_{channel}" for channel in RESCUE_CHANNELS)
MIXED_PRECISION_VARIANT = "mixed_precision_candidate"
MIXED_NORMAL_EXTRA_VARIANT = "mixed_precision_normal_extra_candidate"
MIXED_NORMAL_OPACITY_VARIANT = "mixed_precision_normal_opacity_candidate"
MIXED_FIXED_ORDER_VARIANT = "mixed_fixed_order_candidate"
MIXED_CONIC_FP64_VARIANT = "mixed_conic_fp64_candidate"
SHARED_DETERMINISTIC_FP32_VARIANT = "shared_deterministic_fp32"
VARIANTS = (
    "stable", "optimized_stable", "source_compatible", "fp32_candidate",
    "fp32_deterministic", *RESCUE_VARIANTS, MIXED_PRECISION_VARIANT,
    MIXED_NORMAL_EXTRA_VARIANT, MIXED_NORMAL_OPACITY_VARIANT,
    MIXED_FIXED_ORDER_VARIANT, MIXED_CONIC_FP64_VARIANT,
    SHARED_DETERMINISTIC_FP32_VARIANT)
PRODUCTION_VARIANT = "stable"
VARIANT_DEFINES = {
    "stable": "GANG_RASTER_VARIANT_STABLE",
    "optimized_stable": "GANG_RASTER_VARIANT_OPTIMIZED_STABLE",
    "source_compatible": "GANG_RASTER_VARIANT_SOURCE_COMPATIBLE",
    "fp32_candidate": "GANG_RASTER_VARIANT_FP32_CANDIDATE",
    "fp32_deterministic": "GANG_RASTER_VARIANT_FP32_DETERMINISTIC",
    MIXED_PRECISION_VARIANT: "GANG_RASTER_VARIANT_MIXED_PRECISION_CANDIDATE",
    MIXED_NORMAL_EXTRA_VARIANT:
        "GANG_RASTER_VARIANT_MIXED_PRECISION_NORMAL_EXTRA_CANDIDATE",
    MIXED_NORMAL_OPACITY_VARIANT:
        "GANG_RASTER_VARIANT_MIXED_PRECISION_NORMAL_OPACITY_CANDIDATE",
    MIXED_FIXED_ORDER_VARIANT:
        "GANG_RASTER_VARIANT_MIXED_FIXED_ORDER_CANDIDATE",
    MIXED_CONIC_FP64_VARIANT:
        "GANG_RASTER_VARIANT_MIXED_CONIC_FP64_CANDIDATE",
    SHARED_DETERMINISTIC_FP32_VARIANT:
        "GANG_RASTER_VARIANT_SHARED_DETERMINISTIC_FP32",
    **{
        variant: f"GANG_RASTER_VARIANT_{variant.upper()}"
        for variant in RESCUE_VARIANTS
    },
}
FORBIDDEN_VARIANT_DEFINES = {
    "candidate_a": "GANG_RASTER_VARIANT_CANDIDATE_A",
    "candidate_b": "GANG_RASTER_VARIANT_CANDIDATE_B",
}
CONTROLLED_SOURCES = (
    "cuda_rasterizer/backward.cu",
    "cuda_rasterizer/backward.h",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.cu",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
CONTROLLED_SOURCE_TREES = ("third_party/glm",)
COMPILE_UNITS = (
    "cuda_rasterizer/backward.cu",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/rasterizer_impl.cu",
)
INCLUDE_DIRECTORIES = ("cuda_rasterizer", "third_party/glm")
SOURCE_COMPATIBLE_GENERATED = "build/source_compatible/generated"
OPTIMIZED_STABLE_GENERATED = "build/optimized_stable/generated"
FP32_CANDIDATE_GENERATED = "build/fp32_candidate/generated"
FP32_DETERMINISTIC_GENERATED = "build/fp32_deterministic/generated"
FP32_RESCUE_GENERATED = {
    variant: f"build/{variant}/generated" for variant in RESCUE_VARIANTS
}
MIXED_PRECISION_GENERATED = "build/mixed_precision_candidate/generated"
MIXED_NORMAL_EXTRA_GENERATED = (
    "build/mixed_precision_normal_extra_candidate/generated")
MIXED_NORMAL_OPACITY_GENERATED = (
    "build/mixed_precision_normal_opacity_candidate/generated")
MIXED_FIXED_ORDER_GENERATED = "build/mixed_fixed_order_candidate/generated"
MIXED_CONIC_FP64_GENERATED = "build/mixed_conic_fp64_candidate/generated"
SHARED_DETERMINISTIC_FP32_GENERATED = (
    "build/shared_deterministic_fp32/generated")
VARIANT_COMPILE_UNITS = {
    "stable": COMPILE_UNITS,
    "optimized_stable": (
        f"{OPTIMIZED_STABLE_GENERATED}/backward.cu",
        "cuda_rasterizer/forward.cu",
        f"{OPTIMIZED_STABLE_GENERATED}/rasterizer_impl.cu",
    ),
    "source_compatible": (
        f"{SOURCE_COMPATIBLE_GENERATED}/backward.cu",
        "cuda_rasterizer/forward.cu",
        f"{SOURCE_COMPATIBLE_GENERATED}/rasterizer_impl.cu",
    ),
    "fp32_candidate": (
        f"{FP32_CANDIDATE_GENERATED}/backward.cu",
        "cuda_rasterizer/forward.cu",
        f"{FP32_CANDIDATE_GENERATED}/rasterizer_impl.cu",
    ),
    "fp32_deterministic": (
        f"{FP32_DETERMINISTIC_GENERATED}/backward.cu",
        "cuda_rasterizer/forward.cu",
        f"{FP32_DETERMINISTIC_GENERATED}/rasterizer_impl.cu",
    ),
}
VARIANT_COMPILE_UNITS.update({
    variant: (
        f"{generated}/backward.cu",
        "cuda_rasterizer/forward.cu",
        f"{generated}/rasterizer_impl.cu",
    )
    for variant, generated in FP32_RESCUE_GENERATED.items()
})
VARIANT_COMPILE_UNITS[MIXED_PRECISION_VARIANT] = (
    f"{MIXED_PRECISION_GENERATED}/backward.cu",
    "cuda_rasterizer/forward.cu",
    f"{MIXED_PRECISION_GENERATED}/rasterizer_impl.cu",
)
VARIANT_COMPILE_UNITS[MIXED_NORMAL_EXTRA_VARIANT] = (
    f"{MIXED_NORMAL_EXTRA_GENERATED}/backward.cu",
    "cuda_rasterizer/forward.cu",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/rasterizer_impl.cu",
)
VARIANT_COMPILE_UNITS[MIXED_NORMAL_OPACITY_VARIANT] = (
    f"{MIXED_NORMAL_OPACITY_GENERATED}/backward.cu",
    "cuda_rasterizer/forward.cu",
    f"{MIXED_NORMAL_OPACITY_GENERATED}/rasterizer_impl.cu",
)
VARIANT_COMPILE_UNITS[MIXED_FIXED_ORDER_VARIANT] = (
    f"{MIXED_FIXED_ORDER_GENERATED}/backward.cu",
    "cuda_rasterizer/forward.cu",
    f"{MIXED_FIXED_ORDER_GENERATED}/rasterizer_impl.cu",
)
VARIANT_COMPILE_UNITS[MIXED_CONIC_FP64_VARIANT] = (
    f"{MIXED_CONIC_FP64_GENERATED}/backward.cu",
    "cuda_rasterizer/forward.cu",
    f"{MIXED_CONIC_FP64_GENERATED}/rasterizer_impl.cu",
)
VARIANT_COMPILE_UNITS[SHARED_DETERMINISTIC_FP32_VARIANT] = (
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/backward.cu",
    "cuda_rasterizer/forward.cu",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/rasterizer_impl.cu",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/shared_deterministic_fp32_api.cu",
)
VARIANT_INCLUDE_DIRECTORIES = {
    "stable": INCLUDE_DIRECTORIES,
    "optimized_stable": (
        OPTIMIZED_STABLE_GENERATED,
        "cuda_rasterizer",
        "third_party/glm",
    ),
    "source_compatible": (
        SOURCE_COMPATIBLE_GENERATED,
        "cuda_rasterizer",
        "third_party/glm",
    ),
    "fp32_candidate": (
        FP32_CANDIDATE_GENERATED,
        "cuda_rasterizer",
        "third_party/glm",
    ),
    "fp32_deterministic": (
        FP32_DETERMINISTIC_GENERATED,
        "cuda_rasterizer",
        "third_party/glm",
    ),
}
VARIANT_INCLUDE_DIRECTORIES.update({
    variant: (generated, "cuda_rasterizer", "third_party/glm")
    for variant, generated in FP32_RESCUE_GENERATED.items()
})
VARIANT_INCLUDE_DIRECTORIES[MIXED_PRECISION_VARIANT] = (
    MIXED_PRECISION_GENERATED, "cuda_rasterizer", "third_party/glm")
VARIANT_INCLUDE_DIRECTORIES[MIXED_NORMAL_EXTRA_VARIANT] = (
    MIXED_NORMAL_EXTRA_GENERATED, "cuda_rasterizer", "third_party/glm")
VARIANT_INCLUDE_DIRECTORIES[MIXED_NORMAL_OPACITY_VARIANT] = (
    MIXED_NORMAL_OPACITY_GENERATED, "cuda_rasterizer", "third_party/glm")
VARIANT_INCLUDE_DIRECTORIES[MIXED_FIXED_ORDER_VARIANT] = (
    MIXED_FIXED_ORDER_GENERATED, "cuda_rasterizer", "third_party/glm")
VARIANT_INCLUDE_DIRECTORIES[MIXED_CONIC_FP64_VARIANT] = (
    MIXED_CONIC_FP64_GENERATED, "cuda_rasterizer", "third_party/glm")
VARIANT_INCLUDE_DIRECTORIES[SHARED_DETERMINISTIC_FP32_VARIANT] = (
    SHARED_DETERMINISTIC_FP32_GENERATED, "cuda_rasterizer", "third_party/glm")
SOURCE_COMPATIBLE_SOURCES = (
    f"{SOURCE_COMPATIBLE_GENERATED}/backward.cu",
    f"{SOURCE_COMPATIBLE_GENERATED}/backward.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/auxiliary.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/config.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/forward.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/rasterizer_impl.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/rasterizer.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/vec_math.h",
    f"{SOURCE_COMPATIBLE_GENERATED}/rasterizer_impl.cu",
    "source_compatible/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
OPTIMIZED_STABLE_SOURCES = (
    f"{OPTIMIZED_STABLE_GENERATED}/backward.cu",
    f"{OPTIMIZED_STABLE_GENERATED}/backward.h",
    f"{OPTIMIZED_STABLE_GENERATED}/auxiliary.h",
    f"{OPTIMIZED_STABLE_GENERATED}/config.h",
    f"{OPTIMIZED_STABLE_GENERATED}/forward.h",
    f"{OPTIMIZED_STABLE_GENERATED}/rasterizer_impl.h",
    f"{OPTIMIZED_STABLE_GENERATED}/rasterizer.h",
    f"{OPTIMIZED_STABLE_GENERATED}/vec_math.h",
    f"{OPTIMIZED_STABLE_GENERATED}/rasterizer_impl.cu",
    "optimized_stable/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
FP32_CANDIDATE_SOURCES = (
    f"{FP32_CANDIDATE_GENERATED}/backward.cu",
    f"{FP32_CANDIDATE_GENERATED}/backward.h",
    f"{FP32_CANDIDATE_GENERATED}/auxiliary.h",
    f"{FP32_CANDIDATE_GENERATED}/config.h",
    f"{FP32_CANDIDATE_GENERATED}/forward.h",
    f"{FP32_CANDIDATE_GENERATED}/rasterizer_impl.h",
    f"{FP32_CANDIDATE_GENERATED}/rasterizer.h",
    f"{FP32_CANDIDATE_GENERATED}/vec_math.h",
    f"{FP32_CANDIDATE_GENERATED}/rasterizer_impl.cu",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
FP32_DETERMINISTIC_SOURCES = (
    f"{FP32_DETERMINISTIC_GENERATED}/backward.cu",
    f"{FP32_DETERMINISTIC_GENERATED}/backward.h",
    f"{FP32_DETERMINISTIC_GENERATED}/auxiliary.h",
    f"{FP32_DETERMINISTIC_GENERATED}/config.h",
    f"{FP32_DETERMINISTIC_GENERATED}/forward.h",
    f"{FP32_DETERMINISTIC_GENERATED}/rasterizer_impl.h",
    f"{FP32_DETERMINISTIC_GENERATED}/rasterizer.h",
    f"{FP32_DETERMINISTIC_GENERATED}/vec_math.h",
    f"{FP32_DETERMINISTIC_GENERATED}/rasterizer_impl.cu",
    "fp32_deterministic/materialize.py",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
FP32_RESCUE_SOURCES = {
    variant: (
        f"{generated}/backward.cu",
        f"{generated}/backward.h",
        f"{generated}/auxiliary.h",
        f"{generated}/config.h",
        f"{generated}/forward.h",
        f"{generated}/rasterizer_impl.h",
        f"{generated}/rasterizer.h",
        f"{generated}/vec_math.h",
        f"{generated}/rasterizer_impl.cu",
        "mixed_precision_candidate/materialize.py",
        "fp32_deterministic/materialize.py",
        "fp32_candidate/materialize.py",
        "cuda_rasterizer/forward.cu",
        "cuda_rasterizer/forward.h",
        "cuda_rasterizer/rasterizer_impl.h",
        "cuda_rasterizer/rasterizer.h",
        "cuda_rasterizer/auxiliary.h",
        "cuda_rasterizer/config.h",
        "cuda_rasterizer/vec_math.h",
        "rasterize_points_jt.py",
        "light_gaussian/__init__.py",
    )
    for variant, generated in FP32_RESCUE_GENERATED.items()
}
MIXED_PRECISION_SOURCES = (
    f"{MIXED_PRECISION_GENERATED}/backward.cu",
    f"{MIXED_PRECISION_GENERATED}/backward.h",
    f"{MIXED_PRECISION_GENERATED}/auxiliary.h",
    f"{MIXED_PRECISION_GENERATED}/config.h",
    f"{MIXED_PRECISION_GENERATED}/forward.h",
    f"{MIXED_PRECISION_GENERATED}/rasterizer_impl.h",
    f"{MIXED_PRECISION_GENERATED}/rasterizer.h",
    f"{MIXED_PRECISION_GENERATED}/vec_math.h",
    f"{MIXED_PRECISION_GENERATED}/rasterizer_impl.cu",
    "mixed_precision_candidate/materialize.py",
    "fp32_deterministic/materialize.py",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
MIXED_NORMAL_EXTRA_SOURCES = (
    f"{MIXED_NORMAL_EXTRA_GENERATED}/backward.cu",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/backward.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/auxiliary.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/config.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/forward.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/rasterizer_impl.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/rasterizer.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/vec_math.h",
    f"{MIXED_NORMAL_EXTRA_GENERATED}/rasterizer_impl.cu",
    "mixed_precision_candidate/materialize.py",
    "fp32_deterministic/materialize.py",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
MIXED_NORMAL_OPACITY_SOURCES = tuple(
    path.replace(MIXED_NORMAL_EXTRA_GENERATED, MIXED_NORMAL_OPACITY_GENERATED)
    for path in MIXED_NORMAL_EXTRA_SOURCES)
MIXED_FIXED_ORDER_SOURCES = (
    f"{MIXED_FIXED_ORDER_GENERATED}/backward.cu",
    f"{MIXED_FIXED_ORDER_GENERATED}/backward.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/auxiliary.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/config.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/forward.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/rasterizer_impl.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/rasterizer.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/vec_math.h",
    f"{MIXED_FIXED_ORDER_GENERATED}/rasterizer_impl.cu",
    "mixed_fixed_order_candidate/materialize.py",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
MIXED_CONIC_FP64_SOURCES = (
    f"{MIXED_CONIC_FP64_GENERATED}/backward.cu",
    f"{MIXED_CONIC_FP64_GENERATED}/backward.h",
    f"{MIXED_CONIC_FP64_GENERATED}/auxiliary.h",
    f"{MIXED_CONIC_FP64_GENERATED}/config.h",
    f"{MIXED_CONIC_FP64_GENERATED}/forward.h",
    f"{MIXED_CONIC_FP64_GENERATED}/rasterizer_impl.h",
    f"{MIXED_CONIC_FP64_GENERATED}/rasterizer.h",
    f"{MIXED_CONIC_FP64_GENERATED}/vec_math.h",
    f"{MIXED_CONIC_FP64_GENERATED}/rasterizer_impl.cu",
    "mixed_conic_fp64_candidate/materialize.py",
    "mixed_fixed_order_candidate/materialize.py",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
)
SHARED_DETERMINISTIC_FP32_SOURCES = (
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/backward.cu",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/backward.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/auxiliary.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/config.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/forward.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/rasterizer_impl.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/rasterizer.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/vec_math.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/rasterizer_impl.cu",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/shared_deterministic_fp32_api.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/shared_deterministic_fp32_api.cu",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/torch_rasterize_points.cu",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/torch_rasterize_points.h",
    f"{SHARED_DETERMINISTIC_FP32_GENERATED}/torch_ext.cpp",
    "shared_deterministic_fp32/materialize.py",
    "shared_deterministic_fp32/jittor_backward.py",
    "shared_deterministic_fp32/jittor_saved_state.py",
    "mixed_fixed_order_candidate/materialize.py",
    "fp32_candidate/materialize.py",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/forward.h",
    "cuda_rasterizer/rasterizer_impl.h",
    "cuda_rasterizer/rasterizer.h",
    "cuda_rasterizer/auxiliary.h",
    "cuda_rasterizer/config.h",
    "cuda_rasterizer/vec_math.h",
    "rasterize_points_jt.py",
    "light_gaussian/__init__.py",
    "../../GANG-master/submodules/light_gaussian/rasterize_points.cu",
    "../../GANG-master/submodules/light_gaussian/rasterize_points.h",
)
VARIANT_CONTROLLED_SOURCES = {
    "stable": CONTROLLED_SOURCES,
    "optimized_stable": OPTIMIZED_STABLE_SOURCES,
    "source_compatible": SOURCE_COMPATIBLE_SOURCES,
    "fp32_candidate": FP32_CANDIDATE_SOURCES,
    "fp32_deterministic": FP32_DETERMINISTIC_SOURCES,
    **FP32_RESCUE_SOURCES,
    MIXED_PRECISION_VARIANT: MIXED_PRECISION_SOURCES,
    MIXED_NORMAL_EXTRA_VARIANT: MIXED_NORMAL_EXTRA_SOURCES,
    MIXED_NORMAL_OPACITY_VARIANT: MIXED_NORMAL_OPACITY_SOURCES,
    MIXED_FIXED_ORDER_VARIANT: MIXED_FIXED_ORDER_SOURCES,
    MIXED_CONIC_FP64_VARIANT: MIXED_CONIC_FP64_SOURCES,
    SHARED_DETERMINISTIC_FP32_VARIANT: SHARED_DETERMINISTIC_FP32_SOURCES,
}
REQUIRED_COMPILE_FLAGS = ("-shared", "-Xcompiler", "-fPIC", "--std=c++17", "-O3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_build_dir(root: Path, variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported rasterizer variant: {variant}")
    return root / "build" if variant == "stable" else root / "build" / variant


def manifest_path(root: Path, variant: str) -> Path:
    return variant_build_dir(root, variant) / "rasterizer_manifest.json"


def compile_units_for_variant(variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported rasterizer variant: {variant}")
    return VARIANT_COMPILE_UNITS[variant]


def include_directories_for_variant(variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported rasterizer variant: {variant}")
    return VARIANT_INCLUDE_DIRECTORIES[variant]


def jit_header_dir(root: Path, variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported rasterizer variant: {variant}")
    relative = {
        "stable": "cuda_rasterizer",
        "optimized_stable": OPTIMIZED_STABLE_GENERATED,
        "source_compatible": SOURCE_COMPATIBLE_GENERATED,
        "fp32_candidate": FP32_CANDIDATE_GENERATED,
        "fp32_deterministic": FP32_DETERMINISTIC_GENERATED,
        **FP32_RESCUE_GENERATED,
        MIXED_PRECISION_VARIANT: MIXED_PRECISION_GENERATED,
        MIXED_NORMAL_EXTRA_VARIANT: MIXED_NORMAL_EXTRA_GENERATED,
        MIXED_NORMAL_OPACITY_VARIANT: MIXED_NORMAL_OPACITY_GENERATED,
        MIXED_FIXED_ORDER_VARIANT: MIXED_FIXED_ORDER_GENERATED,
        MIXED_CONIC_FP64_VARIANT: MIXED_CONIC_FP64_GENERATED,
        SHARED_DETERMINISTIC_FP32_VARIANT:
            SHARED_DETERMINISTIC_FP32_GENERATED,
    }[variant]
    return Path(root).resolve() / relative


def controlled_source_paths(
        root: Path, variant: str = PRODUCTION_VARIANT) -> tuple[str, ...]:
    """Return every file whose bytes may affect the native/JIT rasterizer."""
    root = Path(root).resolve()
    if variant not in VARIANTS:
        raise ValueError(f"unsupported rasterizer variant: {variant}")
    paths = list(VARIANT_CONTROLLED_SOURCES[variant])
    for relative_tree in CONTROLLED_SOURCE_TREES:
        tree = (root / relative_tree).resolve()
        try:
            tree.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"controlled source tree escapes rasterizer root: {relative_tree}") from exc
        if not tree.is_dir():
            raise FileNotFoundError(f"missing controlled source tree: {tree}")
        for path in sorted(tree.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"controlled source tree contains a symlink: {path}")
            if path.is_file():
                paths.append(path.relative_to(root).as_posix())
    if len(paths) != len(set(paths)):
        raise ValueError("controlled rasterizer source paths contain duplicates")
    return tuple(sorted(paths))


def source_hashes(
        root: Path, variant: str = PRODUCTION_VARIANT) -> dict[str, str]:
    paths = controlled_source_paths(root, variant)
    missing = [relative for relative in paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing controlled rasterizer sources: {missing}")
    return {relative: sha256(root / relative) for relative in paths}


def _validate_build_contract(payload: dict, root: Path, requested: str) -> None:
    expected_define = VARIANT_DEFINES[requested]
    if payload.get("variant_define") != expected_define:
        raise RuntimeError(
            f"rasterizer variant define mismatch: {payload.get('variant_define')} "
            f"!= {expected_define}")

    flags = payload.get("compile_flags")
    if not isinstance(flags, list) or not all(isinstance(flag, str) and flag for flag in flags):
        raise RuntimeError("rasterizer manifest has invalid compile_flags")
    missing_flags = [flag for flag in REQUIRED_COMPILE_FLAGS if flag not in flags]
    define_flag = f"-D{expected_define}"
    if define_flag not in flags:
        missing_flags.append(define_flag)
    foreign_define_names = {
        *FORBIDDEN_VARIANT_DEFINES.values(),
        *(define for name, define in VARIANT_DEFINES.items() if name != requested),
    }
    foreign_defines = [
        f"-D{define}" for define in sorted(foreign_define_names)
        if f"-D{define}" in flags
    ]
    if foreign_defines:
        raise RuntimeError(
            f"rasterizer manifest mixes variant defines: {foreign_defines}")
    architectures = [
        flag for flag in flags
        if flag.startswith("arch=compute_") and ",code=sm_" in flag
    ]
    if not architectures:
        missing_flags.append("arch=compute_<N>,code=sm_<N>")
    if missing_flags:
        raise RuntimeError(
            f"rasterizer manifest is missing compile flags: {missing_flags}")

    argv = payload.get("build_argv")
    if not isinstance(argv, list) or not all(isinstance(arg, str) and arg for arg in argv):
        raise RuntimeError("rasterizer manifest has invalid build_argv")
    expected_units = compile_units_for_variant(requested)
    expected_includes = include_directories_for_variant(requested)
    declared_units = payload.get("compile_units")
    if declared_units != list(expected_units):
        raise RuntimeError("rasterizer manifest compile units differ from the build contract")
    declared_includes = payload.get("include_directories")
    if declared_includes != list(expected_includes):
        raise RuntimeError("rasterizer manifest include directories differ from the build contract")
    for relative in (*expected_units, *expected_includes):
        absolute = str((root / relative).resolve())
        if absolute not in argv:
            raise RuntimeError(f"rasterizer build argv omits {relative}")
    output = str((variant_build_dir(root, requested) / "libCudaRasterizer.so").resolve())
    if output not in argv or "-o" not in argv:
        raise RuntimeError("rasterizer build argv omits its exact output path")
    # _rebuild.sh places the complete compiler flag vector immediately after
    # nvcc. Require the manifest to preserve exact order and multiplicity,
    # including each repeated -gencode pair.
    if argv[1:1 + len(flags)] != flags:
        raise RuntimeError(
            "rasterizer compile_flags do not match the leading build_argv flags")
    command = payload.get("build_command")
    if not isinstance(command, str) or not command.strip():
        raise RuntimeError("rasterizer manifest has no complete build_command")


def validate_rasterizer_build(root: str | Path, variant: str | None = None) -> dict:
    root = Path(root).resolve()
    configured = os.environ.get("GANG_RASTER_BACKWARD_VARIANT")
    if variant is None:
        requested = PRODUCTION_VARIANT
        if configured is not None and configured != PRODUCTION_VARIANT:
            raise RuntimeError(
                "production rasterizer imports are fixed to stable; "
                f"GANG_RASTER_BACKWARD_VARIANT={configured!r} is not permitted")
    else:
        requested = variant
    if requested not in VARIANTS:
        raise RuntimeError(f"invalid GANG_RASTER_BACKWARD_VARIANT={requested!r}")
    path = manifest_path(root, requested)
    if not path.is_file():
        raise RuntimeError(f"missing rasterizer build manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gang.rasterizer_build.v1":
        raise RuntimeError(f"unsupported rasterizer manifest schema: {path}")
    if payload.get("abi_version") != ABI_VERSION:
        raise RuntimeError(f"rasterizer ABI mismatch: {path}")
    if payload.get("variant") != requested:
        raise RuntimeError(
            f"rasterizer variant mismatch: requested={requested}, "
            f"manifest={payload.get('variant')}")
    _validate_build_contract(payload, root, requested)
    build_dir = variant_build_dir(root, requested)
    libraries = {
        name: build_dir / name for name in ("libCudaRasterizer.so", "librasterizer.so")
    }
    for name, library in libraries.items():
        if not library.is_file():
            raise RuntimeError(f"missing rasterizer library: {library}")
        actual = sha256(library)
        expected = payload.get("libraries", {}).get(name, {}).get("sha256")
        if actual != expected:
            raise RuntimeError(
                f"rasterizer library hash mismatch for {library}: {actual} != {expected}")
    current_sources = source_hashes(root, requested)
    if current_sources != payload.get("source_sha256"):
        changed = sorted(set(current_sources) | set(payload.get("source_sha256", {})))
        changed = [name for name in changed
                   if current_sources.get(name) != payload.get("source_sha256", {}).get(name)]
        raise RuntimeError(f"rasterizer source/build mismatch: {changed}")
    return {**payload, "build_dir": str(build_dir)}
