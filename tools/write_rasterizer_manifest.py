"""Write or verify the native rasterizer build manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "submodules" / "light_gaussian"
sys.path.insert(0, str(MODULE_ROOT))

from rasterizer_provenance import (  # noqa: E402
    ABI_VERSION, VARIANT_DEFINES, VARIANTS, compile_units_for_variant,
    include_directories_for_variant, sha256, source_hashes, variant_build_dir,
)


def _nvcc_version(nvcc: str) -> str:
    completed = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True,
        text=True, timeout=20)
    return (completed.stdout or completed.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="stable")
    parser.add_argument("--variant-define", required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--build-command", required=True)
    parser.add_argument("--compile-flag", action="append", default=[])
    parser.add_argument("--build-arg", action="append", default=[])
    args = parser.parse_args()
    expected_define = VARIANT_DEFINES[args.variant]
    if args.variant_define != expected_define:
        raise ValueError(
            f"variant define mismatch: {args.variant_define} != {expected_define}")
    if not args.compile_flag:
        raise ValueError("at least one explicit compile flag is required")
    if not args.build_arg or args.build_arg[0] != args.nvcc:
        raise ValueError("build argv must start with the declared nvcc executable")
    define_flag = f"-D{expected_define}"
    if define_flag not in args.compile_flag or define_flag not in args.build_arg:
        raise ValueError("build contract does not contain the selected variant define")
    foreign_defines = [
        f"-D{define}" for name, define in VARIANT_DEFINES.items()
        if name != args.variant
        and (f"-D{define}" in args.compile_flag or f"-D{define}" in args.build_arg)
    ]
    if foreign_defines:
        raise ValueError(f"build contract mixes variant defines: {foreign_defines}")
    build_dir = variant_build_dir(MODULE_ROOT, args.variant)
    compile_units = compile_units_for_variant(args.variant)
    include_directories = include_directories_for_variant(args.variant)
    libraries = {}
    for name in ("libCudaRasterizer.so", "librasterizer.so"):
        path = build_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        libraries[name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    if libraries["libCudaRasterizer.so"]["sha256"] != libraries["librasterizer.so"]["sha256"]:
        raise RuntimeError("the two rasterizer library hashes differ")
    payload = {
        "schema": "gang.rasterizer_build.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "variant_define": expected_define,
        "abi_version": ABI_VERSION,
        "nvcc_version": _nvcc_version(args.nvcc),
        "compile_flags": args.compile_flag,
        "build_argv": args.build_arg,
        "build_command": args.build_command,
        "compile_units": list(compile_units),
        "include_directories": list(include_directories),
        "source_sha256": source_hashes(MODULE_ROOT, args.variant),
        "libraries": libraries,
    }
    output = build_dir / "rasterizer_manifest.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
