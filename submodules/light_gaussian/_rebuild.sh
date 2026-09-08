#!/bin/bash
set -euo pipefail

# Build one isolated rasterizer variant and bind it to a source manifest.
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$MODULE_DIR/../.." && pwd)"
VARIANT="${1:-stable}"
case "$VARIANT" in
  stable) BUILD_DIR="$MODULE_DIR/build"; VARIANT_DEFINE="GANG_RASTER_VARIANT_STABLE" ;;
  optimized_stable)
    BUILD_DIR="$MODULE_DIR/build/optimized_stable"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_OPTIMIZED_STABLE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/optimized_stable/materialize.py" --output "$GENERATED_DIR"
    ;;
  source_compatible)
    BUILD_DIR="$MODULE_DIR/build/source_compatible"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_SOURCE_COMPATIBLE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/source_compatible/materialize.py" --output "$GENERATED_DIR"
    ;;
  fp32_candidate)
    BUILD_DIR="$MODULE_DIR/build/fp32_candidate"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_FP32_CANDIDATE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/fp32_candidate/materialize.py" --output "$GENERATED_DIR"
    ;;
  fp32_deterministic)
    BUILD_DIR="$MODULE_DIR/build/fp32_deterministic"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_FP32_DETERMINISTIC"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/fp32_deterministic/materialize.py" --output "$GENERATED_DIR"
    ;;
  fp32_rescue_opacity|fp32_rescue_color|fp32_rescue_depth|fp32_rescue_normal|fp32_rescue_extra)
    CHANNEL="${VARIANT#fp32_rescue_}"
    BUILD_DIR="$MODULE_DIR/build/$VARIANT"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_${VARIANT^^}"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/mixed_precision_candidate/materialize.py" \
      --output "$GENERATED_DIR" --fp64-channels "$CHANNEL"
    ;;
  mixed_precision_candidate)
    BUILD_DIR="$MODULE_DIR/build/mixed_precision_candidate"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_MIXED_PRECISION_CANDIDATE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/mixed_precision_candidate/materialize.py" \
      --output "$GENERATED_DIR" --fp64-channels normal
    ;;
  mixed_precision_normal_extra_candidate)
    BUILD_DIR="$MODULE_DIR/build/mixed_precision_normal_extra_candidate"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_MIXED_PRECISION_NORMAL_EXTRA_CANDIDATE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/mixed_precision_candidate/materialize.py" \
      --output "$GENERATED_DIR" --fp64-channels normal,extra
    ;;
  mixed_precision_normal_opacity_candidate)
    BUILD_DIR="$MODULE_DIR/build/mixed_precision_normal_opacity_candidate"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_MIXED_PRECISION_NORMAL_OPACITY_CANDIDATE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/mixed_precision_candidate/materialize.py" \
      --output "$GENERATED_DIR" --fp64-channels normal,opacity
    ;;
  mixed_fixed_order_candidate)
    BUILD_DIR="$MODULE_DIR/build/mixed_fixed_order_candidate"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_MIXED_FIXED_ORDER_CANDIDATE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/mixed_fixed_order_candidate/materialize.py" \
      --output "$GENERATED_DIR"
    ;;
  mixed_conic_fp64_candidate)
    BUILD_DIR="$MODULE_DIR/build/mixed_conic_fp64_candidate"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_MIXED_CONIC_FP64_CANDIDATE"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/mixed_conic_fp64_candidate/materialize.py" \
      --output "$GENERATED_DIR"
    ;;
  shared_deterministic_fp32)
    BUILD_DIR="$MODULE_DIR/build/shared_deterministic_fp32"
    VARIANT_DEFINE="GANG_RASTER_VARIANT_SHARED_DETERMINISTIC_FP32"
    GENERATED_DIR="$BUILD_DIR/generated"
    python3 "$MODULE_DIR/shared_deterministic_fp32/materialize.py" \
      --output "$GENERATED_DIR"
    ;;
  *) echo "unsupported rasterizer variant: $VARIANT" >&2; exit 2 ;;
esac

CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_ROOT/bin/nvcc"
CUDA_ARCHS="${GANG_CUDA_ARCHS:-70 75 86}"
mkdir -p "$BUILD_DIR"
LOG_PATH="/tmp/gang_rasterizer_${VARIANT}_build.log"

GENCODE_ARGS=()
for arch in $CUDA_ARCHS; do
  case "$arch" in
    ''|*[!0-9]*) echo "invalid CUDA architecture: $arch" >&2; exit 2 ;;
  esac
  GENCODE_ARGS+=("-gencode" "arch=compute_${arch},code=sm_${arch}")
done
test "${#GENCODE_ARGS[@]}" -gt 0

COMPILE_FLAGS=(
  "-shared" "-Xcompiler" "-fPIC" "--std=c++17" "-O3"
  "-D$VARIANT_DEFINE"
  "${GENCODE_ARGS[@]}"
)
if [[ "$VARIANT" == "stable" ]]; then
  INCLUDE_DIRS=(
    "$MODULE_DIR/cuda_rasterizer"
    "$MODULE_DIR/third_party/glm"
  )
  COMPILE_UNITS=(
    "$MODULE_DIR/cuda_rasterizer/backward.cu"
    "$MODULE_DIR/cuda_rasterizer/forward.cu"
    "$MODULE_DIR/cuda_rasterizer/rasterizer_impl.cu"
  )
else
  INCLUDE_DIRS=(
    "$GENERATED_DIR"
    "$MODULE_DIR/cuda_rasterizer"
    "$MODULE_DIR/third_party/glm"
  )
  COMPILE_UNITS=(
    "$GENERATED_DIR/backward.cu"
    "$MODULE_DIR/cuda_rasterizer/forward.cu"
    "$GENERATED_DIR/rasterizer_impl.cu"
  )
  if [[ "$VARIANT" == "shared_deterministic_fp32" ]]; then
    COMPILE_UNITS+=("$GENERATED_DIR/shared_deterministic_fp32_api.cu")
  fi
fi
BUILD_ARGV=("$NVCC" "${COMPILE_FLAGS[@]}")
for include_dir in "${INCLUDE_DIRS[@]}"; do
  BUILD_ARGV+=("-I" "$include_dir")
done
BUILD_ARGV+=("-o" "$BUILD_DIR/libCudaRasterizer.so" "${COMPILE_UNITS[@]}")
printf -v BUILD_COMMAND '%q ' "${BUILD_ARGV[@]}"
BUILD_COMMAND="${BUILD_COMMAND% }"

"${BUILD_ARGV[@]}" \
  >"$LOG_PATH" 2>&1

cp -f "$BUILD_DIR/libCudaRasterizer.so" "$BUILD_DIR/librasterizer.so"
SYMBOLS_PATH="/tmp/gang_rasterizer_${VARIANT}_symbols.txt"
nm -D "$BUILD_DIR/librasterizer.so" >"$SYMBOLS_PATH"
REQUIRED_SYMBOLS=(
  markVisible depthToNormal forward_0 forward_1 backward visible_filter
  lite_forward receiver_forward
)
for symbol in "${REQUIRED_SYMBOLS[@]}"; do
  grep -qi "$symbol" "$SYMBOLS_PATH"
done
if [[ "$VARIANT" == "shared_deterministic_fp32" ]]; then
  grep -q "gang_shared_deterministic_fp32_abi_version" "$SYMBOLS_PATH"
  grep -q "gang_shared_deterministic_fp32_workspace_bytes" "$SYMBOLS_PATH"
  grep -q "gang_shared_deterministic_fp32_backward" "$SYMBOLS_PATH"
  grep -q "gang_shared_deterministic_fp32_export_saved_state" "$SYMBOLS_PATH"
fi
test "$(sha256sum "$BUILD_DIR/libCudaRasterizer.so" | cut -d' ' -f1)" = \
     "$(sha256sum "$BUILD_DIR/librasterizer.so" | cut -d' ' -f1)"

cd "$PROJECT_DIR"
MANIFEST_FLAG_ARGS=()
for flag in "${COMPILE_FLAGS[@]}"; do
  MANIFEST_FLAG_ARGS+=("--compile-flag=$flag")
done
MANIFEST_BUILD_ARGS=()
for arg in "${BUILD_ARGV[@]}"; do
  MANIFEST_BUILD_ARGS+=("--build-arg=$arg")
done
python3 tools/write_rasterizer_manifest.py \
  --variant "$VARIANT" \
  --variant-define "$VARIANT_DEFINE" \
  --nvcc "$NVCC" \
  --build-command "$BUILD_COMMAND" \
  "${MANIFEST_FLAG_ARGS[@]}" \
  "${MANIFEST_BUILD_ARGS[@]}"

tail -30 "$LOG_PATH"
sha256sum "$BUILD_DIR/libCudaRasterizer.so" "$BUILD_DIR/librasterizer.so"
