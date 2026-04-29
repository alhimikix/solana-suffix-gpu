#!/bin/bash
# Build script for Linux/macOS.
# Override SM_ARCH for non-Ampere GPUs:
#   sm_75 = RTX 20xx, GTX 16xx
#   sm_86 = RTX 30xx, A100/A40 (default)
#   sm_89 = RTX 40xx, L40
#   sm_90 = H100, GH200
set -euo pipefail

SM_ARCH="${SM_ARCH:-sm_86}"
NVCC="${NVCC:-nvcc}"

"$NVCC" -O3 -arch="$SM_ARCH" -o vanity_gpu vanity.cu
echo "Build OK: vanity_gpu (arch=$SM_ARCH)"
