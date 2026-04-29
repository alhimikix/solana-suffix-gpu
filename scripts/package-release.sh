#!/bin/bash
# Build all SM architectures and package into release tarball.
# Usage: scripts/package-release.sh v1.0.0
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0.0"
    exit 1
fi

VERSION="$1"
OUTDIR="release/solana-vanity-gpu-${VERSION}-linux-x64"
NVCC="${NVCC:-nvcc}"

mkdir -p "$OUTDIR"

for arch in 75 86 89; do
    echo "Building sm_${arch}..."
    "$NVCC" -O3 -arch="sm_${arch}" -o "$OUTDIR/vanity_gpu_sm${arch}" vanity.cu
done

cp README.md LICENSE CHANGELOG.md build.sh "$OUTDIR/"

cd release
tar -czf "solana-vanity-gpu-${VERSION}-linux-x64.tar.gz" -C "solana-vanity-gpu-${VERSION}-linux-x64" .
cd ..

echo
echo "Done: release/solana-vanity-gpu-${VERSION}-linux-x64.tar.gz"
