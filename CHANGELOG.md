# Changelog

## v1.0.0 - 2026-04-29

Initial release.

### Performance
- **44M keys/sec on RTX 3090** (97x faster than CPU baseline).

### Features
- 8-bit signed-digit comb fixed-base scalar multiplication.
- ref10-style fe10 (radix-2^25.5) field arithmetic.
- Within-thread Montgomery batched inversion (N=4).
- Suffix mod-58^K prefilter to skip base58 encoding for 99.99% of non-matching candidates.
- CSV output (`<suffix>_results.csv`) with `public_key,private_key` columns in base58.
- Sustained throughput reporter every 2 seconds.
- Configurable target match count, or unlimited mode.

### Build
- Windows build script (`build.bat`) with auto-detection of Visual Studio 2019/2022/2025.
- Linux/macOS build script (`build.sh`) with configurable SM architecture.
- CUDA Toolkit ≥ 13.1 required.
