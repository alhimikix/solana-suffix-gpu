# solana-vanity-gpu

GPU grinder for Solana vanity addresses. Brute-forces ed25519 keypairs on CUDA until the base58 public key ends with the desired suffix.

**~44M keys/sec on RTX 3090.** ~97x faster than CPU baseline (16 threads on 8/16-core CPU).

## Requirements

- **NVIDIA GPU** with Compute Capability ≥ 7.5 (RTX 20xx and newer). Default build targets `sm_86` (RTX 30xx). Change `-arch=sm_XX` in `build.bat` / `build.sh` for other GPUs.
- **CUDA Toolkit** ≥ 12.0 (`nvcc`). Tested with 13.1.
- **MSVC** (Visual Studio 2019/2022, or 2025 with `-allow-unsupported-compiler`) on Windows. GCC on Linux.

## Installation

### Pre-built binaries (recommended)

Download from [Releases](../../releases):

- `solana-vanity-gpu-vX.Y.Z-windows-x64.zip` — Windows
- `solana-vanity-gpu-vX.Y.Z-linux-x64.tar.gz` — Linux

Each archive contains three binaries for different GPUs:
- `vanity_gpu_sm75` — RTX 20xx, GTX 16xx (Turing)
- `vanity_gpu_sm86` — RTX 30xx, A100 (Ampere)
- `vanity_gpu_sm89` — RTX 40xx, L40 (Ada)

Run the one matching your GPU. If unsure, try one — incompatible binaries fail with a CUDA error immediately.

### Build from source

**Windows:**
```bash
.\build.bat
```

**Linux/macOS:**
```bash
./build.sh                        # default: sm_86 (Ampere)
SM_ARCH=sm_89 ./build.sh          # for RTX 40xx
SM_ARCH=sm_90 ./build.sh          # for H100
```

Produces `vanity_gpu.exe` (or `vanity_gpu`).

### Local release packaging

```bash
# Windows: builds sm_75/sm_86/sm_89 + zip
scripts\package-release.bat v1.0.0

# Linux: builds sm_75/sm_86/sm_89 + tar.gz
scripts/package-release.sh v1.0.0
```

Output: `release/solana-vanity-gpu-v1.0.0-{windows,linux}-x64.{zip,tar.gz}`.

## Usage

```bash
# Run forever, write matches to pump_results.csv (Ctrl+C to stop)
.\vanity_gpu.exe pump

# Find 100 matches and exit
.\vanity_gpu.exe pump 100
```

CSV format: `public_key,private_key` (both base58). The file `<suffix>_results.csv` is created automatically; the header is written once. Subsequent runs append to it.

Allowed suffix characters: base58 alphabet (`123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz`). Note: `0`, `O`, `I`, `l` are excluded by base58 — using them in a suffix produces an error.

## Importing keys

The `private_key` column is a 64-byte base58 string in Phantom/Solflare format:

- **Phantom:** Settings → Add/Connect Wallet → Import Private Key → paste the string.
- **Solflare:** Wallet → Import → Private Key.
- **Solana CLI:** convert to a 64-byte JSON array via `solana-keygen` or use directly.

## Performance

| Suffix length | Combinations | Average wait at 44M k/s |
|---------------|-------------|-------------------------|
| 4 (`pump`)    | 11.3 M       | <1 second               |
| 5             | 656 M        | ~15 seconds             |
| 6             | 38 G         | ~15 minutes             |
| 7             | 2.2 T        | ~14 hours               |
| 8             | 128 T        | ~1 month                |

The distribution is geometric — you may get lucky in seconds, or it may take 2–3x the average.

## How it works

Optimization journey:

| Version | k/s | vs CPU |
|---------|-----|--------|
| CPU 16 threads (TweetNaCl-style ed25519) | 455k | 1x |
| GPU naive (TweetNaCl port to CUDA) | 311k | 0.68x |
| + 8-bit signed-digit comb scalar mult | 4.65M | 10x |
| + Donna fe10 (radix-2^25.5) field arithmetic | 31M | 68x |
| + Within-thread Montgomery batched inversion (N=4) | 38M | 84x |
| + Suffix mod-58^K prefilter | **44M** | **97x** |

Implementation details:

1. **8-bit signed-digit comb** for fixed-base scalar multiplication. The 256-bit scalar splits into 32 signed bytes in `[-128, 128]`, each indexing a precomputed table `(j * 256^w * G)` for `j ∈ [1..128]`, `w ∈ [0..31]` (4096 points × 3 fe10 = 480 KB in device global memory, fits in L2 cache). One scalar mult = 32 mixed additions instead of 256 doublings + 256 conditional adds.

2. **ref10 fe10** — field representation in radix-2^25.5 (10 signed i32 limbs alternating 26/25 bits). `fe_mul` does 100 partial products vs 256 in TweetNaCl 16x16, and uses `mad.wide.s32` (1-cycle Ampere instruction) instead of slow i64 multiplication.

3. **Within-thread Montgomery batched inversion (N=4)** — each thread processes 4 candidates in parallel. Stage 1: four `scalarbase` calls saving `(X, Y, Z)` to local memory. Stage 2: forward prefix products + one `inv25519` + backward walk. One inversion is amortized across 4 candidates instead of 4 separate inversions. Saves ~75% of inversion cost.

4. **Suffix mod-58^K prefilter** — after `pack25519` we compute `pubkey_int mod 58^K` (where K is the suffix length) and compare against a precomputed target. 99.99% of non-matching candidates skip the full base58 encode + match check.

Hot path per pubkey:
- SHA-512 → clamp → fast_scalarbase (32 madd) → batched inversion (1/4 of inv25519 amortized) → 2× fe_mul (X·invZ, Y·invZ) → pack25519 → mod-58^K check
- ~25K cycles per pubkey effective. Measured 44M k/s = ~15% of theoretical peak. Bottleneck: register pressure (255 registers/thread = Ampere maximum → 8 warps/SM occupancy).

Further optimization would require hand-tuned PTX/SASS or fundamental architectural rework.

## Security

- The starting seed mixes `rand()` (host) + `time(NULL)` (host) + `tid` (per thread). Not cryptographic randomness, but for a vanity grinder only the unpredictability of the starting point matters — the seed is then incremented as a 256-bit counter.
- **If you need crypto-grade randomness for the starting seed** (e.g., a key holding long-term funds): replace `random_bytes()` in `vanity.cu` with reads from `/dev/urandom` (Linux) or `BCryptGenRandom` (Windows).
- No network calls. No telemetry. All private keys stay on your machine.
- The ed25519 implementation is a port of ref10 (the standard reference). Correctness verified against `ed25519-dalek`.
- The `<suffix>_results.csv` file contains private keys in plaintext. **Never commit it.** Consider encrypting it after the run. The default `.gitignore` excludes `*_results.csv`.

## License

MIT. See `LICENSE`.

## Credits

- **ed25519 ref10** by Daniel J. Bernstein, Niels Duif, Tanja Lange, Peter Schwabe, Bo-Yin Yang — public domain (via SUPERCOP). The field constants (`D`, `D2`) and the inversion addition chain are taken directly from ref10.
- The 8-bit signed-digit comb structure for fixed-base scalar multiplication is a standard construction; see e.g. Hisil-Wong-Carter-Dawson "Twisted Edwards Curves Revisited" (2008) and the ed25519 paper.
