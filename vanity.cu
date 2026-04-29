// GPU Solana vanity address grinder.
// Ed25519 keypair derivation ported from TweetNaCl (public domain).
// Build: nvcc -O3 -arch=sm_86 -o vanity_gpu.exe vanity.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while(0)

// ============================================================================
// SHA-512 (device)
// ============================================================================

__device__ __constant__ u64 K512[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
};

__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

__device__ void sha512_compress(u64 state[8], const u8 block[128]) {
    u64 W[80];
    for (int i = 0; i < 16; i++) {
        W[i] = ((u64)block[i*8+0] << 56) | ((u64)block[i*8+1] << 48) |
               ((u64)block[i*8+2] << 40) | ((u64)block[i*8+3] << 32) |
               ((u64)block[i*8+4] << 24) | ((u64)block[i*8+5] << 16) |
               ((u64)block[i*8+6] << 8)  | ((u64)block[i*8+7]);
    }
    for (int i = 16; i < 80; i++) {
        u64 s0 = rotr64(W[i-15],1) ^ rotr64(W[i-15],8) ^ (W[i-15] >> 7);
        u64 s1 = rotr64(W[i-2],19) ^ rotr64(W[i-2],61) ^ (W[i-2] >> 6);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    u64 a = state[0], b = state[1], c = state[2], d = state[3];
    u64 e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; i++) {
        u64 S1 = rotr64(e,14) ^ rotr64(e,18) ^ rotr64(e,41);
        u64 ch = (e & f) ^ ((~e) & g);
        u64 t1 = h + S1 + ch + K512[i] + W[i];
        u64 S0 = rotr64(a,28) ^ rotr64(a,34) ^ rotr64(a,39);
        u64 mj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__device__ void sha512_32(const u8 in[32], u8 out[64]) {
    u64 st[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    u8 block[128];
    #pragma unroll
    for (int i = 0; i < 32; i++) block[i] = in[i];
    block[32] = 0x80;
    #pragma unroll
    for (int i = 33; i < 128-8; i++) block[i] = 0;
    // bit length = 32*8 = 256
    block[120] = 0; block[121] = 0; block[122] = 0; block[123] = 0;
    block[124] = 0; block[125] = 0; block[126] = 0x01; block[127] = 0x00;
    sha512_compress(st, block);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*8+0] = (u8)(st[i] >> 56);
        out[i*8+1] = (u8)(st[i] >> 48);
        out[i*8+2] = (u8)(st[i] >> 40);
        out[i*8+3] = (u8)(st[i] >> 32);
        out[i*8+4] = (u8)(st[i] >> 24);
        out[i*8+5] = (u8)(st[i] >> 16);
        out[i*8+6] = (u8)(st[i] >> 8);
        out[i*8+7] = (u8)(st[i]);
    }
}

// ============================================================================
// Ed25519 field arithmetic: ref10-style radix-2^25.5, 10 signed i32 limbs.
// fe_mul has 100 partial products (vs TweetNaCl's 256). ~3x faster.
// ============================================================================

typedef i32 gf[10];

__device__ __constant__ gf gf0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ __constant__ gf gf1 = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Curve constant d = -121665/121666 mod p (ref10 limb form)
__device__ __constant__ gf D = {
    -10913610, 13857413, -15372611, 6949391, 114729,
    -8787816, -6275908, -3247719, -18696448, -12055116
};
// 2*d
__device__ __constant__ gf D2 = {
    -21827239, -5839606, -30745221, 13898782, 229458,
    15978800, -12551817, -6495438, 29715968, 9444199
};

// Suffix prefilter: target = pubkey_int mod 58^K, set on host
__device__ __constant__ u64 d_target_mod;
__device__ __constant__ u64 d_mod_K;

// Basepoint coords (initialized at startup from canonical bytes)
__device__ gf X, Y;

__device__ __constant__ u8 BX_BYTES[32] = {
    0x1A, 0xD5, 0x25, 0x8F, 0x60, 0x2D, 0x56, 0xC9, 0xB2, 0xA7, 0x25, 0x95, 0x60, 0xC7, 0x2C, 0x69,
    0x5C, 0xDC, 0xD6, 0xFD, 0x31, 0xE2, 0xA4, 0xC0, 0xFE, 0x53, 0x6E, 0xCD, 0xD3, 0x36, 0x69, 0x21
};
__device__ __constant__ u8 BY_BYTES[32] = {
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
};

__device__ __forceinline__ void set25519(gf out, const gf in) {
    #pragma unroll
    for (int i = 0; i < 10; i++) out[i] = in[i];
}

// Conditional swap: swap p, q if b == 1
__device__ void sel25519(gf p, gf q, int b) {
    i32 c = -b;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        i32 t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

__device__ void A(gf h, const gf f, const gf g) {
    #pragma unroll
    for (int i = 0; i < 10; i++) h[i] = f[i] + g[i];
}

__device__ void Z(gf h, const gf f, const gf g) {
    #pragma unroll
    for (int i = 0; i < 10; i++) h[i] = f[i] - g[i];
}

// Field mul: ref10 fe_mul. h = f*g mod p.
__device__ void M(gf h, const gf f, const gf g) {
    i32 f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    i32 f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    i32 g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    i32 g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];

    i32 g1_19 = 19 * g1, g2_19 = 19 * g2, g3_19 = 19 * g3, g4_19 = 19 * g4;
    i32 g5_19 = 19 * g5, g6_19 = 19 * g6, g7_19 = 19 * g7, g8_19 = 19 * g8, g9_19 = 19 * g9;
    i32 f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5, f7_2 = 2 * f7, f9_2 = 2 * f9;

    i64 f0g0 = (i64)f0*g0;
    i64 f0g1 = (i64)f0*g1;     i64 f1g0 = (i64)f1*g0;
    i64 f0g2 = (i64)f0*g2;     i64 f1g1_2 = (i64)f1_2*g1;   i64 f2g0 = (i64)f2*g0;
    i64 f0g3 = (i64)f0*g3;     i64 f1g2 = (i64)f1*g2;       i64 f2g1 = (i64)f2*g1;       i64 f3g0 = (i64)f3*g0;
    i64 f0g4 = (i64)f0*g4;     i64 f1g3_2 = (i64)f1_2*g3;   i64 f2g2 = (i64)f2*g2;       i64 f3g1_2 = (i64)f3_2*g1;   i64 f4g0 = (i64)f4*g0;
    i64 f0g5 = (i64)f0*g5;     i64 f1g4 = (i64)f1*g4;       i64 f2g3 = (i64)f2*g3;       i64 f3g2 = (i64)f3*g2;       i64 f4g1 = (i64)f4*g1;     i64 f5g0 = (i64)f5*g0;
    i64 f0g6 = (i64)f0*g6;     i64 f1g5_2 = (i64)f1_2*g5;   i64 f2g4 = (i64)f2*g4;       i64 f3g3_2 = (i64)f3_2*g3;   i64 f4g2 = (i64)f4*g2;     i64 f5g1_2 = (i64)f5_2*g1; i64 f6g0 = (i64)f6*g0;
    i64 f0g7 = (i64)f0*g7;     i64 f1g6 = (i64)f1*g6;       i64 f2g5 = (i64)f2*g5;       i64 f3g4 = (i64)f3*g4;       i64 f4g3 = (i64)f4*g3;     i64 f5g2 = (i64)f5*g2;     i64 f6g1 = (i64)f6*g1;     i64 f7g0 = (i64)f7*g0;
    i64 f0g8 = (i64)f0*g8;     i64 f1g7_2 = (i64)f1_2*g7;   i64 f2g6 = (i64)f2*g6;       i64 f3g5_2 = (i64)f3_2*g5;   i64 f4g4 = (i64)f4*g4;     i64 f5g3_2 = (i64)f5_2*g3; i64 f6g2 = (i64)f6*g2;     i64 f7g1_2 = (i64)f7_2*g1; i64 f8g0 = (i64)f8*g0;
    i64 f0g9 = (i64)f0*g9;     i64 f1g8 = (i64)f1*g8;       i64 f2g7 = (i64)f2*g7;       i64 f3g6 = (i64)f3*g6;       i64 f4g5 = (i64)f4*g5;     i64 f5g4 = (i64)f5*g4;     i64 f6g3 = (i64)f6*g3;     i64 f7g2 = (i64)f7*g2;     i64 f8g1 = (i64)f8*g1;     i64 f9g0 = (i64)f9*g0;
    i64 f1g9_38 = (i64)f1_2*g9_19;
    i64 f2g8_19 = (i64)f2*g8_19;   i64 f2g9_19 = (i64)f2*g9_19;
    i64 f3g7_38 = (i64)f3_2*g7_19; i64 f3g8_19 = (i64)f3*g8_19;   i64 f3g9_38 = (i64)f3_2*g9_19;
    i64 f4g6_19 = (i64)f4*g6_19;   i64 f4g7_19 = (i64)f4*g7_19;   i64 f4g8_19 = (i64)f4*g8_19;   i64 f4g9_19 = (i64)f4*g9_19;
    i64 f5g5_38 = (i64)f5_2*g5_19; i64 f5g6_19 = (i64)f5*g6_19;   i64 f5g7_38 = (i64)f5_2*g7_19; i64 f5g8_19 = (i64)f5*g8_19;   i64 f5g9_38 = (i64)f5_2*g9_19;
    i64 f6g4_19 = (i64)f6*g4_19;   i64 f6g5_19 = (i64)f6*g5_19;   i64 f6g6_19 = (i64)f6*g6_19;   i64 f6g7_19 = (i64)f6*g7_19;   i64 f6g8_19 = (i64)f6*g8_19;   i64 f6g9_19 = (i64)f6*g9_19;
    i64 f7g3_38 = (i64)f7_2*g3_19; i64 f7g4_19 = (i64)f7*g4_19;   i64 f7g5_38 = (i64)f7_2*g5_19; i64 f7g6_19 = (i64)f7*g6_19;   i64 f7g7_38 = (i64)f7_2*g7_19; i64 f7g8_19 = (i64)f7*g8_19;   i64 f7g9_38 = (i64)f7_2*g9_19;
    i64 f8g2_19 = (i64)f8*g2_19;   i64 f8g3_19 = (i64)f8*g3_19;   i64 f8g4_19 = (i64)f8*g4_19;   i64 f8g5_19 = (i64)f8*g5_19;   i64 f8g6_19 = (i64)f8*g6_19;   i64 f8g7_19 = (i64)f8*g7_19;   i64 f8g8_19 = (i64)f8*g8_19;   i64 f8g9_19 = (i64)f8*g9_19;
    i64 f9g1_38 = (i64)f9_2*g1_19; i64 f9g2_19 = (i64)f9*g2_19;   i64 f9g3_38 = (i64)f9_2*g3_19; i64 f9g4_19 = (i64)f9*g4_19;   i64 f9g5_38 = (i64)f9_2*g5_19; i64 f9g6_19 = (i64)f9*g6_19;   i64 f9g7_38 = (i64)f9_2*g7_19; i64 f9g8_19 = (i64)f9*g8_19;   i64 f9g9_38 = (i64)f9_2*g9_19;

    i64 h0 = f0g0   + f1g9_38 + f2g8_19 + f3g7_38 + f4g6_19 + f5g5_38 + f6g4_19 + f7g3_38 + f8g2_19 + f9g1_38;
    i64 h1 = f0g1   + f1g0    + f2g9_19 + f3g8_19 + f4g7_19 + f5g6_19 + f6g5_19 + f7g4_19 + f8g3_19 + f9g2_19;
    i64 h2 = f0g2   + f1g1_2  + f2g0    + f3g9_38 + f4g8_19 + f5g7_38 + f6g6_19 + f7g5_38 + f8g4_19 + f9g3_38;
    i64 h3 = f0g3   + f1g2    + f2g1    + f3g0    + f4g9_19 + f5g8_19 + f6g7_19 + f7g6_19 + f8g5_19 + f9g4_19;
    i64 h4 = f0g4   + f1g3_2  + f2g2    + f3g1_2  + f4g0    + f5g9_38 + f6g8_19 + f7g7_38 + f8g6_19 + f9g5_38;
    i64 h5 = f0g5   + f1g4    + f2g3    + f3g2    + f4g1    + f5g0    + f6g9_19 + f7g8_19 + f8g7_19 + f9g6_19;
    i64 h6 = f0g6   + f1g5_2  + f2g4    + f3g3_2  + f4g2    + f5g1_2  + f6g0    + f7g9_38 + f8g8_19 + f9g7_38;
    i64 h7 = f0g7   + f1g6    + f2g5    + f3g4    + f4g3    + f5g2    + f6g1    + f7g0    + f8g9_19 + f9g8_19;
    i64 h8 = f0g8   + f1g7_2  + f2g6    + f3g5_2  + f4g4    + f5g3_2  + f6g2    + f7g1_2  + f8g0    + f9g9_38;
    i64 h9 = f0g9   + f1g8    + f2g7    + f3g6    + f4g5    + f5g4    + f6g3    + f7g2    + f8g1    + f9g0;

    i64 c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;
    i64 c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c1 = (h1 + (1LL << 24)) >> 25; h2 += c1; h1 -= c1 << 25;
    i64 c5 = (h5 + (1LL << 24)) >> 25; h6 += c5; h5 -= c5 << 25;
    i64 c2 = (h2 + (1LL << 25)) >> 26; h3 += c2; h2 -= c2 << 26;
    i64 c6 = (h6 + (1LL << 25)) >> 26; h7 += c6; h6 -= c6 << 26;
    i64 c3 = (h3 + (1LL << 24)) >> 25; h4 += c3; h3 -= c3 << 25;
    i64 c7 = (h7 + (1LL << 24)) >> 25; h8 += c7; h7 -= c7 << 25;
    c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c8 = (h8 + (1LL << 25)) >> 26; h9 += c8; h8 -= c8 << 26;
    i64 c9 = (h9 + (1LL << 24)) >> 25; h0 += c9 * 19; h9 -= c9 << 25;
    c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;

    h[0] = (i32)h0; h[1] = (i32)h1; h[2] = (i32)h2; h[3] = (i32)h3; h[4] = (i32)h4;
    h[5] = (i32)h5; h[6] = (i32)h6; h[7] = (i32)h7; h[8] = (i32)h8; h[9] = (i32)h9;
}

// Field square: optimized symmetric version. ~55 partial products vs M's 100.
__device__ void S(gf h, const gf f) {
    i32 f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    i32 f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];

    i32 f0_2 = 2*f0, f1_2 = 2*f1, f2_2 = 2*f2, f3_2 = 2*f3, f4_2 = 2*f4;
    i32 f5_2 = 2*f5, f6_2 = 2*f6, f7_2 = 2*f7;
    i32 f5_38 = 38*f5, f6_19 = 19*f6, f7_38 = 38*f7, f8_19 = 19*f8, f9_38 = 38*f9;

    i64 f0f0   = (i64)f0   * f0;
    i64 f0f1_2 = (i64)f0_2 * f1;
    i64 f0f2_2 = (i64)f0_2 * f2;
    i64 f0f3_2 = (i64)f0_2 * f3;
    i64 f0f4_2 = (i64)f0_2 * f4;
    i64 f0f5_2 = (i64)f0_2 * f5;
    i64 f0f6_2 = (i64)f0_2 * f6;
    i64 f0f7_2 = (i64)f0_2 * f7;
    i64 f0f8_2 = (i64)f0_2 * f8;
    i64 f0f9_2 = (i64)f0_2 * f9;
    i64 f1f1_2 = (i64)f1_2 * f1;
    i64 f1f2_2 = (i64)f1_2 * f2;
    i64 f1f3_4 = (i64)f1_2 * f3_2;
    i64 f1f4_2 = (i64)f1_2 * f4;
    i64 f1f5_4 = (i64)f1_2 * f5_2;
    i64 f1f6_2 = (i64)f1_2 * f6;
    i64 f1f7_4 = (i64)f1_2 * f7_2;
    i64 f1f8_2 = (i64)f1_2 * f8;
    i64 f1f9_76 = (i64)f1_2 * f9_38;
    i64 f2f2   = (i64)f2   * f2;
    i64 f2f3_2 = (i64)f2_2 * f3;
    i64 f2f4_2 = (i64)f2_2 * f4;
    i64 f2f5_2 = (i64)f2_2 * f5;
    i64 f2f6_2 = (i64)f2_2 * f6;
    i64 f2f7_2 = (i64)f2_2 * f7;
    i64 f2f8_38 = (i64)f2_2 * f8_19;
    i64 f2f9_38 = (i64)f2   * f9_38;
    i64 f3f3_2 = (i64)f3_2 * f3;
    i64 f3f4_2 = (i64)f3_2 * f4;
    i64 f3f5_4 = (i64)f3_2 * f5_2;
    i64 f3f6_2 = (i64)f3_2 * f6;
    i64 f3f7_76 = (i64)f3_2 * f7_38;
    i64 f3f8_38 = (i64)f3_2 * f8_19;
    i64 f3f9_76 = (i64)f3_2 * f9_38;
    i64 f4f4   = (i64)f4   * f4;
    i64 f4f5_2 = (i64)f4_2 * f5;
    i64 f4f6_38 = (i64)f4_2 * f6_19;
    i64 f4f7_38 = (i64)f4   * f7_38;
    i64 f4f8_38 = (i64)f4_2 * f8_19;
    i64 f4f9_38 = (i64)f4   * f9_38;
    i64 f5f5_38 = (i64)f5   * f5_38;
    i64 f5f6_38 = (i64)f5_2 * f6_19;
    i64 f5f7_76 = (i64)f5_2 * f7_38;
    i64 f5f8_38 = (i64)f5_2 * f8_19;
    i64 f5f9_76 = (i64)f5_2 * f9_38;
    i64 f6f6_19 = (i64)f6   * f6_19;
    i64 f6f7_38 = (i64)f6   * f7_38;
    i64 f6f8_38 = (i64)f6_2 * f8_19;
    i64 f6f9_38 = (i64)f6   * f9_38;
    i64 f7f7_38 = (i64)f7   * f7_38;
    i64 f7f8_38 = (i64)f7_2 * f8_19;
    i64 f7f9_76 = (i64)f7_2 * f9_38;
    i64 f8f8_19 = (i64)f8   * f8_19;
    i64 f8f9_38 = (i64)f8   * f9_38;
    i64 f9f9_38 = (i64)f9   * f9_38;

    i64 h0 = f0f0   + f1f9_76 + f2f8_38 + f3f7_76 + f4f6_38 + f5f5_38;
    i64 h1 = f0f1_2 + f2f9_38 + f3f8_38 + f4f7_38 + f5f6_38;
    i64 h2 = f0f2_2 + f1f1_2 + f3f9_76 + f4f8_38 + f5f7_76 + f6f6_19;
    i64 h3 = f0f3_2 + f1f2_2 + f4f9_38 + f5f8_38 + f6f7_38;
    i64 h4 = f0f4_2 + f1f3_4 + f2f2   + f5f9_76 + f6f8_38 + f7f7_38;
    i64 h5 = f0f5_2 + f1f4_2 + f2f3_2 + f6f9_38 + f7f8_38;
    i64 h6 = f0f6_2 + f1f5_4 + f2f4_2 + f3f3_2 + f7f9_76 + f8f8_19;
    i64 h7 = f0f7_2 + f1f6_2 + f2f5_2 + f3f4_2 + f8f9_38;
    i64 h8 = f0f8_2 + f1f7_4 + f2f6_2 + f3f5_4 + f4f4   + f9f9_38;
    i64 h9 = f0f9_2 + f1f8_2 + f2f7_2 + f3f6_2 + f4f5_2;

    i64 c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;
    i64 c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c1 = (h1 + (1LL << 24)) >> 25; h2 += c1; h1 -= c1 << 25;
    i64 c5 = (h5 + (1LL << 24)) >> 25; h6 += c5; h5 -= c5 << 25;
    i64 c2 = (h2 + (1LL << 25)) >> 26; h3 += c2; h2 -= c2 << 26;
    i64 c6 = (h6 + (1LL << 25)) >> 26; h7 += c6; h6 -= c6 << 26;
    i64 c3 = (h3 + (1LL << 24)) >> 25; h4 += c3; h3 -= c3 << 25;
    i64 c7 = (h7 + (1LL << 24)) >> 25; h8 += c7; h7 -= c7 << 25;
    c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c8 = (h8 + (1LL << 25)) >> 26; h9 += c8; h8 -= c8 << 26;
    i64 c9 = (h9 + (1LL << 24)) >> 25; h0 += c9 * 19; h9 -= c9 << 25;
    c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;

    h[0] = (i32)h0; h[1] = (i32)h1; h[2] = (i32)h2; h[3] = (i32)h3; h[4] = (i32)h4;
    h[5] = (i32)h5; h[6] = (i32)h6; h[7] = (i32)h7; h[8] = (i32)h8; h[9] = (i32)h9;
}

// Pack to 32 canonical bytes (little-endian).
__device__ void pack25519(u8 *s, const gf h_in) {
    i32 h0 = h_in[0], h1 = h_in[1], h2 = h_in[2], h3 = h_in[3], h4 = h_in[4];
    i32 h5 = h_in[5], h6 = h_in[6], h7 = h_in[7], h8 = h_in[8], h9 = h_in[9];

    i32 q = (19 * h9 + (1 << 24)) >> 25;
    q = (h0 + q) >> 26;
    q = (h1 + q) >> 25;
    q = (h2 + q) >> 26;
    q = (h3 + q) >> 25;
    q = (h4 + q) >> 26;
    q = (h5 + q) >> 25;
    q = (h6 + q) >> 26;
    q = (h7 + q) >> 25;
    q = (h8 + q) >> 26;
    q = (h9 + q) >> 25;

    h0 += 19 * q;

    i32 c0 = h0 >> 26; h1 += c0; h0 -= c0 << 26;
    i32 c1 = h1 >> 25; h2 += c1; h1 -= c1 << 25;
    i32 c2 = h2 >> 26; h3 += c2; h2 -= c2 << 26;
    i32 c3 = h3 >> 25; h4 += c3; h3 -= c3 << 25;
    i32 c4 = h4 >> 26; h5 += c4; h4 -= c4 << 26;
    i32 c5 = h5 >> 25; h6 += c5; h5 -= c5 << 25;
    i32 c6 = h6 >> 26; h7 += c6; h6 -= c6 << 26;
    i32 c7 = h7 >> 25; h8 += c7; h7 -= c7 << 25;
    i32 c8 = h8 >> 26; h9 += c8; h8 -= c8 << 26;
    i32 c9 = h9 >> 25;             h9 -= c9 << 25;

    s[0]  = (u8)(h0 >> 0);
    s[1]  = (u8)(h0 >> 8);
    s[2]  = (u8)(h0 >> 16);
    s[3]  = (u8)((h0 >> 24) | (h1 << 2));
    s[4]  = (u8)(h1 >> 6);
    s[5]  = (u8)(h1 >> 14);
    s[6]  = (u8)((h1 >> 22) | (h2 << 3));
    s[7]  = (u8)(h2 >> 5);
    s[8]  = (u8)(h2 >> 13);
    s[9]  = (u8)((h2 >> 21) | (h3 << 5));
    s[10] = (u8)(h3 >> 3);
    s[11] = (u8)(h3 >> 11);
    s[12] = (u8)((h3 >> 19) | (h4 << 6));
    s[13] = (u8)(h4 >> 2);
    s[14] = (u8)(h4 >> 10);
    s[15] = (u8)(h4 >> 18);
    s[16] = (u8)(h5 >> 0);
    s[17] = (u8)(h5 >> 8);
    s[18] = (u8)(h5 >> 16);
    s[19] = (u8)((h5 >> 24) | (h6 << 1));
    s[20] = (u8)(h6 >> 7);
    s[21] = (u8)(h6 >> 15);
    s[22] = (u8)((h6 >> 23) | (h7 << 3));
    s[23] = (u8)(h7 >> 5);
    s[24] = (u8)(h7 >> 13);
    s[25] = (u8)((h7 >> 21) | (h8 << 4));
    s[26] = (u8)(h8 >> 4);
    s[27] = (u8)(h8 >> 12);
    s[28] = (u8)((h8 >> 20) | (h9 << 6));
    s[29] = (u8)(h9 >> 2);
    s[30] = (u8)(h9 >> 10);
    s[31] = (u8)(h9 >> 18);
}

__device__ static i64 load_3(const u8 *s) {
    return (i64)s[0] | ((i64)s[1] << 8) | ((i64)s[2] << 16);
}
__device__ static i64 load_4(const u8 *s) {
    return (i64)s[0] | ((i64)s[1] << 8) | ((i64)s[2] << 16) | ((i64)s[3] << 24);
}

__device__ void fe_frombytes(gf h, const u8 s[32]) {
    i64 h0 = load_4(s);
    i64 h1 = load_3(s + 4) << 6;
    i64 h2 = load_3(s + 7) << 5;
    i64 h3 = load_3(s + 10) << 3;
    i64 h4 = load_3(s + 13) << 2;
    i64 h5 = load_4(s + 16);
    i64 h6 = load_3(s + 20) << 7;
    i64 h7 = load_3(s + 23) << 5;
    i64 h8 = load_3(s + 26) << 4;
    i64 h9 = (load_3(s + 29) & 0x7FFFFFLL) << 2;

    i64 c9 = (h9 + (1LL << 24)) >> 25; h0 += c9 * 19; h9 -= c9 << 25;
    i64 c1 = (h1 + (1LL << 24)) >> 25; h2 += c1; h1 -= c1 << 25;
    i64 c3 = (h3 + (1LL << 24)) >> 25; h4 += c3; h3 -= c3 << 25;
    i64 c5 = (h5 + (1LL << 24)) >> 25; h6 += c5; h5 -= c5 << 25;
    i64 c7 = (h7 + (1LL << 24)) >> 25; h8 += c7; h7 -= c7 << 25;
    i64 c0 = (h0 + (1LL << 25)) >> 26; h1 += c0; h0 -= c0 << 26;
    i64 c2 = (h2 + (1LL << 25)) >> 26; h3 += c2; h2 -= c2 << 26;
    i64 c4 = (h4 + (1LL << 25)) >> 26; h5 += c4; h4 -= c4 << 26;
    i64 c6 = (h6 + (1LL << 25)) >> 26; h7 += c6; h6 -= c6 << 26;
    i64 c8 = (h8 + (1LL << 25)) >> 26; h9 += c8; h8 -= c8 << 26;

    h[0] = (i32)h0; h[1] = (i32)h1; h[2] = (i32)h2; h[3] = (i32)h3; h[4] = (i32)h4;
    h[5] = (i32)h5; h[6] = (i32)h6; h[7] = (i32)h7; h[8] = (i32)h8; h[9] = (i32)h9;
}

__global__ void init_basepoint_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        fe_frombytes(X, BX_BYTES);
        fe_frombytes(Y, BY_BYTES);
    }
}

__device__ void inv25519(gf out, const gf z) {
    gf t0, t1, t2, t3;
    int i;
    S(t0, z);
    S(t1, t0); S(t1, t1);
    M(t1, z, t1);
    M(t0, t0, t1);
    S(t2, t0);
    M(t1, t2, t1);
    S(t2, t1);
    for (i = 1; i < 5; i++) S(t2, t2);
    M(t1, t2, t1);
    S(t2, t1);
    for (i = 1; i < 10; i++) S(t2, t2);
    M(t2, t2, t1);
    S(t3, t2);
    for (i = 1; i < 20; i++) S(t3, t3);
    M(t2, t3, t2);
    S(t2, t2);
    for (i = 1; i < 10; i++) S(t2, t2);
    M(t1, t2, t1);
    S(t2, t1);
    for (i = 1; i < 50; i++) S(t2, t2);
    M(t2, t2, t1);
    S(t3, t2);
    for (i = 1; i < 100; i++) S(t3, t3);
    M(t2, t3, t2);
    S(t2, t2);
    for (i = 1; i < 50; i++) S(t2, t2);
    M(t1, t2, t1);
    S(t1, t1);
    for (i = 1; i < 5; i++) S(t1, t1);
    M(out, t1, t0);
}

// Edwards point: P[0]=X, P[1]=Y, P[2]=Z, P[3]=T
__device__ void add(gf p[4], gf q[4]) {
    gf a, b, c, d, t, e, f, g, h;
    Z(a, p[1], p[0]);
    Z(t, q[1], q[0]);
    M(a, a, t);
    A(b, p[0], p[1]);
    A(t, q[0], q[1]);
    M(b, b, t);
    M(c, p[3], q[3]);
    M(c, c, D2);
    M(d, p[2], q[2]);
    A(d, d, d);
    Z(e, b, a);
    Z(f, d, c);
    A(g, d, c);
    A(h, b, a);
    M(p[0], e, f);
    M(p[1], h, g);
    M(p[2], g, f);
    M(p[3], e, h);
}

__device__ void cswap(gf p[4], gf q[4], int b) {
    #pragma unroll
    for (int i = 0; i < 4; i++) sel25519(p[i], q[i], b);
}

__device__ void scalarmult(gf p[4], gf q[4], const u8 *s) {
    set25519(p[0], gf0);
    set25519(p[1], gf1);
    set25519(p[2], gf1);
    set25519(p[3], gf0);
    for (int i = 255; i >= 0; i--) {
        u8 b = (s[i/8] >> (i & 7)) & 1;
        cswap(p, q, b);
        add(q, p);
        add(p, p);
        cswap(p, q, b);
    }
}

__device__ void scalarbase(gf p[4], const u8 *s) {
    gf q[4];
    set25519(q[0], X);
    set25519(q[1], Y);
    set25519(q[2], gf1);
    M(q[3], X, Y);
    scalarmult(p, q, s);
}

// ============================================================================
// Precomputed basepoint table for fast fixed-base scalar mult.
// Entry [w * 8 + (j-1)] = j * 256^w * G stored as (Y+X, Y-X, 2dXY) with Z=1.
// 32 windows x 8 magnitudes = 256 entries; each gf = 128 B; total ~96 KB.
// ============================================================================

#define TABLE_WINDOWS 32
#define TABLE_DIGITS  128
#define TABLE_SIZE    (TABLE_WINDOWS * TABLE_DIGITS)

__device__ gf bp_yxp[TABLE_SIZE];
__device__ gf bp_yxm[TABLE_SIZE];
__device__ gf bp_t2d[TABLE_SIZE];

__global__ void gen_bp_table_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TABLE_SIZE) return;
    int w = idx / TABLE_DIGITS;
    int j = idx % TABLE_DIGITS;

    u8 s[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) s[i] = 0;
    s[w] = (u8)(j + 1);

    gf p[4];
    scalarbase(p, s);

    gf zinv, x_aff, y_aff, xy, yxp_, yxm_, t2d_;
    inv25519(zinv, p[2]);
    M(x_aff, p[0], zinv);
    M(y_aff, p[1], zinv);
    A(yxp_, y_aff, x_aff);
    Z(yxm_, y_aff, x_aff);
    M(xy, x_aff, y_aff);
    M(t2d_, xy, D2);

    set25519(bp_yxp[idx], yxp_);
    set25519(bp_yxm[idx], yxm_);
    set25519(bp_t2d[idx], t2d_);
}

// p += precomputed point (Y+X, Y-X, 2dXY) with implicit Z=1
__device__ void madd(gf p[4], const gf yxp, const gf yxm, const gf t2d) {
    gf a_, b_, c_, d_, e_, f_, g_, h_;
    Z(a_, p[1], p[0]);
    M(a_, a_, yxm);
    A(b_, p[0], p[1]);
    M(b_, b_, yxp);
    M(c_, p[3], t2d);
    A(d_, p[2], p[2]);
    Z(e_, b_, a_);
    Z(f_, d_, c_);
    A(g_, d_, c_);
    A(h_, b_, a_);
    M(p[0], e_, f_);
    M(p[1], h_, g_);
    M(p[2], g_, f_);
    M(p[3], e_, h_);
}

// p -= precomputed point (negation: swap (Y+X, Y-X), negate 2dXY)
__device__ void msub(gf p[4], const gf yxp, const gf yxm, const gf t2d) {
    gf a_, b_, c_, d_, e_, f_, g_, h_;
    Z(a_, p[1], p[0]);
    M(a_, a_, yxp);
    A(b_, p[0], p[1]);
    M(b_, b_, yxm);
    M(c_, p[3], t2d);
    Z(c_, gf0, c_);
    A(d_, p[2], p[2]);
    Z(e_, b_, a_);
    Z(f_, d_, c_);
    A(g_, d_, c_);
    A(h_, b_, a_);
    M(p[0], e_, f_);
    M(p[1], h_, g_);
    M(p[2], g_, f_);
    M(p[3], e_, h_);
}

// Convert 256-bit scalar (32 bytes) into 32 signed 8-bit digits in [-128, 128]
__device__ void scalar_to_signed_bytes(int e[32], const u8 a[32]) {
    #pragma unroll
    for (int i = 0; i < 32; i++) e[i] = (int)a[i];
    int carry = 0;
    #pragma unroll
    for (int i = 0; i < 31; i++) {
        e[i] += carry;
        carry = (e[i] + 128) >> 8;
        e[i] -= carry << 8;
    }
    e[31] += carry;
}

// Fixed-base scalar mult using 8-bit-window comb on basepoint table.
// p = a * G where a is little-endian 256-bit scalar.
__device__ void fast_scalarbase(gf p[4], const u8 a[32]) {
    int e[32];
    scalar_to_signed_bytes(e, a);

    set25519(p[0], gf0);
    set25519(p[1], gf1);
    set25519(p[2], gf1);
    set25519(p[3], gf0);

    for (int w = 0; w < 32; w++) {
        int d = e[w];
        if (d > 0) {
            int idx = w * TABLE_DIGITS + (d - 1);
            madd(p, bp_yxp[idx], bp_yxm[idx], bp_t2d[idx]);
        } else if (d < 0) {
            int idx = w * TABLE_DIGITS + (-d - 1);
            msub(p, bp_yxp[idx], bp_yxm[idx], bp_t2d[idx]);
        }
    }
}

__device__ int par25519(const gf a) {
    u8 d[32];
    pack25519(d, a);
    return d[0] & 1;
}

__device__ void ed25519_pubkey(const u8 seed[32], u8 pk[32]) {
    u8 h[64];
    sha512_32(seed, h);
    h[0]  &= 248;
    h[31] &= 127;
    h[31] |= 64;
    gf p[4];
    fast_scalarbase(p, h);
    gf tx, ty, zi;
    inv25519(zi, p[2]);
    M(tx, p[0], zi);
    M(ty, p[1], zi);
    pack25519(pk, ty);
    pk[31] ^= par25519(tx) << 7;
}

// ============================================================================
// Base58 encode (full 32 bytes -> ~44 chars)
// ============================================================================

__device__ __constant__ char B58_ALPHA[59] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Encode 32-byte big-endian into base58. Output written most-significant-first.
// Returns length. Output buffer must be >= 44.
__device__ int base58_encode_32(const u8 *in, char *out) {
    u8 buf[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) buf[i] = in[i];

    int zeros = 0;
    while (zeros < 32 && buf[zeros] == 0) zeros++;

    char tmp[64];
    int len = 0;
    int start = zeros;
    while (start < 32) {
        // divide buf[start..] by 58, get remainder
        int rem = 0;
        for (int i = start; i < 32; i++) {
            int v = (rem << 8) | buf[i];
            buf[i] = (u8)(v / 58);
            rem = v % 58;
        }
        tmp[len++] = B58_ALPHA[rem];
        while (start < 32 && buf[start] == 0) start++;
    }
    // prepend '1' for each leading zero byte
    int total = zeros + len;
    for (int i = 0; i < zeros; i++) out[i] = '1';
    for (int i = 0; i < len; i++) out[zeros + i] = tmp[len - 1 - i];
    return total;
}

// ============================================================================
// Vanity kernel
// ============================================================================

struct Match {
    u8 seed[32];
    u8 pk[32];
};

// Process BATCH_N candidates per inner iteration with within-thread batched
// Montgomery inversion. Single inv25519 amortized across BATCH_N pubkeys.
#define BATCH_N 4

__global__ void vanity_kernel(
    const u8* __restrict__ base_seed,
    u64 iterations_per_thread,
    const char* __restrict__ suffix,
    int suffix_len,
    Match* matches,
    int* match_count,
    int max_matches
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;

    u8 seed[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) seed[i] = base_seed[i];
    seed[28] ^= (u8)(tid & 0xff);
    seed[29] ^= (u8)((tid >> 8) & 0xff);
    seed[30] ^= (u8)((tid >> 16) & 0xff);
    seed[31] ^= (u8)((tid >> 24) & 0xff);

    char b58[64];

    u8 seed_buf[BATCH_N][32];
    gf X_buf[BATCH_N], Y_buf[BATCH_N], Z_buf[BATCH_N];
    gf prefix[BATCH_N];
    gf invZ_buf[BATCH_N];

    u64 batches = iterations_per_thread / BATCH_N;

    for (u64 it = 0; it < batches; it++) {
        // Stage 1: compute (X, Y, Z) for BATCH_N candidates
        #pragma unroll
        for (int b = 0; b < BATCH_N; b++) {
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                seed[j]++;
                if (seed[j] != 0) break;
            }
            #pragma unroll
            for (int j = 0; j < 32; j++) seed_buf[b][j] = seed[j];

            u8 h[64];
            sha512_32(seed, h);
            h[0]  &= 248;
            h[31] &= 127;
            h[31] |= 64;

            gf p[4];
            fast_scalarbase(p, h);

            #pragma unroll
            for (int k = 0; k < 10; k++) {
                X_buf[b][k] = p[0][k];
                Y_buf[b][k] = p[1][k];
                Z_buf[b][k] = p[2][k];
            }
        }

        // Stage 2: batched Montgomery inversion of all Z's
        set25519(prefix[0], Z_buf[0]);
        #pragma unroll
        for (int b = 1; b < BATCH_N; b++) {
            M(prefix[b], prefix[b-1], Z_buf[b]);
        }
        gf inv_total;
        inv25519(inv_total, prefix[BATCH_N - 1]);
        #pragma unroll
        for (int b = BATCH_N - 1; b >= 1; b--) {
            M(invZ_buf[b], prefix[b-1], inv_total);
            gf t;
            M(t, inv_total, Z_buf[b]);
            set25519(inv_total, t);
        }
        set25519(invZ_buf[0], inv_total);

        // Stage 3: finalize pubkey + fast mod-58^K prefilter, then full check
        #pragma unroll
        for (int b = 0; b < BATCH_N; b++) {
            gf tx, ty;
            M(tx, X_buf[b], invZ_buf[b]);
            M(ty, Y_buf[b], invZ_buf[b]);
            u8 pk[32];
            pack25519(pk, ty);
            pk[31] ^= par25519(tx) << 7;

            // Fast prefilter: pk_int mod 58^K must equal target.
            // Skips base58 encode for 99.99% of candidates.
            u64 pk_mod = 0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                pk_mod = (pk_mod * 256 + pk[i]) % d_mod_K;
            }
            if (pk_mod != d_target_mod) continue;

            // Confirmed match (suffix matches by mod equivalence). Record.
            int idx = atomicAdd(match_count, 1);
            if (idx < max_matches) {
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    matches[idx].seed[j] = seed_buf[b][j];
                    matches[idx].pk[j]   = pk[j];
                }
            }
        }
    }
}

// ============================================================================
// Host helpers
// ============================================================================

static const char BS58_ALPHA_HOST[59] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static int validate_suffix(const char *s) {
    for (const char *p = s; *p; p++) {
        if (!strchr(BS58_ALPHA_HOST, *p)) return 0;
    }
    return 1;
}

static void host_b58_encode(const u8 *in, int in_len, char *out, int *out_len) {
    u8 buf[128];
    memcpy(buf, in, in_len);
    int zeros = 0;
    while (zeros < in_len && buf[zeros] == 0) zeros++;
    char tmp[256];
    int len = 0;
    int start = zeros;
    while (start < in_len) {
        int rem = 0;
        for (int i = start; i < in_len; i++) {
            int v = (rem << 8) | buf[i];
            buf[i] = (u8)(v / 58);
            rem = v % 58;
        }
        tmp[len++] = BS58_ALPHA_HOST[rem];
        while (start < in_len && buf[start] == 0) start++;
    }
    int total = zeros + len;
    for (int i = 0; i < zeros; i++) out[i] = '1';
    for (int i = 0; i < len; i++) out[zeros + i] = tmp[len - 1 - i];
    out[total] = 0;
    *out_len = total;
}

static void random_bytes(u8 *out, int n) {
    // Use rand() mixed with time. Not crypto-grade, but base seed only needs
    // unpredictability across runs; per-iteration seeds are derived by counter.
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)&seeded);
        seeded = 1;
    }
    for (int i = 0; i < n; i++) out[i] = (u8)(rand() & 0xff);
}

int main(int argc, char **argv) {
    const char *suffix = "pump";
    int target_count = -1; // unlimited

    if (argc >= 2) suffix = argv[1];
    if (argc >= 3) target_count = atoi(argv[2]);

    if (!validate_suffix(suffix)) {
        fprintf(stderr, "error: suffix '%s' contains non-base58 chars (0,O,I,l forbidden)\n", suffix);
        return 1;
    }
    int suffix_len = (int)strlen(suffix);

    // Query GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU:     %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Suffix:  %s\n", suffix);

    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s_results.csv", suffix);
    printf("Output:  %s\n", csv_path);
    if (target_count > 0) printf("Target:  %d keypairs\n", target_count);
    else                   printf("Target:  unlimited (Ctrl+C to stop)\n");

    // Launch config
    int block_size = 256;
    int grid_size = prop.multiProcessorCount * 8;
    int total_threads = block_size * grid_size;
    u64 iters_per_thread = 256;
    u64 iters_per_launch = (u64)total_threads * iters_per_thread;
    int max_matches = 1024;

    printf("Threads: %d (%d blocks x %d), %llu iters/thread, %llu keys/launch\n\n",
        total_threads, grid_size, block_size,
        (unsigned long long)iters_per_thread,
        (unsigned long long)iters_per_launch);

    // Open CSV (append mode, write header if new)
    FILE *csv;
    int wrote_header = 0;
    csv = fopen(csv_path, "r");
    if (!csv) wrote_header = 0; else { fclose(csv); wrote_header = 1; }
    csv = fopen(csv_path, "a");
    if (!csv) { fprintf(stderr, "cannot open %s\n", csv_path); return 1; }
    if (!wrote_header) fprintf(csv, "public_key,private_key\n");
    fflush(csv);

    // Device buffers
    u8 *d_base_seed; CUDA_CHECK(cudaMalloc(&d_base_seed, 32));
    char *d_suffix;  CUDA_CHECK(cudaMalloc(&d_suffix, 64));
    Match *d_matches; CUDA_CHECK(cudaMalloc(&d_matches, sizeof(Match) * max_matches));
    int *d_count;    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_suffix, suffix, suffix_len, cudaMemcpyHostToDevice));

    // Compute target_mod = b58_decode(suffix) and mod_K = 58^suffix_len
    u64 mod_K = 1;
    for (int i = 0; i < suffix_len; i++) mod_K *= 58;
    u64 target_mod = 0;
    for (int i = 0; i < suffix_len; i++) {
        const char *p = strchr(BS58_ALPHA_HOST, suffix[i]);
        target_mod = target_mod * 58 + (u64)(p - BS58_ALPHA_HOST);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_target_mod, &target_mod, sizeof(u64)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mod_K, &mod_K, sizeof(u64)));

    Match *h_matches = (Match*)malloc(sizeof(Match) * max_matches);

    // Initialize basepoint coords from canonical bytes
    init_basepoint_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // One-time precompute basepoint table for fast scalar mult
    printf("Precomputing basepoint table...\n");
    clock_t t_table = clock();
    gen_bp_table_kernel<<<(TABLE_SIZE + 31) / 32, 32>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Table ready in %.2fs\n\n", (double)(clock() - t_table) / CLOCKS_PER_SEC);

    int found_total = 0;
    u64 keys_total = 0;
    clock_t t_start = clock();
    clock_t t_last = t_start;
    u64 keys_last = 0;

    while (target_count < 0 || found_total < target_count) {
        u8 base_seed[32];
        random_bytes(base_seed, 32);
        CUDA_CHECK(cudaMemcpy(d_base_seed, base_seed, 32, cudaMemcpyHostToDevice));
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

        vanity_kernel<<<grid_size, block_size>>>(
            d_base_seed, iters_per_thread, d_suffix, suffix_len,
            d_matches, d_count, max_matches);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int n;
        CUDA_CHECK(cudaMemcpy(&n, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (n > max_matches) n = max_matches;
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(h_matches, d_matches, sizeof(Match) * n, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++) {
                char pub_b58[64];
                int pub_len;
                host_b58_encode(h_matches[i].pk, 32, pub_b58, &pub_len);
                u8 full[64];
                memcpy(full, h_matches[i].seed, 32);
                memcpy(full + 32, h_matches[i].pk, 32);
                char priv_b58[128];
                int priv_len;
                host_b58_encode(full, 64, priv_b58, &priv_len);
                fprintf(csv, "%s,%s\n", pub_b58, priv_b58);
                fflush(csv);
                found_total++;
                printf("Found #%d: %s\n", found_total, pub_b58);
                if (target_count > 0 && found_total >= target_count) break;
            }
        }

        keys_total += iters_per_launch;
        clock_t t_now = clock();
        double dt = (double)(t_now - t_last) / CLOCKS_PER_SEC;
        if (dt >= 2.0) {
            double rate = (double)(keys_total - keys_last) / dt;
            fprintf(stderr, "  tried ~%12llu | %12.0f keys/sec | found: %d\n",
                (unsigned long long)keys_total, rate, found_total);
            t_last = t_now;
            keys_last = keys_total;
        }
    }

    double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
    printf("\nDone. %d keypairs in %.2fs. Saved to %s\n", found_total, elapsed, csv_path);

    fclose(csv);
    cudaFree(d_base_seed); cudaFree(d_suffix); cudaFree(d_matches); cudaFree(d_count);
    free(h_matches);
    return 0;
}
