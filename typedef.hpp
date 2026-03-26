#ifndef __TYPEDEF_H__
#define __TYPEDEF_H__
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "buffer.hpp"
#include <immintrin.h>
typedef std::bfloat16_t dtype_in;
typedef std::bfloat16_t dtype_out;
using bfloat16_t = uint16_t;

typedef uint8_t qs_uint8_t;

struct dequantize_params {
    std::uint8_t    qx[128 * 16];
    std::bfloat16_t sx_min[128 * 2]; // [min, scale] repeated 128 times
  };

/////////////////////////////////////////////////////////////////
static inline uint16_t float_to_bfloat16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  // round-to-nearest-even
  uint32_t r = ((x >> 16) & 1u) + 0x7FFFu;
  return (uint16_t)((x + r) >> 16);
}


static inline __m512 load_bf16_to_fp32(const uint16_t* p) {
  // load 16×u16, widen to 16×u32, shift into f32 exponent bits, reinterpret as f32
  __m256i u16 = _mm256_loadu_si256((const __m256i*)p);
  __m512i u32 = _mm512_cvtepu16_epi32(u16);   // AVX-512BW
  u32 = _mm512_slli_epi32(u32, 16);
  return _mm512_castsi512_ps(u32);
}

static inline void store_fp32_to_bf16(uint16_t* p, __m512 v) {
  // portable RNE pack; still fast enough (only on stores)
  alignas(32) float tmp[16];
  _mm256_store_ps(tmp,     _mm512_castps512_ps256(v));
  _mm256_store_ps(tmp + 8, _mm512_extractf32x8_ps(v, 1));
  for (int i = 0; i < 16; ++i) p[i] = float_to_bfloat16(tmp[i]);
}

static inline float  bf16_to_f32(uint16_t x) {
  uint32_t u = uint32_t(x) << 16;
  return *reinterpret_cast<float*>(&u);
}
static inline uint16_t f32_to_bf16(float f) {
  uint32_t u = *reinterpret_cast<uint32_t*>(&f);
  return uint16_t(u >> 16);
}

void convert_bf16_to_f32_avx2(const uint16_t *src16,
  float        *dst32,
  size_t        N)
{
size_t i = 0;
for (; i + 8 <= N; i += 8) {
__m128i u16 = _mm_loadu_si128((__m128i const*)(src16 + i));
__m256i u32 = _mm256_cvtepu16_epi32(u16);
u32 = _mm256_slli_epi32(u32, 16);
__m256 f32 = _mm256_castsi256_ps(u32);
_mm256_storeu_ps(dst32 + i, f32);
}
for (; i < N; ++i) {
dst32[i] = bf16_to_f32(src16[i]);
}
}

static inline __m512 bf16_load16_to_f32(const dtype_in* p) {
  __m256i u16 = _mm256_loadu_si256((const __m256i*)p);   // 16x u16
  __m512i u32 = _mm512_cvtepu16_epi32(u16);              // 16x u32
  u32 = _mm512_slli_epi32(u32, 16);
  return _mm512_castsi512_ps(u32);
}

static inline void f32_store16_to_bf16_rne(dtype_out* p, __m512 v) {
  #if defined(__AVX512BF16__)
      __m256bh b16 = _mm512_cvtneps_pbh(v);                  // RNE
      _mm256_storeu_si256((__m256i*)p, (__m256i)b16);
  #else
      // Software RNE fallback (rarely needed on modern CPUs). Ask if you need it.
      __m512i x   = _mm512_castps_si512(v);
      __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(x,16), _mm512_set1_epi32(1));
      __m512i t   = _mm512_add_epi32(x, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb));
      __m512i y   = _mm512_srli_epi32(t, 16);
      __m256i lo  = _mm512_extracti64x4_epi64(y, 0);
      __m256i hi  = _mm512_extracti64x4_epi64(y, 1);
      __m256i pk  = _mm256_packus_epi32(lo, hi);
      _mm256_storeu_si256((__m256i*)p, pk);
  #endif
  }

  static inline void f32_to_bf16_rn_16(uint16_t* out16, __m512 x) {
    __m512i u = _mm512_castps_si512(x);
    __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    __m512i bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    u = _mm512_add_epi32(u, bias);
    __m512i hi = _mm512_srli_epi32(u, 16);                       // bf16 bits in low 16
    __m256i pack16 = _mm512_cvtepi32_epi16(hi);                  // pack 16x u16
    _mm256_storeu_si256((__m256i*)out16, pack16);
}

static inline float bfloat16_to_float(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
// static inline uint16_t float_to_bfloat16(float f) {
//   uint32_t x;
//   std::memcpy(&x, &f, sizeof(x));
//   // round-to-nearest-even
//   uint32_t r = ((x >> 16) & 1u) + 0x7FFFu;
//   return (uint16_t)((x + r) >> 16);
// }

#endif
