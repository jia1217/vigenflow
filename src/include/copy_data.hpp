#ifndef COPY_DATA_HPP
#define COPY_DATA_HPP

#include <cstdint>
#include <cstring>
#include "pad_qk.hpp"


using bf16_u16 = std::uint16_t;

inline void split_qkv_4096x12288_to_3x4096x4096_avx512_bf16(
    const bf16_u16* __restrict src, // [4096, 12288] row-major, bf16 bits
    bf16_u16* __restrict q,         // [4096, 4096]
    bf16_u16* __restrict k,         // [4096, 4096]
    bf16_u16* __restrict v          // [4096, 4096]
) {
    constexpr std::size_t R = 4096;
    constexpr std::size_t C = 4096;              // per Q/K/V
    constexpr std::size_t SRC_COLS = 3 * C;      // 12288

    constexpr std::size_t row_bytes_src = SRC_COLS * sizeof(bf16_u16); // 24576
    constexpr std::size_t row_bytes_blk = C * sizeof(bf16_u16);        // 8192

    const std::uint8_t* s  = reinterpret_cast<const std::uint8_t*>(src);
    std::uint8_t* q8       = reinterpret_cast<std::uint8_t*>(q);
    std::uint8_t* k8       = reinterpret_cast<std::uint8_t*>(k);
    std::uint8_t* v8       = reinterpret_cast<std::uint8_t*>(v);

    // 64 bytes per ZMM
    constexpr std::size_t VEC = 64;
    static_assert(row_bytes_blk % VEC == 0, "Row bytes must be divisible by 64");
    const std::size_t iters = row_bytes_blk / VEC; // 8192/64 = 128

    for (std::size_t r = 0; r < R; ++r) {
        const std::uint8_t* srow = s + r * row_bytes_src;

        const std::uint8_t* sq = srow + 0 * row_bytes_blk;
        const std::uint8_t* sk = srow + 1 * row_bytes_blk;
        const std::uint8_t* sv = srow + 2 * row_bytes_blk;

        std::uint8_t* qrow = q8 + r * row_bytes_blk;
        std::uint8_t* krow = k8 + r * row_bytes_blk;
        std::uint8_t* vrow = v8 + r * row_bytes_blk;

        // Copy Q/K/V row in one loop (3 loads + 3 stores per 64B)
        for (std::size_t i = 0; i < iters; ++i) {
            const std::size_t off = i * VEC;

            __m512i qv = _mm512_loadu_si512((const void*)(sq + off));
            __m512i kv = _mm512_loadu_si512((const void*)(sk + off));
            __m512i vv = _mm512_loadu_si512((const void*)(sv + off));

            _mm512_storeu_si512((void*)(qrow + off), qv);
            _mm512_storeu_si512((void*)(krow + off), kv);
            _mm512_storeu_si512((void*)(vrow + off), vv);
        }
    }
}

// static inline bool bf16_is_nan_bits(uint16_t h){
//     uint16_t exp = (h >> 7) & 0xFF;
//     uint16_t man = h & 0x7F;
//     return (exp == 0xFF) && (man != 0);
// }
// auto count_nan = [](const uint16_t* p, size_t n){
//     size_t c=0;
//     for(size_t i=0;i<n;i++){
//         uint16_t h=p[i];
//         uint16_t exp=(h>>7)&0xFF;
//         uint16_t man=h&0x7F;
//         if(exp==0xFF && man!=0) c++;
//     }
//     return c;
// };

static inline __m512 load_bf16x16_as_f32(const uint16_t* p) {
    #if defined(__AVX512BF16__)
        __m256i  raw = _mm256_loadu_si256((const __m256i*)p); // load 32 bytes
        __m256bh bf  = (__m256bh)raw;                         // bit-cast to bf16 type
        return _mm512_cvtpbh_ps(bf);
    #else
        __m256i v16 = _mm256_loadu_si256((const __m256i*)p);
        __m512i v32 = _mm512_cvtepu16_epi32(v16);
        v32 = _mm512_slli_epi32(v32, 16);
        return _mm512_castsi512_ps(v32);
    #endif
    }
    
    // 16xf32 -> 16xbf16(u16) (RN-even)
    static inline void store_f32x16_as_bf16(uint16_t* p, __m512 x) {
    #if defined(__AVX512BF16__)
        __m256bh bf  = _mm512_cvtneps_pbh(x);                 // returns __m256bh
        __m256i  raw = (__m256i)bf;                           // bit-cast to integer type
        _mm256_storeu_si256((__m256i*)p, raw);
    #else
        __m512i xi  = _mm512_castps_si512(x);
        __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(xi, 16), _mm512_set1_epi32(1));
        __m512i bias= _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
        __m512i rnd = _mm512_add_epi32(xi, bias);
        __m512i hi  = _mm512_srli_epi32(rnd, 16);
        __m256i out16 = _mm512_cvtepi32_epi16(hi);
        _mm256_storeu_si256((__m256i*)p, out16);
    #endif
    }
inline void bf16_mul_avx512(uint16_t* out, const uint16_t* a, const uint16_t* b, size_t n) {
    constexpr size_t VEC = 16;
    size_t i = 0;
    for (; i + VEC <= n; i += VEC) {
        __m512 va = load_bf16x16_as_f32(a + i);
        __m512 vb = load_bf16x16_as_f32(b + i);
        __m512 vc = _mm512_mul_ps(va, vb);
        store_f32x16_as_bf16(out + i, vc);
    }
    // tail (rare for your sizes)
    for (; i < n; ++i) {
        uint32_t fa = (uint32_t)a[i] << 16;
        uint32_t fb = (uint32_t)b[i] << 16;
        float aa = *reinterpret_cast<float*>(&fa);
        float bb = *reinterpret_cast<float*>(&fb);
        float cc = aa * bb;

        uint32_t bits = *reinterpret_cast<uint32_t*>(&cc);
        uint32_t lsb  = (bits >> 16) & 1u;
        bits += 0x7FFFu + lsb;
        out[i] = (uint16_t)(bits >> 16);
    }
}

inline void bf16_add_4096x3840_avx512(
    uint16_t*       out_bf16,
    const uint16_t* a_bf16,
    const uint16_t* b_bf16,
    int rows = 4096,
    int cols = 3840
) {
    constexpr int VEC = 16; // 16 elements per chunk
    const size_t n = (size_t)rows * (size_t)cols;

    size_t i = 0;
    for (; i + VEC <= n; i += VEC) {
        __m512 va = load_bf16x16_as_f32(a_bf16 + i);
        __m512 vb = load_bf16x16_as_f32(b_bf16 + i);
        __m512 vc = _mm512_add_ps(va, vb);
        store_f32x16_as_bf16(out_bf16 + i, vc);
    }

    // tail (not needed for 4096*3840 since divisible by 16, but safe)
    for (; i < n; ++i) {
        uint32_t fa = (uint32_t)a_bf16[i] << 16;
        uint32_t fb = (uint32_t)b_bf16[i] << 16;
        float aa = *reinterpret_cast<float*>(&fa);
        float bb = *reinterpret_cast<float*>(&fb);
        float cc = aa + bb;

        uint32_t bits = *reinterpret_cast<uint32_t*>(&cc);
        uint32_t lsb  = (bits >> 16) & 1u;
        bits += 0x7FFFu + lsb;
        out_bf16[i] = (uint16_t)(bits >> 16);
    }
}

static inline void unpatchify_bf16_no_vector(
    const dtype_out* x_in,
    int64_t     L,
    int64_t     K,
    dtype_out*       x_out,
    int64_t     F,
    int64_t     H,
    int64_t     W,
    int64_t     patch_size,   // pH = pW
    int64_t     f_patch_size, // pF
    int64_t     outC
) {
    const int64_t pH = patch_size;
    const int64_t pW = patch_size;
    const int64_t pF = f_patch_size;

    // divisibility required by Python code's integer divisions
    assert(F % pF == 0);
    assert(H % pH == 0);
    assert(W % pW == 0);

    const int64_t Fp = F / pF;
    const int64_t Hp = H / pH;
    const int64_t Wp = W / pW;

    const int64_t ori_len = Fp * Hp * Wp;          // x[:ori_len]
    const int64_t expectK = pF * pH * pW * outC;   // last dim after view
    assert(K == expectK);
    assert(L >= ori_len);

    // token index mapping: t <-> (fb,hb,wb)
    // t = (fb*Hp + hb)*Wp + wb
    for (int64_t t = 0; t < ori_len; ++t) {
        int64_t tmp = t;
        const int64_t wb = tmp % Wp; tmp /= Wp;
        const int64_t hb = tmp % Hp; tmp /= Hp;
        const int64_t fb = tmp; // 0..Fp-1

        const dtype_out* token_base = x_in + t * K; // row t, length K

        for (int64_t fo = 0; fo < pF; ++fo) {
            const int64_t f = fb * pF + fo;
            for (int64_t ho = 0; ho < pH; ++ho) {
                const int64_t h = hb * pH + ho;
                for (int64_t wo = 0; wo < pW; ++wo) {
                    const int64_t w = wb * pW + wo;

                    // within-token offset for (fo,ho,wo, c)
                    const int64_t patch_idx = ( (fo * pH + ho) * pW + wo ); // 0..pF*pH*pW-1
                    const int64_t in_off = patch_idx * outC;

                    // write all channels to out[c,f,h,w]
                    for (int64_t c = 0; c < outC; ++c) {
                        const dtype_out v = token_base[in_off + c];

                        // out index for [outC, F, H, W] contiguous
                        const int64_t out_idx = (((c * F + f) * H + h) * W + w);
                        x_out[out_idx] = v;
                    }
                }
            }
        }
    }
}


static inline void copy64_u16_to_bf16_avx512(const uint16_t* src_u16,
    std::bfloat16_t* dst_bf16) {
// store as 16-bit lanes; two 512b moves = 64 * u16
__m512i a = _mm512_loadu_si512((const void*)(src_u16));
__m512i b = _mm512_loadu_si512((const void*)(src_u16 + 32));
_mm512_storeu_si512((void*)(dst_bf16),      a);
_mm512_storeu_si512((void*)(dst_bf16 + 32), b);
}

// A_u16: [rows x Cin]  (Cin=64, elements are bf16 bits in uint16_t)
// B_out: [rows x Cout] (Cout=512, elements are std::bfloat16_t)
// tile_8x=true  -> replicate each row 8× across columns
// tile_8x=false -> put in first 64 cols, zero-fill the rest
void copy_into_B_u16_to_bf16(const uint16_t*       A_u16,
std::bfloat16_t*      B_out,
size_t rows,
size_t Cin,   // 64
size_t Cout,  // 512
bool   tile_8x)
{
// Fast path for your exact sizes
if (Cin == 64 && Cout == 512) {
for (size_t r = 0; r < rows; ++r) {
const uint16_t*      src = A_u16 + r*Cin;
std::bfloat16_t*     dst = B_out + r*Cout;

if (tile_8x) {
// replicate 8 tiles of 64
for (int t = 0; t < 8; ++t)
copy64_u16_to_bf16_avx512(src, dst + t*Cin);
} else {
// copy first 64, zero the remaining 448
copy64_u16_to_bf16_avx512(src, dst);
// bf16 zero is 0x0000 → byte zeroing is correct
std::memset(dst + Cin, 0, (Cout - Cin) * sizeof(std::bfloat16_t));
}
}
return;
}

// Generic fallback (any Cin/Cout); still bitwise copy
for (size_t r = 0; r < rows; ++r) {
const uint16_t*      src = A_u16 + r*Cin;
std::bfloat16_t*     dst = B_out + r*Cout;

if (tile_8x && Cout % Cin == 0) {
const size_t reps = Cout / Cin;
for (size_t t = 0; t < reps; ++t)
std::memcpy(dst + t*Cin, src, Cin * sizeof(std::bfloat16_t));
} else {
// place on the left, zero-pad the rest
std::memcpy(dst, src, Cin * sizeof(std::bfloat16_t));
std::memset(dst + Cin, 0, (Cout - Cin) * sizeof(std::bfloat16_t));
}
}
}
static inline void copy_bf16_to_bf16_avx512(const std::bfloat16_t* src,
     std::bfloat16_t* dst,
     size_t count) {
    const uint16_t* src_u16 = reinterpret_cast<const uint16_t*>(src);
    uint16_t*       dst_u16 = reinterpret_cast<uint16_t*>(dst);

    size_t i = 0;
    const size_t V = 32;
    for (; i + V <= count; i += V) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src_u16 + i));
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_u16 + i), v);
    }
    if (i < count) {
    std::memcpy(dst_u16 + i, src_u16 + i, (count - i) * sizeof(uint16_t));
    }
}

static inline void set_bf16_zero_avx512(std::bfloat16_t* dst, size_t count) {
    uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(dst);
    size_t i = 0;
    const size_t V = 32;
    const __m512i z = _mm512_setzero_si512();
    for (; i + V <= count; i += V) {
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst_u16 + i), z);
    }
    if (i < count) {
    std::memset(dst_u16 + i, 0, (count - i) * sizeof(uint16_t));
    }
}

/**
* AVX-512 version of your code
*
* N_size: length of bias vectors
* K_size, M_size as in your loops
* All buffers are sized appropriately:
*   bias, bias_img             : [N_size]
*   B_in                       : [K_size * N_size]
*   A_in                       : [M_size * K_size]
*   vector_data_silu           : [K_size]
*/
void do_work_avx512(const uint16_t* u16_bias,
const uint16_t* u16_bias_img,
const uint16_t* u16_B,
size_t N_size,
size_t K_size,
size_t M_size,
std::bfloat16_t*       bias,
std::bfloat16_t*       bias_img,
std::bfloat16_t*       B_in,
std::bfloat16_t*       A_in,
const std::bfloat16_t* vector_data_silu)
{
    // 1) bias and bias_img: bit-pattern copy u16 -> bf16
    copy_u16_to_bf16_avx512(u16_bias,     bias,     N_size);
    copy_u16_to_bf16_avx512(u16_bias_img, bias_img, N_size);

    // 2) B_in: bit-pattern copy u16 -> bf16, total K*N elements
    const size_t KN = K_size * N_size;
    copy_u16_to_bf16_avx512(u16_B, B_in, KN);

    // 3) A_in:
    //    - row 0: copy vector_data_silu
    //    - rows 1..M-1: zero
    if (M_size > 0) {
    copy_bf16_to_bf16_avx512(vector_data_silu, A_in, K_size);
    }
    if (M_size > 1) {
    set_bf16_zero_avx512(A_in + K_size, (M_size - 1) * K_size);
    // (memset would also work because bf16 zero is 0x0000)
    // std::memset(reinterpret_cast<uint16_t*>(A_in + K_size), 0,
    //             (M_size - 1) * K_size * sizeof(uint16_t));
    }
}

#endif // COPY_DATA_HPP