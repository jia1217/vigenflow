#ifndef HEAD_QK_HPP
#define HEAD_QK_HPP

#include <bit>      // for std::bit_cast
#include <cstddef>  // for size_t
#include "typedef.hpp"
#include "buffer.hpp"
#include <immintrin.h>


void pad_square_bf16_3840_to_4096(
    const std::vector<uint16_t>& src,
    buffer<dtype_out>&           dst,
    int                          offset              // column offset in dst
)
{
    const int oldN = 3840;
    const int newN = 4096;

    // 1) fill dst with 0 (bf16 zero)
    std::fill(dst.data(), dst.data() + newN * newN, dtype_out(0.0f));

    // reinterpret raw u16 data as dtype_out (bf16) elements
    const dtype_out* src_base =
        reinterpret_cast<const dtype_out*>(src.data());

    // 2) copy each 3840-wide row into the top-left of the 4096-wide dst
    for (int r = 0; r < oldN; ++r) {
        const dtype_out* src_row = src_base + r * oldN;
        dtype_out*       dst_row = dst.data() + r * newN + offset;

        std::memcpy(dst_row, src_row, oldN * sizeof(dtype_out));
    }
}

void pad_bf16_10240x3840_to_10240x4096(
    const std::vector<uint16_t>& src_u16,  // size = 10240*3840 (bf16 bits)
    buffer<__bf16>&              dst,      // size = 10240*4096
    int                          offset
) {
    const int rows = 10240;
    const int oldN = 3840;
    const int newN = 4096;

    // optional asserts if you have size()
    // assert((int)src_u16.size() == rows * oldN);
    // assert((int)dst.size() == rows * newN);
    // assert(offset >= 0 && offset + oldN <= newN);

    // fill dst with bf16 zero
    std::fill(dst.data(), dst.data() + (size_t)rows * newN, (__bf16)0);

    // reinterpret src bits as __bf16 storage (no conversion)
    const __bf16* src = reinterpret_cast<const __bf16*>(src_u16.data());

    for (int r = 0; r < rows; ++r) {
        const __bf16* src_row = src + (size_t)r * oldN;
        __bf16*       dst_row = dst.data() + (size_t)r * newN + offset;
        std::memcpy(dst_row, src_row, (size_t)oldN * sizeof(__bf16));
    }
}
void concat_rows_stream(
    const dtype_out* __restrict src1,
    const dtype_out* __restrict src2,
    dtype_out* __restrict dst,
    size_t rows)
{
    constexpr size_t COL_SRC1 = 3072;
    constexpr size_t COL_SRC2 = 3072 * 4;
    constexpr size_t COL_DST  = 3072 * 5;

    constexpr size_t VEC_BYTES = 64;
    constexpr size_t ELEM_SIZE = sizeof(dtype_out); // bf16 -> 2
    constexpr size_t VEC_ELEMS = VEC_BYTES / ELEM_SIZE; // 32 elems

    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t* s1 = reinterpret_cast<const uint16_t*>(src1) + r * COL_SRC1;
        const uint16_t* s2 = reinterpret_cast<const uint16_t*>(src2) + r * COL_SRC2;
        uint16_t*       d  = reinterpret_cast<uint16_t*>(dst)  + r * COL_DST;

        // copy block0
        for (size_t c = 0; c < COL_SRC1; c += VEC_ELEMS) {
            __m512i v = _mm512_load_si512((const __m512i*)(s1 + c));
            _mm512_stream_si512((__m512i*)(d + c), v);
        }
        // copy block1
        for (size_t c = 0; c < COL_SRC2; c += VEC_ELEMS) {
            __m512i v = _mm512_load_si512((const __m512i*)(s2 + c));
            _mm512_stream_si512((__m512i*)(d + COL_SRC1 + c), v);
        }
    }
    _mm_mfence();   // ensure streaming stores are visible
}
void concat_last_dim_bf16(
    const buffer<dtype_out> &a,   // [R, Da]
    const buffer<dtype_out> &b,   // [R, Db]
          buffer<dtype_out> &out, // [R, Da+Db]
    int R, int Da, int Db)
{
    const int Dout = Da + Db;
    out.resize( (size_t)R * (size_t)Dout );

    const dtype_out *a_ptr = a.data();
    const dtype_out *b_ptr = b.data();
          dtype_out *o_ptr = out.data();

    for (int r = 0; r < R; ++r) {
        // pointers to this row
        const dtype_out *a_row = a_ptr + (size_t)r * Da;
        const dtype_out *b_row = b_ptr + (size_t)r * Db;
              dtype_out *o_row = o_ptr + (size_t)r * Dout;

        // copy A row
        std::memcpy(o_row,
                    a_row,
                    (size_t)Da * sizeof(dtype_out));

        // append B row
        std::memcpy(o_row + Da,
                    b_row,
                    (size_t)Db * sizeof(dtype_out));
    }
}

void concat_2d_bf16(
    const buffer<dtype_out> &a,
    const buffer<dtype_out> &b,
          buffer<dtype_out> &out,
    int rows_a,
    int rows_b,
    int D)
{
    const size_t elems_a = (size_t)rows_a * (size_t)D;
    const size_t elems_b = (size_t)rows_b * (size_t)D;

    out.resize(elems_a + elems_b);

    dtype_out *out_ptr = out.data();

    // copy [0 .. rows_a-1]
    std::memcpy(out_ptr,
                a.data(),
                elems_a * sizeof(dtype_out));

    // copy [rows_a .. rows_a+rows_b-1]
    std::memcpy(out_ptr + elems_a,
                b.data(),
                elems_b * sizeof(dtype_out));
}


template<typename T>
void reorder_heads_avx512(
    const buffer<T> &src,    // [num_heads][seq_len][head_dim]
          buffer<T> &dst,    // [seq_len][num_heads][head_dim]
    size_t num_heads,
    size_t seq_len,
    size_t head_dim
) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "T must be trivially copyable");

    // Disallow in-place: layouts overlap unpredictably
    if (src.data() == dst.data()) {
        // If you need in-place, use a temp buffer.
        return;
    }

    const size_t row_width_elems = num_heads * head_dim;
    const size_t elem_bytes      = sizeof(T);
    const size_t block_bytes     = head_dim * elem_bytes;

    const char* __restrict sbase = reinterpret_cast<const char*>(src.data());
          char* __restrict dbase = reinterpret_cast<char*>(dst.data());

    // Process 64-byte chunks with AVX-512; memcpy the tiny tail.
    auto copy_block_avx512 = [&](const char* sp, char* dp) {
        size_t j = 0;
        // 64B per iteration
        for (; j + 64 <= block_bytes; j += 64) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(sp + j));
            _mm512_storeu_si512(reinterpret_cast<void*>(dp + j), v);
        }
        if (j < block_bytes) {
            std::memcpy(dp + j, sp + j, block_bytes - j);
        }
    };

    // Outer loop over heads keeps source reads sequential in memory.
    // You can parallelize over heads if desired:
    // #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < num_heads; ++h) {
        const size_t src_head_off_e = h * seq_len * head_dim;   // in elements
        for (size_t t = 0; t < seq_len; ++t) {
            const size_t src_base_e = src_head_off_e + t * head_dim;
            const size_t dst_base_e = (t * row_width_elems) + h * head_dim;

            const char* sp = sbase + src_base_e * elem_bytes;
                  char* dp = dbase + dst_base_e * elem_bytes;

            copy_block_avx512(sp, dp);
        }
    }
}


template<typename T>
void reorder_heads(
    const buffer<T> &src,    // layout: [num_heads][seq_len][head_dim]
    buffer<T>       &dst,    // layout: [seq_len][num_heads][head_dim]
    size_t           num_heads,
    size_t           seq_len,
    size_t           head_dim
) {
    // how wide one “time‐step” row is in dst:
    const size_t row_width = num_heads * head_dim;

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t t = 0; t < seq_len; ++t) {
            // base offsets into flat buffers:
            size_t src_base = (h * seq_len + t) * head_dim;
            size_t dst_base = (t * row_width + h * head_dim);

            for (size_t f = 0; f < head_dim; ++f) {
                dst[dst_base + f] = src[src_base + f];
            }
        }
    }
}

void reorder_bf16_avx2(const buffer<dtype_out> &in_buf,
    buffer<dtype_out>       &out_buf)
{
    constexpr int ROWS       = 4336;
    constexpr int COLS       = 3072;
    constexpr int CHUNKS     = 24;
    constexpr int CHUNK_SIZE = 128;
    constexpr int LANES      = 16;   // 16 × 16-bit = 256 bits

    // raw pointers to the underlying bf16 array:
    const dtype_out *in  = in_buf.data();
    dtype_out *out = out_buf.data();

    for (int j = 0; j < CHUNKS; ++j) {
    // each row’s j-th block starts at offset j*128 in that row
    const dtype_out *in_base  = in  + j * CHUNK_SIZE;
    dtype_out *out_base = out + j * ROWS * CHUNK_SIZE;

        for (int i = 0; i < ROWS; ++i) {
        // pointer to the start of that block in row i
        const dtype_out *in_row  = in_base  + i * COLS;
            dtype_out *out_row = out_base + i * CHUNK_SIZE;

        // copy 128 bf16’s in chunks of 16 (256 bits) via _mm256_loadu_si256
            for (int k = 0; k < CHUNK_SIZE; k += LANES) {
                // load 16×16-bit lanes
                __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(in_row + k)
                );
                // store them back out
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(out_row + k), v
                );
            }
        }
    }
}

void reorder_bf16_avx2_txt(const buffer<dtype_out> &in_buf,
    buffer<dtype_out>       &out_buf)
{
    constexpr int ROWS       = 256;
    constexpr int COLS       = 3072;
    constexpr int CHUNKS     = 24;
    constexpr int CHUNK_SIZE = 128;
    constexpr int LANES      = 16;   // 16 × 16-bit = 256 bits

    // raw pointers to the underlying bf16 array:
    const dtype_out *in  = in_buf.data();
    dtype_out *out = out_buf.data();

    for (int j = 0; j < CHUNKS; ++j) {
    // each row’s j-th block starts at offset j*128 in that row
    const dtype_out *in_base  = in  + j * CHUNK_SIZE;
    dtype_out *out_base = out + j * ROWS * CHUNK_SIZE;

        for (int i = 0; i < ROWS; ++i) {
        // pointer to the start of that block in row i
        const dtype_out *in_row  = in_base  + i * COLS;
            dtype_out *out_row = out_base + i * CHUNK_SIZE;

        // copy 128 bf16’s in chunks of 16 (256 bits) via _mm256_loadu_si256
            for (int k = 0; k < CHUNK_SIZE; k += LANES) {
                // load 16×16-bit lanes
                __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(in_row + k)
                );
                // store them back out
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(out_row + k), v
                );
            }
        }
    }
}

void reorder_bf16_avx2_img(const buffer<dtype_out> &in_buf,
    buffer<dtype_out>       &out_buf)
{
    constexpr int ROWS       = 4096;
    constexpr int COLS       = 3072;
    constexpr int CHUNKS     = 24;
    constexpr int CHUNK_SIZE = 128;
    constexpr int LANES      = 16;   // 16 × 16-bit = 256 bits

    // raw pointers to the underlying bf16 array:
    const dtype_out *in  = in_buf.data();
    dtype_out *out = out_buf.data();

    for (int j = 0; j < CHUNKS; ++j) {
    // each row’s j-th block starts at offset j*128 in that row
    const dtype_out *in_base  = in  + j * CHUNK_SIZE;
    dtype_out *out_base = out + j * ROWS * CHUNK_SIZE;

        for (int i = 0; i < ROWS; ++i) {
        // pointer to the start of that block in row i
        const dtype_out *in_row  = in_base  + i * COLS;
            dtype_out *out_row = out_base + i * CHUNK_SIZE;

        // copy 128 bf16’s in chunks of 16 (256 bits) via _mm256_loadu_si256
            for (int k = 0; k < CHUNK_SIZE; k += LANES) {
                // load 16×16-bit lanes
                __m256i v = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(in_row + k)
                );
                // store them back out
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(out_row + k), v
                );
            }
        }
    }
}


void reorder_2D_to_3D_crop_avx2(const buffer<dtype_out>& in_buf,
    buffer<dtype_out>&       out_buf)
{
static_assert(sizeof(dtype_out) == 2, "BF16 storage must be 16-bit");

constexpr int Lin  = 4336;     // input rows
constexpr int C    = 3072;     // input cols
constexpr int B    = 24;       // heads/chunks
constexpr int D    = 128;      // per-chunk width
constexpr int Lout = 4336;     // keep first 4336 rows (drop 16)

const dtype_out* __restrict in  = in_buf.data();   // shape [Lin, C] row-major
dtype_out* __restrict out = out_buf.data();  // shape [B, Lout, D] packed

// Parallelize over (b, l)
#pragma omp parallel for collapse(2) schedule(static)
for (int b = 0; b < B; ++b) {
for (int l = 0; l < Lout; ++l) {
// source: row l, columns [b*D ... b*D + D)
const dtype_out* __restrict src_e = in  + std::size_t(l) * C + b * D;
// dest:   out[b, l, :]
dtype_out*       __restrict dst_e = out + (std::size_t(b) * Lout + l) * D;

// prefetch next row chunk to help streaming
_mm_prefetch(reinterpret_cast<const char*>(src_e + C), _MM_HINT_T0);

// copy D=128 bf16 = 256 bytes as 8 × 32B moves
const __m256i* src = reinterpret_cast<const __m256i*>(src_e);
__m256i* dst = reinterpret_cast<__m256i*>(dst_e);

__m256i v0 = _mm256_loadu_si256(src + 0); _mm256_storeu_si256(dst + 0, v0);
__m256i v1 = _mm256_loadu_si256(src + 1); _mm256_storeu_si256(dst + 1, v1);
__m256i v2 = _mm256_loadu_si256(src + 2); _mm256_storeu_si256(dst + 2, v2);
__m256i v3 = _mm256_loadu_si256(src + 3); _mm256_storeu_si256(dst + 3, v3);
__m256i v4 = _mm256_loadu_si256(src + 4); _mm256_storeu_si256(dst + 4, v4);
__m256i v5 = _mm256_loadu_si256(src + 5); _mm256_storeu_si256(dst + 5, v5);
__m256i v6 = _mm256_loadu_si256(src + 6); _mm256_storeu_si256(dst + 6, v6);
__m256i v7 = _mm256_loadu_si256(src + 7); _mm256_storeu_si256(dst + 7, v7);
}
}
}



// NOTE: works best when your buffers are 64-byte aligned.
// Compile:  -O3 -mavx512f -mavx512bw -fopenmp   (add -mavx512bf16 only if you later need BF16 math)

inline void concat5_bf16_avx512(
    const uint16_t* __restrict img_attention_reorder, // [seg_size][seg_num]
    const uint16_t* __restrict mlp_bf16_gelu_all,     // [seg_size][seg_num*4]
    uint16_t* __restrict       all_concat_out,        // [seg_size][seg_num*5]
    size_t seg_size,
    size_t seg_num)
{
    constexpr size_t VW = 32;                 // 32 bf16 per 512b register
    const size_t out_stride = seg_num * 5;    // per-row stride in output
    const size_t mlp_stride = seg_num * 4;    // per-row stride in mlp

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)seg_size; ++i) {
        // Base pointers for row i
        const uint16_t* src0 = img_attention_reorder + i*seg_num;
        const uint16_t* src1 = mlp_bf16_gelu_all   + i*mlp_stride + 0*seg_num;
        const uint16_t* src2 = mlp_bf16_gelu_all   + i*mlp_stride + 1*seg_num;
        const uint16_t* src3 = mlp_bf16_gelu_all   + i*mlp_stride + 2*seg_num;
        const uint16_t* src4 = mlp_bf16_gelu_all   + i*mlp_stride + 3*seg_num;

        uint16_t* dst0 = all_concat_out + i*out_stride + 0*seg_num;
        uint16_t* dst1 = all_concat_out + i*out_stride + 1*seg_num;
        uint16_t* dst2 = all_concat_out + i*out_stride + 2*seg_num;
        uint16_t* dst3 = all_concat_out + i*out_stride + 3*seg_num;
        uint16_t* dst4 = all_concat_out + i*out_stride + 4*seg_num;

        size_t j = 0;
        for (; j + VW <= seg_num; j += VW) {
            // (optional) prefetch next chunk
            _mm_prefetch((const char*)(src0 + j + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(src1 + j + 256), _MM_HINT_T0);

            __m512i a0 = _mm512_loadu_si512((const void*)(src0 + j));
            __m512i b0 = _mm512_loadu_si512((const void*)(src1 + j));
            __m512i b1 = _mm512_loadu_si512((const void*)(src2 + j));
            __m512i b2 = _mm512_loadu_si512((const void*)(src3 + j));
            __m512i b3 = _mm512_loadu_si512((const void*)(src4 + j));

            _mm512_storeu_si512((void*)(dst0 + j), a0);
            _mm512_storeu_si512((void*)(dst1 + j), b0);
            _mm512_storeu_si512((void*)(dst2 + j), b1);
            _mm512_storeu_si512((void*)(dst3 + j), b2);
            _mm512_storeu_si512((void*)(dst4 + j), b3);
        }
        // Tail (handles non-multiple-of-32 widths)
        for (; j < seg_num; ++j) {
            dst0[j] = src0[j];
            dst1[j] = src1[j];
            dst2[j] = src2[j];
            dst3[j] = src3[j];
            dst4[j] = src4[j];
        }
    }
}



inline void copy_u16_to_bf16_avx512(const uint16_t* __restrict src,
    dtype_out* __restrict dst, size_t n)
{
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {                       // 32×u16 per 512b reg
    __m512i v = _mm512_loadu_si512((const void*)(src + i));
    _mm512_storeu_si512((void*)(dst + i), v);        // bit-identical copy
    }
    // tiny tail
    for (; i < n; ++i) dst[i] = std::bit_cast<dtype_out>(src[i]);
}

inline void copy_row_bf16_avx512(const dtype_out* __restrict src,
 dtype_out* __restrict dst, size_t n)
{
size_t i = 0;
for (; i + 32 <= n; i += 32) {
__m512i v = _mm512_loadu_si512((const void*)(src + i));
_mm512_storeu_si512((void*)(dst + i), v);
}
for (; i < n; ++i) dst[i] = src[i];
}

inline void zero_bf16_avx512(dtype_out* __restrict dst, size_t n)
{
__m512i z = _mm512_setzero_si512();
size_t i = 0;
for (; i + 32 <= n; i += 32) _mm512_storeu_si512((void*)(dst + i), z);
if (i < n) std::memset(dst + i, 0, (n - i) * sizeof(dtype_out));
}

// Cache-blocked transpose (handles any sizes, no slow gathers)
inline void transpose_NxK_to_KxN_blocked(const dtype_out* __restrict src_NK, // [N][K]
         dtype_out* __restrict dst_KN,       // [K][N]
         size_t N, size_t K)
{
    constexpr size_t T = 64;            // tile; 64×64 works well on most CPUs
    for (size_t rb = 0; rb < N; rb += T) {
    const size_t rn = std::min(T, N - rb);
    for (size_t cb = 0; cb < K; cb += T) {
    const size_t cn = std::min(T, K - cb);

    // For each tile-column 'c', write a contiguous column into dst.
    for (size_t c = 0; c < cn; ++c) {
    const dtype_out* s = src_NK + rb * K + (cb + c);
    dtype_out*       d = dst_KN + (cb + c) * N + rb;

    size_t r = 0;
    // manual unroll (good ILP; avoids costly gathers for 16-bit)
    for (; r + 32 <= rn; r += 32) {
    d[r +  0] = s[(r +  0) * K];
    d[r +  1] = s[(r +  1) * K];
    d[r +  2] = s[(r +  2) * K];
    d[r +  3] = s[(r +  3) * K];
    d[r +  4] = s[(r +  4) * K];
    d[r +  5] = s[(r +  5) * K];
    d[r +  6] = s[(r +  6) * K];
    d[r +  7] = s[(r +  7) * K];
    d[r +  8] = s[(r +  8) * K];
    d[r +  9] = s[(r +  9) * K];
    d[r + 10] = s[(r + 10) * K];
    d[r + 11] = s[(r + 11) * K];
    d[r + 12] = s[(r + 12) * K];
    d[r + 13] = s[(r + 13) * K];
    d[r + 14] = s[(r + 14) * K];
    d[r + 15] = s[(r + 15) * K];
    d[r + 16] = s[(r + 16) * K];
    d[r + 17] = s[(r + 17) * K];
    d[r + 18] = s[(r + 18) * K];
    d[r + 19] = s[(r + 19) * K];
    d[r + 20] = s[(r + 20) * K];
    d[r + 21] = s[(r + 21) * K];
    d[r + 22] = s[(r + 22) * K];
    d[r + 23] = s[(r + 23) * K];
    d[r + 24] = s[(r + 24) * K];
    d[r + 25] = s[(r + 25) * K];
    d[r + 26] = s[(r + 26) * K];
    d[r + 27] = s[(r + 27) * K];
    d[r + 28] = s[(r + 28) * K];
    d[r + 29] = s[(r + 29) * K];
    d[r + 30] = s[(r + 30) * K];
    d[r + 31] = s[(r + 31) * K];
    }
    for (; r < rn; ++r) d[r] = s[r * K];
    }
    }
    }
}
////////////////////////////////////////////////////////////////////////////////////

void prepare_inputs_avx512(
    const uint16_t* __restrict u16_bias_img, size_t N_size,
    const uint16_t* __restrict u16_B_img,   size_t K_size, // B has size K*N
    dtype_out* __restrict bias_img,
    dtype_out* __restrict B_in_T_img,                                // [N][K]
    dtype_out* __restrict A_in, const dtype_out* __restrict vector_data_silu,
    size_t M_size,
    dtype_out* __restrict B_in)                                      // [K][N]
{
    // 1) u16 -> bf16 (bit copy)
    copy_u16_to_bf16_avx512(u16_bias_img, bias_img, N_size);

    // 2) u16 -> bf16 for the whole B table
    copy_u16_to_bf16_avx512(u16_B_img, B_in_T_img, (size_t)K_size * N_size);

    // 3) A_in: first row = vector_data_silu, remaining rows = 0
    if (M_size) {
        copy_row_bf16_avx512(vector_data_silu, A_in, K_size);
        zero_bf16_avx512(A_in + K_size, (M_size - 1) * (size_t)K_size);
    }

    // 4) B transpose: [N×K] (row-major) -> [K×N] (row-major)
    transpose_NxK_to_KxN_blocked(B_in_T_img, B_in, N_size, K_size);
}
///////////////////////////////////////////////////////////////////////////////////////

inline void copy_bf16_avx512(const dtype_out* __restrict src,
    dtype_out* __restrict dst, size_t n) {
size_t i=0;
for (; i+32<=n; i+=32) {
__m512i v = _mm512_loadu_si512((const void*)(src+i));
_mm512_storeu_si512((void*)(dst+i), v);
}
if (i<n) std::memcpy(dst+i, src+i, (n-i)*sizeof(dtype_out));
}

// convert 3×u16 → 3×bf16 in one pass
inline void cvt3_u16_to_bf16(const uint16_t* __restrict s0,
    const uint16_t* __restrict s1,
    const uint16_t* __restrict s2,
    dtype_out* __restrict d0,
    dtype_out* __restrict d1,
    dtype_out* __restrict d2,
    size_t n) {
size_t i=0;
for (; i+32<=n; i+=32) {
__m512i a=_mm512_loadu_si512((const void*)(s0+i));
__m512i b=_mm512_loadu_si512((const void*)(s1+i));
__m512i c=_mm512_loadu_si512((const void*)(s2+i));
_mm512_storeu_si512((void*)(d0+i), a);
_mm512_storeu_si512((void*)(d1+i), b);
_mm512_storeu_si512((void*)(d2+i), c);
}
for (; i<n; ++i) {
d0[i] = std::bit_cast<dtype_out>(s0[i]);
d1[i] = std::bit_cast<dtype_out>(s1[i]);
d2[i] = std::bit_cast<dtype_out>(s2[i]);
}
}

/**
* Fused convert+transpose for Q/K/V:
*   inputs  (u16):  Wq, Wk, Wv  shaped [N x K] row-major
*   outputs (bf16): Bq, Bk, Bv  shaped [K x N] row-major
*/
void transpose_qkv_u16_to_bf16_KN(
size_t K, size_t N,

const uint16_t* __restrict Wq,   // [N*K]
const uint16_t* __restrict Wk,   // [N*K]
const uint16_t* __restrict Wv,   // [N*K]

dtype_out* __restrict Bq,             // [K*N]
dtype_out* __restrict Bk,             // [K*N]
dtype_out* __restrict Bv)             // [K*N]
{
// tile to keep working set in L2
constexpr size_t T = 64;
    for (size_t rb=0; rb<N; rb+=T) {
    size_t rn = std::min(T, N - rb);
    for (size_t cb=0; cb<K; cb+=T) {
    size_t cn = std::min(T, K - cb);

    // process each column of the tile at once (best for 16-bit, no gathers)
    for (size_t c=0; c<cn; ++c) {
    const uint16_t* sq = Wq + (rb*K) + (cb + c);
    const uint16_t* sk = Wk + (rb*K) + (cb + c);
    const uint16_t* sv = Wv + (rb*K) + (cb + c);

    dtype_out* dq = Bq + (cb + c)*N + rb;
    dtype_out* dk = Bk + (cb + c)*N + rb;
    dtype_out* dv = Bv + (cb + c)*N + rb;

    size_t r=0;
    // unrolled scalar stores (fast for 16-bit strided loads)
    for (; r+32<=rn; r+=32) {
    dq[r+ 0]=std::bit_cast<dtype_out>(sq[(r+ 0)*K]);
    dq[r+ 1]=std::bit_cast<dtype_out>(sq[(r+ 1)*K]);
    dq[r+ 2]=std::bit_cast<dtype_out>(sq[(r+ 2)*K]);
    dq[r+ 3]=std::bit_cast<dtype_out>(sq[(r+ 3)*K]);
    dq[r+ 4]=std::bit_cast<dtype_out>(sq[(r+ 4)*K]);
    dq[r+ 5]=std::bit_cast<dtype_out>(sq[(r+ 5)*K]);
    dq[r+ 6]=std::bit_cast<dtype_out>(sq[(r+ 6)*K]);
    dq[r+ 7]=std::bit_cast<dtype_out>(sq[(r+ 7)*K]);
    dq[r+ 8]=std::bit_cast<dtype_out>(sq[(r+ 8)*K]);
    dq[r+ 9]=std::bit_cast<dtype_out>(sq[(r+ 9)*K]);
    dq[r+10]=std::bit_cast<dtype_out>(sq[(r+10)*K]);
    dq[r+11]=std::bit_cast<dtype_out>(sq[(r+11)*K]);
    dq[r+12]=std::bit_cast<dtype_out>(sq[(r+12)*K]);
    dq[r+13]=std::bit_cast<dtype_out>(sq[(r+13)*K]);
    dq[r+14]=std::bit_cast<dtype_out>(sq[(r+14)*K]);
    dq[r+15]=std::bit_cast<dtype_out>(sq[(r+15)*K]);
    dq[r+16]=std::bit_cast<dtype_out>(sq[(r+16)*K]);
    dq[r+17]=std::bit_cast<dtype_out>(sq[(r+17)*K]);
    dq[r+18]=std::bit_cast<dtype_out>(sq[(r+18)*K]);
    dq[r+19]=std::bit_cast<dtype_out>(sq[(r+19)*K]);
    dq[r+20]=std::bit_cast<dtype_out>(sq[(r+20)*K]);
    dq[r+21]=std::bit_cast<dtype_out>(sq[(r+21)*K]);
    dq[r+22]=std::bit_cast<dtype_out>(sq[(r+22)*K]);
    dq[r+23]=std::bit_cast<dtype_out>(sq[(r+23)*K]);
    dq[r+24]=std::bit_cast<dtype_out>(sq[(r+24)*K]);
    dq[r+25]=std::bit_cast<dtype_out>(sq[(r+25)*K]);
    dq[r+26]=std::bit_cast<dtype_out>(sq[(r+26)*K]);
    dq[r+27]=std::bit_cast<dtype_out>(sq[(r+27)*K]);
    dq[r+28]=std::bit_cast<dtype_out>(sq[(r+28)*K]);
    dq[r+29]=std::bit_cast<dtype_out>(sq[(r+29)*K]);
    dq[r+30]=std::bit_cast<dtype_out>(sq[(r+30)*K]);
    dq[r+31]=std::bit_cast<dtype_out>(sq[(r+31)*K]);

    dk[r+ 0]=std::bit_cast<dtype_out>(sk[(r+ 0)*K]);
    dk[r+ 1]=std::bit_cast<dtype_out>(sk[(r+ 1)*K]);
    dk[r+ 2]=std::bit_cast<dtype_out>(sk[(r+ 2)*K]);
    dk[r+ 3]=std::bit_cast<dtype_out>(sk[(r+ 3)*K]);
    dk[r+ 4]=std::bit_cast<dtype_out>(sk[(r+ 4)*K]);
    dk[r+ 5]=std::bit_cast<dtype_out>(sk[(r+ 5)*K]);
    dk[r+ 6]=std::bit_cast<dtype_out>(sk[(r+ 6)*K]);
    dk[r+ 7]=std::bit_cast<dtype_out>(sk[(r+ 7)*K]);
    dk[r+ 8]=std::bit_cast<dtype_out>(sk[(r+ 8)*K]);
    dk[r+ 9]=std::bit_cast<dtype_out>(sk[(r+ 9)*K]);
    dk[r+10]=std::bit_cast<dtype_out>(sk[(r+10)*K]);
    dk[r+11]=std::bit_cast<dtype_out>(sk[(r+11)*K]);
    dk[r+12]=std::bit_cast<dtype_out>(sk[(r+12)*K]);
    dk[r+13]=std::bit_cast<dtype_out>(sk[(r+13)*K]);
    dk[r+14]=std::bit_cast<dtype_out>(sk[(r+14)*K]);
    dk[r+15]=std::bit_cast<dtype_out>(sk[(r+15)*K]);
    dk[r+16]=std::bit_cast<dtype_out>(sk[(r+16)*K]);
    dk[r+17]=std::bit_cast<dtype_out>(sk[(r+17)*K]);
    dk[r+18]=std::bit_cast<dtype_out>(sk[(r+18)*K]);
    dk[r+19]=std::bit_cast<dtype_out>(sk[(r+19)*K]);
    dk[r+20]=std::bit_cast<dtype_out>(sk[(r+20)*K]);
    dk[r+21]=std::bit_cast<dtype_out>(sk[(r+21)*K]);
    dk[r+22]=std::bit_cast<dtype_out>(sk[(r+22)*K]);
    dk[r+23]=std::bit_cast<dtype_out>(sk[(r+23)*K]);
    dk[r+24]=std::bit_cast<dtype_out>(sk[(r+24)*K]);
    dk[r+25]=std::bit_cast<dtype_out>(sk[(r+25)*K]);
    dk[r+26]=std::bit_cast<dtype_out>(sk[(r+26)*K]);
    dk[r+27]=std::bit_cast<dtype_out>(sk[(r+27)*K]);
    dk[r+28]=std::bit_cast<dtype_out>(sk[(r+28)*K]);
    dk[r+29]=std::bit_cast<dtype_out>(sk[(r+29)*K]);
    dk[r+30]=std::bit_cast<dtype_out>(sk[(r+30)*K]);
    dk[r+31]=std::bit_cast<dtype_out>(sk[(r+31)*K]);

    dv[r+ 0]=std::bit_cast<dtype_out>(sv[(r+ 0)*K]);
    dv[r+ 1]=std::bit_cast<dtype_out>(sv[(r+ 1)*K]);
    dv[r+ 2]=std::bit_cast<dtype_out>(sv[(r+ 2)*K]);
    dv[r+ 3]=std::bit_cast<dtype_out>(sv[(r+ 3)*K]);
    dv[r+ 4]=std::bit_cast<dtype_out>(sv[(r+ 4)*K]);
    dv[r+ 5]=std::bit_cast<dtype_out>(sv[(r+ 5)*K]);
    dv[r+ 6]=std::bit_cast<dtype_out>(sv[(r+ 6)*K]);
    dv[r+ 7]=std::bit_cast<dtype_out>(sv[(r+ 7)*K]);
    dv[r+ 8]=std::bit_cast<dtype_out>(sv[(r+ 8)*K]);
    dv[r+ 9]=std::bit_cast<dtype_out>(sv[(r+ 9)*K]);
    dv[r+10]=std::bit_cast<dtype_out>(sv[(r+10)*K]);
    dv[r+11]=std::bit_cast<dtype_out>(sv[(r+11)*K]);
    dv[r+12]=std::bit_cast<dtype_out>(sv[(r+12)*K]);
    dv[r+13]=std::bit_cast<dtype_out>(sv[(r+13)*K]);
    dv[r+14]=std::bit_cast<dtype_out>(sv[(r+14)*K]);
    dv[r+15]=std::bit_cast<dtype_out>(sv[(r+15)*K]);
    dv[r+16]=std::bit_cast<dtype_out>(sv[(r+16)*K]);
    dv[r+17]=std::bit_cast<dtype_out>(sv[(r+17)*K]);
    dv[r+18]=std::bit_cast<dtype_out>(sv[(r+18)*K]);
    dv[r+19]=std::bit_cast<dtype_out>(sv[(r+19)*K]);
    dv[r+20]=std::bit_cast<dtype_out>(sv[(r+20)*K]);
    dv[r+21]=std::bit_cast<dtype_out>(sv[(r+21)*K]);
    dv[r+22]=std::bit_cast<dtype_out>(sv[(r+22)*K]);
    dv[r+23]=std::bit_cast<dtype_out>(sv[(r+23)*K]);
    dv[r+24]=std::bit_cast<dtype_out>(sv[(r+24)*K]);
    dv[r+25]=std::bit_cast<dtype_out>(sv[(r+25)*K]);
    dv[r+26]=std::bit_cast<dtype_out>(sv[(r+26)*K]);
    dv[r+27]=std::bit_cast<dtype_out>(sv[(r+27)*K]);
    dv[r+28]=std::bit_cast<dtype_out>(sv[(r+28)*K]);
    dv[r+29]=std::bit_cast<dtype_out>(sv[(r+29)*K]);
    dv[r+30]=std::bit_cast<dtype_out>(sv[(r+30)*K]);
    dv[r+31]=std::bit_cast<dtype_out>(sv[(r+31)*K]);
    }
    for (; r<rn; ++r) {
    dq[r] = std::bit_cast<dtype_out>(sq[r*K]);
    dk[r] = std::bit_cast<dtype_out>(sk[r*K]);
    dv[r] = std::bit_cast<dtype_out>(sv[r*K]);
    }
    }
    }
    }
}

/** Full prep when you have three B_in (Q, K, V) */
void prep_qkv_three_Bin(
    size_t K_size_1, size_t N_size_1, size_t all_size,

    // weights u16 [N×K]
    const uint16_t* __restrict u16_Wq,
    const uint16_t* __restrict u16_Wk,
    const uint16_t* __restrict u16_Wv,

    // biases u16 [N]
    const uint16_t* __restrict u16_bq,
    const uint16_t* __restrict u16_bk,
    const uint16_t* __restrict u16_bv,

    const dtype_out* __restrict image_md_data,

    // outputs
    dtype_out* __restrict B_in_q,   // [K×N]
    dtype_out* __restrict B_in_k,   // [K×N]
    dtype_out* __restrict B_in_v,   // [K×N]
    dtype_out* __restrict A_in_1,   // [all_size]
    dtype_out* __restrict bias_q,   // [N]
    dtype_out* __restrict bias_k,   // [N]
    dtype_out* __restrict bias_v)   // [N]
{
    // 1) transpose & convert all three
    transpose_qkv_u16_to_bf16_KN(K_size_1, N_size_1,
            u16_Wq, u16_Wk, u16_Wv,
            B_in_q, B_in_k, B_in_v);

    // 2) copy A
    copy_bf16_avx512(image_md_data, A_in_1, all_size);

    // 3) convert biases in one pass
    cvt3_u16_to_bf16(u16_bq, u16_bk, u16_bv,
    bias_q, bias_k, bias_v, N_size_1);
}
///////////////////////////////////////////////////////////////////////////////////////
inline void u16_to_bf16_copy_avx512(dtype_out* __restrict dst,
    const uint16_t* __restrict src,
    size_t n) {
size_t i = 0;
for (; i + 32 <= n; i += 32) {
__m512i v = _mm512_loadu_si512((const __m512i*)(src + i));
_mm512_storeu_si512((__m512i*)(dst + i), v);
}
// tail (alias-safe)
for (; i < n; ++i) dst[i] = std::bit_cast<dtype_out>(src[i]);
}

// Blocked transpose: dst[c*N + r] = bit_cast<bf16>(src[r*K + c])
inline void transpose_u16_to_bf16_blocked(const uint16_t* __restrict src,
    dtype_out* __restrict dst,
          size_t N, size_t K) {
if (!src || !dst) throw std::invalid_argument("null ptr");
if (N == 0 || K == 0) return;

constexpr int TC = 64;
constexpr int TR = 64;

// Parallelize if you want: add #pragma omp parallel for collapse(2)
for (size_t c0 = 0; c0 < K; c0 += TC) {
for (size_t r0 = 0; r0 < N; r0 += TR) {
const size_t cmax = (c0 + TC < K) ? (c0 + TC) : K;
const size_t rmax = (r0 + TR < N) ? (r0 + TR) : N;

for (size_t c = c0; c < cmax; ++c) {
    dtype_out* __restrict dcol = dst + c * N + r0;
const uint16_t* __restrict scol = src + r0 * K + c;

size_t o = 0;
// unrolled by 8; alias-safe stores via bit_cast
for (; o + 8 <= (rmax - r0); o += 8) {
dcol[o + 0] = std::bit_cast<dtype_out>(scol[0 * K]);
dcol[o + 1] = std::bit_cast<dtype_out>(scol[1 * K]);
dcol[o + 2] = std::bit_cast<dtype_out>(scol[2 * K]);
dcol[o + 3] = std::bit_cast<dtype_out>(scol[3 * K]);
dcol[o + 4] = std::bit_cast<dtype_out>(scol[4 * K]);
dcol[o + 5] = std::bit_cast<dtype_out>(scol[5 * K]);
dcol[o + 6] = std::bit_cast<dtype_out>(scol[6 * K]);
dcol[o + 7] = std::bit_cast<dtype_out>(scol[7 * K]);
scol += 8 * K;
}
for (; o < (rmax - r0); ++o) {
dcol[o] = std::bit_cast<dtype_out>(*scol);
scol += K;
}
}
}
}
}

inline void copy_then_zero_tail_avx512(dtype_out* __restrict dst,
       const dtype_out* __restrict src,
       size_t n_copy,
       size_t n_total) {
    if (!dst || !src) throw std::invalid_argument("null ptr");
    if (n_copy > n_total) throw std::invalid_argument("n_copy>n_total");

    size_t i = 0;
    for (; i + 32 <= n_copy; i += 32) {
    __m512i v = _mm512_loadu_si512((const __m512i*)(src + i));
    _mm512_storeu_si512((__m512i*)(dst + i), v);
    }
    for (; i < n_copy; ++i) dst[i] = src[i];

    __m512i z = _mm512_setzero_si512();
    for (; i + 32 <= n_total; i += 32) _mm512_storeu_si512((__m512i*)(dst + i), z);
    for (; i < n_total; ++i) dst[i] = dtype_out{};
    }

// High-level API (mirrors your original)
inline void prepare_inputs_avx512(
    dtype_out* __restrict bias_img2,                // [N]
    const uint16_t* __restrict u16_bias_img2, // [N]
    dtype_out* __restrict B_in_4,                   // [K*N]
    const uint16_t* __restrict u16_B_img2,    // [N*K]
    dtype_out* __restrict A_in_4,                   // [M*K]
    const dtype_out* __restrict img_atten_reorder,  // [(M-16)*K] valid
    size_t N, size_t K, size_t M)
    {
    // 1) bias copy (bit-cast)
    u16_to_bf16_copy_avx512(bias_img2, u16_bias_img2, N);

    // 2) transpose without temporary
    transpose_u16_to_bf16_blocked(u16_B_img2, B_in_4, N, K);

    // 3) copy + zero-pad (guard M >= 16)
    const size_t copy_rows = (M > 16) ? (M - 16) : 0;
    const size_t copyN     = copy_rows * K;
    const size_t total     = M * K;
    copy_then_zero_tail_avx512(A_in_4, img_atten_reorder, copyN, total);
}

////////////////////////////////////////////////////////////////////////////////////////////

static inline void transpose_u16_to_bf16_blocked(const uint16_t* __restrict src,
          dtype_out* __restrict dst,
          int N, int K) {
constexpr int TC = 64; // tile over columns
constexpr int TR = 64; // tile over rows

for (int c0 = 0; c0 < K; c0 += TC) {
for (int r0 = 0; r0 < N; r0 += TR) {
const int cmax = (c0 + TC < K) ? (c0 + TC) : K;
const int rmax = (r0 + TR < N) ? (r0 + TR) : N;

for (int c = c0; c < cmax; ++c) {
dtype_out* __restrict dcol = dst + (size_t)c * N + r0;
const uint16_t* __restrict scol = src + (size_t)r0 * K + c;

int r = r0;
// unrolled stride-walk; destination is contiguous
for (; r + 8 <= rmax; r += 8) {
dcol[r - r0 + 0] = std::bit_cast<dtype_out>(scol[0 * K]);
dcol[r - r0 + 1] = std::bit_cast<dtype_out>(scol[1 * K]);
dcol[r - r0 + 2] = std::bit_cast<dtype_out>(scol[2 * K]);
dcol[r - r0 + 3] = std::bit_cast<dtype_out>(scol[3 * K]);
dcol[r - r0 + 4] = std::bit_cast<dtype_out>(scol[4 * K]);
dcol[r - r0 + 5] = std::bit_cast<dtype_out>(scol[5 * K]);
dcol[r - r0 + 6] = std::bit_cast<dtype_out>(scol[6 * K]);
dcol[r - r0 + 7] = std::bit_cast<dtype_out>(scol[7 * K]);
scol += 8 * K;
}
for (; r < rmax; ++r) {
dcol[r - r0] = std::bit_cast<dtype_out>(*scol);
scol += K;
}
}
}
}
}



// ------------------------------------------------------------------
// M = 4602, K = 3072, N = 12288 ( = 3072*4 ); valid rows for A = 4592
// ------------------------------------------------------------------
static inline void prepare_inputs_avx512(
    dtype_out* __restrict bias_img2,                 // [N]
    const uint16_t* __restrict u16_bias_img2,   // [N]
    dtype_out* __restrict B_in_4,                    // [K * N] (transposed)
    const uint16_t* __restrict u16_B_img2,      // [N * K] (row-major)
    dtype_out* __restrict A_in_4,                    // [M * K]
    const dtype_out* __restrict image_md_data,       // [VALID_M * K]
    int M_size, int K_size, int N_size, int VALID_M_size)
{
    // runtime sizes (avoid constexpr here)
    const size_t M       = static_cast<size_t>(M_size);
    const size_t K       = static_cast<size_t>(K_size);
    const size_t N       = static_cast<size_t>(N_size);
    const size_t VALID_M = static_cast<size_t>(VALID_M_size);

    // basic safety checks (keep or replace with your own error handling)
    if (!bias_img2 || !u16_bias_img2 || !B_in_4 || !u16_B_img2 || !A_in_4 || !image_md_data)
        return; // or throw
    if (VALID_M > M) return;                    // or throw
    if (K > INT_MAX || N > INT_MAX) return;     // transpose helper takes int

    // 1) bias: bit-cast u16 -> bf16 (wide AVX-512 moves inside helper)
    u16_to_bf16_copy_avx512(bias_img2, u16_bias_img2, N);

    // 2) B: transpose directly from u16 -> bf16 (no temporary buffer)
    //    B_in_4[c*N + r] = bit_cast<bf16>(u16_B_img2[r*K + c]);
    transpose_u16_to_bf16_blocked(u16_B_img2, B_in_4,
                                   static_cast<int>(N),
                                   static_cast<int>(K));

    // 3) A: copy first VALID_M rows, then zero-pad the tail (M - VALID_M rows)
    const size_t elems_copy  = VALID_M * K;
    const size_t elems_total = M * K;
    copy_then_zero_tail_avx512(A_in_4, image_md_data, elems_copy, elems_total);
}

template<int M, int K, int N, int VALID_M>
static inline void prepare_inputs_avx512_fixed(
    dtype_out* __restrict bias_img2,
    const uint16_t* __restrict u16_bias_img2,
    dtype_out* __restrict B_in_4,
    const uint16_t* __restrict u16_B_img2,
    dtype_out* __restrict A_in_4,
    const dtype_out* __restrict image_md_data)
{
    static_assert(VALID_M <= M, "VALID_M must be <= M");
    u16_to_bf16_copy_avx512(bias_img2, u16_bias_img2, (size_t)N);
    transpose_u16_to_bf16_blocked(u16_B_img2, B_in_4, N, K);
    copy_then_zero_tail_avx512(A_in_4, image_md_data,
                               (size_t)VALID_M * K,
                               (size_t)M * K);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline void copy6_blocks_bf16_avx512(
    const dtype_out* __restrict base,  // [6*stride]
    dtype_out* __restrict out0,        // [stride]
    dtype_out* __restrict out1,        // [stride]
    dtype_out* __restrict out2,        // [stride]
    dtype_out* __restrict out3,        // [stride]
    dtype_out* __restrict out4,        // [stride]
    dtype_out* __restrict out5,        // [stride]
    size_t stride)
{
    const dtype_out* __restrict b0 = base + 0*stride;
    const dtype_out* __restrict b1 = base + 1*stride;
    const dtype_out* __restrict b2 = base + 2*stride;
    const dtype_out* __restrict b3 = base + 3*stride;
    const dtype_out* __restrict b4 = base + 4*stride;
    const dtype_out* __restrict b5 = base + 5*stride;

    size_t i = 0;
    // 32 bf16 per vector
    for (; i + 32 <= stride; i += 32) {
        __m512i v0 = _mm512_loadu_si512((const __m512i*)(b0 + i));
        __m512i v1 = _mm512_loadu_si512((const __m512i*)(b1 + i));
        __m512i v2 = _mm512_loadu_si512((const __m512i*)(b2 + i));
        __m512i v3 = _mm512_loadu_si512((const __m512i*)(b3 + i));
        __m512i v4 = _mm512_loadu_si512((const __m512i*)(b4 + i));
        __m512i v5 = _mm512_loadu_si512((const __m512i*)(b5 + i));

        _mm512_storeu_si512((__m512i*)(out0 + i), v0);
        _mm512_storeu_si512((__m512i*)(out1 + i), v1);
        _mm512_storeu_si512((__m512i*)(out2 + i), v2);
        _mm512_storeu_si512((__m512i*)(out3 + i), v3);
        _mm512_storeu_si512((__m512i*)(out4 + i), v4);
        _mm512_storeu_si512((__m512i*)(out5 + i), v5);
    }
    // tail (rare here; 3072 % 32 == 0 so this won't run)
    for (; i < stride; ++i) {
        out0[i] = b0[i];
        out1[i] = b1[i];
        out2[i] = b2[i];
        out3[i] = b3[i];
        out4[i] = b4[i];
        out5[i] = b5[i];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline __m256i to_bf16_rne(__m512 f32) {
    #ifdef __AVX512BF16__
        // HW path returns __m256bh -> bit-cast to __m256i for stores
        __m256bh bh = _mm512_cvtneps_pbh(f32);
        return std::bit_cast<__m256i>(bh);      // no cost, type-only cast
    #else
        // software RNE pack
        __m512i u   = _mm512_castps_si512(f32);
        __m512i lsb = _mm512_srli_epi32(u, 16);
        lsb         = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
        __m512i add = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
        u           = _mm512_add_epi32(u, add);
        u           = _mm512_srli_epi32(u, 16);               // now 16-bit payloads in 32-bit lanes
        return _mm512_cvtepi32_epi16(u);                      // pack to __m256i (16×u16)
    #endif
    }
    
    // ---- main kernel ----
    void add_bias_per_head_row_avx512_bf16(
        const dtype_out* __restrict bias_img_q,   // [HEADS * D_dim]
        const dtype_out* __restrict C_out_1,      // [HEADS * ROWS * D_dim]
        dtype_out*       __restrict img_q_out_bias,
        int HEADS, int ROWS, int D_dim)
    {
        // Parallelize outer loops if desired:
        // #pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < HEADS; ++h) {
            const dtype_out* __restrict bias = bias_img_q + (size_t)h * D_dim;
    
            for (int i = 0; i < ROWS; ++i) {
                const size_t base = ((size_t)h * ROWS + i) * (size_t)D_dim;
    
                int d = 0;
                for (; d + 32 <= D_dim; d += 32) {
                    // load 32 bf16 from C and bias
                    __m512i c16 = _mm512_loadu_si512((const __m512i*)(C_out_1 + base + d));
                    __m512i b16 = _mm512_loadu_si512((const __m512i*)(bias       + d));
    
                    // expand low 16 lanes bf16 -> fp32
                    __m256i c_lo16 = _mm512_castsi512_si256(c16);
                    __m256i b_lo16 = _mm512_castsi512_si256(b16);
                    __m512i c_lo32 = _mm512_cvtepu16_epi32(c_lo16);
                    __m512i b_lo32 = _mm512_cvtepu16_epi32(b_lo16);
                    c_lo32 = _mm512_slli_epi32(c_lo32, 16);
                    b_lo32 = _mm512_slli_epi32(b_lo32, 16);
                    __m512  c_lo  = _mm512_castsi512_ps(c_lo32);
                    __m512  b_lo  = _mm512_castsi512_ps(b_lo32);
    
                    // expand high 16 lanes
                    __m256i c_hi16 = _mm512_extracti64x4_epi64(c16, 1);
                    __m256i b_hi16 = _mm512_extracti64x4_epi64(b16, 1);
                    __m512i c_hi32 = _mm512_cvtepu16_epi32(c_hi16);
                    __m512i b_hi32 = _mm512_cvtepu16_epi32(b_hi16);
                    c_hi32 = _mm512_slli_epi32(c_hi32, 16);
                    b_hi32 = _mm512_slli_epi32(b_hi32, 16);
                    __m512  c_hi  = _mm512_castsi512_ps(c_hi32);
                    __m512  b_hi  = _mm512_castsi512_ps(b_hi32);
    
                    // add in fp32
                    __m512 y_lo = _mm512_add_ps(c_lo, b_lo);
                    __m512 y_hi = _mm512_add_ps(c_hi, b_hi);
    
                    // convert back to bf16 and store
                    __m256i y_lo_bf16 = to_bf16_rne(y_lo);
                    __m256i y_hi_bf16 = to_bf16_rne(y_hi);
                    _mm256_storeu_si256((__m256i*)(img_q_out_bias + base + d),       y_lo_bf16);
                    _mm256_storeu_si256((__m256i*)(img_q_out_bias + base + d + 16), y_hi_bf16);
                }
    
                // tail (runs only if D_dim % 32 != 0)
                for (; d < D_dim; ++d) {
                    float x = std::bit_cast<float>(uint32_t(std::bit_cast<uint16_t>(C_out_1[base + d])) << 16);
                    float b = std::bit_cast<float>(uint32_t(std::bit_cast<uint16_t>(bias[d]))        << 16);
                    float y = x + b;
    #ifdef __AVX512BF16__
                    // scalar fallback using bf16 conversion
                    img_q_out_bias[base + d] = (dtype_out)y;
    #else
                    // manual RNE
                    uint32_t ui = std::bit_cast<uint32_t>(y);
                    ui += 0x7FFF + ((ui >> 16) & 1);
                    img_q_out_bias[base + d] = std::bit_cast<dtype_out>(uint16_t(ui >> 16));
    #endif
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    static inline __m512 bf16_load32_to_f32(const dtype_out* p) {
        __m512i u16x32 = _mm512_loadu_si512((const __m512i*)p);   // 32×u16
        __m256i lo16   = _mm512_castsi512_si256(u16x32);
        __m256i hi16   = _mm512_extracti64x4_epi64(u16x32, 1);
        __m512i lo32   = _mm512_cvtepu16_epi32(lo16);
        __m512i hi32   = _mm512_cvtepu16_epi32(hi16);
        lo32           = _mm512_slli_epi32(lo32, 16);
        hi32           = _mm512_slli_epi32(hi32, 16);
        // return as two packs; caller handles both
        // (we return lo now; hi returned by caller via ref)
        return _mm512_castsi512_ps(lo32);
    }
    static inline __m512 bf16_load32_hi_to_f32(const dtype_out* p) {
        __m512i u16x32 = _mm512_loadu_si512((const __m512i*)p);
        __m256i hi16   = _mm512_extracti64x4_epi64(u16x32, 1);
        __m512i hi32   = _mm512_cvtepu16_epi32(hi16);
        hi32           = _mm512_slli_epi32(hi32, 16);
        return _mm512_castsi512_ps(hi32);
    }
    
    // pack 16×fp32 -> 16×bf16 (RNE) as __m256i (for easy storing)
    static inline __m256i f32_to_bf16_rne_pack16(__m512 f32) {
    #ifdef __AVX512BF16__
        // HW path returns __m256bh; bit-cast to __m256i for store
        __m256bh bh = _mm512_cvtneps_pbh(f32);
        return std::bit_cast<__m256i>(bh);
    #else
        __m512i u   = _mm512_castps_si512(f32);
        __m512i lsb = _mm512_srli_epi32(u, 16);
        lsb         = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
        __m512i add = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
        u           = _mm512_add_epi32(u, add);
        u           = _mm512_srli_epi32(u, 16);
        return _mm512_cvtepi32_epi16(u);  // 16×u16 in __m256i
    #endif
    }
    
    // -------------------------------------------------------------
    // AVX-512 bf16 kernel: txt_v_out_bias[row*N + j] = C[row*N + j] + bias[j]
    // -------------------------------------------------------------
    void add_bias_rows_bf16_avx512(
        const dtype_out* __restrict C_out_2,   // [M*N]
        const dtype_out* __restrict bias_txt_v,// [N]
        dtype_out*       __restrict txt_v_out_bias,
        int M, int N)
    {
        // Optional: parallelize rows
        // #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            size_t base = (size_t)i * (size_t)N;
    
            int j = 0;
            for (; j + 32 <= N; j += 32) {
                // load 32 bf16 from C and bias -> fp32
                __m512 c_lo = bf16_load32_to_f32 (C_out_2   + base + j);
                __m512 b_lo = bf16_load32_to_f32 (bias_txt_v + j);
                __m512 c_hi = bf16_load32_hi_to_f32 (C_out_2   + base + j);
                __m512 b_hi = bf16_load32_hi_to_f32 (bias_txt_v + j);
    
                // add in fp32
                __m512 y_lo = _mm512_add_ps(c_lo, b_lo);
                __m512 y_hi = _mm512_add_ps(c_hi, b_hi);
    
                // convert back to bf16 and store
                __m256i y0 = f32_to_bf16_rne_pack16(y_lo);
                __m256i y1 = f32_to_bf16_rne_pack16(y_hi);
                _mm256_storeu_si256((__m256i*)(txt_v_out_bias + base + j),      y0);
                _mm256_storeu_si256((__m256i*)(txt_v_out_bias + base + j + 16), y1);
            }
    
            // tail (runs only if N % 32 != 0)
            for (; j < N; ++j) {
                // scalar bf16 add via fp32
                uint32_t uiC = (uint32_t)std::bit_cast<uint16_t>(C_out_2[base + j]) << 16;
                uint32_t uiB = (uint32_t)std::bit_cast<uint16_t>(bias_txt_v[j])     << 16;
                float y = std::bit_cast<float>(uiC) + std::bit_cast<float>(uiB);
    #ifdef __AVX512BF16__
                txt_v_out_bias[base + j] = (dtype_out)y;
    #else
                uint32_t ui = std::bit_cast<uint32_t>(y);
                ui += 0x7FFF + ((ui >> 16) & 1);                 // RNE
                txt_v_out_bias[base + j] = std::bit_cast<dtype_out>((uint16_t)(ui >> 16));
    #endif
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline void bitcast_copy_u16_to_16b_avx512(
        T* __restrict dst,               // e.g., std::bfloat16_t*
        const uint16_t* __restrict src,  // source buffer (u16)
        size_t n)
    {
        static_assert(sizeof(T) == 2, "dtype_out must be 16-bit");
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        size_t i = 0;
    
        // 32 × 16-bit elements per 512-bit vector
        for (; i + 32 <= n; i += 32) {
            __m512i v = _mm512_loadu_si512((const __m512i*)(src + i));
            _mm512_storeu_si512((__m512i*)(dst + i), v);
        }
    
        // scalar tail (<= 31 elements)
        for (; i < n; ++i) {
            // exact bit copy; no numeric conversion
            reinterpret_cast<uint16_t&>(dst[i]) = src[i];
        }
    #else
        // Fallback if AVX-512 not available
        for (size_t i = 0; i < n; ++i) {
            reinterpret_cast<uint16_t&>(dst[i]) = src[i];
        }
    #endif
    }



    static inline void transpose_u16row_to_bf16col_avx512(const uint16_t* __restrict src_u16,
        std::bfloat16_t* __restrict dst_col,
        int N, int K) {
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;
alignas(64) std::bfloat16_t tile[TILE_K * TILE_N]; // tile[t][r]

for (int c0 = 0; c0 < K; c0 += TILE_K) {
int tk = std::min(TILE_K, K - c0);
for (int r0 = 0; r0 < N; r0 += TILE_N) {
int tn = std::min(TILE_N, N - r0);

// Fill tile: tile[t][r] = bf16(src[r0+r][c0+t])
for (int t = 0; t < tk; ++t) {
const uint16_t* col_src = src_u16 + (c0 + t); // start of column
std::bfloat16_t* trow   = tile + (size_t)t * TILE_N;
for (int r = 0; r < tn; ++r) {
if ((r & 7) == 0)
_mm_prefetch((const char*)(col_src + (size_t)(r0 + r + 16) * K), _MM_HINT_T0);
trow[r] = std::bit_cast<std::bfloat16_t>(col_src[(size_t)(r0 + r) * K]);
}
}

// Store tile rows contiguously into dst
for (int t = 0; t < tk; ++t) {
std::bfloat16_t* d = dst_col + (size_t)(c0 + t) * N + r0;
const std::bfloat16_t* s = tile + (size_t)t * TILE_N;
int r = 0;
for (; r + 32 <= tn; r += 32) {
__m512i v = _mm512_loadu_si512((const void*)(s + r));
_mm512_storeu_si512((void*)(d + r), v);
}
for (; r < tn; ++r) d[r] = s[r];
}
}
}
}

/* ------------------------------------------------------------------------- */
/* Combined one-pass routine (no B_in_T_img2 temporary)                      */
/* ------------------------------------------------------------------------- */
void pack_inputs_onepass_avx512(
// inputs
const uint16_t*        __restrict u16_bias_img2, // [N]
const uint16_t*        __restrict u16_B_img2,    // [N*K] row-major (u16 BF16 bits)
const std::bfloat16_t* __restrict img_md_data,   // [(M-16)*K] bf16
// outputs
std::bfloat16_t*       __restrict bias_img2,     // [N] bf16
std::bfloat16_t*       __restrict B_in_4,        // [K*N] bf16 col-major
std::bfloat16_t*       __restrict A_in_4,        // [(M-16)*K] bf16
// sizes
int N_size_4, int K_size_4, int M_size_4)
{
// 1) bias copy: u16 → bf16
copy_u16_to_bf16_avx512(u16_bias_img2, bias_img2, (size_t)N_size_4);

// 2) direct transpose: u16 row-major → bf16 col-major
transpose_u16row_to_bf16col_avx512(u16_B_img2, B_in_4, N_size_4, K_size_4);

// 3) A copy: bf16 → bf16
size_t A_len = (size_t)(M_size_4 - 16) * (size_t)K_size_4;
copy_bf16_avx512(img_md_data, A_in_4, A_len);
}
/////////////////////////////////////////////////add bias/////////////////////////////////////////////////////
static inline __m512 load16_bf16_to_f32(const std::bfloat16_t* src) {
// load 16 x u16 as 256-bit
__m256i u16  = _mm256_loadu_si256((const __m256i*)src);
// widen to 16 x u32 (in 512-bit), shift into high 16 bits of float layout
__m512i u32  = _mm512_cvtepu16_epi32(u16);
u32          = _mm512_slli_epi32(u32, 16);
return _mm512_castsi512_ps(u32);
}

// Store 16 f32 -> 16 bf16 (round-to-nearest-even)
static inline void store16_f32_as_bf16(std::bfloat16_t* dst, __m512 v) {
__m512i x  = _mm512_castps_si512(v);
// add 0x7FFF + LSB of high part for RNE
__m512i lsb = _mm512_srli_epi32(x, 16);
__m512i rnd = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                            _mm512_and_si512(lsb, _mm512_set1_epi32(1)));
x = _mm512_add_epi32(x, rnd);
x = _mm512_srli_epi32(x, 16);               // keep top 16 bits in low half
// down-convert 16 x u32 -> 16 x u16 (saturating; fine for bf16)
__m256i u16 = _mm512_cvtusepi32_epi16(x);   // requires AVX-512BW
_mm256_storeu_si256((__m256i*)dst, u16);
}

// Load the 64-element bias once as four 16-wide f32 vectors
static inline void load_bias64_f32(const std::bfloat16_t* bias,
                            __m512& b0, __m512& b1,
                            __m512& b2, __m512& b3) {
b0 = load16_bf16_to_f32(bias +  0);
b1 = load16_bf16_to_f32(bias + 16);
b2 = load16_bf16_to_f32(bias + 32);
b3 = load16_bf16_to_f32(bias + 48);
}

// --- main kernel -----------------------------------------------------------
// Adds 64-len bf16 bias to each row of bf16 matrix with row stride 'row_stride'
// Reads first 64 columns from C_out_4[i*row_stride + 0..63], writes 64 bf16 to all_fc_out
void add_bias_final_bf16_avx512(const std::bfloat16_t* __restrict C_out_4,
                        const std::bfloat16_t* __restrict bias_img2, // [64]
                        std::bfloat16_t*       __restrict all_fc_out,
                        int rows, int row_stride /*=512*/) {
__m512 b0, b1, b2, b3;  // bias chunks of 16
load_bias64_f32(bias_img2, b0, b1, b2, b3);

for (int i = 0; i < rows; ++i) {
 const std::bfloat16_t* src = C_out_4    + (size_t)i * row_stride;
 std::bfloat16_t*       dst = all_fc_out + (size_t)i * 64;

 _mm_prefetch((const char*)(src + row_stride), _MM_HINT_T0);

 __m512 x0 = load16_bf16_to_f32(src +  0);
 __m512 x1 = load16_bf16_to_f32(src + 16);
 __m512 x2 = load16_bf16_to_f32(src + 32);
 __m512 x3 = load16_bf16_to_f32(src + 48);

 x0 = _mm512_add_ps(x0, b0);
 x1 = _mm512_add_ps(x1, b1);
 x2 = _mm512_add_ps(x2, b2);
 x3 = _mm512_add_ps(x3, b3);

 store16_f32_as_bf16(dst +  0, x0);
 store16_f32_as_bf16(dst + 16, x1);
 store16_f32_as_bf16(dst + 32, x2);
 store16_f32_as_bf16(dst + 48, x3);
}
}

/////////////////////////////////////////////////update img/////////////////////////////////////////////////////
#include <immintrin.h>
#include <stdfloat>
#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(__AVX512BF16__) && \
   ((defined(__clang__) && __clang_major__ >= 14) || \
    (defined(__GNUC__)  && __GNUC__        >= 12))
  #define USE_AVX512_BF16_INTRINSICS 1
#else
  #define USE_AVX512_BF16_INTRINSICS 0
#endif

// ---------- BF16 <-> FP32 helpers (16 lanes) ----------
static inline __m512 bf16bits_to_ps(__m256i u16v) {
    __m512i u32v = _mm512_cvtepu16_epi32(u16v);
    u32v = _mm512_slli_epi32(u32v, 16);
    return _mm512_castsi512_ps(u32v);
}
static inline __m256i ps_to_bf16bits(__m512 ps) {
    __m512i x   = _mm512_castps_si512(ps);
    __m512i rnd = _mm512_set1_epi32(0x00007FFF);
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(x,16), _mm512_set1_epi32(1));
    x = _mm512_add_epi32(x, _mm512_add_epi32(rnd, lsb));
    x = _mm512_srli_epi32(x, 16);
    return _mm512_cvtusepi32_epi16(x);
}

// ---------- Raw-pointer kernel (BF16 stored as u16) ----------
static inline void bf16_axpy_into_u16(
        uint16_t*       dst,     // update_img (BF16 bits)
        const uint16_t* base,    // denoise_noise (BF16 bits)
        const uint16_t* x,       // all_fc_out (BF16 bits)
        float t_prev, float t_curr,
        size_t N)                // element count
{
    const float  delta  = t_prev - t_curr;
    const __m512 vdelta = _mm512_set1_ps(delta);

    const size_t vecN = N & ~size_t(15);
    size_t i = 0;

    // Vector body
    for (; i < vecN; i += 16) {
        __m256i vbase_u16 = _mm256_loadu_si256((const __m256i*)(base + i));
        __m256i vx_u16    = _mm256_loadu_si256((const __m256i*)(x    + i));

        __m512 base_ps = bf16bits_to_ps(vbase_u16);
        __m512 x_ps    = bf16bits_to_ps(vx_u16);

        __m512 out_ps  = _mm512_fmadd_ps(vdelta, x_ps, base_ps); // base + delta*x
        __m256i out_u16= ps_to_bf16bits(out_ps);

        _mm256_storeu_si256((__m256i*)(dst + i), out_u16);
    }

    // Scalar tail
    for (; i < N; ++i) {
        uint32_t b = uint32_t(base[i]) << 16;
        uint32_t v = uint32_t(x[i])    << 16;
        float fb, fx;
        std::memcpy(&fb, &b, 4);
        std::memcpy(&fx, &v, 4);
        float fr = fb + delta * fx;

        uint32_t bits; std::memcpy(&bits, &fr, 4);
        bits += ((bits >> 16) & 1u) + 0x7FFFu;     // RNE -> BF16
        dst[i] = uint16_t(bits >> 16);
    }
}

// ---------- Buffer overload for buffer<std::bfloat16_t> ----------
template <class Buffer>
inline void bf16_axpy_into(Buffer& update_img,
                           const Buffer& denoise_noise,
                           const Buffer& all_fc_out,
                           float t_prev, float t_curr)
{
    static_assert(sizeof(std::bfloat16_t)==2, "BF16 must be 16 bits");
    const size_t N   = update_img.size(); // ELEMENTS (ensure your buffer returns elements)
    auto* dst  = reinterpret_cast<uint16_t*>(update_img.data());
    auto* base = reinterpret_cast<const uint16_t*>(denoise_noise.data());
    auto* x    = reinterpret_cast<const uint16_t*>(all_fc_out.data());
    bf16_axpy_into_u16(dst, base, x, t_prev, t_curr, N);
}
/////



inline void fill_bias_512x3072_and_copy_u16_tail_to_bf16_avx512(
    dtype_out*             __restrict dst,          // [512*3072 + n_u16]
    const uint16_t*       __restrict bias_bf16,    // [3072] (one row)
    const uint16_t*        __restrict src_u16_tail, // u16 payload to append
    size_t                               n_u16,
    size_t                               n_N)     // how many u16 to append
{
    // Part 1: replicate bias row (512 rows × 3072 cols)
    const size_t ROWS = 512;
    const size_t COLS = n_N;
    const size_t BIAS_TOTAL = ROWS * COLS;

#if defined(__AVX512F__)
    // Copy one bias row into all 512 rows
    for (size_t r = 0; r < ROWS; ++r) {
        dtype_out* __restrict drow = dst + r * COLS;

        size_t j = 0;
        for (; j + 32 <= COLS; j += 32) {
            __m512i v = _mm512_loadu_si512((const void*)(bias_bf16 + j));  // 32×u16
            _mm512_storeu_si512((void*)(drow + j), v);
        }
        // tiny tail for the row
        for (; j < COLS; ++j) drow[j] = bias_bf16[j];
    }

    // Part 2: append u16 payload as bf16 (bit-identical)
    dtype_out* __restrict tail = dst + BIAS_TOTAL;

    size_t i = 0;
    for (; i + 32 <= n_u16; i += 32) {
        __m512i v = _mm512_loadu_si512((const void*)(src_u16_tail + i)); // 32×u16
        _mm512_storeu_si512((void*)(tail + i), v);
    }
    for (; i < n_u16; ++i) {
        // bit-identical move into bf16 slot
        tail[i] = std::bit_cast<dtype_out>(src_u16_tail[i]);
    }
#else
    // -------- Non-AVX512 fallback --------
    // Replicate bias
    for (size_t r = 0; r < ROWS; ++r) {
        std::memcpy(dst + r * COLS, bias_bf16, COLS * sizeof(dtype_out));
    }
    // Append tail
    dtype_out* __restrict tail = dst + BIAS_TOTAL;
    for (size_t i = 0; i < n_u16; ++i) {
        tail[i] = std::bit_cast<dtype_out>(src_u16_tail[i]);
    }
#endif
}

inline void fill_bias_512x3072_and_copy_u16_tail_to_bf16_avx512_scale(
    dtype_out*       __restrict dst,           // [512*3072 + n_u16]
    const uint16_t*  __restrict bias_bf16,     // [3072]
    const uint16_t*  __restrict scale_u16,     // [128]
    const uint16_t*  __restrict src_u16_tail,  // [n_u16]
    size_t                        n_u16,
    size_t                        n_N)         // should be 3072
{
    const size_t ROWS       = 512;
    const size_t COLS       = n_N;             // 3072
    const size_t BIAS_TOTAL = ROWS * COLS;     // 512*3072

    // =======================
    // Part 1: rows 0 and 1
    // =======================

#if defined(__AVX512F__)
    constexpr size_t VEC_ELEMS = 32;

    // --- row 0: bias ---
    {
        dtype_out* __restrict row0 = dst;

        size_t j = 0;
        for (; j + VEC_ELEMS <= COLS; j += VEC_ELEMS) {
            __m512i v = _mm512_loadu_si512((const void*)(bias_bf16 + j));
            _mm512_storeu_si512((void*)(row0 + j), v);
        }
        for (; j < COLS; ++j) {
            row0[j] = std::bit_cast<dtype_out>(bias_bf16[j]);
        }
    }

    // --- row 1: scale pattern (128) repeated 24× ---
    if (ROWS > 1) {
        constexpr size_t PATTERN = 128;
        const size_t REPEATS = COLS / PATTERN;   // 24 when COLS=3072

        dtype_out* __restrict row1 = dst + COLS;

        for (size_t rep = 0; rep < REPEATS; ++rep) {
            const size_t base = rep * PATTERN;

            size_t j = 0;
            for (; j + VEC_ELEMS <= PATTERN; j += VEC_ELEMS) {
                __m512i v = _mm512_loadu_si512((const void*)(scale_u16 + j));
                _mm512_storeu_si512((void*)(row1 + base + j), v);
            }
            for (; j < PATTERN; ++j) {
                row1[base + j] = std::bit_cast<dtype_out>(scale_u16[j]);
            }
        }
    }

#else   // -------- non-AVX512 fallback --------

    // row 0: bias
    std::memcpy(dst,
                bias_bf16,
                COLS * sizeof(dtype_out));

    // row 1: scale pattern repeated
    if (ROWS > 1) {
        constexpr size_t PATTERN = 128;
        const size_t REPEATS = COLS / PATTERN;   // 24 for 3072

        dtype_out* row1 = dst + COLS;
        for (size_t rep = 0; rep < REPEATS; ++rep) {
            std::memcpy(row1 + rep * PATTERN,
                        scale_u16,
                        PATTERN * sizeof(dtype_out));
        }
    }

#endif  // __AVX512F__

    // rows 2..511: intentionally left untouched

    // =======================
    // Part 2: append tail
    // =======================
    dtype_out* __restrict tail = dst + BIAS_TOTAL;

#if defined(__AVX512F__)
    {
        size_t i = 0;
        for (; i + 32 <= n_u16; i += 32) {
            __m512i v = _mm512_loadu_si512((const void*)(src_u16_tail + i));
            _mm512_storeu_si512((void*)(tail + i), v);
        }
        for (; i < n_u16; ++i) {
            tail[i] = std::bit_cast<dtype_out>(src_u16_tail[i]);
        }
    }
#else
    for (size_t i = 0; i < n_u16; ++i) {
        tail[i] = std::bit_cast<dtype_out>(src_u16_tail[i]);
    }
#endif
}


#endif