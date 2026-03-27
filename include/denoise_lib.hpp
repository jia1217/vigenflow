#ifndef DENoise_LIB_HPP
#define DENoise_LIB_HPP

#include "typedef.hpp"




static inline __m512 exp512_ps_approx(__m512 x) {
    // clamp
    const __m512 max_x = _mm512_set1_ps(+80.0f);
    const __m512 min_x = _mm512_set1_ps(-80.0f);
    x = _mm512_max_ps(min_x, _mm512_min_ps(max_x, x));

    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    const __m512 ln2   = _mm512_set1_ps(0.6931471805599453f);

    // fx = x * (1/ln2)
    __m512 fx = _mm512_mul_ps(x, log2e);
    // round down for integer exponent
    __m512i ix = _mm512_cvttps_epi32(fx);
    // build 2^n via bit-twiddling: (ix + 127) << 23
    __m512i e_i = _mm512_slli_epi32(_mm512_add_epi32(ix, _mm512_set1_epi32(127)), 23);
    __m512  exp_n = _mm512_castsi512_ps(e_i);

    // fractional remainder f = x – ix*ln2
    __m512 f = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(ix), ln2, x);

    // 5-term minimax on f∈[0,1]:
    const __m512 c5 = _mm512_set1_ps(1.0f/120.0f);
    const __m512 c4 = _mm512_set1_ps(1.0f/24.0f);
    const __m512 c3 = _mm512_set1_ps(1.0f/6.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c1 = _mm512_set1_ps(1.0f);

    __m512 p = _mm512_fmadd_ps(c5, f, c4);
    p = _mm512_fmadd_ps(p,  f, c3);
    p = _mm512_fmadd_ps(p,  f, c2);
    p = _mm512_fmadd_ps(p,  f, c1);
    p = _mm512_fmadd_ps(p,  f, _mm512_set1_ps(1.0f));

    return _mm512_mul_ps(exp_n, p);
}

static inline void silu_bf16_avx512_noexp(
    const buffer<dtype_in> &src_buf,
          buffer<dtype_out> &dst_buf,
    size_t                N        // total __bf16 elements
) {
    auto const *src_bits = reinterpret_cast<const uint16_t*>(src_buf.data());
          auto *dst_bits = reinterpret_cast<      uint16_t*>(dst_buf.data());

    size_t i = 0;
    // AVX-512 main loop: 16 lanes per iteration
    for (; i + 15 < N; i += 16) {
        // unpack BFloat16 → FP32
        __m512 x   = load_bf16_to_fp32(src_bits + i);

        // sigmoid: 1/(1+exp(−x))
        __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
        __m512 e   = exp512_ps_approx(neg);
        __m512 sig = _mm512_div_ps(_mm512_set1_ps(1.0f),
                                   _mm512_add_ps(e, _mm512_set1_ps(1.0f)));

        // SiLU = x * sigmoid(x)
        __m512 y   = _mm512_mul_ps(x, sig);

        // repack FP32 → BFloat16
        store_fp32_to_bf16(dst_bits + i, y);
    }

    // scalar tail
    for (; i < N; ++i) {
        // unpack
        uint32_t u; memcpy(&u, &src_bits[i],4);
        float    xv; memcpy(&xv, &u,4);

        float sv = xv/(1.0f + std::exp(-xv));   // scalar SiLU

        uint32_t ru; memcpy(&ru, &sv,4);
        dst_bits[i] = uint16_t(ru >> 16);
    }
}

void timestep_embedding_python_exact(
    // const float* t,        // [N]
    const std::vector<float>& t,
    int          N,
    int          dim,
    float*       out,      // [N, dim] row-major
    float        max_period
) {
    const int half = dim / 2;
    const float log_max = std::log(max_period);

    // Precompute freqs[j] = exp(-log(max_period) * j / half)
    // NOTE: when half==0 (dim<2), handle gracefully.
    std::vector<float> freqs(half);
    if (half > 0) {
        const float inv_half = 1.0f / float(half);
        for (int j = 0; j < half; ++j) {
            freqs[j] = std::exp(-log_max * (float(j) * inv_half));
        }
    }

    // Compute embeddings
    #pragma omp parallel for if (N > 256)
    for (int i = 0; i < N; ++i) {
        const float ti = t[i];          // <-- IMPORTANT: no time_factor
        float* row = out + i * dim;

        // cos part
        for (int j = 0; j < half; ++j) {
            const float angle = ti * freqs[j];
            row[j] = std::cos(angle);
        }

        // sin part
        for (int j = 0; j < half; ++j) {
            const float angle = ti * freqs[j];
            row[j + half] = std::sin(angle);
        }

        // if dim is odd, last element is 0
        if (dim & 1) {
            row[dim - 1] = 0.0f;
        }
    }
}

void build_pos_cs_from_pos_all(
    const buffer<float> &pos_all,
    buffer<float>       &pos_c_all,
    buffer<float>       &pos_s_all,
    int L, int D)
  {
    const size_t pos_stride = 2u * (size_t)D;   // floats per token in pos_all
  
    // make sure storage is large enough
    if (pos_c_all.size() < (size_t)L * (size_t)D)
        pos_c_all.resize((size_t)L * (size_t)D);
    if (pos_s_all.size() < (size_t)L * (size_t)D)
        pos_s_all.resize((size_t)L * (size_t)D);
  
    for (int t = 0; t < L; ++t) {
        const float *pos_ptr = pos_all.data() + (size_t)t * pos_stride;
        float *c_row = pos_c_all.data() + (size_t)t * D;
        float *s_row = pos_s_all.data() + (size_t)t * D;
  
        const int n_pairs = D / 2;      // assume D is even
  
        for (int pair = 0; pair < n_pairs; ++pair) {
            const size_t m = (size_t)pair * 4;
            float c = pos_ptr[m + 0];   // cos φ
            float s = pos_ptr[m + 2];   // sin φ
  
            const int d0 = pair * 2;
            const int d1 = d0 + 1;
  
            c_row[d0] = c;
            c_row[d1] = c;
            s_row[d0] = s;
            s_row[d1] = s;
        }
    }
  }

void rope_avx(
    const std::vector<float> &pos,  // u16_pos[4336, 3]
    int N,  // L_all 4336
    int dim, // axes_dim[3] = [16, 56, 56]
    float theta, // theta 10000.0
    std::vector<float> &out // pos[4336, 128]
) {
    assert(dim % 2 == 0);
    int D = dim / 2;
    float *omega = static_cast<float*>(_mm_malloc(sizeof(float) * D, 32)); //meaning 32-byte aligned
    float log_theta = logf(theta);
    for(int k = 0; k < D; ++k) {
        omega[k] = expf(-log_theta * ((float)k / (float)D));
    }
    // --- 2) scratch buffer for pos[n]*omega vector
    float *tmp = static_cast<float*>(_mm_malloc(sizeof(float) * D, 32));

    // --- 3) main loop over positions
    for(int n = 0; n < N; ++n) {
        // broadcast pos[n] into an AVX register
        __m256 pv = _mm256_set1_ps(pos[n]);

        // 3a) vectorize the multiplication: tmp[k] = pos[n] * omega[k]
        int k = 0;
        for(; k + 7 < D; k += 8) {
            __m256 w = _mm256_load_ps(&omega[k]);         // aligned load
            __m256 x = _mm256_mul_ps(pv, w);             // elementwise mul
            _mm256_store_ps(&tmp[k], x);                 // aligned store
        }
        for(; k < D; ++k) {
            tmp[k] = pos[n] * omega[k];\
            // std::cout << "tmp: " << tmp[k] << std::endl;
        }

        // 3b) compute sin/cos and pack into out[]
        k = 0;
        for(; k + 7 < D; k += 8) {
            __m256 x = _mm256_load_ps(&tmp[k]);
            // Replace _mm256_cos_ps and _mm256_sin_ps with scalar implementations
            float cos_vals[8] __attribute__((aligned(32)));
            float sin_vals[8] __attribute__((aligned(32)));
            for(int i = 0; i < 8; ++i){
            cos_vals[i] = cosf(((float*)&x)[i]);
            sin_vals[i] = sinf(((float*)&x)[i]);
            }
            __m256 c = _mm256_load_ps(cos_vals);  // now safe
            __m256 s = _mm256_load_ps(sin_vals);

            // scatter the 8 results
            for(int i = 0; i < 8; ++i) {
                float cc = ((float*)&c)[i];
                float ss = ((float*)&s)[i];
                int base = ((n * D + (k + i)) * 4);
                out[base + 0] =  cc;
                out[base + 1] = -ss;
                out[base + 2] =  ss;
                out[base + 3] =  cc;
            }
        }
        // tail-scalar for any remaining k
        for(; k < D; ++k) {
            float x  = tmp[k];
            float cc = cosf(x);
            float ss = sinf(x);
            int base = ((n * D + k) * 4);
            out[base + 0] =  cc;
            out[base + 1] = -ss;
            out[base + 2] =  ss;
            out[base + 3] =  cc;
        }
    }
    free(tmp);
    free(omega);
}

int32_t* create_coordinate_grid_c(
    const int64_t* size,
    const int64_t* start,
    int ndim
) {
    if (ndim <= 0) return NULL;

    // total number of spatial points = product(size)
    int64_t total = 1;
    for (int d = 0; d < ndim; ++d) {
        if (size[d] <= 0) return NULL;
        total *= size[d];
    }

    // output shape is [total, ndim] in flattened form
    int32_t* out = (int32_t*)malloc((size_t)(total * ndim * sizeof(int32_t)));
    if (!out) return NULL;

    // For each linear point in the N-D grid
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t tmp = linear;

        // Decode linear index into N-D indices, row-major, "ij" style
        int64_t idx[32];  // simple version; enough if ndim <= 32
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = tmp % size[d];
            tmp /= size[d];
        }

        // Write stacked coordinates
        for (int d = 0; d < ndim; ++d) {
            out[linear * ndim + d] = (int32_t)(start[d] + idx[d]);
        }
    }

    return out;
}


void rmsnorm_avx2(float *buf32, int D, float eps = 1e-6f) {
    __m256 sum = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 v = _mm256_loadu_ps(buf32 + d);
        sum = _mm256_fmadd_ps(v, v, sum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float acc = tmp[0]+tmp[1]+tmp[2]+tmp[3]
              +tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; d < D; ++d) {
        float v = buf32[d];
        acc += v*v;
    }

    float inv = 1.f / std::sqrt(acc / D + eps);
    __m256 invv = _mm256_set1_ps(inv);

    d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 v = _mm256_loadu_ps(buf32 + d);
        v = _mm256_mul_ps(v, invv);
        _mm256_storeu_ps(buf32 + d, v);
    }
    for (; d < D; ++d) {
        buf32[d] *= inv;
    }
}

void mul_and_store_bf16_avx2(const float  *in32,
    const float  *scale_f32,
          uint16_t*out16,
    int             D)
{
static const __m128i pack_mask = _mm_setr_epi8(
0,1,  4,5,  8,9, 12,13,
-1,-1,-1,-1,-1,-1,-1,-1
);

int i = 0, V = 8;
for (; i + V <= D; i += V) {
// load 8 f32 activations
__m256 v = _mm256_loadu_ps(in32 + i);

// truncate → bf16 bits → back to f32
__m256i vi   = _mm256_castps_si256(v);
__m256i hi   = _mm256_srli_epi32(vi, 16);
__m256  v_bf = _mm256_castsi256_ps(_mm256_slli_epi32(hi, 16));

// load 8 f32 scales
__m256 s_f   = _mm256_loadu_ps(scale_f32 + i);

// multiply in f32
__m256 prod  = _mm256_mul_ps(v_bf, s_f);

// truncate product → bf16 bits
__m256i pi   = _mm256_castps_si256(prod);
__m256i phi  = _mm256_srli_epi32(pi, 16);

// pack 8→4+4 bf16 words and store
__m128i lo128 = _mm256_extracti128_si256(phi, 0);
__m128i hi128 = _mm256_extracti128_si256(phi, 1);
__m128i p0    = _mm_shuffle_epi8(lo128, pack_mask);
__m128i p1    = _mm_shuffle_epi8(hi128, pack_mask);
_mm_storel_epi64((__m128i*)(out16 + i    ), p0);
_mm_storel_epi64((__m128i*)(out16 + i + 4), p1);
}
// tail
for (; i < D; ++i) {
uint16_t v_bh = f32_to_bf16(in32[i]);
float    vf  = bf16_to_f32(v_bh);
float    prod= vf * scale_f32[i];
out16[i]     = f32_to_bf16(prod);
}
}

inline void rmsnorm_mat_bf16_with_scale(
    const buffer<dtype_out> &in_bf16,  // [L, D] bf16
    const uint16_t*          scale16,  // [D] bf16 bits (gamma)
          buffer<dtype_out> &out_bf16, // [L, D] bf16
    int L,
    int D
) {
    static_assert(sizeof(dtype_out) == 2, "bf16 must be 16 bits");

    const size_t dim = (size_t)D;

    const uint16_t* in16  = reinterpret_cast<const uint16_t*>(in_bf16.data());
          uint16_t* out16 = reinterpret_cast<uint16_t*>(out_bf16.data());

    // unpack gamma once
    std::vector<float> gamma_f32((size_t)D);
    convert_bf16_to_f32_avx2(scale16, gamma_f32.data(), D);

    #pragma omp parallel
    {
        std::vector<float> row_f32((size_t)D);

        #pragma omp for schedule(static)
        for (int l = 0; l < L; ++l) {
            const size_t base = (size_t)l * dim;

            // bf16 -> f32
            convert_bf16_to_f32_avx2(in16 + base, row_f32.data(), D);

            // RMSNorm (in-place)
            rmsnorm_avx2(row_f32.data(), D);

            // (row * gamma) and store bf16
            mul_and_store_bf16_avx2(row_f32.data(),
                                    gamma_f32.data(),
                                    out16 + base,
                                    D);
        }
    }
}

inline void rmsnorm_mat_bf16_with_scale_partial(
    const buffer<dtype_out> &in_bf16,   // [L, D_out] bf16
    const uint16_t*          scale16,   // [D_norm] bf16 bits (gamma)
          buffer<dtype_out> &out_bf16,  // [L, D_out] bf16
    int L,
    int D_out,   // e.g. 4096
    int D_norm   // e.g. 3840  (must be <= D_out)
) {
    static_assert(sizeof(dtype_out) == 2, "bf16 must be 16 bits");
    if (D_norm > D_out) return; // or assert(false)

    const size_t dim_out  = (size_t)D_out;
    const size_t dim_norm = (size_t)D_norm;

    const uint16_t* in16  = reinterpret_cast<const uint16_t*>(in_bf16.data());
          uint16_t* out16 = reinterpret_cast<uint16_t*>(out_bf16.data());

    // unpack gamma once (only for normalized part)
    std::vector<float> gamma_f32((size_t)D_norm);
    convert_bf16_to_f32_avx2(scale16, gamma_f32.data(), D_norm);

    #pragma omp parallel
    {
        std::vector<float> row_f32;
        row_f32.resize((size_t)D_norm);

        #pragma omp for schedule(static)
        for (int l = 0; l < L; ++l) {
            const size_t base = (size_t)l * dim_out;

            // ---- 1) process first D_norm ----
            convert_bf16_to_f32_avx2(in16 + base, row_f32.data(), D_norm);
            rmsnorm_avx2(row_f32.data(), D_norm);
            mul_and_store_bf16_avx2(row_f32.data(),
                                    gamma_f32.data(),
                                    out16 + base,
                                    D_norm);

            // ---- 2) handle tail [D_norm .. D_out) ----
            // Option A (common): copy tail unchanged
            const size_t tail = dim_out - dim_norm;
            if (tail) {
                std::memcpy(out16 + base + dim_norm,
                            in16  + base + dim_norm,
                            tail * sizeof(uint16_t));
            }

            // Option B: zero tail instead (uncomment if you want this)
            // for (size_t i = dim_norm; i < dim_out; ++i) out16[base + i] = 0;
        }
    }
}

inline void process_qk_end2end_bf16_new(
    const buffer<dtype_out> &q_bias_bf16,   // layout: [L, H*D]  BF16
    const buffer<dtype_out> &k_bias_bf16,   // layout: [L, H*D]  BF16
    const uint16_t*         q_scale16,      // [D]      BF16 bits (γ_Q)
    const uint16_t*         k_scale16,      // [D]      BF16 bits (γ_K)
          buffer<dtype_out> &q_out_bf16,    // [L, H*D]  BF16
          buffer<dtype_out> &k_out_bf16,    // [L, H*D]  BF16
    int H, int L, int D)
{
    static_assert(sizeof(dtype_out) == 2, "bf16 must be 16 bits");

    const size_t heads     = static_cast<size_t>(H);
    const size_t tokens    = static_cast<size_t>(L);
    const size_t head_dim  = static_cast<size_t>(D);
    const size_t row_cols  = heads * head_dim;   // 3072

    // Reinterpret BF16 storage as raw u16 bit-patterns (no copy)
    const uint16_t* q16  = reinterpret_cast<const uint16_t*>(q_bias_bf16.data());
    const uint16_t* k16  = reinterpret_cast<const uint16_t*>(k_bias_bf16.data());
          uint16_t* qo16 = reinterpret_cast<uint16_t*>(q_out_bf16.data());
          uint16_t* ko16 = reinterpret_cast<uint16_t*>(k_out_bf16.data());

    // Unpack scales once to FP32 (per-head-dim, length D=128)
    std::vector<float> scale_q_f32(D), scale_k_f32(D);
    convert_bf16_to_f32_avx2(q_scale16, scale_q_f32.data(), D);
    convert_bf16_to_f32_avx2(k_scale16, scale_k_f32.data(), D);

    // One scratch row per thread
    #pragma omp parallel
    {
        std::vector<float> row_f32(D);

        // iterate over heads & tokens for better cache behaviour
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; ++h) {
            for (int l = 0; l < L; ++l) {
                // base index for (h, l) block of length D in [L, H*D] layout:
                // row = l, col_block_start = h * D
                const size_t base =
                    static_cast<size_t>(l) * row_cols +   // row offset (token)
                    static_cast<size_t>(h) * head_dim;    // head offset inside row

                // --- Q path ---
                convert_bf16_to_f32_avx2(q16 + base, row_f32.data(), D); // bf16 -> f32
                rmsnorm_avx2(row_f32.data(), D);                         // RMSNorm in f32
                mul_and_store_bf16_avx2(row_f32.data(),
                                         scale_q_f32.data(),
                                         qo16 + base,
                                         D);                             // scale and store bf16

                // --- K path ---
                convert_bf16_to_f32_avx2(k16 + base, row_f32.data(), D);
                rmsnorm_avx2(row_f32.data(), D);
                mul_and_store_bf16_avx2(row_f32.data(),
                                         scale_k_f32.data(),
                                         ko16 + base,
                                         D);
            }
        }
    }
}


void apply_rope_bf16_avx512_new(
    const buffer<dtype_in>  &q_in,       // [B, L, H*D] bf16
          buffer<dtype_out> &q_out,      // [B, L, H*D] bf16
    const buffer<float>     &pos_c_all,  // [L * D]
    const buffer<float>     &pos_s_all,  // [L * D]
    int B, int H, int L, int D)
  {
    const size_t row_stride  = (size_t)H * (size_t)D; // elements per (b, t)
    const size_t head_stride = (size_t)D;
  
    // sign mask with signbit set for *even* lanes (flip sign only in even lanes)
    const __m512 sign_even = _mm512_castsi512_ps(
        _mm512_maskz_set1_epi32(0x5555, 0x80000000u)
    );
  
    // Parallelize over (b, t) – each thread owns a full row [H*D]
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
  
            const size_t row_off = ((size_t)b * L + (size_t)t) * row_stride;
  
            const float *c_row = pos_c_all.data() + (size_t)t * D;
            const float *s_row = pos_s_all.data() + (size_t)t * D;
  
            for (int h = 0; h < H; ++h) {
                const size_t base_off = row_off + (size_t)h * head_stride;
  
                const dtype_in  *q_in_base  = q_in.data()  + base_off;
                      dtype_out *q_out_base = q_out.data() + base_off;
  
                int d = 0;
                for (; d + 16 <= D; d += 16) {
                    // 1) load 16 bf16 -> f32
                    __m512 v_q = bf16_load16_to_f32(q_in_base + d);
  
                    // 2) load precomputed cos/sin (already duplicated per dim)
                    __m512 v_c = _mm512_loadu_ps(c_row + d);
                    __m512 v_s = _mm512_loadu_ps(s_row + d);
  
                    // 3) rotate:
                    // q_even, q_odd interleaved in v_q
                    __m512 q_sw  = _mm512_permute_ps(v_q, 0b10110001); // swap within pairs
                    __m512 term2 = _mm512_mul_ps(v_s, q_sw);
                    term2        = _mm512_xor_ps(term2, sign_even);   // flip even lanes
                    __m512 v_out = _mm512_fmadd_ps(v_c, v_q, term2);  // c*q + (±s)*q_sw
  
                    // 4) store bf16
                    f32_store16_to_bf16_rne(q_out_base + d, v_out);
                }
  
                // scalar tail
                for (; d < D; d += 2) {
                    float qe = static_cast<float>(q_in_base[d]);
                    float qo = static_cast<float>(q_in_base[d + 1]);
  
                    float c = c_row[d];
                    float s = s_row[d];
  
                    q_out_base[d]     = (dtype_out)( c*qe - s*qo );
                    q_out_base[d + 1] = (dtype_out)( s*qe + c*qo );
                }
            }
        }
    }
  }


  void interleave_kv(
    const dtype_in *K,   // [rows, cols]
    const dtype_in *V,   // [rows, cols]
    dtype_in *KV,        // [2 * rows, cols]
    int rows,
    int cols
  ) {
    const int BLOCK = 64;   // 64-row block
  
    for (int i = 0; i < rows; ++i) {
        // which 64-row block and offset inside that block
        size_t g       = (size_t)i / BLOCK;   // block index
        size_t off     = (size_t)i % BLOCK;   // row offset in block
  
        size_t kv_k_id = g * (size_t)(2 * BLOCK) + off;        // row index in KV for K
        size_t kv_v_id = g * (size_t)(2 * BLOCK) + BLOCK + off; // row index in KV for V
  
        const dtype_in *k_row   = K  + (size_t)i       * cols;
        const dtype_in *v_row   = V  + (size_t)i       * cols;
        dtype_in       *kv_krow = KV + kv_k_id * (size_t)cols;
        dtype_in       *kv_vrow = KV + kv_v_id * (size_t)cols;
  
        // copy rows
        memcpy(kv_krow, k_row, (size_t)cols * sizeof(dtype_in));
        memcpy(kv_vrow, v_row, (size_t)cols * sizeof(dtype_in));
    }
  }

  std::vector<float> get_noise_3d_cpp(
    int64_t num_samples,
    int64_t height,
    int64_t width,
    uint64_t seed
) {
    size_t total_size = static_cast<size_t>(num_samples * height * width);
    std::vector<float> noise(total_size);

    // 64-bit Mersenne Twister generator seeded
    std::mt19937_64 gen(seed);
    
    // Normal distribution with mean = 0.0 and stddev = 1.0
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < total_size; ++i) {
        noise[i] = dist(gen);
    }

    return noise;
}

template <typename T>
void patchify_fcHW_to_tokens_cpp(
    const T* input,  // Actual memory layout: [F, C, H, W]
    T* output,       // Output layout: [F_tokens*H_tokens*W_tokens, pF*pH*pW*C]
    int64_t F, int64_t C, int64_t H, int64_t W, // Note: F and C order swapped to match tensor
    int64_t pF, int64_t pH, int64_t pW) 
{
    // 1. Calculate token dimensions
    int64_t F_tokens = F / pF;
    int64_t H_tokens = H / pH;
    int64_t W_tokens = W / pW;

    // Pre-compute strides for [F, C, H, W] layout
    int64_t HW = H * W;
    int64_t CHW = C * HW;

    // 2. Multithread over the output tokens (f, h, w)
    #pragma omp parallel for collapse(3)
    for (int64_t f = 0; f < F_tokens; ++f) {
        for (int64_t h = 0; h < H_tokens; ++h) {
            for (int64_t w = 0; w < W_tokens; ++w) {
                
                // Calculate the base output index for this specific token/patch
                int64_t patch_idx = (f * H_tokens * W_tokens) + (h * W_tokens) + w;
                int64_t patch_size = pF * pH * pW * C;
                int64_t out_base = patch_idx * patch_size;
                
                int64_t local_out_idx = 0;
                
                // 3. Iterate inside the patch (pf, ph, pw, c)
                for (int64_t pf = 0; pf < pF; ++pf) {
                    for (int64_t ph = 0; ph < pH; ++ph) {
                        for (int64_t pw = 0; pw < pW; ++pw) {
                            
                            // Map local patch coordinates to global spatial coordinates
                            int64_t in_f = f * pF + pf;
                            int64_t in_h = h * pH + ph;
                            int64_t in_w = w * pW + pw;
                            
                            // Iterate over channels last to match PyTorch permute(..., 0)
                            for (int64_t c = 0; c < C; ++c) {
                                // FIXED: Correct flat index for [F, C, H, W] layout
                                int64_t in_idx = (in_f * CHW) + (c * HW) + (in_h * W) + in_w;
                                
                                // Write sequentially to output
                                output[out_base + local_out_idx] = input[in_idx];
                                local_out_idx++;
                            }
                        }
                    }
                }
            }
        }
    }
}

buffer<dtype_out> convert_float_vector_to_bf16_buffer(const std::vector<float>& input_float) {
    size_t size = input_float.size();
    
    // 1. Allocate the destination buffer
    buffer<dtype_out> out_bf16(size);

    const float* in_ptr = input_float.data();
    dtype_out* out_ptr = out_bf16.data();

    size_t i = 0;

    // 2. Main Loop: Process 16 elements (512 bits of float) per iteration
    for (; i + 16 <= size; i += 16) {
        // A. Load 16 Float32 values
        __m512 f_vec = _mm512_loadu_ps(in_ptr + i);
        
        // B. Hardware conversion Float32 -> BF16
        __m256bh bf_vec = _mm512_cvtneps_pbh(f_vec);
        
        // C. Store 16 BF16 values (256 bits) into the buffer
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_ptr + i), (__m256i)bf_vec);
    }

    // 3. Tail Handling: Process remaining elements if size is not a multiple of 16
    if (i < size) {
        int rem = size - i;
        __mmask16 mask = (1U << rem) - 1U;

        // Masked load
        __m512 f_vec = _mm512_maskz_loadu_ps(mask, in_ptr + i);
        
        // Convert
        __m256bh bf_vec = _mm512_cvtneps_pbh(f_vec);
        
        // Masked store
        _mm256_mask_storeu_epi16(out_ptr + i, mask, (__m256i)bf_vec);
    }

    return out_bf16;
}
inline void rmsnorm_mat_bf16_with_two_scales(
    const buffer<dtype_out> &in_bf16,     // [L, D]  = [4096, 3840] bf16
    const uint16_t*          scale16,     // [D] bf16 bits (γ from RMSNorm)
    const buffer<dtype_out> &extra_bf16,  // [D] bf16 (extra vector)
          buffer<dtype_out> &out_bf16,    // [L, D] bf16
    int L,                                // 4096
    int D)                                // 3840
{
    static_assert(sizeof(dtype_out) == 2, "bf16 must be 16 bits");

    const size_t tokens = static_cast<size_t>(L);
    const size_t dim    = static_cast<size_t>(D);

    // reinterpret BF16 storage as u16
    const uint16_t* in16    =
        reinterpret_cast<const uint16_t*>(in_bf16.data());
          uint16_t* out16   =
        reinterpret_cast<uint16_t*>(out_bf16.data());
    const uint16_t* extra16 =
        reinterpret_cast<const uint16_t*>(extra_bf16.data());

    // ---- 1) unpack both scales to fp32 and combine them ----
    std::vector<float> scale_f32(D);
    std::vector<float> extra_f32(D);
    std::vector<float> combined_f32(D);

    convert_bf16_to_f32_avx2(scale16, scale_f32.data(), D);   // γ
    convert_bf16_to_f32_avx2(extra16, extra_f32.data(), D);   // extra

    for (int i = 0; i < D; ++i) {
        combined_f32[i] = scale_f32[i] * (extra_f32[i]);
    }

    // ---- 2) process each row [D] ----
    #pragma omp parallel
    {
        std::vector<float> row_f32(D);

        #pragma omp for schedule(static)
        for (int l = 0; l < L; ++l) {
            const size_t base = static_cast<size_t>(l) * dim;

            // bf16 -> f32
            convert_bf16_to_f32_avx2(in16 + base, row_f32.data(), D);

            // RMSNorm in f32
            rmsnorm_avx2(row_f32.data(), D);

            // multiply by combined scale and store bf16
            mul_and_store_bf16_avx2(row_f32.data(),
                                     combined_f32.data(),
                                     out16 + base,
                                     D);
        }
    }
}

inline void rmsnorm_mlp_fused_two_scales_exact(
    const buffer<dtype_out> &noise_out_C_out,   // [L,4096] bf16
    const buffer<dtype_out> &x_noise_in,        // [L,D]    bf16  (D=3840)
    const uint16_t*          scale1_16,         // [D] bf16 bits  (γ1 = noise_atten_norm2_u16)
    const buffer<dtype_out> &extra1_bf16,       // [D] bf16       (gate_msa)
    const uint16_t*          scale2_16,         // [D] bf16 bits  (γ2 = noise_ffn_norm1_u16)
    const buffer<dtype_out> &extra2_bf16,       // [D] bf16       (scale_mlp)
          buffer<dtype_out> &x_noise_norm1_ff,  // [L,D] compact final
          buffer<dtype_out> &x_noise_in_norm2_mlp, // [L,D] intermediate
          buffer<dtype_out> &noise_mlp0,        // [L,4096]
          buffer<dtype_out> &noise_mlp1,        // [L,4096]
    int L,
    int D                                      // 3840
)
{
    static_assert(sizeof(dtype_out) == 2, "dtype_out must be 16 bits (bf16)");

    constexpr int D_INLD = 4096;                 // input/output stride
    const size_t dim = static_cast<size_t>(D);   // 3840

    // reinterpret BF16 storage as u16
    const uint16_t* in0_16 =
        reinterpret_cast<const uint16_t*>(noise_out_C_out.data());      // [L,4096]
    const uint16_t* res16 =
        reinterpret_cast<const uint16_t*>(x_noise_in.data());           // [L,D]
          uint16_t* out16 =
        reinterpret_cast<uint16_t*>(x_noise_norm1_ff.data());           // [L,D]
          dtype_out* mlp_base =
        x_noise_in_norm2_mlp.data();                                    // [L,D]

          uint16_t* mlp0_u16_base =
        reinterpret_cast<uint16_t*>(noise_mlp0.data());                 // [L,4096]
          uint16_t* mlp1_u16_base =
        reinterpret_cast<uint16_t*>(noise_mlp1.data());                 // [L,4096]

    const uint16_t* extra1_16 =
        reinterpret_cast<const uint16_t*>(extra1_bf16.data());          // [D]
    const uint16_t* extra2_16 =
        reinterpret_cast<const uint16_t*>(extra2_bf16.data());          // [D]

    // ---- 1) unpack scales and build combined vectors ----
    std::vector<float> scale1_f32(D);
    std::vector<float> extra1_f32(D);
    std::vector<float> combined1_f32(D);

    std::vector<float> scale2_f32(D);
    std::vector<float> extra2_f32(D);
    std::vector<float> combined2_f32(D);

    // γ1 and gate_msa
    convert_bf16_to_f32_avx2(scale1_16, scale1_f32.data(), D);
    convert_bf16_to_f32_avx2(extra1_16, extra1_f32.data(), D);
    for (int i = 0; i < D; ++i) {
        combined1_f32[i] = scale1_f32[i] * extra1_f32[i];  // γ1 * gate
    }

    // γ2 and scale_mlp
    convert_bf16_to_f32_avx2(scale2_16, scale2_f32.data(), D);
    convert_bf16_to_f32_avx2(extra2_16, extra2_f32.data(), D);
    for (int i = 0; i < D; ++i) {
        combined2_f32[i] = scale2_f32[i] * extra2_f32[i];  // γ2 * mlp_scale
    }

    // ---- 2) process each row ----
    #pragma omp parallel
    {
        std::vector<float>     row_f32(D);        // f32 scratch per thread
        std::vector<uint16_t>  row_norm2_u16(D);  // bf16 bits after norm2
        std::vector<dtype_out> row_mlp_bf16(D);   // bf16 values after residual add

        #pragma omp for schedule(static)
        for (int l = 0; l < L; ++l) {
            const size_t base_compact = static_cast<size_t>(l) * dim;    // [L,D]
            const size_t base_inld    = static_cast<size_t>(l) * D_INLD; // [L,4096]

            // pointers for this row
            const uint16_t* row_noise = in0_16 + base_inld;              // 4096 stride
            const uint16_t* row_res16 = res16   + base_compact;          // 3840 stride

            dtype_out*  row_mlp_global = mlp_base      + base_compact;   // [D]
            uint16_t*   row_out16      = out16         + base_compact;   // [D]
            uint16_t*   row_mlp0_u16   = mlp0_u16_base + base_inld;      // [4096]
            uint16_t*   row_mlp1_u16   = mlp1_u16_base + base_inld;      // [4096]

            // ========= stage 1: RMSNorm(noise_out_C_out[:3840]) * combined1 =========
            // bf16 -> f32 (only first D elements)
            convert_bf16_to_f32_avx2(row_noise, row_f32.data(), D);

            // RMSNorm in f32 (no γ inside)
            rmsnorm_avx2(row_f32.data(), D);

            // multiply by combined1 and store to local bf16 buffer
            mul_and_store_bf16_avx2(
                row_f32.data(),
                combined1_f32.data(),
                row_norm2_u16.data(),
                D);

            // ========= residual add in bf16 =========
            dtype_out*       row_norm2_bf16 =
                reinterpret_cast<dtype_out*>(row_norm2_u16.data());
            const dtype_out* row_res_bf16 =
                reinterpret_cast<const dtype_out*>(row_res16);

            for (int j = 0; j < D; ++j) {
                row_mlp_bf16[j] = row_norm2_bf16[j] + row_res_bf16[j];
            }

            // write intermediate x_noise_in_norm2_mlp
            for (int j = 0; j < D; ++j) {
                row_mlp_global[j] = row_mlp_bf16[j];
            }

            // ========= stage 2: RMSNorm(mlp_input) * combined2 =========
            const uint16_t* row_mlp_u16 =
                reinterpret_cast<const uint16_t*>(row_mlp_bf16.data());

            // bf16 -> f32
            convert_bf16_to_f32_avx2(row_mlp_u16, row_f32.data(), D);

            // RMSNorm again in f32
            rmsnorm_avx2(row_f32.data(), D);

            // multiply by combined2 and write compact bf16 row
            mul_and_store_bf16_avx2(
                row_f32.data(),
                combined2_f32.data(),
                row_out16,
                D);

            // copy compact row into first D columns of noise_mlp0 / noise_mlp1
            std::memcpy(row_mlp0_u16, row_out16,
                        static_cast<size_t>(D) * sizeof(uint16_t));
            std::memcpy(row_mlp1_u16, row_out16,
                        static_cast<size_t>(D) * sizeof(uint16_t));

            // zero-pad [D,4096) for both MLP outputs
            std::memset(row_mlp0_u16 + D, 0,
                        (D_INLD - D) * sizeof(uint16_t));
            std::memset(row_mlp1_u16 + D, 0,
                        (D_INLD - D) * sizeof(uint16_t));
        }
    }
}

static inline void fused_norm2_add_4096x3840_avx512(
    const uint16_t* all_mlp2_out_bf16,   // [L,4096]
    const uint16_t* gamma_norm2_u16,     // [3840]
    const uint16_t* gate_mlp_bf16,       // [3840]
    const uint16_t* x_in_norm2_mlp_bf16, // [L,3840]
          uint16_t* out_sum_bf16,        // [L,3840]  (all_result_buffers[i+1])
    int L,
    float eps = 1e-6f)
{
    constexpr int D_NORM = 3840;   // dim we norm on
    constexpr int D_INLD = 4096;   // input row stride
    constexpr int V      = 16;

    // ---- 1) precompute combined scale γ * gate in fp32 ----
    alignas(64) float combined[D_NORM];

    for (int j = 0; j < D_NORM; j += V) {
        __m512 g    = load_bf16_to_fp32(gamma_norm2_u16 + j);
        __m512 gate = load_bf16_to_fp32(gate_mlp_bf16   + j);
        __m512 c    = _mm512_mul_ps(g, gate);
        _mm512_store_ps(combined + j, c);
    }

    // ---- 2) per-row loop ----
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < L; ++r) {
        const uint16_t* in_row_4096  = all_mlp2_out_bf16   + (size_t)r * D_INLD;
        const uint16_t* add_row_3840 = x_in_norm2_mlp_bf16 + (size_t)r * D_NORM;

        uint16_t* sum_row_3840 = out_sum_bf16 + (size_t)r * D_NORM;

        // ---- (A) RMSNorm stats over first 3840 of in_row ----
        __m512 acc = _mm512_setzero_ps();
        for (int j = 0; j < D_NORM; j += V) {
            __m512 x = load_bf16_to_fp32(in_row_4096 + j);
            acc = _mm512_fmadd_ps(x, x, acc);
        }
        float sumsq   = reduce_add_ps(acc);
        float mean    = sumsq / float(D_NORM);
        float inv_rms = 1.0f / std::sqrt(mean + eps);
        __m512 v_inv  = _mm512_set1_ps(inv_rms);

        // ---- (B) compute sum only: norm_row + x_in_norm2_mlp ----
        for (int j = 0; j < D_NORM; j += V) {
            __m512 x    = load_bf16_to_fp32(in_row_4096 + j);   // from [L,4096]
            __m512 c    = _mm512_load_ps(combined + j);      // γ * gate
            __m512 x_n  = _mm512_mul_ps(x, v_inv);           // RMSNorm
            x_n         = _mm512_mul_ps(x_n, c);             // scale

            __m512 add  = load_bf16_to_fp32(add_row_3840 + j);  // x_all_in_norm2_mlp
            __m512 sum  = _mm512_add_ps(x_n, add);

            f32_to_bf16_rn_16(sum_row_3840 + j, sum);
        }
    }
}


inline void layernorm_bf16_avx512_noaffine_ptr(
    const dtype_out* src,   // [B*H] bf16 storage
    dtype_out*       dst,   // [B*H] bf16 storage
    size_t B, size_t H, float eps = 1e-6f)
{
    // Use only the u16 views for all memory access to avoid aliasing UB
    const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(src);
    uint16_t*       dst_bits = reinterpret_cast<uint16_t*>(dst);

    for (size_t b = 0; b < B; ++b) {
        size_t off = b * H;
        const uint16_t* row = src_bits + off;

        // mean
        __m512 vsum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 15 < H; i += 16)
            vsum = _mm512_add_ps(vsum, load_bf16_to_fp32(row + i));
        float mean = _mm512_reduce_add_ps(vsum);
        for (; i < H; ++i) mean += bfloat16_to_float(row[i]);
        mean /= float(H);
        __m512 vmean = _mm512_set1_ps(mean);

        // variance
        __m512 vsq = _mm512_setzero_ps();
        i = 0;
        for (; i + 15 < H; i += 16) {
            __m512 x = load_bf16_to_fp32(row + i);
            __m512 d = _mm512_sub_ps(x, vmean);
            vsq = _mm512_fmadd_ps(d, d, vsq);
        }
        float var = _mm512_reduce_add_ps(vsq);
        for (; i < H; ++i) { float d = bfloat16_to_float(row[i]) - mean; var += d*d; }
        var /= float(H);
        float invstd = 1.0f / std::sqrt(var + eps);
        __m512 vinv = _mm512_set1_ps(invstd);

        // normalize & store
        i = 0;
        for (; i + 15 < H; i += 16) {
            __m512 x = load_bf16_to_fp32(row + i);
            __m512 y = _mm512_mul_ps(_mm512_sub_ps(x, vmean), vinv);
            store_fp32_to_bf16(dst_bits + off + i, y);
        }
        for (; i < H; ++i) {
            float x = bfloat16_to_float(row[i]);
            dst_bits[off + i] = float_to_bfloat16((x - mean) * invstd);
        }
    }
}

inline void layernorm_bf16_avx512_noaffine(
    const buffer<dtype_out>& src_buf,
    buffer<dtype_out>&       dst_buf,
    size_t B, size_t H, float eps = 1e-6f)
{
    layernorm_bf16_avx512_noaffine_ptr(src_buf.data(), dst_buf.data(), B, H, eps);
}
#endif