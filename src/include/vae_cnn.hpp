#ifndef VAE_CNN_HPP
#define VAE_CNN_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include "npu_utils.hpp"
#include "denoise_lib.hpp"
// #include "img_to_col.hpp"

using bfloat16 = uint16_t;
inline bool bf16_isnan(uint16_t v) {
    uint16_t exp  = (v >> 7) & 0xFF;
    uint16_t mant = v & 0x7F;
    return (exp == 0xFF) && (mant != 0);
}

size_t count_nan_bf16(const uint16_t* data, size_t n) {
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        cnt += bf16_isnan(data[i]);
    }
    return cnt;
}

struct vae_decoder_weights {
    static constexpr std::size_t CONV_IN_W  = 512ull * 16 * 3 * 3;
    static constexpr std::size_t CONV_IN_B  = 512ull;
    static constexpr std::size_t NORM_OUT   = 128ull;
    static constexpr std::size_t CONV_OUT_W = 3ull * 128 * 3 * 3;
    static constexpr std::size_t CONV_OUT_B = 3ull;
    static constexpr std::size_t D          = 512ull;
    static constexpr std::size_t D2         = 512ull * 512;
  
    std::vector<uint16_t> conv_in_weight;
    std::vector<uint16_t> conv_in_bias;

    buffer<dtype_in> mid_block_resnets_0_norm1_weight;
    buffer<dtype_in> mid_block_resnets_0_norm1_bias;
    buffer<dtype_in> mid_block_resnets_0_weight;
    buffer<dtype_in> mid_block_resnets_0_bias;
    buffer<dtype_in> mid_block_resnets_0_norm2_weight;
    buffer<dtype_in> mid_block_resnets_0_norm2_bias;
    buffer<dtype_in> mid_block_resnets_0_conv2_weight;
    buffer<dtype_in> mid_block_resnets_0_conv2_bias;

    buffer<dtype_in> mid_atten_group_norm_weight;
    buffer<dtype_in> mid_atten_group_norm_bias;
    // buffer<dtype_in> mid_atten_to_k;
    // buffer<dtype_in> mid_atten_to_k_bias;
    // buffer<dtype_in> mid_atten_to_q;
    // buffer<dtype_in> mid_atten_to_q_bias;
    // buffer<dtype_in> mid_atten_to_v;
    // buffer<dtype_in> mid_atten_to_v_bias;
    // buffer<dtype_in> mid_atten_to_out;
    // buffer<dtype_in> mid_atten_to_out_bias;

    buffer<dtype_out> updecoder0_upsample_weight;
    buffer<dtype_out> updecoder0_upsample_bias;
   
    buffer<dtype_out> updecoder1_upsample_weight;
    buffer<dtype_out> updecoder1_upsample_bias;
   
    buffer<dtype_out> updecoder2_upsample_weight;
    buffer<dtype_out> updecoder2_upsample_bias;
    
    buffer<dtype_out> final_conv_norm_out_weight;
    buffer<dtype_out> final_conv_norm_out_bias;
    buffer<dtype_out> final_conv_out_weight;
    buffer<dtype_out> final_conv_out_bias;



  
    vae_decoder_weights()
      : conv_in_weight(CONV_IN_W),
        conv_in_bias(CONV_IN_B),
        mid_block_resnets_0_norm1_weight(D),
        mid_block_resnets_0_norm1_bias(D),
        mid_block_resnets_0_norm2_weight(512),
        mid_block_resnets_0_norm2_bias(512),
        mid_block_resnets_0_weight(512*512*3*3),
        mid_block_resnets_0_bias(512),
        mid_block_resnets_0_conv2_weight(512*512*3*3),
        mid_block_resnets_0_conv2_bias(512),
        
        updecoder0_upsample_weight(512*512*3*3),
        updecoder0_upsample_bias(512),
        updecoder1_upsample_weight(512*512*3*3),
        updecoder1_upsample_bias(512),
        updecoder2_upsample_weight(256*256*3*3),
        updecoder2_upsample_bias(256),
      
        final_conv_norm_out_weight(128),
        final_conv_norm_out_bias(128),
        final_conv_out_weight(3*128*3*3),
        final_conv_out_bias(3),

        
        
        mid_atten_group_norm_weight(D),
        mid_atten_group_norm_bias(D)
        // mid_atten_to_k(D2),
        // mid_atten_to_k_bias(D),
        // mid_atten_to_q(D2),
        // mid_atten_to_q_bias(D),
        // mid_atten_to_v(D2),
        // mid_atten_to_v_bias(D),
        // mid_atten_to_out(D2),
        // mid_atten_to_out_bias(D) 
        {}
  };


// Utility: Convert Float to BF16 (Truncation method)
bfloat16 float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    return (bfloat16)(x >> 16); // Take upper 16 bits
}

// Utility: Convert BF16 to Float
float bf16_to_float(bfloat16 bf) {
    uint32_t x = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

inline uint16_t float_to_bf16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    return static_cast<uint16_t>(x >> 16);
}

void ProcessVaeOutputBF16(const uint16_t* input, uint8_t* output, int width, int height, int channels) {
    const int pixels_per_channel = width * height;
    const float scale_factor = 127.5f;

    for (int i = 0; i < pixels_per_channel; ++i) {
        for (int c = 0; c < channels; ++c) {
            // 1. Calculate Indexing (Planar CHW)
            int in_idx = (c * pixels_per_channel) + i;
            
            // 2. Convert bfloat16 to float32
            float val = bf16_to_float(input[in_idx]);

            // 3. Clamp (matches .clamp(-1.0f, 1.0f))
            if (val < -1.0f) val = -1.0f;
            if (val > 1.0f)  val = 1.0f;

            // 4. Normalize & Scale
            val = (val + 1.0f) * scale_factor;

            // 5. Round and Cast
            output[i * channels + c] = static_cast<uint8_t>(val + 0.5f);
        }
    }
}

void conv2d_nchw_native_padded( float* input, 
    uint16_t* weights, 
     uint16_t* bias, 
    uint16_t* output,
    int H, int W, int C_IN, int C_OUT, 
    int K_H, int K_W, int STRIDE, int PAD) {

// Helper constant
int plane_size = H * W;

// Loop Output Channels
for (int co = 0; co < C_OUT; ++co) {

// 1. Prepare Bias (Broadcast to vector)
uint16_t b_val = bias[co];
uint32_t b_u32 = static_cast<uint32_t>(b_val) << 16;
float bias_f;
std::memcpy(&bias_f, &b_u32, sizeof(float));
__m512 bias_vec = _mm512_set1_ps(bias_f);

// Loop Spatial Output (Height)
for (int h = 0; h < H; ++h) {

// Loop Spatial Output (Width) in chunks of 16
for (int w = 0; w < W; w += 16) {

// Initialize Sum with Bias
__m512 sum = bias_vec;

// Loop Kernel (Standard 3x3 loop)
for (int kh = 0; kh < K_H; ++kh) {
// Calculate Vertical Input Index (Virtual Padding)
int in_h = h * STRIDE - PAD + kh;

// Optimization: If this entire row is padding (outside H), skip it!
if (in_h < 0 || in_h >= H) continue;

for (int kw = 0; kw < K_W; ++kw) {
// Calculate Horizontal Input Start Index
int in_w_start = w * STRIDE - PAD + kw;

// -----------------------------------------------------
// VIRTUAL PADDING MASK
// -----------------------------------------------------
// We are loading 16 pixels: [in_w_start, ..., in_w_start+15]
// Some might be < 0 (Left Pad) or >= W (Right Pad).
// We construct a mask where 1 = Valid Pixel, 0 = Padding.

// Default: All ones (0xFFFF)
int mask_int = 0xFFFF;

// Left Clipping (if starting before 0)
if (in_w_start < 0) {
 // Example: in_w_start = -1. We need to mask out bit 0.
 // Shift mask left by abs(in_w_start) to clear bottom bits
 mask_int &= ~((1 << (-in_w_start)) - 1);
}

// Right Clipping (if ending after W)
int in_w_end = in_w_start + 16;
if (in_w_end > W) {
 // Example: W=128, end=129. We need to mask out top bit.
 int overflow = in_w_end - W;
 mask_int &= (1 << (16 - overflow)) - 1;
}

// Convert to AVX mask
__mmask16 load_mask = (__mmask16)mask_int;

// If mask is 0 (all padding), skip compute
if (load_mask == 0) continue;

// -----------------------------------------------------
// ACCUMULATION
// -----------------------------------------------------
// Loop Input Channels (Sequentially for NCHW layout)
// Note: C_IN=16 is small, so we unroll or just loop.
for (int ci = 0; ci < C_IN; ++ci) {
 
 // A. Load Weight (Scalar -> Broadcast)
 // PyTorch NCHW Weight Index: co * (Cin*K*K) + ci * (K*K) + ...
 int w_idx = (co * C_IN * K_H * K_W) + 
             (ci * K_H * K_W) + 
             (kh * K_W) + kw;
 
 uint16_t w_bf16 = weights[w_idx];
 // Convert bf16 -> float32 for FMA
 uint32_t w_u32 = static_cast<uint32_t>(w_bf16) << 16;
 float w_f32;
 std::memcpy(&w_f32, &w_u32, sizeof(float));
 
 __m512 w_vec = _mm512_set1_ps(w_f32);

 // B. Load Input (Masked Load)
 // Pointer to start of this channel plane
 const float* ch_ptr = input + (ci * plane_size) + (in_h * W);
 
 // Offset to specific column (in_w_start)
 // Note: pointer arithmetic handles negative index logic gracefully
 // as long as the mask prevents access.
 __m512 in_vec = _mm512_maskz_loadu_ps(load_mask, ch_ptr + in_w_start);

 // C. Math: Sum += Input * Weight
 sum = _mm512_fmadd_ps(in_vec, w_vec, sum);
}
}
}

// -----------------------------------------------------
// STORE RESULT (Convert to BF16)
// -----------------------------------------------------
// Handle edge case where Width is not a multiple of 16 (optional)
__mmask16 store_mask = (w + 16 <= W) ? 0xFFFF : (1 << (W - w)) - 1;

__m256bh res_bf16 = _mm512_cvtneps_pbh(sum);

int out_idx = (co * plane_size) + (h * W) + w;
_mm256_mask_storeu_epi16(output + out_idx, store_mask, (__m256i)res_bf16);
}
}
}
}


// --------------------------------------------------------------------------
// TUNING PARAMETERS FOR 512x512
// --------------------------------------------------------------------------


// ---------------------------------------------------------------------
// HELPER: Fix for "load_bf16_to_ps not declared"
// ---------------------------------------------------------------------
inline __m512 load_bf16_to_ps(const uint16_t* ptr) {
    // Load 16x BF16 (256 bits)
    __m256i loaded = _mm256_loadu_si256((const __m256i*)ptr);
    // Convert to FP32 (AVX512_BF16)
    // If compiler fails here, verify -march=native and AVX512 support
    return _mm512_cvtpbh_ps((__m256bh)loaded);
}


/////////////////////////////////////////////////////////////////////////

static inline __m512 _mm512_load_bf16_to_ps(const uint16_t* addr) {
    __m256i raw_bf16 = _mm256_loadu_si256((const __m256i*)addr); // Load 16x BF16
    __m512i expanded = _mm512_cvtepu16_epi32(raw_bf16);          // Expand to 32-bit int
    __m512i shifted  = _mm512_slli_epi32(expanded, 16);           // Shift to Float position
    return _mm512_castsi512_ps(shifted);                          // Reinterpret as Float
}

// Masked Load for Padding (Handle boundaries)
static inline __m512 _mm512_maskz_load_bf16_to_ps(__mmask16 mask, const uint16_t* addr) {
    __m256i raw_bf16 = _mm256_maskz_loadu_epi16(mask, addr);      // Masked Load
    __m512i expanded = _mm512_cvtepu16_epi32(raw_bf16);
    __m512i shifted  = _mm512_slli_epi32(expanded, 16);
    return _mm512_castsi512_ps(shifted);
}

// Convert Float32 (512-bit) -> BFloat16 (256-bit) and Store
// Method: Round to nearest even (check AVX512_BF16 support) or Truncate
static inline void _mm512_store_ps_as_bf16(uint16_t* addr, __m512 v) {
    // Fast Truncation Method (Standard for DL inference)
    // 1. Cast to Int32
    __m512i v_int = _mm512_castps_si512(v);
    
    // 2. Add 0x8000 for Rounding (Optional, but better precision)
    // __m512i rounding = _mm512_set1_epi32(0x8000);
    // v_int = _mm512_add_epi32(v_int, rounding);

    // 3. Shift Right Logical by 16 to keep top bits
    __m512i v_shifted = _mm512_srli_epi32(v_int, 16);
    
    // 4. Pack 32-bit integers down to 16-bit
    __m256i v_bf16 = _mm512_cvtepi32_epi16(v_shifted);
    
    _mm256_storeu_si256((__m256i*)addr, v_bf16);
}

// Masked Store
static inline void _mm512_mask_store_ps_as_bf16(uint16_t* addr, __mmask16 mask, __m512 v) {
    __m512i v_int = _mm512_castps_si512(v);
    __m512i v_shifted = _mm512_srli_epi32(v_int, 16);
    __m256i v_bf16 = _mm512_cvtepi32_epi16(v_shifted);
    _mm256_mask_storeu_epi16(addr, mask, v_bf16);
}

// =============================================================
// 2. The Conv2d Kernel (3x3, Pad 1, Stride 1)
// =============================================================
#define BLOCK_W_1 4 

void conv2d_bf16_avx512_optimized(
    const uint16_t* input,    // [C_in, H, W]
    const uint16_t* weights,  // [C_out, C_in, 3, 3]
    const uint16_t* bias,     // [C_out]
    uint16_t* output,         // [C_out, H, W]
    int H, int W, 
    int C_IN, int C_OUT
) {
    int plane_size = H * W;

    // Parallel Output Channels
    #pragma omp parallel for schedule(dynamic)
    for (int co = 0; co < C_OUT; ++co) {
        
        // 1. Prepare Bias (Load once -> Broadcast)
        uint32_t b_val = static_cast<uint32_t>(bias[co]) << 16;
        float b_f; std::memcpy(&b_f, &b_val, 4);
        __m512 bias_vec = _mm512_set1_ps(b_f);

        // Pre-calculate weight pointer for this output channel
        const uint16_t* w_ptr_co = weights + (co * C_IN * 9);

        for (int h = 0; h < H; ++h) {
            
            // Loop Width in chunks of (16 * BLOCK_W) = 64 pixels
            for (int w = 0; w < W; w += (16 * BLOCK_W_1)) {
                
                // ---------------------------------------------------------
                // A. Init Accumulators
                // ---------------------------------------------------------
                __m512 accum[BLOCK_W_1];
                
                // Identify valid columns in this block (handling right edge of image)
                bool col_valid[BLOCK_W_1];
                for (int b = 0; b < BLOCK_W_1; ++b) {
                    accum[b] = bias_vec;
                    col_valid[b] = (w + (b * 16) < W);
                }

                // ---------------------------------------------------------
                // B. Pre-calculate Padding Masks (HOISTED)
                // ---------------------------------------------------------
                // We calculate masks for the 3 horizontal kernel positions (KW=0,1,2)
                // This removes integer logic from the inner hot loop.
                __mmask16 cached_masks[3][BLOCK_W_1];
                int       cached_offsets[3][BLOCK_W_1];

                for (int kw = 0; kw < 3; ++kw) {
                    for (int b = 0; b < BLOCK_W_1; ++b) {
                        if (!col_valid[b]) continue;

                        int w_curr = w + (b * 16);
                        int in_w_start = w_curr - 1 + kw; // Pad 1 logic

                        cached_offsets[kw][b] = in_w_start;
                        
                        // Default Mask (handling image right edge clamping)
                        __mmask16 m = 0xFFFF;
                        int rem = W - w_curr;
                        if (rem < 16) m = (1 << rem) - 1;

                        // Padding Logic (Left/Right Kernel Overhang)
                        // Note: We use negative offsets allowed by maskz_load
                        if (in_w_start < 0) {
                            // Clear bits for the left padding (e.g. index -1)
                            m &= ~((1 << -in_w_start) - 1);
                        } else if (in_w_start + 16 > W) {
                            // Clear bits for right padding
                            int over = (in_w_start + 16) - W;
                            if (over > 0) m &= (1 << (16 - over)) - 1;
                        }
                        cached_masks[kw][b] = m;
                    }
                }

                // ---------------------------------------------------------
                // C. Compute (Accumulate C_IN)
                // ---------------------------------------------------------
                const uint16_t* w_ptr = w_ptr_co;
                
                for (int ci = 0; ci < C_IN; ++ci) {
                    const uint16_t* in_plane = input + (ci * plane_size);

                    // 3x3 Kernel Loop
                    for (int kh = 0; kh < 3; ++kh) {
                        int in_h = h - 1 + kh;

                        // Vertical Padding Optimization: Skip entire row if OOB
                        if (in_h < 0 || in_h >= H) {
                            w_ptr += 3; // Advance weight pointer past this row
                            continue;
                        }

                        const uint16_t* in_row_ptr = in_plane + (in_h * W);

                        for (int kw = 0; kw < 3; ++kw) {
                            
                            // 1. Load Weight (Scalar BF16 -> Broadcast Float32)
                            // Note: w_ptr increments linearly
                            uint32_t w_bits = static_cast<uint32_t>(*w_ptr++) << 16;
                            float w_f_local; std::memcpy(&w_f_local, &w_bits, 4);
                            __m512 w_vec = _mm512_set1_ps(w_f_local);

                            // 2. Compute for the blocked width
                            #pragma GCC unroll 4
                            for (int b = 0; b < BLOCK_W_1; ++b) {
                                if (!col_valid[b]) continue;

                                // Load cached pre-calculated mask/offset
                                __mmask16 m = cached_masks[kw][b];
                                int off = cached_offsets[kw][b];

                                // LOAD + CONVERT
                                // _mm256_maskz_loadu_epi16 handles the negative offset safely 
                                // because the mask bits for the negative part are 0.
                                __m256i raw = _mm256_maskz_loadu_epi16(m, (const void*)(in_row_ptr + off));
                                __m512 in_vec = _mm512_cvtpbh_ps((__m256bh)raw);

                                // FMA
                                accum[b] = _mm512_fmadd_ps(in_vec, w_vec, accum[b]);
                            }
                        }
                    }
                }

                // ---------------------------------------------------------
                // D. Store Result
                // ---------------------------------------------------------
                for (int b = 0; b < BLOCK_W_1; ++b) {
                    if (col_valid[b]) {
                        int w_curr = w + (b * 16);
                        int out_idx = (co * plane_size) + (h * W) + w_curr;

                        // Recalculate store mask for edge safety
                        int remain = W - w_curr;
                        __mmask16 store_mask = (remain >= 16) ? 0xFFFF : (1 << remain) - 1;

                        // Convert FP32 -> BF16 and Store
                        __m256bh res = _mm512_cvtneps_pbh(accum[b]);
                        _mm256_mask_storeu_epi16(output + out_idx, store_mask, (__m256i)res);
                    }
                }
            }
        }
    }
}



//////////////////////////////////////////////////////////////////////
void conv2d_nchw_kernel1x1_bf16(
    const bfloat16* input, 
    const uint16_t* weights, 
    const uint16_t* bias, 
    uint16_t* output,
    int H, int W, int C_IN, int C_OUT) 
{
    // Precompute strides
    // For 1x1 Stride=1, Input H/W match Output H/W
    int in_plane_size = H * W; 
    int out_plane_size = H * W;

    // Loop Output Channels (Parallelize here)
    #pragma omp parallel for
    for (int co = 0; co < C_OUT; ++co) {

        // 1. Prepare Bias (Broadcast to vector once per Output Channel)
        // Convert bf16 bias -> float32
        uint16_t b_val = bias[co];
        uint32_t b_u32 = static_cast<uint32_t>(b_val) << 16;
        float bias_f;
        std::memcpy(&bias_f, &b_u32, sizeof(float));
        __m512 bias_vec = _mm512_set1_ps(bias_f);

        // Loop Spatial Output (Height)
        for (int h = 0; h < H; ++h) {
            
            // Loop Spatial Output (Width) in chunks of 16
            for (int w = 0; w < W; w += 16) {

                // 2. Initialize Sum with Bias
                __m512 sum = bias_vec;

                // 3. Determine Mask (Only needed for the last chunk)
                // If we are at the edge, calculate remainder. Otherwise all 1s.
                int remain = W - w;
                __mmask16 mask = (remain >= 16) ? 0xFFFF : (1 << remain) - 1;

                // 4. Loop Input Channels (Reduction)
                for (int ci = 0; ci < C_IN; ++ci) {
                    
                    // A. Load Weight (Scalar -> Broadcast)
                    // Index: [co, ci, 0, 0] -> flattened: co * C_IN + ci
                    int w_idx = (co * C_IN) + ci;
                    
                    uint16_t w_bf16 = weights[w_idx];
                    uint32_t w_u32 = static_cast<uint32_t>(w_bf16) << 16;
                    float w_f32;
                    std::memcpy(&w_f32, &w_u32, sizeof(float));
                    
                    __m512 w_vec = _mm512_set1_ps(w_f32);

                    // B. Load Input (BF16 -> Float32)
                    // Input Ptr: Plane(ci) + Row(h) + Col(w)
                    const bfloat16* in_ptr = input + (ci * in_plane_size) + (h * W) + w;

                    // Load 16 bf16 values (masked for safety at edges)
                    __m256i input_bf16_raw = _mm256_maskz_loadu_epi16(mask, in_ptr);

                    // Convert Packed BF16 (YMM) -> Packed Float (ZMM)
                    // Requires AVX-512_BF16 or VL support. 
                    // _mm512_cvtpbh_ps takes __m256bh (bfloat16 vector)
                    __m512 in_vec = _mm512_cvtpbh_ps((__m256bh)input_bf16_raw);

                    // C. FMA: Sum += Input * Weight
                    sum = _mm512_fmadd_ps(in_vec, w_vec, sum);
                }

                // 5. Store Result (Convert FP32 -> BF16)
                // Convert Accumulator back to BF16
                __m256bh res_bf16 = _mm512_cvtneps_pbh(sum);

                // Store to Output: Plane(co) + Row(h) + Col(w)
                int out_idx = (co * out_plane_size) + (h * W) + w;
                _mm256_mask_storeu_epi16(output + out_idx, mask, (__m256i)res_bf16);
            }
        }
    }
}

void conv2d_nchw_native_padded_bf16_opt(
    const uint16_t* input,      
    const uint16_t* weights,    
    const uint16_t* bias,       
    uint16_t* output,           
    int H, int W, int C_IN, int C_OUT, 
    int K_H, int K_W, int STRIDE, int PAD) 
{
    // -----------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------
    constexpr int BLOCK_CO = 16; // Process 16 Output Channels (Fits in ZMM)
    int plane_size = H * W;
    int kernel_vol = K_H * K_W;

    // -----------------------------------------------------------------
    // 1. GLOBAL WEIGHT PACKING
    // -----------------------------------------------------------------
    // Layout: [C_OUT_PADDED/16, C_IN, K_H, K_W, 16]
    // This linearizes the access pattern for the inner loop.
    int C_OUT_PADDED = (C_OUT + BLOCK_CO - 1) / BLOCK_CO * BLOCK_CO;
    std::vector<float> packed_weights(C_OUT_PADDED * C_IN * kernel_vol);

    #pragma omp parallel for schedule(static)
    for (int co_blk = 0; co_blk < C_OUT; co_blk += BLOCK_CO) {
        size_t blk_offset = (size_t)co_blk * C_IN * kernel_vol;

        for (int ci = 0; ci < C_IN; ++ci) {
            for (int k = 0; k < kernel_vol; ++k) {
                // We pack 16 output channels together for this specific (ci, k)
                for (int r = 0; r < BLOCK_CO; ++r) {
                    int co = co_blk + r;
                    float val = 0.0f;
                    
                    if (co < C_OUT) {
                        // Original Weight Index: [CO, CI, K]
                        int src_idx = (co * C_IN * kernel_vol) + (ci * kernel_vol) + k;
                        val = bf16_to_f32(weights[src_idx]);
                    }
                    
                    // Dest Index: Block-major
                    size_t dst_idx = blk_offset + ((size_t)ci * kernel_vol * BLOCK_CO) + (k * BLOCK_CO) + r;
                    packed_weights[dst_idx] = val;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // 2. COMPUTE LOOP
    // -----------------------------------------------------------------
    #pragma omp parallel for schedule(dynamic)
    for (int h = 0; h < H; ++h) {
        
        // Loop over Output Channels in blocks of 16
        for (int co_blk = 0; co_blk < C_OUT; co_blk += BLOCK_CO) {
            
            // Pointer to the start of weights for this CO block
            const float* w_ptr_base = packed_weights.data() + ((size_t)co_blk * C_IN * kernel_vol);

            // Loop over Width (Output) in chunks of 16
            for (int w = 0; w < W; w += 16) {

                // --- Init Accumulators with Bias ---
                __m512 accum[BLOCK_CO];
                for (int r = 0; r < BLOCK_CO; ++r) {
                    if (co_blk + r < C_OUT) {
                        accum[r] = _mm512_set1_ps(bf16_to_f32(bias[co_blk + r]));
                    } else {
                        accum[r] = _mm512_setzero_ps();
                    }
                }

                // --- Convolution ---
                const float* w_runner = w_ptr_base;

                for (int ci = 0; ci < C_IN; ++ci) {
                    const uint16_t* in_plane = input + (ci * plane_size);

                    for (int kh = 0; kh < K_H; ++kh) {
                        int in_h = h * STRIDE - PAD + kh;

                        // Vertical Padding: Skip entire row if invalid
                        if (in_h < 0 || in_h >= H) {
                            w_runner += (K_W * BLOCK_CO); // Skip weights
                            continue;
                        }

                        const uint16_t* in_row_ptr = in_plane + (in_h * W);

                        for (int kw = 0; kw < K_W; ++kw) {
                            
                            // -------------------------------------------------
                            // ROBUST INPUT LOADING
                            // -------------------------------------------------
                            __m512 in_vec;

                            // Calculate input X position for the first pixel in this block
                            int w_start_in = w * STRIDE - PAD + kw;
                            
                            // Check if this entire 16-pixel block falls within valid image bounds
                            // Condition: Stride must be 1 (for contiguous load) AND no padding needed
                            bool fast_path = (STRIDE == 1) && 
                                             (w_start_in >= 0) && 
                                             (w_start_in + 16 <= W);

                            if (fast_path) {
                                // FAST PATH: Unaligned Load directly from memory
                                // Since W_start_in >= 0, pointer arithmetic is safe
                                __m256i raw = _mm256_loadu_si256((const __m256i*)(in_row_ptr + w_start_in));
                                in_vec = _mm512_cvtpbh_ps((__m256bh)raw);
                            } 
                            else {
                                // SAFE PATH: Handle Padding, Stride > 1, and Edges
                                // We construct the vector manually. This guarantees correctness.
                                // It is slower but only runs on the edges of the image.
                                float temp_buf[16];
                                for (int i = 0; i < 16; ++i) {
                                    int curr_w_out = w + i;
                                    // Calculate Input Index
                                    int curr_w_in = curr_w_out * STRIDE - PAD + kw;
                                    
                                    if (curr_w_out < W && curr_w_in >= 0 && curr_w_in < W) {
                                        temp_buf[i] = bf16_to_f32(in_row_ptr[curr_w_in]);
                                    } else {
                                        temp_buf[i] = 0.0f; // Padding (Zero)
                                    }
                                }
                                in_vec = _mm512_loadu_ps(temp_buf);
                            }

                            // -------------------------------------------------
                            // COMPUTE (FMA)
                            // -------------------------------------------------
                            // Unroll loop for the 16 Output Channels
                            #pragma GCC unroll 16
                            for (int r = 0; r < BLOCK_CO; ++r) {
                                __m512 w_val = _mm512_set1_ps(*w_runner++);
                                accum[r] = _mm512_fmadd_ps(in_vec, w_val, accum[r]);
                            }
                        }
                    }
                }

                // --- Store Results ---
                for (int r = 0; r < BLOCK_CO; ++r) {
                    if (co_blk + r < C_OUT) {
                        for (int i = 0; i < 16; ++i) {
                            int curr_w = w + i;
                            if (curr_w < W) {
                                int out_idx = ((co_blk + r) * plane_size) + (h * W) + curr_w;
                                // Extract float, convert to bf16, store
                                // (Note: In a pure optimization, we would use vector stores with masks,
                                //  but scalar extraction here ensures debugging clarity given previous errors)
                                float res_f = ((float*)&accum[r])[i];
                                output[out_idx] = f32_to_bf16(res_f);
                            }
                        }
                    }
                }
            }
        }
    }
}
// -----------------------------------------------------------------------------
// AVX-512 GroupNorm (Mixed Precision)
// Input/Output/Weights: BFloat16
// Computation: Float32 (Accumulators)
// -----------------------------------------------------------------------------
void groupnorm_avx512_bf16(const bfloat16* input,   // [N, C, H, W]
                           bfloat16* output,        // [N, C, H, W]
                           const bfloat16* gamma,   // [C]
                           const bfloat16* beta,    // [C]
                           int N, int C, int H, int W, 
                           int G, 
                           float eps = 1e-5f) {

    int HW = H * W;
    int channels_per_group = C / G;
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            
            // -------------------------------------------------------
            // PASS 1: Statistics (Mean & Variance) in Float32
            // -------------------------------------------------------
            __m512 v_sum = _mm512_setzero_ps();
            __m512 v_sq_sum = _mm512_setzero_ps();

            int group_start_c = g * channels_per_group;
            int group_end_c = group_start_c + channels_per_group;

            for (int c = group_start_c; c < group_end_c; ++c) {
                const bfloat16* ptr = input + (n * C * HW) + (c * HW);

                // Iterate Spatial Pixels
                for (int i = 0; i < HW; i += 16) {
                    // 1. Mask Handling for non-multiple of 16
                    __mmask16 mask = (i + 16 <= HW) ? 0xFFFF : (1 << (HW - i)) - 1;
                    
                    // 2. Load BF16 (16 values = 256 bits)
                    // We load as integer, but masked
                    __m256i bf_vec = _mm256_maskz_loadu_epi16(mask, ptr + i);
                    
                    // 3. Convert to Float32
                    // _mm512_cvtpbh_ps takes __m256bh (which is essentially __m256i bits)
                    __m512 val_f32 = _mm512_cvtpbh_ps((__m256bh)bf_vec);
                    
                    // 4. Accumulate
                    v_sum = _mm512_add_ps(v_sum, val_f32);
                    v_sq_sum = _mm512_fmadd_ps(val_f32, val_f32, v_sq_sum);
                }
            }

            // Reduction to Scalar
            float sum = _mm512_reduce_add_ps(v_sum);
            float sq_sum = _mm512_reduce_add_ps(v_sq_sum);

            float count = (float)(channels_per_group * HW);
            float mu = sum / count;
            float var = (sq_sum / count) - (mu * mu);
            float inv_std = 1.0f / std::sqrt(var + eps);

            // -------------------------------------------------------
            // PASS 2: Normalize & Store (Float32 -> BF16)
            // -------------------------------------------------------
            for (int c = group_start_c; c < group_end_c; ++c) {
                
                // Convert gamma/beta to float ONCE per channel
                float gm = bf16_to_float(gamma[c]);
                float bt = bf16_to_float(beta[c]);

                // Pre-calc Fused constants: y = x * A + B
                float term_A = inv_std * gm;
                float term_B = bt - (mu * term_A);

                __m512 v_term_A = _mm512_set1_ps(term_A);
                __m512 v_term_B = _mm512_set1_ps(term_B);

                const bfloat16* in_ptr = input + (n * C * HW) + (c * HW);
                bfloat16* out_ptr = output + (n * C * HW) + (c * HW);

                for (int i = 0; i < HW; i += 16) {
                    __mmask16 mask = (i + 16 <= HW) ? 0xFFFF : (1 << (HW - i)) - 1;

                    // A. Load BF16 -> Float32
                    __m256i bf_raw = _mm256_maskz_loadu_epi16(mask, in_ptr + i);
                    __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_raw);

                    // B. Math (Fused)
                    __m512 y = _mm512_fmadd_ps(x, v_term_A, v_term_B);

                    // C. Convert Float32 -> BF16
                    __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                    // D. Store BF16
                    _mm256_mask_storeu_epi16(out_ptr + i, mask, (__m256i)y_bf16);
                }
            }
        }
    }
}

constexpr float LOG2E = 1.44269504089f; // 1 / ln(2)
constexpr float LN2 = 0.69314718056f;   // ln(2)

constexpr float P5_COEFF_1 = 1.0f;
constexpr float P5_COEFF_2 = 1.0f / 2.0f;
constexpr float P5_COEFF_3 = 1.0f / 6.0f;
constexpr float P5_COEFF_4 = 1.0f / 24.0f;
constexpr float P5_COEFF_5 = 1.0f / 120.0f;

inline __m512 fast_exp_avx512(__m512 x) {
    // --- Pre-load constants into vector registers ---
    const __m512 v_log2e = _mm512_set1_ps(LOG2E);
    const __m512 v_ln2 = _mm512_set1_ps(LN2);
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512i v_127 = _mm512_set1_epi32(127); // FP32 exponent bias

    // --- 1. Range Reduction: x = n*ln(2) + r ---
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, v_log2e), 0);
    __m512 r = _mm512_fnmadd_ps(n, v_ln2, x); // Efficiently computes x - (n * ln2)

    // --- 2. Evaluate polynomial for e^r using Horner's method ---
    __m512 poly = _mm512_set1_ps(P5_COEFF_5);
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(P5_COEFF_4));
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(P5_COEFF_3));
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(P5_COEFF_2));
    poly = _mm512_fmadd_ps(poly, r, _mm512_set1_ps(P5_COEFF_1));
    poly = _mm512_fmadd_ps(poly, r, v_one);

    // --- 3. Calculate 2^n using fast bit manipulation ---
    __m512i n_int = _mm512_cvttps_epi32(n);
    __m512i power_of_2_bits = _mm512_add_epi32(n_int, v_127);
    power_of_2_bits = _mm512_slli_epi32(power_of_2_bits, 23);
    __m512 power_of_2 = _mm512_castsi512_ps(power_of_2_bits);

    // --- 4. Combine results: e^x ≈ 2^n * polynomial(r) ---
    return _mm512_mul_ps(power_of_2, poly);
}


// -----------------------------------------------------------------------------
// Helper: Sigmoid (1 / (1 + exp(-x)))
// -----------------------------------------------------------------------------
inline __m512 fast_sigmoid_avx512(__m512 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 zero = _mm512_setzero_ps();
    
    // Calculate exp(-x)
    __m512 neg_x = _mm512_sub_ps(zero, x);
    __m512 exp_neg_x = fast_exp_avx512(neg_x);
    
    // 1 + exp(-x)
    __m512 den = _mm512_add_ps(one, exp_neg_x);
    
    // 1 / (1 + exp(-x))
    // rcp14 is an approximate reciprocal (1/x), fast but slightly less precise (14 bits).
    // For activation functions, this is usually acceptable.
    // For higher precision, use _mm512_div_ps(one, den).
    return _mm512_div_ps(one, den); 
}



void swish_avx512_bf16_pad_h_only_chw(
    const bfloat16* input,
    bfloat16* output,
    int C, int H, int W)
{
    const int Hp = H + 2;   // pad only height (top + bottom)
    const int Wp = W;       // width unchanged

    // Output shape: [C, H+2, W]
    // Zero everything first so first row / last row are all zeros
    std::memset(output, 0, static_cast<size_t>(C) * Hp * Wp * sizeof(bfloat16));

#pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            const size_t in_base  = (static_cast<size_t>(c) * H  + h)     * W;
            const size_t out_base = (static_cast<size_t>(c) * Hp + (h+1)) * W;
            //              ^ write to row (h+1), leaving row 0 and row H+1 as zero rows

            int w = 0;

            // Vectorized body: 16 BF16 elements at a time
            for (; w + 16 <= W; w += 16) {
                // Load 16 bf16 values (32 bytes)
                __m256i bf_vec_i = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(input + in_base + w));

                // BF16 -> FP32
                __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_vec_i);

                // sigmoid(x)
                __m512 sig = fast_sigmoid_avx512(x);

                // swish = x * sigmoid(x)
                __m512 y = _mm512_mul_ps(x, sig);

                // FP32 -> BF16 (16 lanes)
                __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                // Store into output center rows (no left/right padding)
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(output + out_base + w),
                    (__m256i)y_bf16);
            }

            // Tail (< 16 elements)
            if (w < W) {
                const int rem = W - w;
                __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);

                __m256i bf_vec_i = _mm256_maskz_loadu_epi16(
                    mask,
                    reinterpret_cast<const void*>(input + in_base + w));

                __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_vec_i);
                __m512 sig = fast_sigmoid_avx512(x);
                __m512 y = _mm512_mul_ps(x, sig);
                __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                _mm256_mask_storeu_epi16(
                    reinterpret_cast<void*>(output + out_base + w),
                    mask,
                    (__m256i)y_bf16);
            }
        }
    }
}

void groupnorm_swish_pad_h_avx512_bf16(
    const bfloat16* input,   // [N, C, H, W]
    bfloat16* output,        // [N, C, H+2, W]
    const bfloat16* gamma,   // [C]
    const bfloat16* beta,    // [C]
    int N, int C, int H, int W, 
    int G, 
    float eps = 1e-5f) 
{
    const int HW = H * W;
    const int Hp = H + 2; 
    const int Wp = W;
    const int channels_per_group = C / G;

    // 1. Zero out the entire output buffer to handle top/bottom padding inherently
    std::memset(output, 0, static_cast<size_t>(N) * C * Hp * Wp * sizeof(bfloat16));

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            
            // -------------------------------------------------------
            // PASS 1: Statistics (Mean & Variance) in Float32
            // -------------------------------------------------------
            __m512 v_sum = _mm512_setzero_ps();
            __m512 v_sq_sum = _mm512_setzero_ps();

            int group_start_c = g * channels_per_group;
            int group_end_c = group_start_c + channels_per_group;

            for (int c = group_start_c; c < group_end_c; ++c) {
                const bfloat16* ptr = input + (n * C * HW) + (c * HW);

                for (int i = 0; i < HW; i += 16) {
                    __mmask16 mask = (i + 16 <= HW) ? 0xFFFF : (1 << (HW - i)) - 1;
                    
                    __m256i bf_vec = _mm256_maskz_loadu_epi16(mask, ptr + i);
                    __m512 val_f32 = _mm512_cvtpbh_ps((__m256bh)bf_vec);
                    
                    v_sum = _mm512_add_ps(v_sum, val_f32);
                    v_sq_sum = _mm512_fmadd_ps(val_f32, val_f32, v_sq_sum);
                }
            }

            float sum = _mm512_reduce_add_ps(v_sum);
            float sq_sum = _mm512_reduce_add_ps(v_sq_sum);

            float count = (float)(channels_per_group * HW);
            float mu = sum / count;
            float var = (sq_sum / count) - (mu * mu);
            float inv_std = 1.0f / std::sqrt(var + eps);

            // -------------------------------------------------------
            // PASS 2: Normalize -> Swish -> Pad -> Store
            // -------------------------------------------------------
            for (int c = group_start_c; c < group_end_c; ++c) {
                
                float gm = bf16_to_float(gamma[c]);
                float bt = bf16_to_float(beta[c]);

                float term_A = inv_std * gm;
                float term_B = bt - (mu * term_A);

                __m512 v_term_A = _mm512_set1_ps(term_A);
                __m512 v_term_B = _mm512_set1_ps(term_B);

                for (int h = 0; h < H; ++h) {
                    // Base indices: read from standard H, write to H+2 (offset by h+1)
                    const size_t in_base  = (static_cast<size_t>(n) * C * H  * W)  + (c * H  * W)  + (h * W);
                    const size_t out_base = (static_cast<size_t>(n) * C * Hp * Wp) + (c * Hp * Wp) + ((h + 1) * Wp);

                    int w = 0;
                    for (; w + 16 <= W; w += 16) {
                        // A. Load BF16 -> Float32
                        __m256i bf_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + in_base + w));
                        __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_raw);

                        // B. GroupNorm Math (Fused)
                        __m512 norm_x = _mm512_fmadd_ps(x, v_term_A, v_term_B);

                        // C. Swish Math
                        __m512 sig = fast_sigmoid_avx512(norm_x);
                        __m512 y = _mm512_mul_ps(norm_x, sig);

                        // D. Convert Float32 -> BF16
                        __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                        // E. Store BF16
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + out_base + w), (__m256i)y_bf16);
                    }

                    // Tail handling (< 16 elements)
                    if (w < W) {
                        const int rem = W - w;
                        __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);

                        __m256i bf_raw = _mm256_maskz_loadu_epi16(mask, input + in_base + w);
                        __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_raw);

                        __m512 norm_x = _mm512_fmadd_ps(x, v_term_A, v_term_B);
                        __m512 sig = fast_sigmoid_avx512(norm_x);
                        __m512 y = _mm512_mul_ps(norm_x, sig);

                        __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                        _mm256_mask_storeu_epi16(output + out_base + w, mask, (__m256i)y_bf16);
                    }
                }
            }
        }
    }
}
void add_avx512_bf16(const bfloat16* input_a, 
    const bfloat16* input_b, 
    bfloat16* output, 
    int size) { // Total number of elements (N*C*H*W)

// Process 16 elements per iteration (256 bits of BF16 data)
for (int i = 0; i < size; i += 16) {

// 1. Calculate Mask (Handle edges where size is not multiple of 16)
__mmask16 mask = (i + 16 <= size) ? 0xFFFF : (1 << (size - i)) - 1;

// 2. Load Inputs (Load 16 bfloat16 values into YMM registers)
__m256i a_raw = _mm256_maskz_loadu_epi16(mask, input_a + i);
__m256i b_raw = _mm256_maskz_loadu_epi16(mask, input_b + i);

// 3. Convert BF16 -> Float32 (YMM -> ZMM)
// We must convert to float32 to perform arithmetic
__m512 a_vec = _mm512_cvtpbh_ps((__m256bh)a_raw);
__m512 b_vec = _mm512_cvtpbh_ps((__m256bh)b_raw);

// 4. Add (Float32)
__m512 sum_vec = _mm512_add_ps(a_vec, b_vec);

// 5. Convert Float32 -> BF16 (ZMM -> YMM)
__m256bh res_bf16 = _mm512_cvtneps_pbh(sum_vec);

// 6. Store Result
_mm256_mask_storeu_epi16(output + i, mask, (__m256i)res_bf16);
}
}


///////////////////////////////////////////////updecoderBlock2d_0 modules ////////////////////////////////////////////////////////

struct ResNetBlockBF16Weights {
    const bfloat16* norm1_weight;
    const bfloat16* norm1_bias;

    const uint16_t* conv1_weight;
    const uint16_t* conv1_bias;

    const bfloat16* norm2_weight;
    const bfloat16* norm2_bias;

    const uint16_t* conv2_weight;
    const uint16_t* conv2_bias;
};


struct DecoderResNetBF16Weights {
    const bfloat16* norm1_weight;
    const bfloat16* norm1_bias;

    const uint16_t* conv1_weight;   // Cin -> Cout
    const uint16_t* conv1_bias;

    const bfloat16* norm2_weight;
    const bfloat16* norm2_bias;

    const uint16_t* conv2_weight;   // Cout -> Cout
    const uint16_t* conv2_bias;

    const uint16_t* skip_weight;    // Cin -> Cout
    const uint16_t* skip_bias;
};



// ---------------------------------------------------------------------
// OPTIMIZED CONVOLUTION KERNEL
// Strategy: Block Output Channels (16) inside Spatial Loops
// ---------------------------------------------------------------------
#include <cstring>
#include <omp.h>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------
// HIGH-PERFORMANCE KERNEL (Weight Packing + Register Blocking)
// ---------------------------------------------------------------------




void nearest_neighbor_512x128x128_avx512(const bfloat16* src, bfloat16* dst, int N, int H, int W) {
    // Dimensions

    
    // Strides (in number of elements)
    const int src_stride_row = W;       // 128
    const int src_stride_plane = H * W; // 16,384
    
    const int dst_stride_row = W * 2;           // 256
    const int dst_stride_plane = (H * 2) * (W * 2); // 65,536

    // --- Precompute Shuffle Indices ---
    // Lower half duplicates: 0,0, 1,1 ... 15,15
    // _mm512_set_epi16 reads in reverse (index 31...0), so we input 15...0
    __m512i idx_lo = _mm512_set_epi16(
        15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
        7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0
    );

    // Upper half duplicates: 16,16 ... 31,31
    __m512i idx_hi = _mm512_set_epi16(
        31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
        23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16
    );

    // Loop over the 512 channels/batch
    for (int n = 0; n < N; ++n) {
        
        const bfloat16* src_plane = src + n * src_stride_plane;
        bfloat16* dst_plane = dst + n * dst_stride_plane;

        for (int y = 0; y < H; ++y) {
            const bfloat16* src_row = src_plane + y * src_stride_row;
            bfloat16* dst_row_1 = dst_plane + (2 * y) * dst_stride_row;
            bfloat16* dst_row_2 = dst_plane + (2 * y + 1) * dst_stride_row;

            // Inner Loop: W = 128. 
            // We process 32 elements per step, so this runs exactly 4 times (128/32).
            // No cleanup loop needed.
            for (int x = 0; x < W; x += 32) {
                
                // 1. Load 32 bfloat16 elements (512 bits)
                __m512i input = _mm512_loadu_si512((const void*)(src_row + x));

                // 2. Permute/Duplicate bits
                __m512i lower = _mm512_permutexvar_epi16(idx_lo, input);
                __m512i upper = _mm512_permutexvar_epi16(idx_hi, input);

                // 3. Store to first output row (Current Y)
                _mm512_storeu_si512((void*)(dst_row_1 + 2 * x), lower);
                _mm512_storeu_si512((void*)(dst_row_1 + 2 * x + 32), upper);

                // 4. Store to second output row (Next Y - Vertical Scale)
                _mm512_storeu_si512((void*)(dst_row_2 + 2 * x), lower);
                _mm512_storeu_si512((void*)(dst_row_2 + 2 * x + 32), upper);
            }
        }
    }
}

///////////////////////////////////////////////updecoderBlock2d_1 modules ////////////////////////////////////////////////////////

static inline __m512 _mm512_exp_ps_custom(__m512 x) {
    // Coefficients for Degree 5 Polynomial Approx
    const __m512 v_log2e    = _mm512_set1_ps(1.442695041f);
    const __m512 v_ln2      = _mm512_set1_ps(0.693147181f);
    const __m512 c0         = _mm512_set1_ps(1.0f);
    const __m512 c1         = _mm512_set1_ps(1.0f);
    const __m512 c2         = _mm512_set1_ps(0.5f);
    const __m512 c3         = _mm512_set1_ps(0.166666667f);
    const __m512 c4         = _mm512_set1_ps(0.041666667f);
    const __m512 c5         = _mm512_set1_ps(0.008333333f);
    
    // Clamping limits to prevent overflow (exp(88) fits in float)
    const __m512 v_hi = _mm512_set1_ps(88.0f);
    const __m512 v_lo = _mm512_set1_ps(-88.0f);
    x = _mm512_max_ps(v_lo, _mm512_min_ps(v_hi, x));

    // 1. Range Reduction: e^x = 2^(x * log2(e))
    // k = round(x * log2(e))
    __m512 v_k_float = _mm512_roundscale_ps(_mm512_mul_ps(x, v_log2e), 
                                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // r = x - k * ln(2)
    __m512 r = _mm512_fnmadd_ps(v_k_float, v_ln2, x);

    // 2. Polynomial Evaluation (Horner's Scheme)
    __m512 poly = _mm512_fmadd_ps(r, c5, c4);
    poly = _mm512_fmadd_ps(r, poly, c3);
    poly = _mm512_fmadd_ps(r, poly, c2);
    poly = _mm512_fmadd_ps(r, poly, c1);
    poly = _mm512_fmadd_ps(r, poly, c0);

    // 3. Reconstruct: result = poly * 2^k
    return _mm512_scalef_ps(poly, v_k_float);
}

static inline __m512 _mm512_silu_ps(__m512 x) {
    const __m512 v_one = _mm512_set1_ps(1.0f);
    const __m512 v_zero = _mm512_setzero_ps();

    // 1. Calculate exp(-x)
    __m512 neg_x = _mm512_sub_ps(v_zero, x);
    __m512 exp_val = _mm512_exp_ps_custom(neg_x);

    // 2. Denominator = 1 + exp(-x)
    __m512 denom = _mm512_add_ps(v_one, exp_val);

    // 3. Sigmoid = 1 / Denominator
    // _mm512_rcp14_ps is faster but less precise. 
    // For inference, it is usually sufficient. 
    // Use _mm512_div_ps(v_one, denom) if you need strict precision.
    __m512 sigmoid = _mm512_rcp14_ps(denom); 

    // 4. Result = x * sigmoid
    return _mm512_mul_ps(x, sigmoid);
}
void silu_bf16_avx512(const uint16_t* input, uint16_t* output, int n) {
    int i = 0;
    
    // Process 16 elements (bfloat16) per loop
    for (; i <= n - 16; i += 16) {
        // 1. Load BF16 and convert to Float32
        // Load 16x 16-bit values (256 bits)
        __m256i bf16_raw = _mm256_loadu_si256((const __m256i*)&input[i]);
        // Expand to 32-bit integers
        __m512i u32_vec = _mm512_cvtepu16_epi32(bf16_raw);
        // Shift Left 16 to move bits to float position
        __m512i shifted = _mm512_slli_epi32(u32_vec, 16);
        // Cast to float32
        __m512 x_f32 = _mm512_castsi512_ps(shifted);

        // 2. Compute SiLU (All math happens in FP32)
        __m512 result_f32 = _mm512_silu_ps(x_f32);

        // 3. Convert back to BF16
        // Cast float to int32
        __m512i res_int = _mm512_castps_si512(result_f32);
        // Shift Right 16 (Truncation - standard for fast BF16)
        __m512i res_shifted = _mm512_srli_epi32(res_int, 16);
        // Compress 32-bit integers down to 16-bit
        __m256i out_bf16 = _mm512_cvtepi32_epi16(res_shifted);

        // 4. Store
        _mm256_storeu_si256((__m256i*)&output[i], out_bf16);
    }

    // Handle remainder (scalar fallback)
    for (; i < n; ++i) {
        // Convert input BF16 to float
        uint32_t in_bits = static_cast<uint32_t>(input[i]) << 16;
        float x; 
        std::memcpy(&x, &in_bits, 4);

        // Math
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        float res = x * sigmoid;

        // Convert result float to BF16
        uint32_t res_bits; 
        std::memcpy(&res_bits, &res, 4);
        output[i] = static_cast<uint16_t>(res_bits >> 16);
    }
}

void swish_avx512_bf16_pad_hw_chw(
    const bfloat16* input,
    bfloat16* output,
    int C, int H, int W, int H_pad, int W_pad)
{
    const int Hp = H_pad;   // 128 -> 130 (pad 1 row top, 1 row bottom)
    const int Wp = W_pad;     // 96 -> 128  (pad 32 columns to the right)

    // Output shape: [C, Hp, Wp]
    // Zero everything first so padded regions (top, bottom, right) are all zeros
    std::memset(output, 0, static_cast<size_t>(C) * Hp * Wp * sizeof(bfloat16));

#pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            const size_t in_base  = (static_cast<size_t>(c) * H  + h) * W;
            
            // FIX: Must multiply by the new stride (Wp), not the old stride (W)
            const size_t out_base = (static_cast<size_t>(c) * Hp + (h+1)) * Wp; 
            // ^ write to row (h+1), leaving row 0 and row H+1 as zero rows

            int w = 0;

            // Vectorized body: 16 BF16 elements at a time
            for (; w + 16 <= W; w += 16) {
                // Load 16 bf16 values (32 bytes)
                __m256i bf_vec_i = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(input + in_base + w));

                // BF16 -> FP32 (Hardware instruction)
                __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_vec_i);

                // sigmoid(x) - assuming fast_sigmoid_avx512 is defined in your header
                __m512 sig = fast_sigmoid_avx512(x);

                // swish = x * sigmoid(x)
                __m512 y = _mm512_mul_ps(x, sig);

                // FP32 -> BF16 (Hardware instruction)
                __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                // Store into output (No padding logic needed here thanks to memset!)
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(output + out_base + w),
                    (__m256i)y_bf16);
            }

            // Tail (< 16 elements)
            if (w < W) {
                const int rem = W - w;
                __mmask16 mask = static_cast<__mmask16>((1u << rem) - 1u);

                __m256i bf_vec_i = _mm256_maskz_loadu_epi16(
                    mask,
                    reinterpret_cast<const void*>(input + in_base + w));

                __m512 x = _mm512_cvtpbh_ps((__m256bh)bf_vec_i);
                __m512 sig = fast_sigmoid_avx512(x);
                __m512 y = _mm512_mul_ps(x, sig);
                __m256bh y_bf16 = _mm512_cvtneps_pbh(y);

                _mm256_mask_storeu_epi16(
                    reinterpret_cast<void*>(output + out_base + w),
                    mask,
                    (__m256i)y_bf16);
            }
        }
    }
}

void groupnorm_unpad_avx512_bf16(const bfloat16* input,   // Input shape:  [N, C, H_pad, W_pad]
    bfloat16* output,        // Output shape: [N, C, H, W] (Tightly packed!)
    const bfloat16* gamma,   // [C]
    const bfloat16* beta,    // [C]
    int N, int C, int H, int W, 
    int H_pad, int W_pad,    // Physical dimensions (e.g., 130, 128)
    int pad_top,             // Rows to skip at the top (e.g., 1)
    int pad_left,            // Cols to skip at the left (e.g., 0)
    int G, 
    float eps = 1e-5f) {

int channels_per_group = C / G;
float count = (float)(channels_per_group * H * W); // Stats only over VALID elements

#pragma omp parallel for collapse(2)
for (int n = 0; n < N; ++n) {
for (int g = 0; g < G; ++g) {

// -------------------------------------------------------
// PASS 1: Statistics (Mean & Variance)
// -------------------------------------------------------
__m512 v_sum = _mm512_setzero_ps();
__m512 v_sq_sum = _mm512_setzero_ps();

int group_start_c = g * channels_per_group;
int group_end_c = group_start_c + channels_per_group;

for (int c = group_start_c; c < group_end_c; ++c) {
// Correctly jump by the PHYSICAL padded layout
size_t in_channel_offset = (static_cast<size_t>(n) * C * H_pad * W_pad) + 
              (static_cast<size_t>(c) * H_pad * W_pad);

const bfloat16* channel_ptr = input + in_channel_offset;

for (int h = 0; h < H; ++h) {
// Skip the top padding rows and left padding columns
const bfloat16* valid_row_ptr = channel_ptr + ((h + pad_top) * W_pad) + pad_left;

for (int w = 0; w < W; w += 16) {
__mmask16 mask = (w + 16 <= W) ? 0xFFFF : (1 << (W - w)) - 1;

__m256i bf_vec = _mm256_maskz_loadu_epi16(mask, valid_row_ptr + w);
__m512 val_f32 = _mm512_cvtpbh_ps((__m256bh)bf_vec);

v_sum = _mm512_add_ps(v_sum, val_f32);
v_sq_sum = _mm512_fmadd_ps(val_f32, val_f32, v_sq_sum);
}
}
}

float sum = _mm512_reduce_add_ps(v_sum);
float sq_sum = _mm512_reduce_add_ps(v_sq_sum);

float mu = sum / count;
float var = (sq_sum / count) - (mu * mu);
float inv_std = 1.0f / std::sqrt(var + eps);

// -------------------------------------------------------
// PASS 2: Normalize & Store (Fused Unpadding)
// -------------------------------------------------------
for (int c = group_start_c; c < group_end_c; ++c) {
float gm = bf16_to_float(gamma[c]);
float bt = bf16_to_float(beta[c]);

float term_A = inv_std * gm;
float term_B = bt - (mu * term_A);

__m512 v_term_A = _mm512_set1_ps(term_A);
__m512 v_term_B = _mm512_set1_ps(term_B);

size_t in_channel_offset = (static_cast<size_t>(n) * C * H_pad * W_pad) + 
              (static_cast<size_t>(c) * H_pad * W_pad);
const bfloat16* in_channel_ptr = input + in_channel_offset;

// Output remains tightly packed (H * W)
size_t out_channel_offset = (static_cast<size_t>(n) * C * H * W) + 
               (static_cast<size_t>(c) * H * W);
bfloat16* out_channel_ptr = output + out_channel_offset;

for (int h = 0; h < H; ++h) {
// Read from padded memory, skipping top/left padding
const bfloat16* in_ptr = in_channel_ptr + ((h + pad_top) * W_pad) + pad_left;

// Write to tightly packed memory
bfloat16* out_ptr = out_channel_ptr + (h * W);

for (int w = 0; w < W; w += 16) {
__mmask16 mask = (w + 16 <= W) ? 0xFFFF : (1 << (W - w)) - 1;

__m256i bf_raw = _mm256_maskz_loadu_epi16(mask, in_ptr + w);
__m512 x = _mm512_cvtpbh_ps((__m256bh)bf_raw);

__m512 y = _mm512_fmadd_ps(x, v_term_A, v_term_B);

__m256bh y_bf16 = _mm512_cvtneps_pbh(y);

_mm256_mask_storeu_epi16(out_ptr + w, mask, (__m256i)y_bf16);
}
}
}
}
}
}


void add_general_avx512_bf16(
    const bfloat16* input1, // [C, H_pad1, W_pad1]
    const bfloat16* input2, // [C, H_pad2, W_pad2]
    bfloat16* output,       // [C, H, W] (Assumes output is unpadded)
    int C, int H, int W, 
    int H_pad1, int W_pad1, // Physical dimensions of input1
    int H_pad2, int W_pad2) // Physical dimensions of input2
{
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            
            // 1. Input 1 pointer: jumps by its specific physical dimensions
            const bfloat16* in1_row = input1 + (c * H_pad1 * W_pad1) + (h * W_pad1);
            
            // 2. Input 2 pointer: jumps by its specific physical dimensions
            const bfloat16* in2_row = input2 + (c * H_pad2 * W_pad2) + (h * W_pad2);
            
            // 3. Output pointer: jumps by the logical (unpadded) dimensions
            bfloat16* out_row = output + (c * H * W) + (h * W);

            for (int w = 0; w < W; w += 16) {
                // Mask to ensure we strictly process up to logical W
                __mmask16 mask = (w + 16 <= W) ? 0xFFFF : (1 << (W - w)) - 1;

                // Load Input 1 -> Convert to FP32
                __m256i a_bf = _mm256_maskz_loadu_epi16(mask, in1_row + w);
                __m512 a_f32 = _mm512_cvtpbh_ps((__m256bh)a_bf);

                // Load Input 2 -> Convert to FP32
                __m256i b_bf = _mm256_maskz_loadu_epi16(mask, in2_row + w);
                __m512 b_f32 = _mm512_cvtpbh_ps((__m256bh)b_bf);

                // Perform the Addition
                __m512 sum_f32 = _mm512_add_ps(a_f32, b_f32);

                // Convert back to BF16
                __m256bh sum_bf16 = _mm512_cvtneps_pbh(sum_f32);

                // Store tightly into the unpadded output
                _mm256_mask_storeu_epi16(out_row + w, mask, (__m256i)sum_bf16);
            }
        }
    }
}
void add_unpad_avx512_bf16(
    const bfloat16* input_padded,   // e.g., x_conv_out_vae [C, H, W_pad]
    const bfloat16* input_unpadded, // e.g., x_conv_in      [C, H, W]
    bfloat16* output_unpadded,      // e.g., x_resnet_same  [C, H, W]
    int C, int H, int W, 
    int W_pad)                      // The physical width of the padded tensor (128)
{
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            
            // 1. Padded input pointer: jumps by W_pad (128)
            const bfloat16* in_pad_row = input_padded + (c * H * W_pad) + (h * W_pad);
            
            // 2. Unpadded input/output pointers: jump by W (96)
            const bfloat16* in_unpad_row = input_unpadded + (c * H * W) + (h * W);
            bfloat16* out_row = output_unpadded + (c * H * W) + (h * W);

            for (int w = 0; w < W; w += 16) {
                // Mask to ensure we strictly process up to W (96) and ignore the last 32 padded elements
                __mmask16 mask = (w + 16 <= W) ? 0xFFFF : (1 << (W - w)) - 1;

                // Load A (Padded Tensor) -> Convert to FP32
                __m256i a_bf = _mm256_maskz_loadu_epi16(mask, in_pad_row + w);
                __m512 a_f32 = _mm512_cvtpbh_ps((__m256bh)a_bf);

                // Load B (Unpadded Tensor) -> Convert to FP32
                __m256i b_bf = _mm256_maskz_loadu_epi16(mask, in_unpad_row + w);
                __m512 b_f32 = _mm512_cvtpbh_ps((__m256bh)b_bf);

                // Perform the Addition
                __m512 sum_f32 = _mm512_add_ps(a_f32, b_f32);

                // Convert back to BF16
                __m256bh sum_bf16 = _mm512_cvtneps_pbh(sum_f32);

                // Store tightly into the unpadded output
                _mm256_mask_storeu_epi16(out_row + w, mask, (__m256i)sum_bf16);
            }
        }
    }
}

void add_pad_unpad_avx512_bf16(
    const bfloat16* input_padded_1, // e.g., x_conv_out_vae [C, H, W_pad]
    const bfloat16* input_padded_2, // e.g., x_conv_in      [C, H, W_pad]
    bfloat16* output_unpadded,      // e.g., x_resnet_same  [C, H, W]
    int C, int H, int W, 
    int W_pad)                      // The physical width of the padded tensors (128)
{
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            
            // 1. Padded input 1 pointer: jumps by W_pad
            const bfloat16* in_pad_row_1 = input_padded_1 + (c * H * W_pad) + (h * W_pad);
            
            // 2. Padded input 2 pointer: ALSO jumps by W_pad
            const bfloat16* in_pad_row_2 = input_padded_2 + (c * H * W_pad) + (h * W_pad);
            
            // 3. Unpadded output pointer: jumps by W (packs tightly)
            bfloat16* out_row = output_unpadded + (c * H * W) + (h * W);

            for (int w = 0; w < W; w += 16) {
                // Mask to ensure we strictly process up to W (96) and ignore the padding
                __mmask16 mask = (w + 16 <= W) ? 0xFFFF : (1 << (W - w)) - 1;

                // Load A (Padded Tensor 1) -> Convert to FP32
                __m256i a_bf = _mm256_maskz_loadu_epi16(mask, in_pad_row_1 + w);
                __m512 a_f32 = _mm512_cvtpbh_ps((__m256bh)a_bf);

                // Load B (Padded Tensor 2) -> Convert to FP32
                __m256i b_bf = _mm256_maskz_loadu_epi16(mask, in_pad_row_2 + w);
                __m512 b_f32 = _mm512_cvtpbh_ps((__m256bh)b_bf);

                // Perform the Addition
                __m512 sum_f32 = _mm512_add_ps(a_f32, b_f32);

                // Convert back to BF16
                __m256bh sum_bf16 = _mm512_cvtneps_pbh(sum_f32);

                // Store tightly into the unpadded output
                _mm256_mask_storeu_epi16(out_row + w, mask, (__m256i)sum_bf16);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void resnet_block_bf16_128x128_npu(
    const bfloat16* x_in,
    bfloat16* x_out,
    const ResNetBlockBF16Weights& w,
    npu_app& app_vae,          // Pass the NPU app
    buffer<dtype_in>& x_conv_in_vae,  // Pass the input buffer
    buffer<dtype_in>& x_conv_kernel_vae, // Pass the weight buffer
    buffer<dtype_out>& x_conv_out_vae,    // Pass the output buffer
    int H,
    int W,
    int C,
    int H_pad,
    int W_pad,
    int model_size
) {
    const int N = H * W * C;

    buffer<dtype_in> tmp1(N);
    buffer<dtype_in> tmp2(N);

    // GN1

    groupnorm_avx512_bf16(x_in,
        (bfloat16*)tmp1.data(),
    w.norm1_weight, w.norm1_bias,
     1, C, H, W, 32, 1e-6);
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp1.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae.data()), C, H, W, H_pad, W_pad);

    for(int i = 0; i < C*C; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae[i*16 + j] = std::bit_cast<std::bfloat16_t>(w.conv1_weight[i*9 + j]);
        }
    }
    for (int n = 0; n < C; n++) {
        int idx = ((n * C) + (C-1)) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae[idx] = std::bit_cast<std::bfloat16_t>(w.conv1_bias[n]);        // example value
    }
    x_conv_in_vae.sync_to_device();
    x_conv_kernel_vae.sync_to_device();
    app_vae(x_conv_in_vae, x_conv_kernel_vae, x_conv_out_vae);
    x_conv_out_vae.sync_from_device();

    // GN2

    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_out_vae.data()),
    reinterpret_cast<bfloat16*>(tmp2.data()), 
    w.norm2_weight, w.norm2_bias,
     1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp2.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae.data()), C, H, W, H_pad, W_pad);

    for(int i = 0; i < C*C; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae[i*16 + j] = std::bit_cast<std::bfloat16_t>(w.conv2_weight[i*9 + j]);
        }
    }
    for (int n = 0; n < C; n++) {
        int idx = ((n * C) + (C-1)) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae[idx] = std::bit_cast<std::bfloat16_t>(w.conv2_bias[n]);        // example value
    }
    x_conv_in_vae.sync_to_device();
    x_conv_kernel_vae.sync_to_device();
    app_vae(x_conv_in_vae, x_conv_kernel_vae, x_conv_out_vae);
    x_conv_out_vae.sync_from_device();

    
    add_general_avx512_bf16(
        reinterpret_cast<const bfloat16*>(x_conv_out_vae.data()),
        x_in, x_out,
        C, H, W,
        model_size, model_size,
        H, W
    );

}





void resnet_block_bf16_256x256_npu(
    const bfloat16* x_in,
    bfloat16* x_out,
    const ResNetBlockBF16Weights& w,
    npu_app& app_vae_256x512,          // Pass the NPU app
    buffer<dtype_in>& x_conv_in_vae_256x512,  // Pass the input buffer
    buffer<dtype_in>& x_conv_kernel_vae_256x512, // Pass the weight buffer
    buffer<dtype_out>& x_conv_out_vae_256x512,    // Pass the output buffer
    int H,
    int W,
    int C,
    int H_pad,
    int W_pad,
    int model_size
) {
    const int N = H * W * C;

    buffer<dtype_in> tmp1(N);
    buffer<dtype_in> tmp2(N);
    buffer<dtype_in> x_in_temp(H_pad*W_pad*C);

    for(int i = 0; i < H_pad*W_pad*C; i++){
        x_in_temp[i] = std::bit_cast<std::bfloat16_t>(x_in[i]);
    }

    groupnorm_unpad_avx512_bf16(x_in,
        reinterpret_cast<bfloat16*>(tmp1.data()), 
        w.norm1_weight, w.norm1_bias,
        1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp1.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae_256x512.data()), C, H, W, H_pad, W_pad);
    
    for(int i = 0; i < C*C; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae_256x512[i*16 + j] = std::bit_cast<std::bfloat16_t>(w.conv1_weight[i*9 + j]);
        }
    }
    for (int n = 0; n < C; n++) {
        int idx = ((n * C) + (C-1)) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae_256x512[idx] = std::bit_cast<std::bfloat16_t>(w.conv1_bias[n]);        // example value
    }
    x_conv_in_vae_256x512.sync_to_device();
    x_conv_kernel_vae_256x512.sync_to_device();
    app_vae_256x512(x_conv_in_vae_256x512, x_conv_kernel_vae_256x512, x_conv_out_vae_256x512);
    x_conv_out_vae_256x512.sync_from_device();

    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_out_vae_256x512.data()),
    reinterpret_cast<bfloat16*>(tmp2.data()), 
    w.norm2_weight, w.norm2_bias,
     1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp2.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae_256x512.data()), C, H, W, H_pad, W_pad);

    for(int i = 0; i < C*C; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae_256x512[i*16 + j] = std::bit_cast<std::bfloat16_t>(w.conv2_weight[i*9 + j]);
        }
    }
    for (int n = 0; n < C; n++) {
        int idx = ((n * C) + (C-1)) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae_256x512[idx] = std::bit_cast<std::bfloat16_t>(w.conv2_bias[n]);        // example value
    }
    x_conv_in_vae_256x512.sync_to_device();
    x_conv_kernel_vae_256x512.sync_to_device();
    app_vae_256x512(x_conv_in_vae_256x512, x_conv_kernel_vae_256x512, x_conv_out_vae_256x512);
    x_conv_out_vae_256x512.sync_from_device();
   

    add_general_avx512_bf16(
        reinterpret_cast<const bfloat16*>(x_conv_out_vae_256x512.data()),
        reinterpret_cast<const bfloat16*>(x_in_temp.data()),
        x_out,
        C, H, W,
        model_size, model_size,
        model_size, model_size
    );
    
}

void decoder_resnet_block_bf16_512_256_npu(
    const bfloat16* x_in,    // [1, Cin, H, W]
    bfloat16* x_out,         // [1, Cout, H, W]
    const DecoderResNetBF16Weights& w,
    int H,
    int W,
    int Cin,
    int Cout,
    npu_app& app_conv_mm_same,
    buffer<dtype_in>& conv_mm_in_same,
    buffer<dtype_in>& conv_mm_weight_same,
    buffer<dtype_out>& conv_mm_out_same,
    npu_app& app_conv_mm_diff,
    buffer<dtype_in>& conv_mm_in_diff,
    buffer<dtype_in>& conv_mm_weight_diff,
    buffer<dtype_out>& conv_mm_out_diff,
    npu_app& app_skip,
    buffer<dtype_in>& skip_in,
    buffer<dtype_in>& skip_weight,
    buffer<dtype_in>& skip_out,
    int H_pad,
    int W_pad,
    int model_size
) {
    const int N1 = Cin  * H * W;
    const int N2 = Cout * H * W;

    buffer<dtype_in> tmp1(N1);   // GN1
    buffer<dtype_in> tmp2(N2);   // Swish1
   
 

    groupnorm_unpad_avx512_bf16(x_in,
        reinterpret_cast<bfloat16*>(tmp1.data()), 
        w.norm1_weight, w.norm1_bias,
        1, Cin, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp1.data(),
    reinterpret_cast<bfloat16*>(conv_mm_in_diff.data()),
    Cin, H, W, H_pad, W_pad);
      
    for (int i = 0; i < Cout*512; i++){
        for (int j = 0; j < 9; j++){
            conv_mm_weight_diff[i*16 + j] = std::bit_cast<std::bfloat16_t>(w.conv1_weight[i*9 + j]);
        }
        
    }
    for (int n = 0; n < Cout; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        conv_mm_weight_diff[idx] = std::bit_cast<std::bfloat16_t>(w.conv1_bias[n]);        // example value
    }
   
    conv_mm_in_diff.sync_to_device();
    conv_mm_weight_diff.sync_to_device();
    app_conv_mm_diff(conv_mm_in_diff, conv_mm_weight_diff, conv_mm_out_diff);
    conv_mm_out_diff.sync_from_device();
 
    // GN2
    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(conv_mm_out_diff.data()),
    reinterpret_cast<bfloat16*>(tmp2.data()), 
    w.norm2_weight, w.norm2_bias,
     1, Cout, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp2.data(),
    reinterpret_cast<bfloat16*>(conv_mm_in_same.data()), Cout, H, W, H_pad, W_pad);

    for (int n = 0; n < 256; ++n) {
        for (int oc = 0; oc < 256; ++oc) {
            int base_src = ((n * 256) + oc) * 9;
            int base_dst = ((n * 512) + oc) * 16;
    
            // copy 9 valid values
            for (int k = 0; k < 9; ++k) {
                conv_mm_weight_same[base_dst + k] = std::bit_cast<std::bfloat16_t>(w.conv2_weight[base_src + k]);
            }
        }
    }
    for (int n = 0; n < 256; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        conv_mm_weight_same[idx] = std::bit_cast<std::bfloat16_t>(w.conv2_bias[n]);        // example value
    }
    conv_mm_in_same.sync_to_device();
    conv_mm_weight_same.sync_to_device();
    app_conv_mm_same(conv_mm_in_same, conv_mm_weight_same, conv_mm_out_same);
    conv_mm_out_same.sync_from_device();
   
    // Skip projection: Cin -> Cout
    time_utils::time_point one_conv2_bf16_2 = time_utils::now();
    for (int c = 0; c < 512; ++c)
    for (int h = 0; h < 512; ++h) {
        size_t in_base  = ((size_t)c * 512 + h) * 512;
        for (int w = 0; w < 512; ++w) {
        skip_in[in_base + w] = std::bit_cast<std::bfloat16_t>(x_in[in_base + w]); // if cast supporte
       
        }
    }
    for (int i = 0; i < 512*256; i++){
        skip_weight[i*8] = std::bit_cast<std::bfloat16_t>(w.skip_weight[i]);
    }
    skip_in.sync_to_device();
    skip_weight.sync_to_device();
    app_skip(skip_in, skip_weight, skip_out);
    skip_out.sync_from_device();
    
    // Residual add
    add_avx512_bf16(
        reinterpret_cast<const bfloat16*>(conv_mm_out_same.data()),
        reinterpret_cast<const bfloat16*>(skip_out.data()),
        x_out,
        Cout*model_size*model_size);
}


void resnet_block_512_256_all_npu(
    const bfloat16* x_in,
    bfloat16* x_out,
    const ResNetBlockBF16Weights& w,
    npu_app& app_vae_512x512_256_all,          // Pass the NPU app
    buffer<dtype_in>& x_conv_in_vae_512x512_256_all,  // Pass the input buffer
    buffer<dtype_in>& x_conv_kernel_vae_512x512_256_all, // Pass the weight buffer
    buffer<dtype_out>& x_conv_out_vae_512x512_256_all,    // Pass the output buffer
    int H,
    int W,
    int C,
    int H_pad,
    int W_pad,
    int model_size
) {
    const int N = H * W * C;

    buffer<dtype_in> tmp1(N);
    buffer<dtype_in> tmp2(N);
    time_utils::time_point one_512_cnn_bf16_all_start = time_utils::now();
    // GN1
 
    groupnorm_unpad_avx512_bf16(x_in,
        reinterpret_cast<bfloat16*>(tmp1.data()), 
        w.norm1_weight, w.norm1_bias,
        1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp1.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae_512x512_256_all.data()), C, H, W, H_pad, W_pad);
    

    for (int n = 0; n < 256; ++n) {
        for (int oc = 0; oc < 256; ++oc) {
            int base_src = ((n * 256) + oc) * 9;
            int base_dst = ((n * 512) + oc) * 16;
    
            // copy 9 valid values
            for (int k = 0; k < 9; ++k) {
                x_conv_kernel_vae_512x512_256_all[base_dst + k] = std::bit_cast<std::bfloat16_t>(w.conv1_weight[base_src + k]);
            }
        }
    }
    for (int n = 0; n < 256; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae_512x512_256_all[idx] = std::bit_cast<std::bfloat16_t>(w.conv1_bias[n]);        // example value
    }
    x_conv_in_vae_512x512_256_all.sync_to_device();
    x_conv_kernel_vae_512x512_256_all.sync_to_device();
    app_vae_512x512_256_all(x_conv_in_vae_512x512_256_all, x_conv_kernel_vae_512x512_256_all, x_conv_out_vae_512x512_256_all);
    x_conv_out_vae_512x512_256_all.sync_from_device();
      
    // GN2

    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_out_vae_512x512_256_all.data()),
    reinterpret_cast<bfloat16*>(tmp2.data()), 
    w.norm2_weight, w.norm2_bias,
     1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp2.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae_512x512_256_all.data()), C, H, W, H_pad, W_pad);

   
    for (int n = 0; n < 256; ++n) {
        for (int oc = 0; oc < 256; ++oc) {
            int base_src = ((n * 256) + oc) * 9;
            int base_dst = ((n * 512) + oc) * 16;
    
            // copy 9 valid values
            for (int k = 0; k < 9; ++k) {
                x_conv_kernel_vae_512x512_256_all[base_dst + k] = std::bit_cast<std::bfloat16_t>(w.conv2_weight[base_src + k]);
            }
        }
    }
    for (int n = 0; n < 256; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae_512x512_256_all[idx] = std::bit_cast<std::bfloat16_t>(w.conv2_bias[n]);        // example value
    }

    x_conv_in_vae_512x512_256_all.sync_to_device();
    x_conv_kernel_vae_512x512_256_all.sync_to_device();
    app_vae_512x512_256_all(x_conv_in_vae_512x512_256_all, x_conv_kernel_vae_512x512_256_all, x_conv_out_vae_512x512_256_all);
    x_conv_out_vae_512x512_256_all.sync_from_device();

    // Residual
    add_avx512_bf16(reinterpret_cast<const bfloat16*>(
        x_conv_out_vae_512x512_256_all.data()), x_in, x_out,  C*model_size*model_size);
   }

void decoder_resnet_block_bf16_1024_128_npu(
    const bfloat16* x_in,    // [1, Cin, H, W]
    bfloat16* x_out,         // [1, Cout, H, W]
    const DecoderResNetBF16Weights& w,
    int H,
    int W,
    int Cin,
    int Cout,
    npu_app& app_conv_mm_same,
    buffer<dtype_in>& conv_mm_in_same,
    buffer<dtype_in>& conv_mm_weight_same,
    buffer<dtype_out>& conv_mm_out_same,
    npu_app& app_conv_mm_diff,
    buffer<dtype_in>& conv_mm_in_diff,
    buffer<dtype_in>& conv_mm_weight_diff,
    buffer<dtype_out>& conv_mm_out_diff,
    npu_app& app_skip,
    buffer<dtype_in>& skip_in,
    buffer<dtype_in>& skip_weight,
    buffer<dtype_in>& skip_out,
    int H_pad,
    int W_pad,
    int model_size
) {
    const int N1 = Cin  * H * W;
    const int N2 = Cout * H * W;

    buffer<dtype_in> tmp1(N1);   // GN1
    buffer<dtype_in> tmp2(N2);   // Swish1
   
    
    time_utils::time_point one_gn1_bf16 = time_utils::now();
    // GN1

    groupnorm_unpad_avx512_bf16(x_in,
        reinterpret_cast<bfloat16*>(tmp1.data()), 
        w.norm1_weight, w.norm1_bias,
        1, Cin, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp1.data(),
    reinterpret_cast<bfloat16*>(conv_mm_in_diff.data()),
    Cin, H, W, H_pad, W_pad);
  
    for (int i = 0; i < Cout*256; i++){
        for (int j = 0; j < 9; j++){
            conv_mm_weight_diff[i*16 + j] = std::bit_cast<std::bfloat16_t>(w.conv1_weight[i*9 + j]);
        }
        
    }
    for (int n = 0; n < Cout; n++) {
        int idx = ((n * 256) + 255) * 16 + 9;   // [n][255][9]
        conv_mm_weight_diff[idx] = std::bit_cast<std::bfloat16_t>(w.conv1_bias[n]);        // example value
    }
    conv_mm_in_diff.sync_to_device();
    conv_mm_weight_diff.sync_to_device();
    app_conv_mm_diff(conv_mm_in_diff, conv_mm_weight_diff, conv_mm_out_diff);
    conv_mm_out_diff.sync_from_device();
    
    // GN2

    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(conv_mm_out_diff.data()),
    reinterpret_cast<bfloat16*>(tmp2.data()), 
    w.norm2_weight, w.norm2_bias,
     1, Cout, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp2.data(),
    reinterpret_cast<bfloat16*>(conv_mm_in_same.data()),
     Cout, H, W, H_pad, W_pad);

    for (int n = 0; n < 128; ++n) {
        for (int oc = 0; oc < 128; ++oc) {
            int base_src = ((n * 128) + oc) * 9;
            int base_dst = ((n * 256) + oc) * 16;
    
            // copy 9 valid values
            for (int k = 0; k < 9; ++k) {
                conv_mm_weight_same[base_dst + k] = std::bit_cast<std::bfloat16_t>(w.conv2_weight[base_src + k]);
            }
        }
    }
    for (int n = 0; n < 128; n++) {
        int idx = ((n * 256) + 255) * 16 + 9;   // [n][255][9]
        conv_mm_weight_same[idx] = std::bit_cast<std::bfloat16_t>(w.conv2_bias[n]);        // example value
    }
    conv_mm_in_same.sync_to_device();
    conv_mm_weight_same.sync_to_device();
    app_conv_mm_same(conv_mm_in_same, conv_mm_weight_same, conv_mm_out_same);
    conv_mm_out_same.sync_from_device();
    // Skip projection: Cin -> Cout
    
    for (int c = 0; c < 256; ++c)
    for (int h = 0; h < 1024; ++h) {
        size_t in_base  = ((size_t)c * 1024 + h) * 1024;
        for (int w = 0; w < 1024; ++w) {
        skip_in[in_base + w] = std::bit_cast<std::bfloat16_t>(x_in[in_base + w]); // if cast supporte
       
        }
    }
    for (int i = 0; i < 128*256; i++){
        skip_weight[i*8] = std::bit_cast<std::bfloat16_t>(w.skip_weight[i]);
    }
    skip_in.sync_to_device();
    skip_weight.sync_to_device();
    app_skip(skip_in, skip_weight, skip_out);
    skip_out.sync_from_device();
        // Residual add
    add_avx512_bf16(
        reinterpret_cast<const bfloat16*>(conv_mm_out_same.data()),
        // reinterpret_cast<const bfloat16*>(x_out),
        reinterpret_cast<const bfloat16*>(skip_out.data()),
        x_out,
        Cout*model_size*model_size);
}
void nearest_neighbor_pad_avx512_bf16(const bfloat16* src, bfloat16* dst, 
    int C, int latent_h, int latent_w, int H_out_pad, int W_out_pad) 
{
// Fixed output dimensions


// 1. Zero out the entire [C, 256, 256] destination buffer first!
// This perfectly handles BOTH right-side and bottom-side padding automatically.
std::memset(dst, 0, static_cast<size_t>(C) * H_out_pad * W_out_pad * sizeof(bfloat16));

// --- Precompute Shuffle Indices ---
__m512i idx_lo = _mm512_set_epi16(
15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0
);

__m512i idx_hi = _mm512_set_epi16(
31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16
);

#pragma omp parallel for collapse(2)
for (int c = 0; c < C; ++c) {
for (int y = 0; y < latent_h; ++y) {

// Safety bound: Stop if the 2x scaled rows exceed our 256 limit
if ((2 * y) >= H_out_pad) continue; 

// Input physical stride is tightly packed [C, latent_h, latent_w]
const bfloat16* src_row = src + (c * latent_h * latent_w) + (y * latent_w);

// Output physical stride skips the padding [C, 256, 256]
bfloat16* dst_row_1 = dst + (c * H_out_pad * W_out_pad) + ((2 * y) * W_out_pad);
bfloat16* dst_row_2 = dst + (c * H_out_pad * W_out_pad) + ((2 * y + 1) * W_out_pad);

int x = 0;
// Inner Loop: Processes 32 input elements -> 64 output elements per step.
for (; x <= latent_w - 32; x += 32) {

__m512i input = _mm512_loadu_si512((const void*)(src_row + x));

__m512i lower = _mm512_permutexvar_epi16(idx_lo, input);
__m512i upper = _mm512_permutexvar_epi16(idx_hi, input);

// Write to top row of the 2x scale
_mm512_storeu_si512((void*)(dst_row_1 + 2 * x), lower);
_mm512_storeu_si512((void*)(dst_row_1 + 2 * x + 32), upper);

// Write to bottom row of the 2x scale
if ((2 * y + 1) < H_out_pad) { // Boundary safety for the bottom row
_mm512_storeu_si512((void*)(dst_row_2 + 2 * x), lower);
_mm512_storeu_si512((void*)(dst_row_2 + 2 * x + 32), upper);
}
}

// Scalar Remainder Loop: Processes leftover elements if latent_w is not a multiple of 32
for (; x < latent_w; ++x) {
if ((2 * x + 1) < W_out_pad) {
dst_row_1[2 * x]     = src_row[x];
dst_row_1[2 * x + 1] = src_row[x];

if ((2 * y + 1) < H_out_pad) {
dst_row_2[2 * x]     = src_row[x];
dst_row_2[2 * x + 1] = src_row[x];
}
}
}
}
}
}

void resnet_block_1024_128_all_npu(
    const bfloat16* x_in,
    bfloat16* x_out,
    const ResNetBlockBF16Weights& w,
    npu_app& app_vae_1024x1024_128_all,          // Pass the NPU app
    buffer<dtype_in>& x_conv_in_vae_1024x1024_128_all,  // Pass the input buffer
    buffer<dtype_in>& x_conv_kernel_vae_1024x1024_128_all, // Pass the weight buffer
    buffer<dtype_out>& x_conv_out_vae_1024x1024_128_all,    // Pass the output buffer
    int H,
    int W,
    int C,
    int H_pad,
    int W_pad,
    int model_size
) {
    const int N = H * W * C;

    buffer<dtype_in> tmp1(N);
    buffer<dtype_in> tmp2(N);
    time_utils::time_point one_1024_cnn_bf16_all_start = time_utils::now();
    // GN1
  
    groupnorm_unpad_avx512_bf16(x_in,
        reinterpret_cast<bfloat16*>(tmp1.data()), 
        w.norm1_weight, w.norm1_bias,
        1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp1.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae_1024x1024_128_all.data()), C, H, W, H_pad, W_pad);
    
   
    for (int n = 0; n < 128; ++n) {
        for (int oc = 0; oc < 128; ++oc) {
            int base_src = ((n * 128) + oc) * 9;
            int base_dst = ((n * 256) + oc) * 16;
    
            // copy 9 valid values
            for (int k = 0; k < 9; ++k) {
                x_conv_kernel_vae_1024x1024_128_all[base_dst + k] = std::bit_cast<std::bfloat16_t>(w.conv1_weight[base_src + k]);
            }
        }
    }
    for (int n = 0; n < 128; n++) {
        int idx = ((n * 256) + 255) * 16 + 9;   // [n][255][9]
        x_conv_kernel_vae_1024x1024_128_all[idx] = std::bit_cast<std::bfloat16_t>(w.conv1_bias[n]);        // example value
    }
    
    x_conv_in_vae_1024x1024_128_all.sync_to_device();
    x_conv_kernel_vae_1024x1024_128_all.sync_to_device();
    app_vae_1024x1024_128_all(x_conv_in_vae_1024x1024_128_all, x_conv_kernel_vae_1024x1024_128_all, x_conv_out_vae_1024x1024_128_all);
    x_conv_out_vae_1024x1024_128_all.sync_from_device();
    
    
    // GN2
 
    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_out_vae_1024x1024_128_all.data()),
    reinterpret_cast<bfloat16*>(tmp2.data()), 
    w.norm2_weight, w.norm2_bias,
     1, C, H, W, model_size, model_size, 0, 0, 32, 1e-6);
    
    swish_avx512_bf16_pad_hw_chw((bfloat16*)tmp2.data(),
    reinterpret_cast<bfloat16*>(x_conv_in_vae_1024x1024_128_all.data()), C, H, W, H_pad, W_pad);

    for (int n = 0; n < 128; ++n) {
        for (int oc = 0; oc < 128; ++oc) {
            int base_src = ((n * 128) + oc) * 9;
            int base_dst = ((n * 256) + oc) * 16;
    
            // copy 9 valid values
            for (int k = 0; k < 9; ++k) {
                x_conv_kernel_vae_1024x1024_128_all[base_dst + k] = std::bit_cast<std::bfloat16_t>(w.conv2_weight[base_src + k]);
            }
        }
    }
    for (int n = 0; n < 128; n++) {
        int idx = ((n * 256) + 255) * 16 + 9;   // [n][255][9]
        x_conv_kernel_vae_1024x1024_128_all[idx] = std::bit_cast<std::bfloat16_t>(w.conv2_bias[n]);        // example value
    }
   
    x_conv_in_vae_1024x1024_128_all.sync_to_device();
    x_conv_kernel_vae_1024x1024_128_all.sync_to_device();
    app_vae_1024x1024_128_all(x_conv_in_vae_1024x1024_128_all, x_conv_kernel_vae_1024x1024_128_all, x_conv_out_vae_1024x1024_128_all);
    x_conv_out_vae_1024x1024_128_all.sync_from_device();
   
    // Residual
    add_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_out_vae_1024x1024_128_all.data()),
     x_in, x_out, C*model_size*model_size);
    time_utils::time_point end_1024_cnn_bf16_all_add = time_utils::now();
   
}


#endif