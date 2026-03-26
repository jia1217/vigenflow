#ifndef QWEN_LIB_HPP
#define QWEN_LIB_HPP


#include <iostream>
#include "tokenizers_cpp.h"
#include <immintrin.h>







std::string apply_chat_template(const std::string& prompt) {
    return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> tokenize_text_cpu(
    const std::string& prompt, 
    const std::string& tokenizer_json_path
) {
    const int max_len = 512;
    const int32_t pad_id = 151643;

    // 1. Load Tokenizer
    std::ifstream f(tokenizer_json_path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Could not open tokenizer file at: " << tokenizer_json_path << std::endl;
        
        // Fallback: Return padded vectors filled with pad_id and 0s for the mask
        return {
            std::vector<int32_t>(max_len, pad_id), 
            std::vector<int32_t>(max_len, 0)
        };
    }
    
    std::string json_content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_content);

    // 2. Encode
    std::string formatted_text = apply_chat_template(prompt);
    std::vector<int32_t> raw_ids = tokenizer->Encode(formatted_text);

    // 3. Configuration 
    // FIX: We must ALLOW special tokens (151644, 151645, etc.)
    // Your tokenizer says vocab size is 151669. Let's use that as the limit.
    const int64_t VOCAB_REAL_LIMIT = 151669; 

    std::vector<int32_t> final_ids;
    std::vector<int32_t> final_mask;
    final_ids.reserve(max_len);
    final_mask.reserve(max_len);

    // 4. Processing
    int count = 0;
    for (int32_t id : raw_ids) {
        if (count >= max_len) break;

        // Check for garbage, but ALLOW valid special tokens
        if (id < 0 || id >= VOCAB_REAL_LIMIT) {
             final_ids.push_back(pad_id); // Only clamp if truly garbage
        } else {
             final_ids.push_back(id);     // Keep 151644/151645 as is!
        }

        final_mask.push_back(1); 
        count++;
    }

    // 5. Padding
    while (final_ids.size() < max_len) {
        final_ids.push_back(pad_id);
        final_mask.push_back(0); 
    }

    // 6. Return Standard C++ Vectors
    return {final_ids, final_mask};
}

void embed_tokens_forward(
    const int* token_ids, size_t num_tokens, const bfloat16_t* weight_table, 
    size_t vocab_size, size_t hidden_size, int padding_idx, 
    bfloat16_t* output, int is_transposed
) {
    for (size_t i = 0; i < num_tokens; i++) {
        int token_id = token_ids[i];
        bfloat16_t* out_row = output + (i * hidden_size);

        if (token_id == padding_idx) {
            memset(out_row, 0, hidden_size * sizeof(bfloat16_t));
        } else if (token_id >= 0 && (size_t)token_id < vocab_size) {
            if (is_transposed) {
                // Read column-major (strided)
                for (size_t j = 0; j < hidden_size; j++) {
                    out_row[j] = weight_table[j * vocab_size + token_id];
                }
            } else {
                // Read row-major (contiguous)
                const bfloat16_t* weight_row = weight_table + (token_id * hidden_size);
                memcpy(out_row, weight_row, hidden_size * sizeof(bfloat16_t));
            }
        } else {
            memset(out_row, 0, hidden_size * sizeof(bfloat16_t));
        }
    }
}

void compute_rope_embeddings_bf16(
    const int* position_ids,
    size_t batch_size,
    size_t seq_len,
    size_t head_dim,
    float rope_theta,
    float attention_scaling,
    bfloat16_t* cos_out,
    bfloat16_t* sin_out
) {
    size_t half_dim = head_dim / 2;

    // 1. Precompute inverse frequencies in standard float32
    float* inv_freq = (float*)malloc(half_dim * sizeof(float));
    for (size_t i = 0; i < half_dim; i++) {
        inv_freq[i] = 1.0f / powf(rope_theta, (float)(i * 2) / head_dim);
    }

    // 2. Compute freqs, cos, and sin for each position dynamically
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            
            int pos = position_ids[b * seq_len + s];

            // Pointer to the start of the current token's output row
            bfloat16_t* current_cos = cos_out + ((b * seq_len + s) * head_dim);
            bfloat16_t* current_sin = sin_out + ((b * seq_len + s) * head_dim);

            for (size_t i = 0; i < half_dim; i++) {
                // Math is done in float32
                float freq = (float)pos * inv_freq[i];
                float c = (cosf(freq) * attention_scaling);
                float s_val = (sinf(freq) * attention_scaling);

                // Convert to bfloat16
                bfloat16_t c_bf16 = float_to_bfloat16(c);
                bfloat16_t s_bf16 = float_to_bfloat16(s_val);

                // torch.cat((freqs, freqs), dim=-1)
                // Assign identical values to the first and second halves of the head_dim
                current_cos[i]            = c_bf16;
                current_cos[i + half_dim] = c_bf16;
                
                current_sin[i]            = s_bf16;
                current_sin[i + half_dim] = s_bf16;
            }
        }
    }

    free(inv_freq);
}

static inline float reduce_add_ps(__m512 a) {
    return _mm512_reduce_add_ps(a);
}

/**
 * Computes Qwen3 RMSNorm matching PyTorch's RTNE bfloat16 casting using AVX-512.
 * @param input       Input tensor [num_tokens, hidden_size]
 * @param weight      RMSNorm weight tensor [hidden_size]
 * @param output      Output tensor [num_tokens, hidden_size]
 * @param num_tokens  batch_size * seq_len (e.g., 512)
 * @param hidden_size e.g., 2560 (Must be a multiple of 16)
 * @param eps         Variance epsilon (e.g., 1e-6)
 */
void qwen3_rmsnorm_avx512(
    const uint16_t* input, 
    const uint16_t* weight, 
    uint16_t* output, 
    size_t num_tokens, 
    size_t hidden_size, 
    float eps
) {
    for (size_t t = 0; t < num_tokens; t++) {
        const bfloat16_t* in_row = input + (t * hidden_size);
        bfloat16_t* out_row = output + (t * hidden_size);

        // -------------------------------------------------------------
        // PASS 1: Compute Variance (Sum of Squares)
        // -------------------------------------------------------------
        __m512 v_sum_sq = _mm512_setzero_ps();

        for (size_t i = 0; i < hidden_size; i += 16) {
            // Load 16x bfloat16
            __m256i v_in_bf16 = _mm256_loadu_si256((const __m256i*)&in_row[i]);
            // Zero-extend to 32-bit, then shift left by 16 to become valid float32
            __m512i v_in_32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(v_in_bf16), 16);
            __m512 v_in_f32 = _mm512_castsi512_ps(v_in_32);

            // Accumulate sum of squares (sum_sq += x * x)
            v_sum_sq = _mm512_fmadd_ps(v_in_f32, v_in_f32, v_sum_sq);
        }

        // Calculate final scalar inverse RMS
        float sum_sq = reduce_add_ps(v_sum_sq);
        float variance = sum_sq / hidden_size;
        float inv_rms = 1.0f / sqrtf(variance + eps); // Exact math matches PyTorch torch.rsqrt

        __m512 v_inv_rms = _mm512_set1_ps(inv_rms);

        // -------------------------------------------------------------
        // PASS 2: Normalize, Multiply Weight, and Store
        // -------------------------------------------------------------
        for (size_t i = 0; i < hidden_size; i += 16) {
            // Re-load input float32
            __m256i v_in_bf16 = _mm256_loadu_si256((const __m256i*)&in_row[i]);
            __m512 v_in_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_in_bf16), 16));

            // Load weight float32 (Assuming weight is also exported as bfloat16)
            __m256i v_w_bf16 = _mm256_loadu_si256((const __m256i*)&weight[i]);
            __m512 v_w_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_w_bf16), 16));

            // Normalize and scale: out = (in * inv_rms) * weight
            __m512 v_out_f32 = _mm512_mul_ps(v_in_f32, v_inv_rms);
            v_out_f32 = _mm512_mul_ps(v_out_f32, v_w_f32);

            // ---------------------------------------------------------
            // Convert back to bfloat16 matching PyTorch's RTNE Logic
            // f32_bits += 0x7FFF + ((f32_bits >> 16) & 1);
            // ---------------------------------------------------------
            __m512i v_out_32 = _mm512_castps_si512(v_out_f32);
            
            __m512i v_shift16 = _mm512_srli_epi32(v_out_32, 16);
            __m512i v_and1 = _mm512_and_si512(v_shift16, _mm512_set1_epi32(1));
            __m512i v_rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), v_and1);
            
            // Apply bias and shift
            v_out_32 = _mm512_add_epi32(v_out_32, v_rounding_bias);
            v_out_32 = _mm512_srli_epi32(v_out_32, 16);

            // _mm512_cvtepi32_epi16 naturally packs the 16 32-bit ints back into a 256-bit register!
            __m256i v_out_bf16 = _mm512_cvtepi32_epi16(v_out_32);

            // Store directly to output buffer
            _mm256_storeu_si256((__m256i*)&out_row[i], v_out_bf16);
        }
    }
}

void qwen3_rmsnorm_head_avx512(
    const uint16_t* input, 
    const uint16_t* weight, 
    uint16_t* output, 
    size_t num_tokens, 
    size_t hidden_size, 
    size_t head_dim, // <--- Added parameter (e.g., 128)
    float eps
) {
    // Calculate how many heads are in the hidden size (e.g., 4096 / 128 = 32 heads)
    size_t num_heads = hidden_size / head_dim;

    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t h = 0; h < num_heads; h++) {
            
            // Offset for the current head within the current token
            size_t offset = (t * hidden_size) + (h * head_dim);
            
            const uint16_t* in_chunk = input + offset;
            uint16_t* out_chunk = output + offset;
            
            // Offset the weight for this specific head 
            // (Assuming your weights array is size 4096)
            // const uint16_t* w_chunk = weight + (h * head_dim);
            const uint16_t* w_chunk = weight;
            // -------------------------------------------------------------
            // PASS 1: Compute Variance (Sum of Squares) over head_dim
            // -------------------------------------------------------------
            __m512 v_sum_sq = _mm512_setzero_ps();

            // Loop stops at head_dim instead of hidden_size
            for (size_t i = 0; i < head_dim; i += 16) { 
                __m256i v_in_bf16 = _mm256_loadu_si256((const __m256i*)&in_chunk[i]);
                __m512i v_in_32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(v_in_bf16), 16);
                __m512 v_in_f32 = _mm512_castsi512_ps(v_in_32);

                v_sum_sq = _mm512_fmadd_ps(v_in_f32, v_in_f32, v_sum_sq);
            }

            // Calculate local inverse RMS for THIS HEAD only
            float sum_sq = reduce_add_ps(v_sum_sq);
            float variance = sum_sq / head_dim; // <--- Divided by head_dim, not hidden_size
            float inv_rms = 1.0f / sqrtf(variance + eps); 

            __m512 v_inv_rms = _mm512_set1_ps(inv_rms);

            // -------------------------------------------------------------
            // PASS 2: Normalize, Multiply Weight, and Store
            // -------------------------------------------------------------
            for (size_t i = 0; i < head_dim; i += 16) {
                __m256i v_in_bf16 = _mm256_loadu_si256((const __m256i*)&in_chunk[i]);
                __m512 v_in_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_in_bf16), 16));

                __m256i v_w_bf16 = _mm256_loadu_si256((const __m256i*)&w_chunk[i]);
                __m512 v_w_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_w_bf16), 16));

                __m512 v_out_f32 = _mm512_mul_ps(v_in_f32, v_inv_rms);
                v_out_f32 = _mm512_mul_ps(v_out_f32, v_w_f32);

                // Round-to-Nearest-Even logic
                __m512i v_out_32 = _mm512_castps_si512(v_out_f32);
                __m512i v_shift16 = _mm512_srli_epi32(v_out_32, 16);
                __m512i v_and1 = _mm512_and_si512(v_shift16, _mm512_set1_epi32(1));
                __m512i v_rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), v_and1);
                
                v_out_32 = _mm512_add_epi32(v_out_32, v_rounding_bias);
                v_out_32 = _mm512_srli_epi32(v_out_32, 16);

                __m256i v_out_bf16 = _mm512_cvtepi32_epi16(v_out_32);
                _mm256_storeu_si256((__m256i*)&out_chunk[i], v_out_bf16);
            }
        }
    }
}

void reshape_q(
    const bfloat16_t* in_q, 
    bfloat16_t* out_q, 
    size_t seq_len, 
    size_t num_heads, 
    size_t head_dim
) {
    size_t hidden_size = num_heads * head_dim;

    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < num_heads; h++) {
            // Source: token -> head -> head_dim
            const bfloat16_t* src = in_q + (s * hidden_size) + (h * head_dim);
            
            // Destination: head -> token -> head_dim
            bfloat16_t* dst = out_q + (h * seq_len * head_dim) + (s * head_dim);
            
            // Copy the 128 elements
            std::memcpy(dst, src, head_dim * sizeof(bfloat16_t));
        }
    }
}

void apply_rope_avx512(
    const bfloat16_t* in_tensor,
    const bfloat16_t* cos_tensor,
    const bfloat16_t* sin_tensor,
    bfloat16_t* out_tensor,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim
) {
    size_t half_dim = head_dim / 2; // e.g., 64

    for (size_t h = 0; h < num_heads; h++) {
        for (size_t s = 0; s < seq_len; s++) {
            
            // Pointers for the current token in the current head
            const bfloat16_t* in_row = in_tensor + (h * seq_len * head_dim) + (s * head_dim);
            bfloat16_t* out_row = out_tensor + (h * seq_len * head_dim) + (s * head_dim);
            
            // Cos/Sin only depend on sequence position, so we reuse them across heads
            const bfloat16_t* cos_row = cos_tensor + (s * head_dim);
            const bfloat16_t* sin_row = sin_tensor + (s * head_dim);

            // We only need to iterate up to half_dim, because we process both halves simultaneously
            for (size_t i = 0; i < half_dim; i += 16) {
                
                // 1. Load First Half (q1) and Second Half (q2)
                __m256i v_q1_bf16 = _mm256_loadu_si256((const __m256i*)&in_row[i]);
                __m256i v_q2_bf16 = _mm256_loadu_si256((const __m256i*)&in_row[i + half_dim]);

                // 2. Load Cosine and Sine (Remember: cos/sin are identical in both halves of head_dim)
                __m256i v_c_bf16 = _mm256_loadu_si256((const __m256i*)&cos_row[i]);
                __m256i v_s_bf16 = _mm256_loadu_si256((const __m256i*)&sin_row[i]);

                // 3. Convert all to float32
                __m512 v_q1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_q1_bf16), 16));
                __m512 v_q2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_q2_bf16), 16));
                __m512 v_c  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_c_bf16), 16));
                __m512 v_s  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_s_bf16), 16));

                // -------------------------------------------------------------
                // 4. Perform the Math
                // First Half = (q1 * c) - (q2 * s)
                // Second Half = (q2 * c) + (q1 * s)
                // -------------------------------------------------------------
                __m512 v_out1_f32 = _mm512_sub_ps(_mm512_mul_ps(v_q1, v_c), _mm512_mul_ps(v_q2, v_s));
                __m512 v_out2_f32 = _mm512_add_ps(_mm512_mul_ps(v_q2, v_c), _mm512_mul_ps(v_q1, v_s));

                // -------------------------------------------------------------
                // 5. Convert First Half back to bfloat16 (RTNE)
                // -------------------------------------------------------------
                __m512i v_out1_32 = _mm512_castps_si512(v_out1_f32);
                v_out1_32 = _mm512_add_epi32(v_out1_32, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), _mm512_and_si512(_mm512_srli_epi32(v_out1_32, 16), _mm512_set1_epi32(1))));
                __m256i v_out1_bf16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(v_out1_32, 16));

                // -------------------------------------------------------------
                // 6. Convert Second Half back to bfloat16 (RTNE)
                // -------------------------------------------------------------
                __m512i v_out2_32 = _mm512_castps_si512(v_out2_f32);
                v_out2_32 = _mm512_add_epi32(v_out2_32, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), _mm512_and_si512(_mm512_srli_epi32(v_out2_32, 16), _mm512_set1_epi32(1))));
                __m256i v_out2_bf16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(v_out2_32, 16));

                // 7. Store both halves back to the proper locations
                _mm256_storeu_si256((__m256i*)&out_row[i], v_out1_bf16);
                _mm256_storeu_si256((__m256i*)&out_row[i + half_dim], v_out2_bf16);
            }
        }
    }
}

void repeat_kv_only(
    const bfloat16_t* in_kv, 
    bfloat16_t* out_kv, 
    size_t seq_len, 
    size_t num_q_heads, 
    size_t num_kv_heads, 
    size_t head_dim
) {
    size_t n_rep = num_q_heads / num_kv_heads; // e.g., 32 / 8 = 4
    
    // Total elements in one full attention head across all tokens
    size_t elements_per_head = seq_len * head_dim; // e.g., 512 * 128 = 65536
    size_t bytes_per_head = elements_per_head * sizeof(bfloat16_t);

    for (size_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
        // Source: Pointer to the start of the current KV head
        const bfloat16_t* src = in_kv + (kv_h * elements_per_head);
        
        // Write 'n_rep' (e.g., 4) consecutive copies of this entire head
        for (size_t r = 0; r < n_rep; r++) {
            // Calculate the new Query-aligned head index
            size_t q_h = (kv_h * n_rep) + r; 
            
            // Destination: Pointer to the new head location
            bfloat16_t* dst = out_kv + (q_h * elements_per_head);
            
            // Copy the full [seq_len, head_dim] block directly!
            std::memcpy(dst, src, bytes_per_head);
        }
    }
}

#define IDX4D(b, h, l, d, H, L, D) ((b) * (H * L * D) + (h) * (L * D) + (l) * (D) + (d))

void scaled_dot_product_attention_omp(
    bfloat16_t* Q, bfloat16_t* K, bfloat16_t* V, bool* mask, bfloat16_t* Out,
    int B, int H, int L, int D) 
{
    if (D % 32 != 0) {
        fprintf(stderr, "Error: D must be a multiple of 32.\n");
        return;
    }

    float scale = 1.0f / sqrtf((float)D);

    // 1. Spawn a team of threads
    #pragma omp parallel
    {
        // 2. Thread-Local Allocations: Each thread gets its own temporary arrays
        // This completely prevents race conditions and prevents massive allocation overhead.
        float* scores = (float*)aligned_alloc(64, L * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, D * sizeof(float));

        // 3. Distribute the iterations of the 3 outer loops across all threads
        #pragma omp for collapse(3) schedule(static)
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                for (int i = 0; i < L; i++) {
                    
                    float max_score = -INFINITY;
                    int q_base = IDX4D(b, h, i, 0, H, L, D);

                    // --- Dot Product (Q * K^T) ---
                    for (int j = 0; j < L; j++) { 
                        if (!mask[i * L + j]) {
                            scores[j] = -INFINITY;
                            continue;
                        }

                        int k_base = IDX4D(b, h, j, 0, H, L, D);
                        __m512 acc = _mm512_setzero_ps();

                        for(int d = 0; d < D; d += 32) {
                            __m512i q_vec = _mm512_loadu_si512((__m512i*)&Q[q_base + d]);
                            __m512i k_vec = _mm512_loadu_si512((__m512i*)&K[k_base + d]);
                            acc = _mm512_dpbf16_ps(acc, (__m512bh)q_vec, (__m512bh)k_vec);
                        }

                        float dot = _mm512_reduce_add_ps(acc) * scale;
                        scores[j] = dot;
                        if (dot > max_score) max_score = dot;
                    }

                    // --- Softmax ---
                    float sum_exp = 0.0f;
                    for (int j = 0; j < L; j++) {
                        if (scores[j] != -INFINITY) {
                            scores[j] = expf(scores[j] - max_score); 
                            sum_exp += scores[j];
                        } else {
                            scores[j] = 0.0f;
                        }
                    }
                    
                    if (sum_exp > 0.0f) {
                        float inv_sum = 1.0f / sum_exp;
                        for (int j = 0; j < L; j++) scores[j] *= inv_sum; 
                    }

                    // --- Scores * V ---
                    for (int d = 0; d < D; d++) out_fp32[d] = 0.0f; 

                    for (int j = 0; j < L; j++) {
                        if (scores[j] == 0.0f) continue; 

                        __m512 v_score = _mm512_set1_ps(scores[j]);
                        int v_base = IDX4D(b, h, j, 0, H, L, D);

                        for(int d = 0; d < D; d += 16) {
                            __m256i v_bf16 = _mm256_loadu_si256((__m256i*)&V[v_base + d]);
                            __m512 vf32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v_bf16), 16));
                            
                            __m512 acc_vec = _mm512_loadu_ps(&out_fp32[d]);
                            acc_vec = _mm512_fmadd_ps(v_score, vf32, acc_vec);
                            _mm512_storeu_ps(&out_fp32[d], acc_vec);
                        }
                    }

                    // --- Downcast and Store ---
                    int out_base = IDX4D(b, h, i, 0, H, L, D);
                    for(int d = 0; d < D; d += 16) {
                        __m512 acc_vec = _mm512_loadu_ps(&out_fp32[d]);
                        __m256bh out_bf16 = _mm512_cvtneps_pbh(acc_vec); 
                        _mm256_storeu_si256((__m256i*)&Out[out_base + d], (__m256i)out_bf16);
                    }
                }
            }
        }

        // Free the thread-local arrays before the thread dies
        free(scores);
        free(out_fp32);
    } // End of parallel region
}

void reshape_attention_out(
    const bfloat16_t* in_attn, 
    bfloat16_t* out_attn, 
    size_t seq_len, 
    size_t num_heads, 
    size_t head_dim
) {
    size_t hidden_size = num_heads * head_dim; // e.g., 32 * 128 = 4096

    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < num_heads; h++) {
            // Source: Read from [head, token, head_dim]
            const bfloat16_t* src = in_attn + (h * seq_len * head_dim) + (s * head_dim);
            
            // Destination: Write to [token, head, head_dim]
            bfloat16_t* dst = out_attn + (s * hidden_size) + (h * head_dim);
            
            // Copy the 128 elements back into interleaved memory
            std::memcpy(dst, src, head_dim * sizeof(bfloat16_t));
        }
    }
}

static inline __m512 bf16x16_to_fp32(__m256i bf16_u16) {
    __m512i u32 = _mm512_cvtepu16_epi32(bf16_u16);  // 16x u16 -> 16x u32
    u32 = _mm512_slli_epi32(u32, 16);               // place bits in top 16
    return _mm512_castsi512_ps(u32);                // reinterpret as float
}

static inline __m256i fp32x16_to_bf16_u16_rne(__m512 x) {
    __m512i xi = _mm512_castps_si512(x);

    // RN-even: add 0x7FFF + LSB(truncated bf16)
    __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(xi, 16), _mm512_set1_epi32(1));
    __m512i bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    xi = _mm512_add_epi32(xi, bias);

    __m512i bf16_32 = _mm512_srli_epi32(xi, 16);      // bf16 in low 16 of each lane
    return _mm512_cvtepi32_epi16(bf16_32);            // pack to 16x u16
}
static inline __m512 exp512_ps(__m512 x) {
    const __m512 max_x = _mm512_set1_ps(88.3762626647949f);
    const __m512 min_x = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, max_x);
    x = _mm512_max_ps(x, min_x);

    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half  = _mm512_set1_ps(0.5f);

    __m512 fx = _mm512_fmadd_ps(x, log2e, half);
    __m512i emm0 = _mm512_cvttps_epi32(fx);      // floor
    fx = _mm512_cvtepi32_ps(emm0);

    const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);
    x = _mm512_sub_ps(x, _mm512_mul_ps(fx, ln2_hi));
    x = _mm512_sub_ps(x, _mm512_mul_ps(fx, ln2_lo));

    // polynomial approx on reduced x
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201E-1f));

    __m512 x2 = _mm512_mul_ps(x, x);
    y = _mm512_fmadd_ps(y, x2, x);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

    // 2^n
    emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    emm0 = _mm512_slli_epi32(emm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(emm0);

    return _mm512_mul_ps(y, pow2n);
}


void silu_bf16_to_bf16_avx512(const uint16_t* in_bf16, uint16_t* out_bf16, size_t n) {
    size_t i = 0;
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_set1_ps(0.0f);

    for (; i + 16 <= n; i += 16) {
        __m256i x_u16 = _mm256_loadu_si256((const __m256i*)(in_bf16 + i));
        __m512  x     = bf16x16_to_fp32(x_u16);

        // sigmoid(x) = 1 / (1 + exp(-x))
        __m512 negx  = _mm512_sub_ps(zero, x);
        __m512 e     = exp512_ps(negx);
        __m512 sig   = _mm512_div_ps(one, _mm512_add_ps(one, e));

        // silu(x) = x * sigmoid(x)
        __m512 y = _mm512_mul_ps(x, sig);

        __m256i y_u16 = fp32x16_to_bf16_u16_rne(y);
        _mm256_storeu_si256((__m256i*)(out_bf16 + i), y_u16);
    }

    // scalar tail
    for (; i < n; ++i) {
        // BF16 -> FP32
        uint32_t u = ((uint32_t)in_bf16[i]) << 16;
        float x;
        memcpy(&x, &u, sizeof(float));

        float sig = 1.0f / (1.0f + expf(-x));
        float y = x * sig;

        // FP32 -> BF16 RN-even
        uint32_t yu;
        memcpy(&yu, &y, sizeof(uint32_t));
        uint32_t lsb = (yu >> 16) & 1u;
        yu += 0x7FFFu + lsb;
        out_bf16[i] = (uint16_t)(yu >> 16);
    }
}
#endif