
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
#include "debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"
#include "qwen_lib.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
// #include "error.hpp"
#include "read_data_from_files.hpp"
// #include "write_data_to_files.hpp"

// #include "gelu.hpp"
// #include "pad_qk.hpp"
// #include "copy_data.hpp"
// #include "load_weights.hpp"
#include <unordered_map>
#include <cstring>
#include <thread> 
#include <c10/util/BFloat16.h> 
// #include "packet_vector.hpp"
// #include "decoder.hpp"
#include <iostream>
// #include "token.hpp"
// #include "apply_rope.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <malloc.h>
#include "mvm_sequence.hpp"
// #include "vae_cnn.hpp"
 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// #include "qwen_c.hpp"
// #include "denoise_c.hpp"
// #include "lodepng.h"
namespace po = boost::program_options;
namespace fs = std::filesystem;
using host_bf16  = dtype_out;      // your type, e.g. __bf16
using torch_bf16 = c10::BFloat16; // LibTorch's type
buffer<float> test_func(){
    buffer<float> w(100);
    for (int i = 0; i < 100; i++){
        w[i] = i;
    }
    return w;
}


static inline void concat_L(
    buffer<float>&       out,     // [L_a + L_b, D]
    const buffer<float>& a,       // [L_a, D]
    const buffer<float>& b,       // [L_b, D]
    int L_a, int L_b, int D
){
    const size_t bytes_a = size_t(L_a) * size_t(D) * sizeof(float);
    const size_t bytes_b = size_t(L_b) * size_t(D) * sizeof(float);

    out.resize(size_t(L_a + L_b) * size_t(D));

    float*       outp = out.data();
    const float* ap   = a.data();
    const float* bp   = b.data();

    std::memcpy(outp,                          ap, bytes_a);
    std::memcpy(outp + size_t(L_a)*size_t(D),  bp, bytes_b);
}

inline fs::path make_numbered_file(fs::path const &dir,
    char const *prefix,
    int         idx,
    char const *suffix)
{
    fs::path p = dir;           // copy in
    p /= prefix;                // append a path component
    p += std::to_string(idx);   // append the number to that last component
    p += suffix;                // append the extension (or whatever)
    return p;
}

static inline int64_t file_size_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return (int64_t)f.tellg();
}

// static inline void print_tensor_stats(const torch::Tensor& t, const std::string& name) {
//     torch::Tensor cpu = t.detach().to(torch::kCPU).to(torch::kFloat32);
//     int64_t nan_cnt = torch::isnan(cpu).sum().item<int64_t>();
//     float mn = cpu.min().item<float>();
//     float mx = cpu.max().item<float>();
//     float mean = cpu.mean().item<float>();
//     std::cout << std::fixed << std::setprecision(6)
//               << name << " shape=" << cpu.sizes()
//               << " nan=" << nan_cnt
//               << " min=" << mn << " max=" << mx << " mean=" << mean << "\n";
// }

// static inline void save_ppm_u8_hwc(const torch::Tensor& img_u8_hwc, const std::string& path) {
//     torch::Tensor cpu = img_u8_hwc.contiguous().to(torch::kCPU);
//     if (cpu.dtype() != torch::kUInt8 || cpu.dim() != 3 || cpu.size(2) != 3)
//         throw std::runtime_error("save_ppm expects uint8 HWC(3)");

//     int64_t H = cpu.size(0), W = cpu.size(1);
//     std::ofstream ofs(path, std::ios::binary);
//     if (!ofs) throw std::runtime_error("cannot open output: " + path);
//     ofs << "P6\n" << W << " " << H << "\n255\n";
//     ofs.write(reinterpret_cast<const char*>(cpu.data_ptr<uint8_t>()), H * W * 3);
// }

// static inline void print_nonzero_bytes(const torch::Tensor& img_u8_hwc) {
//     torch::Tensor cpu = img_u8_hwc.contiguous().to(torch::kCPU);
//     const uint8_t* p = cpu.data_ptr<uint8_t>();
//     size_t bytes = (size_t)cpu.numel();
//     size_t nz = 0;
//     for (size_t i = 0; i < bytes; ++i) nz += (p[i] != 0);
//     std::cout << "img nonzero bytes: " << nz << " / " << bytes << "\n";
// }
// namespace fs = std::filesystem;

void save_rgb_png(const std::string& output_path,
    const uint8_t* rgb,
    int W, int H)
{
// ensure directory exists
fs::create_directories(fs::path(output_path).parent_path());

int ok = stbi_write_png(output_path.c_str(), W, H, 3, rgb, W * 3);

if (!ok) {
throw std::runtime_error("stbi_write_png failed: " + output_path);
}

// ⚠️ use stderr, NOT stdout
// std::cerr << "saved " << output_path << "\n";
}



void print_progress_bar(const std::string& prefix, int iteration, int total_steps, std::chrono::time_point<std::chrono::steady_clock> start_time) {
    float progress = (float)(iteration + 1) / total_steps;
    int bar_width = 30; 
    
    // Print the custom prefix and percentage
    std::cout << "\r" << prefix << ": " << std::setw(3) << (int)(progress * 100.0) << "%|";
    
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "█";
        else std::cout << " ";
    }
    
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = current_time - start_time;
    
    // Prevent division by zero just in case
    double iters_completed = (iteration + 1);
    double seconds_per_it = elapsed_seconds.count() / iters_completed;
    double remaining_seconds = seconds_per_it * (total_steps - iters_completed);
    
    int elapsed_m = (int)elapsed_seconds.count() / 60;
    int elapsed_s = (int)elapsed_seconds.count() % 60;
    int remaining_m = (int)remaining_seconds / 60;
    int remaining_s = (int)remaining_seconds % 60;

    std::cout << "| " << (iteration + 1) << "/" << total_steps << " "
              << "[" << std::setfill('0') << std::setw(2) << elapsed_m << ":" 
              << std::setfill('0') << std::setw(2) << elapsed_s << "<"
              << std::setfill('0') << std::setw(2) << remaining_m << ":" 
              << std::setfill('0') << std::setw(2) << remaining_s << ", "
              << std::fixed << std::setprecision(2) << seconds_per_it << "s/it]";
              
    std::cout.flush(); 
}


#include "load_weights.hpp"
#include "copy_data.hpp"
#include "denoise_lib.hpp"
#include "vae_cnn.hpp"
int main(int argc, const char *argv[]) {
    std::string weights_path   = "/home/kelsey/NPU_projects/NPU_new/new_test/AUser_host/model_weights";
    std::string npu_files_path = "/home/kelsey/NPU_projects/npu-image-diffusion/build/Z-Image-Turbo";
    int seed                   = 42;
    int image_H                = 1024;
    int image_W                = 768;
    int step                   = 4;
    std::string my_prompt      = "a beautiful landscape";
    std::string output_path    = "./images/output.png";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
    
        // Check for help flag first
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./your_program [options]\n\n"
                      << "Options:\n"
                      << "  --help, -h          Show this help message and exit\n"
                      << "  --weights_path      Path to model weights directory\n"
                      << "  --npu_files_path    Path to NPU files directory\n"
                      << "  --seed              Random seed (default: 42)\n"
                      << "  --image_H           Image height (default: 1024)\n"
                      << "  --image_W           Image width (default: 768)\n"
                      << "  --step              Number of generation steps (default: 4)\n"
                      << "  --prompt            Text prompt (default: \"a beautiful landscape\")\n"
                      << "  --output_path       Path to save the generated image (default: \"./images/output.png\")\n";
            return 0; // Exit the program after printing help
        } 
        // Otherwise, check for the other flags
        else if (arg == "--weights_path" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--npu_files_path" && i + 1 < argc) {
            npu_files_path = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--image_H" && i + 1 < argc) {
            image_H = std::stoi(argv[++i]);
        } else if (arg == "--image_W" && i + 1 < argc) {
            image_W = std::stoi(argv[++i]);
        } else if (arg == "--step" && i + 1 < argc) {
            step = std::stoi(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            my_prompt = argv[++i];
        } else if (arg == "--output_path" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Unknown argument or missing value: " << arg << "\n"
                      << "Run with --help for usage information." << std::endl;
            return 1; // Exit with an error code if the argument is invalid
        }
    }
 
    std::cout << "H = " << image_H << "\n";
    std::cout << "W = " << image_W << "\n";
    std::cout << "Prompt = " << my_prompt << "\n";
    int SEQ_MULTI_OF = 64; 

    int latent_H = image_H/8;
    int latent_W = image_W /8;
    int double_W = latent_W * 2;
    int double_H = latent_H * 2;  
    int tri_W = double_W * 2; 
    int tri_H = double_H * 2;
    int img_seqlen = latent_H*latent_W/4;
    constexpr int pF = 1;
    constexpr int pH = 2;
    constexpr int pW = 2;
    int F_tokens = 1 / pF;  //F//pf
    int H_tokens = latent_H / pH; 
    int W_tokens = latent_W / pW;
    int random_seed = seed;
    int img_size = 0;
    time_utils::time_point start_begin = time_utils::now();
    if (image_W * 9 == image_H * 16 || image_W * 16 == image_H * 9) {
        img_size = 1;
    }
    else {
        img_size = 0;
    }

    time_utils::time_point start_time_tokenizer = time_utils::now();
    // std::string common_path = "/home/kelsey/NPU_projects/model_weights";
    std::string json_path = weights_path + "/Z-Image-Turbo/text_encoder/tokenizer.json" ; 
  
    auto [input_ids_vec, attention_mask_vec] = tokenize_text_cpu(my_prompt, json_path);
    
    size_t num_true = std::count(attention_mask_vec.begin(), attention_mask_vec.end(), true);    
    /////////////////////////////////////////////////////////////////////////////////////////////
  
    float cosine_sim_final, rel_l1_final, rmse_final, rel_l2_final, avg_abs_O_final;
    

    ////////////////////////////////////////////////////////////////////////////////////////
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("M,m", po::value<int>()->default_value(128 * 4 * 4), "M");
    desc.add_options()("K,k", po::value<int>()->default_value(128 * 4), "K");
    desc.add_options()("I,i", po::value<int>()->default_value(1), "Iterations");

    arg_utils::parse_options(argc, argv, desc, vm);
    fs::path base_dir = weights_path + "/Z-Image-Turbo/denoising/bf16_parts"; 

    fs::path qwen_dir = weights_path + "/Z-Image-Turbo/text_encoder";

   
    fs::path vae_decoder_dir = weights_path + "/Z-Image-Turbo/vae_decoder";
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }
    arg_utils::parse_options(argc, argv, desc, vm);
    /////////////////////////////////////////////////////////////////////////
    std::string qwen_qkv_mm_xclbin = npu_files_path + "/xclbins/MM_128.xclbin";
    npu_app_desc accel_desc_q, accel_desc_kv, accel_desc_o, accel_desc_down, accel_desc_gate_up;
    accel_desc_q.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_q.app_name = "qwen_q_mm";
    accel_desc_kv.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_kv.app_name = "qwen_k_mm";
    accel_desc_o.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_o.app_name = "qwen_o_mm";
    accel_desc_down.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_down.app_name = "qwen_down_mm";
    accel_desc_gate_up.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_gate_up.app_name = "qwen_gate_up_mm";

    npu_app app_q = npu_instance.create_app(accel_desc_q);
    npu_app app_kv = npu_instance.create_app(accel_desc_kv);
    npu_app app_o = npu_instance.create_app(accel_desc_o);
    npu_app app_down = npu_instance.create_app(accel_desc_down);
    npu_app app_gate_up = npu_instance.create_app(accel_desc_gate_up);

    app_q.instr_seq->from_file(npu_files_path + "/insts/q_mm.txt");
    app_kv.instr_seq->from_file(npu_files_path + "/insts/kv_mm.txt");
    app_o.instr_seq->from_file(npu_files_path +"/insts/o_mm.txt");
    app_down.instr_seq->from_file(npu_files_path +"/insts/down_mm.txt");
    app_gate_up.instr_seq->from_file(npu_files_path +"/insts/gate_mm.txt");
    
    buffer<float> seq_q = app_q.instr_seq->dump().cast_to<float>();
    buffer<float> seq_kv = app_kv.instr_seq->dump().cast_to<float>();
    buffer<float> seq_o = app_o.instr_seq->dump().cast_to<float>();
    buffer<float> seq_down = app_down.instr_seq->dump().cast_to<float>();
    buffer<float> seq_gate_up = app_gate_up.instr_seq->dump().cast_to<float>();

    
    ///////////////////////////////load weights///////////////////////////////
    buffer<dtype_out> cap_feats(512*2560);
    for(int i = 0; i < 512*2560; i++)
    cap_feats[i] = 0;
    {
    buffer<dtype_out> qwen_q_MM_in = app_q.create_bo_buffer<dtype_out>(512*2560, 3);
    buffer<dtype_out> qwen_q_MM_out = app_q.create_bo_buffer<dtype_out>(512*4096, 5);
    buffer<dtype_out> qwen_kv_MM_in = app_kv.create_bo_buffer<dtype_out>(512*2560, 3);
    buffer<dtype_out> qwen_k_MM_out = app_kv.create_bo_buffer<dtype_out>(512*1024, 5);
    buffer<dtype_out> qwen_v_MM_out = app_kv.create_bo_buffer<dtype_out>(512*1024, 5);
    buffer<dtype_out> qwen_o_MM_in = app_o.create_bo_buffer<dtype_out>(512*4096, 3);
    buffer<dtype_out> qwen_o_MM_out = app_o.create_bo_buffer<dtype_out>(512*2560, 5);
    buffer<dtype_out> qwen_down_MM_in = app_down.create_bo_buffer<dtype_out>(512*9728, 3);
    buffer<dtype_out> qwen_down_MM_out = app_down.create_bo_buffer<dtype_out>(512*2560, 5);
    buffer<dtype_out> qwen_gate_up_MM_in = app_gate_up.create_bo_buffer<dtype_out>(512*2560, 3);
    buffer<dtype_out> qwen_gate_MM_out = app_gate_up.create_bo_buffer<dtype_out>(512*9728, 5);
    buffer<dtype_out> qwen_up_MM_out = app_gate_up.create_bo_buffer<dtype_out>(512*9728, 5);
    struct qwen_MM_weights{
        buffer<dtype_out> w_q_proj;
        std::array<buffer<dtype_out>, 2> w_kv_proj;
        buffer<dtype_out> w_o_proj;
        buffer<dtype_out> w_mlp_down_proj;
        std::array<buffer<dtype_out>, 2> w_mlp_gate_proj;
    };
    constexpr int NUM_qwen_layers = 35;
    std::vector<qwen_MM_weights> qwen_MM_layers(NUM_qwen_layers);
    struct qwen_norm_weights{
        buffer<dtype_out> input_norm;
        buffer<dtype_out> q_norm;
        buffer<dtype_out> k_norm;
        buffer<dtype_out> o_norm;
    };
    std::vector<qwen_norm_weights> qwen_norm_layers(NUM_qwen_layers);
    for (int i = 0; i < NUM_qwen_layers; i++) {
        fs::path qwen_q_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_self_attn_q_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_q_proj = app_q.create_bo_buffer<dtype_out>(4096*2560, 4);
        load_directly_to_tensor(qwen_q_proj_path, qwen_MM_layers[i].w_q_proj.data(), 4096 * 2560);
        fs::path qwen_k_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_self_attn_k_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_kv_proj[0] = app_kv.create_bo_buffer<dtype_out>(1024*2560, 4);
        load_directly_to_tensor(qwen_k_proj_path, qwen_MM_layers[i].w_kv_proj[0].data(), 1024 * 2560);
        fs::path qwen_v_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_self_attn_v_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_kv_proj[1] = app_kv.create_bo_buffer<dtype_out>(1024*2560, 4);
        load_directly_to_tensor(qwen_v_proj_path, qwen_MM_layers[i].w_kv_proj[1].data(), 1024 * 2560);
        fs::path qwen_o_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_self_attn_o_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_o_proj = app_o.create_bo_buffer<dtype_out>(4096*2560, 4);
        load_directly_to_tensor(qwen_o_proj_path, qwen_MM_layers[i].w_o_proj.data(), 4096 * 2560);
        fs::path qwen_mlp_down_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_mlp_down_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_mlp_down_proj = app_down.create_bo_buffer<dtype_out>(9728*2560, 4);
        load_directly_to_tensor(qwen_mlp_down_proj_path, qwen_MM_layers[i].w_mlp_down_proj.data(), 9728 * 2560);
        fs::path qwen_mlp_gate_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_mlp_gate_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_mlp_gate_proj[0] = app_gate_up.create_bo_buffer<dtype_out>(9728*2560, 4);
        load_directly_to_tensor(qwen_mlp_gate_proj_path, qwen_MM_layers[i].w_mlp_gate_proj[0].data(), 9728 * 2560);
        fs::path qwen_mlp_up_proj_path = make_numbered_file(qwen_dir, "layers_", i, "_mlp_up_proj_weight_bf16_u16.bin");
        qwen_MM_layers[i].w_mlp_gate_proj[1] = app_gate_up.create_bo_buffer<dtype_out>(9728*2560, 4);
        load_directly_to_tensor(qwen_mlp_up_proj_path, qwen_MM_layers[i].w_mlp_gate_proj[1].data(), 9728 * 2560);
        fs::path qwen_input_norm_path = make_numbered_file(qwen_dir, "layers_", i, "_input_layernorm_weight_bf16_u16.bin");
        fs::path qwen_q_norm_path = make_numbered_file(qwen_dir, "layers_", i, "_self_attn_q_norm_weight_bf16_u16.bin");
        fs::path qwen_k_norm_path = make_numbered_file(qwen_dir, "layers_", i, "_self_attn_k_norm_weight_bf16_u16.bin");
        fs::path qwen_o_norm_path = make_numbered_file(qwen_dir, "layers_", i, "_post_attention_layernorm_weight_bf16_u16.bin");
        qwen_norm_layers[i].input_norm = read_bf16_from_u16bin<dtype_out>(qwen_input_norm_path, 2560);
        qwen_norm_layers[i].q_norm = read_bf16_from_u16bin<dtype_out>(qwen_q_norm_path, 128);
        qwen_norm_layers[i].k_norm = read_bf16_from_u16bin<dtype_out>(qwen_k_norm_path, 128);
        qwen_norm_layers[i].o_norm = read_bf16_from_u16bin<dtype_out>(qwen_o_norm_path, 2560); 
    }
    
    constexpr size_t N_qwen_all = 512 * 2560;
    std::array<buffer<dtype_in>, 36> all_qwen_result_buffers;
    for (auto& b : all_qwen_result_buffers) b = buffer<dtype_in>(N_qwen_all); // or b.resize(N_qwen_all)
    //////////////////////////////////////////////////////////////////////////////////////////////
    int text_encoder_steps = 1;
    auto start_text = std::chrono::steady_clock::now();
    int vocab_size = 151936;
    int hidden_size = 2560;
    int padding_idx = 0;
    int total_tokens = 512;
    fs::path token_embeddings_path = qwen_dir / "embed_tokens_weight_bf16_u16.bin";
    std::vector<uint16_t> token_embeddings(151936*2560);
    io::read_data_from_files(token_embeddings_path.c_str(), token_embeddings, 151936*2560);
    
    buffer<dtype_out> token_embeddings_output( 512*2560);
    embed_tokens_forward(
        input_ids_vec.data(), 512, token_embeddings.data(), 
        vocab_size, hidden_size, padding_idx, reinterpret_cast<bfloat16_t*>(all_qwen_result_buffers[0].data()), 1
    );
    std::vector<int> position_ids(512);
    for(int i = 0; i < 512; i++) {
        position_ids[i] = i;
    }   
    buffer<dtype_out> cos_out(512*128);
    buffer<dtype_out> sin_out(512*128);

    compute_rope_embeddings_bf16(position_ids.data(), 1, 512, 128, 1000000.0, 1.0, 
    reinterpret_cast<bfloat16_t*>(cos_out.data()), reinterpret_cast<bfloat16_t*>(sin_out.data()));
    int L = 512;
    int valid_k = num_true;
    size_t total_elements = (size_t)L * L;
    bool* inference_mask = (bool*)malloc(total_elements * sizeof(bool));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            bool causal = (j <= i);
            bool key_valid = (j < valid_k);
            
            // Mask mapping: index = i * L + j
            inference_mask[i * L + j] = causal && key_valid;
        }
    }
    ///////////////////////////////////////number of layers/////////////////////////////////////
    time_utils::time_point start_qk_mm = time_utils::now();
    for (int layer_id = 0; layer_id < 35; layer_id++) {
        time_utils::time_point start_input_norm = time_utils::now();
       buffer<dtype_out> input_norm_out(512*2560);
       qwen3_rmsnorm_avx512(
        reinterpret_cast<const bfloat16_t*>(all_qwen_result_buffers[layer_id].data()), // Input
        reinterpret_cast<const bfloat16_t*>(qwen_norm_layers[layer_id].input_norm.data()),          // Weight
        reinterpret_cast<bfloat16_t*>(input_norm_out.data()),                // Output
        512,                                                                 // num_tokens
        2560,                                                                // hidden_size
        1e-6f                                                                // eps (from your config)
        );
        copy_bf16_to_bf16_avx512(input_norm_out.data(), (qwen_q_MM_in.data()), 512*2560);
        time_utils::time_point start_q_mm = time_utils::now();
        qwen_q_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_q_proj.sync_to_device();
        app_q(qwen_q_MM_in, qwen_MM_layers[layer_id].w_q_proj, qwen_q_MM_out);
        qwen_q_MM_out.sync_from_device();

        copy_bf16_to_bf16_avx512(input_norm_out.data(), (qwen_kv_MM_in.data()), 512*2560);
        qwen_kv_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_kv_proj[0].sync_to_device();
        app_kv(qwen_q_MM_in, qwen_MM_layers[layer_id].w_kv_proj[0], qwen_k_MM_out);
        qwen_k_MM_out.sync_from_device();
        
        qwen_kv_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_kv_proj[1].sync_to_device();
        app_kv(qwen_q_MM_in, qwen_MM_layers[layer_id].w_kv_proj[1], qwen_v_MM_out);
        qwen_v_MM_out.sync_from_device();
      
        buffer<dtype_out> q_norm_out(512*4096);
        
        qwen3_rmsnorm_head_avx512(
            reinterpret_cast<const bfloat16_t*>(qwen_q_MM_out.data()), // Input
            reinterpret_cast<const bfloat16_t*>(qwen_norm_layers[layer_id].q_norm.data()),          // Weight
            reinterpret_cast<bfloat16_t*>(q_norm_out.data()),                // Output
            512,                                                                 // num_tokens
            4096,
            128,                                                                // head_dim
            1e-6f                                                                // eps (from your config)
            );
        buffer<dtype_out> k_norm_out(512*1024);
        qwen3_rmsnorm_head_avx512(
            reinterpret_cast<const bfloat16_t*>(qwen_k_MM_out.data()), // Input
            reinterpret_cast<const bfloat16_t*>(qwen_norm_layers[layer_id].k_norm.data()),          // Weight
            reinterpret_cast<bfloat16_t*>(k_norm_out.data()),                // Output
            512,                                                                 // num_tokens
            1024,
            128,                                                                // head_dim
            1e-6f                                                                // eps (from your config)
            );    
        buffer<dtype_out> q_norm_out_rope(512*4096);
        buffer<dtype_out> k_norm_out_rope(512*1024);
        buffer<dtype_out> reshaped_q_buffer(512*4096);
        reshape_q(
            reinterpret_cast<const bfloat16_t*>(q_norm_out.data()),
            reinterpret_cast<bfloat16_t*>(reshaped_q_buffer.data()),
            512, 32, 128
        );
        buffer<dtype_out> reshaped_k_buffer(512*1024);
        reshape_q(
            reinterpret_cast<const bfloat16_t*>(k_norm_out.data()),
            reinterpret_cast<bfloat16_t*>(reshaped_k_buffer.data()),
            512, 8, 128
        );
        buffer<dtype_out> reshaped_v_buffer(512*1024);
        reshape_q(
            reinterpret_cast<const bfloat16_t*>(qwen_v_MM_out.data()),
            reinterpret_cast<bfloat16_t*>(reshaped_v_buffer.data()),
            512, 8, 128
        );
        apply_rope_avx512(reinterpret_cast<const bfloat16_t*>(reshaped_q_buffer.data()), 
        reinterpret_cast<const bfloat16_t*>(cos_out.data()), 
        reinterpret_cast<const bfloat16_t*>(sin_out.data()), 
        reinterpret_cast<bfloat16_t*>(q_norm_out_rope.data()),  32, 512, 128);
        apply_rope_avx512(reinterpret_cast<const bfloat16_t*>(reshaped_k_buffer.data()), 
        reinterpret_cast<const bfloat16_t*>(cos_out.data()), 
        reinterpret_cast<const bfloat16_t*>(sin_out.data()), 
        reinterpret_cast<bfloat16_t*>(k_norm_out_rope.data()),  8, 512, 128);
       
        buffer<dtype_out> repeated_k_buffer(512*4096);
        repeat_kv_only(
            reinterpret_cast<const bfloat16_t*>(k_norm_out_rope.data()),
            reinterpret_cast<bfloat16_t*>(repeated_k_buffer.data()),
            512, 32, 8, 128
        );
        buffer<dtype_out> repeated_v_buffer(512*4096);
        repeat_kv_only(
            reinterpret_cast<const bfloat16_t*>(reshaped_v_buffer.data()),
            reinterpret_cast<bfloat16_t*>(repeated_v_buffer.data()),
            512, 32, 8, 128
        );
        time_utils::time_point start_qk_mm = time_utils::now();
        buffer<dtype_out> o_norm_out(512*4096);
        scaled_dot_product_attention_omp(
             reinterpret_cast< bfloat16_t*>(q_norm_out_rope.data()),
             reinterpret_cast< bfloat16_t*>(repeated_k_buffer.data()), 
             reinterpret_cast< bfloat16_t*>(repeated_v_buffer.data()), 
             inference_mask, reinterpret_cast<bfloat16_t*>(o_norm_out.data()), 1, 32, 512, 128);
       
        buffer<dtype_out> o_norm_out_reshape(512*4096);
        reshape_attention_out(
            reinterpret_cast<const bfloat16_t*>(o_norm_out.data()),
            reinterpret_cast<bfloat16_t*>(qwen_o_MM_in.data()),
            512, 32, 128
        );
        time_utils::time_point start_o_mm = time_utils::now();
       
        qwen_o_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_o_proj.sync_to_device();
        app_o(qwen_o_MM_in, qwen_MM_layers[layer_id].w_o_proj, qwen_o_MM_out);
        qwen_o_MM_out.sync_from_device();
        
       
        buffer<dtype_out> qwen_o_MM_out_add(512*2560);
        for(int i = 0; i < 512; i++) {
          for(int j = 0; j < 2560; j++) {
            qwen_o_MM_out_add[i*2560 + j] = qwen_o_MM_out[i*2560 + j] + all_qwen_result_buffers[layer_id][i*2560 + j];
          }
        }
        buffer<dtype_out> qwen_o_MM_out_add_norm(512*2560);
        qwen3_rmsnorm_avx512(
            reinterpret_cast<const bfloat16_t*>(qwen_o_MM_out_add.data()), // Input
            reinterpret_cast<const bfloat16_t*>(qwen_norm_layers[layer_id].o_norm.data()),          // Weight
            reinterpret_cast<bfloat16_t*>(qwen_gate_up_MM_in.data()),                // Output
            512,                                                                 // num_tokens
            2560,                                                                // hidden_size
            1e-6f                                                                // eps (from your config)
            );
        time_utils::time_point start_gate_mm = time_utils::now();
           
      
        qwen_gate_up_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_mlp_gate_proj[1].sync_to_device();
        app_gate_up(qwen_gate_up_MM_in, qwen_MM_layers[layer_id].w_mlp_gate_proj[1], qwen_up_MM_out);
        qwen_up_MM_out.sync_from_device();  
        
        buffer<dtype_out> gate_silu_out(512*9728);
        time_utils::time_point start_silu_mm = time_utils::now();
        qwen_gate_up_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_mlp_gate_proj[0].sync_to_device();
        app_gate_up(qwen_gate_up_MM_in, qwen_MM_layers[layer_id].w_mlp_gate_proj[0], qwen_gate_MM_out);
        qwen_gate_MM_out.sync_from_device();
        silu_bf16_to_bf16_avx512(
            reinterpret_cast<const uint16_t*>(qwen_gate_MM_out.data()), 
            reinterpret_cast<uint16_t*>(gate_silu_out.data()), 
            9728 * 512
        );
        time_utils::time_point start_down_mm = time_utils::now();
        // std::cout << "------silu time-------: " << time_utils::duration_us(start_silu_mm , start_down_mm ).first / 1000 << "ms" << std::endl;
        bf16_mul_avx512(
            reinterpret_cast<uint16_t*>(qwen_down_MM_in.data()),
             reinterpret_cast<uint16_t*>(qwen_up_MM_out.data()),
              reinterpret_cast<uint16_t*>(gate_silu_out.data()), 512*9728);
        
        qwen_down_MM_in.sync_to_device();
        qwen_MM_layers[layer_id].w_mlp_down_proj.sync_to_device();
        app_down(qwen_down_MM_in, qwen_MM_layers[layer_id].w_mlp_down_proj, qwen_down_MM_out);
        qwen_down_MM_out.sync_from_device();
       
        buffer<dtype_out> qwen_down_MM_out_add(512*2560);
        for(int i = 0; i < 512; i++) {
          for(int j = 0; j < 2560; j++) {
            all_qwen_result_buffers[layer_id + 1][i*2560 + j] = qwen_down_MM_out[i*2560 + j] + qwen_o_MM_out_add[i*2560 + j];
          }
        }
       
    }

  
    copy_bf16_to_bf16_avx512(all_qwen_result_buffers[35].data(), cap_feats.data(), num_true*2560);   ///7.66GB
    print_progress_bar("Text Encoder", 0, text_encoder_steps, start_text);
    std::cout << std::endl;
    }
   
    
   


    int cap_ori_len = num_true;
    if(cap_ori_len < 64)
    {
        cap_ori_len = 128;
    }
    int cap_padding_len = (SEQ_MULTI_OF - (cap_ori_len % SEQ_MULTI_OF)) % SEQ_MULTI_OF;
    
    int cap_total_len = cap_ori_len + cap_padding_len;
    // printf("cap_total_len: %d\n", cap_total_len);
    //////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    time_utils::time_point start_time_pos = time_utils::now();
    
    ////////////////////////////////////////////////////noise layers/////////////////////////////////////////////////////////////////////////////
    std::string xclbin_name_all_v = npu_files_path + "/xclbins/MM_64.xclbin";
    npu_app_desc accel_desc_all_v, accel_desc_all_mlp01, accel_desc_all_mlp_out, accel_desc_all_qk;

    npu_app_desc accel_desc_0, accel_desc_1, accel_desc_2;

    npu_app app_0;
    npu_app app_1;
    npu_app app_2;

    if(img_size == 1)
    {
        accel_desc_0.xclbin_name = xclbin_name_all_v;
        accel_desc_0.app_name = "noise_mm_nobias";
        accel_desc_1.xclbin_name = xclbin_name_all_v;
        accel_desc_1.app_name = "noise_mm_nobias_1";
        accel_desc_2.xclbin_name = xclbin_name_all_v;
        accel_desc_2.app_name = "noise_mm_nobias_2";
        app_0 = npu_instance.create_app(accel_desc_0);
        app_1 = npu_instance.create_app(accel_desc_1);
        app_2 = npu_instance.create_app(accel_desc_2);
        generate_mm_64_double_sequence(*app_0.instr_seq, img_seqlen, 4096, 4096, 32, 4, 8); ///M, K, N, k_size
        generate_mm_64_double_sequence(*app_1.instr_seq, img_seqlen, 4096, 10240, 32, 4, 8); ///M, K, N, k_size
        generate_mm_64_double_sequence(*app_2.instr_seq, img_seqlen, 10240, 4096, 32, 4, 8); ///M, K, N, k_size
        // std::cout << "generate MM 64 for noise layers " << std::endl;
    }
    else
    {
        // std::cout << "noise img_seqlen size is " << std::endl;
        accel_desc_0.xclbin_name = qwen_qkv_mm_xclbin;
        accel_desc_0.app_name = "noise_mm_nobias";
        accel_desc_1.xclbin_name = qwen_qkv_mm_xclbin;
        accel_desc_1.app_name = "noise_mm_nobias_1";
        accel_desc_2.xclbin_name = qwen_qkv_mm_xclbin;
        accel_desc_2.app_name = "noise_mm_nobias_2";
        app_0 = npu_instance.create_app(accel_desc_0);
        app_1 = npu_instance.create_app(accel_desc_1);
        app_2 = npu_instance.create_app(accel_desc_2);
        generate_mm_128_double_sequence(*app_0.instr_seq, img_seqlen, 4096, 4096, 64, 4, 8); ///M, K, N, k_size
        generate_mm_128_double_sequence(*app_1.instr_seq, img_seqlen, 4096, 10240, 64, 4, 8); ///M, K, N, k_size
        generate_mm_128_double_sequence(*app_2.instr_seq, img_seqlen, 10240, 4096, 64, 4, 8); ///M, K, N, k_size
        // std::cout << "generate MM 128 for noise layers " << std::endl;
    }
    
    
    buffer<float> seq_0 = app_0.instr_seq->dump().cast_to<float>();
    buffer<float> seq_1 = app_1.instr_seq->dump().cast_to<float>();
    buffer<float> seq_2 = app_2.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string xclbin_name_first_mm_new = npu_files_path + "/xclbins/mm_test_more.xclbin";
    npu_app_desc accel_desc_adaLN, accel_desc_adaLN_final, accel_desc_final_mm, 
    accel_desc_first_mm, accel_desc_time_emb, accel_desc_time_emb2;
    accel_desc_adaLN.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_adaLN.app_name = "noise_adaLN";
    accel_desc_adaLN_final.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_adaLN_final.app_name = "noise_adaLN_final";
    accel_desc_final_mm.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_final_mm.app_name = "final_mm";
    accel_desc_first_mm.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_first_mm.app_name = "first_mm";
    accel_desc_time_emb.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_time_emb.app_name = "time_emb";
    accel_desc_time_emb2.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_time_emb2.app_name = "time_emb2";
    npu_app app_adaLN = npu_instance.create_app(accel_desc_adaLN);
    npu_app app_adaLN_final = npu_instance.create_app(accel_desc_adaLN_final);
    npu_app app_final_mm = npu_instance.create_app(accel_desc_final_mm);
    npu_app app_first_mm = npu_instance.create_app(accel_desc_first_mm);
    npu_app app_time_emb = npu_instance.create_app(accel_desc_time_emb);
    npu_app app_time_emb2 = npu_instance.create_app(accel_desc_time_emb2);   
    app_adaLN.instr_seq->from_file(npu_files_path +"/insts/noise_mm_adaLN.txt");
    app_adaLN_final.instr_seq->from_file(npu_files_path +"/insts/noise_mm_adaLN_final.txt");
    app_final_mm.instr_seq->from_file(npu_files_path +"/insts/final_mm_new.txt");
    app_first_mm.instr_seq->from_file(npu_files_path +"/insts/first_mm.txt");
    app_time_emb.instr_seq->from_file(npu_files_path +"/insts/mm_time_embed.txt");
    app_time_emb2.instr_seq->from_file(npu_files_path +"/insts/mm_time_embed2.txt");
    buffer<float> seq_time_emb = app_time_emb.instr_seq->dump().cast_to<float>();
    buffer<float> seq_time_emb2 = app_time_emb2.instr_seq->dump().cast_to<float>();
    buffer<float> seq_adaLN = app_adaLN.instr_seq->dump().cast_to<float>();
    buffer<float> seq_adaLN_final = app_adaLN_final.instr_seq->dump().cast_to<float>();
    buffer<float> seq_final_mm = app_final_mm.instr_seq->dump().cast_to<float>();
    buffer<float> seq_first_mm = app_first_mm.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////noise atten rms norm //////////////////////////////////////////////////////
    std::string xclbin_name_attn_noise = npu_files_path +"/xclbins/noise_atten_custom.xclbin";
    npu_app_desc accel_desc_attn_noise;
    accel_desc_attn_noise.xclbin_name = xclbin_name_attn_noise;
    accel_desc_attn_noise.app_name = "noise_attn";
    npu_app app_attn_noise = npu_instance.create_app(accel_desc_attn_noise);
   
    int num_img_kv = img_seqlen/64;

    generate_noise_atten_sequence(*app_attn_noise.instr_seq, 30, 128, num_img_kv, 4096, 4, 8);
    buffer<float> seq_attn_noise = app_attn_noise.instr_seq->dump().cast_to<float>();

    std::string xclbin_name_attn_cap = npu_files_path +"/xclbins/cap_atten_custom.xclbin";
    npu_app_desc accel_desc_attn_cap;
    accel_desc_attn_cap.xclbin_name = xclbin_name_attn_cap;
    accel_desc_attn_cap.app_name = "cap_attn";
    npu_app app_attn_cap = npu_instance.create_app(accel_desc_attn_cap);
   
    //////////////////////generate custom token sequence////////////////////////////////////////////////////////////
    int num_kv = cap_total_len/64;
    generate_cap_atten_sequence(*app_attn_cap.instr_seq, 30, 16, num_kv, 4, 4);  // 512/32=16
    buffer<float> seq_attn_cap = app_attn_cap.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   
    std::string xclbin_name_attn_all = npu_files_path +"/xclbins/all_atten_custom.xclbin";
    npu_app_desc accel_desc_attn_all;
    accel_desc_attn_all.xclbin_name = xclbin_name_attn_all;
    accel_desc_attn_all.app_name = "all_attn";
    npu_app app_attn_all = npu_instance.create_app(accel_desc_attn_all);
    //////////////////////generate custom token sequence////////////////////////////////////////////////////////////
    int num_kv_all = (4096+cap_total_len)/64;
    generate_all_atten_sequence(*app_attn_all.instr_seq, 30, 144, num_kv_all, 4096, 4, 8);
    ////////////////////////////////////////////////////num_head, num_q, num_kv, row_offset, row, col 144 =4608/32
    buffer<float> seq_attn_all = app_attn_all.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    npu_app_desc accel_desc_cap_qkv, accel_desc_cap_mlp01, accel_desc_cap_mlp_out, accel_desc_cap_embedder;
    accel_desc_cap_qkv.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_cap_qkv.app_name = "cap_qkv";
    accel_desc_cap_mlp01.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_cap_mlp01.app_name = "cap_mlp01";
    accel_desc_cap_mlp_out.xclbin_name = qwen_qkv_mm_xclbin;
    accel_desc_cap_mlp_out.app_name = "cap_mlp_out";
    accel_desc_cap_embedder.xclbin_name = xclbin_name_first_mm_new;
    accel_desc_cap_embedder.app_name = "cap_embedder";
    npu_app app_cap_qkv = npu_instance.create_app(accel_desc_cap_qkv);
    npu_app app_cap_mlp01 = npu_instance.create_app(accel_desc_cap_mlp01);
    npu_app app_cap_mlp_out = npu_instance.create_app(accel_desc_cap_mlp_out);
    npu_app app_cap_embedder = npu_instance.create_app(accel_desc_cap_embedder);
  
    app_cap_embedder.instr_seq->from_file(npu_files_path +"/insts/cap_mm_embed.txt");
    app_cap_qkv.instr_seq->from_file(npu_files_path +"/insts/cap_mm_qkvo.txt");
    app_cap_mlp01.instr_seq->from_file(npu_files_path +"/insts/cap_mm_mlp01.txt");
    app_cap_mlp_out.instr_seq->from_file(npu_files_path +"/insts/cap_mm_mlp_out.txt");
    buffer<float> seq_cap_qkv = app_cap_qkv.instr_seq->dump().cast_to<float>();
    buffer<float> seq_cap_mlp01 = app_cap_mlp01.instr_seq->dump().cast_to<float>();
    buffer<float> seq_cap_mlp_out = app_cap_mlp_out.instr_seq->dump().cast_to<float>();
    buffer<float> seq_cap_embedder = app_cap_embedder.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    npu_app app_all_v;
    npu_app app_all_mlp01;
    npu_app app_all_mlp_out;
  
    
    int all_token = img_seqlen + cap_total_len;
     //std::cout << " all token length is " << all_token << std::endl;
    int padding_all = 0;
    if(all_token % 512 == 0)
    {
        accel_desc_all_v.xclbin_name = qwen_qkv_mm_xclbin;
        accel_desc_all_v.app_name = "all_v";
        accel_desc_all_mlp01.xclbin_name = qwen_qkv_mm_xclbin;
        accel_desc_all_mlp01.app_name = "all_mlp01";
        accel_desc_all_mlp_out.xclbin_name = qwen_qkv_mm_xclbin;
        accel_desc_all_mlp_out.app_name = "all_mlp_out";
        app_all_v = npu_instance.create_app(accel_desc_all_v);
        app_all_mlp01 = npu_instance.create_app(accel_desc_all_mlp01);
        app_all_mlp_out = npu_instance.create_app(accel_desc_all_mlp_out);

        generate_mm_128_double_sequence(*app_all_v.instr_seq, all_token, 4096, 4096, 64, 4, 8); ///M, K, N, k_size
        generate_mm_128_double_sequence(*app_all_mlp01.instr_seq, all_token, 4096, 10240, 64, 4, 8); ///M, K, N, k_size
        generate_mm_128_double_sequence(*app_all_mlp_out.instr_seq, all_token, 10240, 4096, 64, 4, 8); ///M, K, N, k_size
      
    }
    else if(all_token % 256 == 0)
    {   
        accel_desc_all_v.xclbin_name = xclbin_name_all_v;
        accel_desc_all_v.app_name = "all_v";
        accel_desc_all_mlp01.xclbin_name = xclbin_name_all_v;
        accel_desc_all_mlp01.app_name = "all_mlp01";
        accel_desc_all_mlp_out.xclbin_name = xclbin_name_all_v;
        accel_desc_all_mlp_out.app_name = "all_mlp_out";
    
        app_all_v = npu_instance.create_app(accel_desc_all_v);
        app_all_mlp01 = npu_instance.create_app(accel_desc_all_mlp01);
        app_all_mlp_out = npu_instance.create_app(accel_desc_all_mlp_out);
      
        generate_mm_64_double_sequence(*app_all_v.instr_seq, all_token, 4096, 4096, 32, 4, 8); ///M, K, N, k_size
        generate_mm_64_double_sequence(*app_all_mlp01.instr_seq, all_token, 4096, 10240, 32, 4, 8); ///M, K, N, k_size
        generate_mm_64_double_sequence(*app_all_mlp_out.instr_seq, all_token, 10240, 4096, 32, 4, 8); ///M, K, N, k_size
         //std::cout << "all image Divisible by 256\n";
    }
    else{
        padding_all = 256 - (all_token % 256);
        all_token = padding_all + all_token;
        if(all_token % 512 == 0)
        {
            accel_desc_all_v.xclbin_name = qwen_qkv_mm_xclbin;
            accel_desc_all_v.app_name = "all_v";
            accel_desc_all_mlp01.xclbin_name = qwen_qkv_mm_xclbin;
            accel_desc_all_mlp01.app_name = "all_mlp01";
            accel_desc_all_mlp_out.xclbin_name = qwen_qkv_mm_xclbin;
            accel_desc_all_mlp_out.app_name = "all_mlp_out";
            app_all_v = npu_instance.create_app(accel_desc_all_v);
            app_all_mlp01 = npu_instance.create_app(accel_desc_all_mlp01);
            app_all_mlp_out = npu_instance.create_app(accel_desc_all_mlp_out);

            generate_mm_128_double_sequence(*app_all_v.instr_seq, all_token, 4096, 4096, 64, 4, 8); ///M, K, N, k_size
            generate_mm_128_double_sequence(*app_all_mlp01.instr_seq, all_token, 4096, 10240, 64, 4, 8); ///M, K, N, k_size
            generate_mm_128_double_sequence(*app_all_mlp_out.instr_seq, all_token, 10240, 4096, 64, 4, 8); ///M, K, N, k_size
             //std::cout << "padding all image Divisible by both 512 and 256\n";
            // std::cout << " all token length is " << all_token << std::endl;
        }
        else
        {
            accel_desc_all_v.xclbin_name = xclbin_name_all_v;
            accel_desc_all_v.app_name = "all_v";
            accel_desc_all_mlp01.xclbin_name = xclbin_name_all_v;
            accel_desc_all_mlp01.app_name = "all_mlp01";
            accel_desc_all_mlp_out.xclbin_name = xclbin_name_all_v;
            accel_desc_all_mlp_out.app_name = "all_mlp_out";
        
            app_all_v = npu_instance.create_app(accel_desc_all_v);
            app_all_mlp01 = npu_instance.create_app(accel_desc_all_mlp01);
            app_all_mlp_out = npu_instance.create_app(accel_desc_all_mlp_out);
        
            generate_mm_64_double_sequence(*app_all_v.instr_seq, all_token, 4096, 4096, 32, 4, 8); ///M, K, N, k_size
            generate_mm_64_double_sequence(*app_all_mlp01.instr_seq, all_token, 4096, 10240, 32, 4, 8); ///M, K, N, k_size
            generate_mm_64_double_sequence(*app_all_mlp_out.instr_seq, all_token, 10240, 4096, 32, 4, 8); ///M, K, N, k_size
             //std::cout << "padding all image Divisible by 256 " << all_token << std::endl;
            // std::cout << " all token length is " << all_token << std::endl;
        }
        
    }
    
    buffer<float> seq_all_v = app_all_v.instr_seq->dump().cast_to<float>();

    buffer<float> seq_all_mlp01 = app_all_mlp01.instr_seq->dump().cast_to<float>();
    buffer<float> seq_all_mlp_out = app_all_mlp_out.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string silu_xclbin_name = npu_files_path +"/xclbins/mm_mlp_silu_custom.xclbin";
    // std::string silu_xclbin_name = npu_files_path +"/xclbins/mm_test.xclbin";
    npu_app_desc accel_desc_3, accel_desc_all_mlp_silu;
    accel_desc_3.xclbin_name = silu_xclbin_name;
    accel_desc_3.app_name = "noise_mlp_silu";
    accel_desc_all_mlp_silu.xclbin_name = silu_xclbin_name;
    accel_desc_all_mlp_silu.app_name = "all_mlp_silu";
    npu_app app_3 = npu_instance.create_app(accel_desc_3);
    npu_app app_all_mlp_silu = npu_instance.create_app(accel_desc_all_mlp_silu);
    int silu_padding = 0;
    int all_silu_token = all_token; 
    if(all_silu_token % 512 == 0)
    {
        silu_padding = 0;
    }
    else
    {
        silu_padding = 512 - (all_silu_token % 512);
    }
    
    // std::cout << "silu padding data size is " << silu_padding << std::endl;
    all_silu_token = all_silu_token + silu_padding;
   
    generate_mm_128silu_double_sequence(*app_3.instr_seq, all_silu_token, 4096, 10240, 64,  4, 8); ///M, K, N, k_size
    generate_mm_128silu_double_sequence(*app_all_mlp_silu.instr_seq, all_silu_token, 4096, 10240, 64,  4, 8); ///M, K, N, k_size
    // std::cout << "silu padding data size is " << all_silu_token << std::endl;

  
    buffer<float> seq_3 = app_3.instr_seq->dump().cast_to<float>();
    buffer<float> seq_all_mlp_silu = app_all_mlp_silu.instr_seq->dump().cast_to<float>();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string dequan_xclbin_name = npu_files_path + "/xclbins/q41_dequan.xclbin";
    npu_app_desc accel_desc_adaLN_dequan, accel_desc_qkv_dequan, accel_desc_w3_dequan, accel_desc_w2_dequan, accel_desc_out_dequan;
    accel_desc_adaLN_dequan.xclbin_name = dequan_xclbin_name;
    accel_desc_adaLN_dequan.app_name = "adaLN_dequan";
    accel_desc_qkv_dequan.xclbin_name = dequan_xclbin_name;
    accel_desc_qkv_dequan.app_name = "qkv_dequan";
    accel_desc_w3_dequan.xclbin_name = dequan_xclbin_name;
    accel_desc_w3_dequan.app_name = "w3_dequan";
    accel_desc_w2_dequan.xclbin_name = dequan_xclbin_name;
    accel_desc_w2_dequan.app_name = "w2_dequan";
    accel_desc_out_dequan.xclbin_name = dequan_xclbin_name;
    accel_desc_out_dequan.app_name = "out_dequan";
    npu_app app_adaLN_dequan = npu_instance.create_app(accel_desc_adaLN_dequan);
    npu_app app_qkv_dequan = npu_instance.create_app(accel_desc_qkv_dequan);
    npu_app app_w3_dequan = npu_instance.create_app(accel_desc_w3_dequan);
    npu_app app_w2_dequan = npu_instance.create_app(accel_desc_w2_dequan);
    npu_app app_out_dequan = npu_instance.create_app(accel_desc_out_dequan);
    app_adaLN_dequan.instr_seq->from_file(npu_files_path +"/insts/adaLN_dequan.txt");
    app_qkv_dequan.instr_seq->from_file(npu_files_path +"/insts/qkv_dequan.txt");
    app_w3_dequan.instr_seq->from_file(npu_files_path +"/insts/feed_w3_dequan.txt");
    app_w2_dequan.instr_seq->from_file(npu_files_path +"/insts/feed_w2_dequan.txt");
    app_out_dequan.instr_seq->from_file(npu_files_path +"/insts/atten_out_dequan.txt");
    buffer<float> seq_adaLN_dequan = app_adaLN_dequan.instr_seq->dump().cast_to<float>();
    buffer<float> seq_qkv_dequan = app_qkv_dequan.instr_seq->dump().cast_to<float>();
    buffer<float> seq_w3_dequan = app_w3_dequan.instr_seq->dump().cast_to<float>();
    buffer<float> seq_w2_dequan = app_w2_dequan.instr_seq->dump().cast_to<float>();
    buffer<float> seq_out_dequan = app_out_dequan.instr_seq->dump().cast_to<float>();
    ////////////////////////////////////////////////////noise layers/////////////////////////////////////////////////////////////////////////////

    int img_cap_size = cap_total_len + img_seqlen;
    // std::cout << "img_cap size is " << img_cap_size << std::endl;
   
    size_t N_img_all = img_cap_size * 3840;
    std::array<buffer<dtype_in>, 31> all_result_buffers;
   
    for (auto& b : all_result_buffers) {
        b = buffer<dtype_in>(N_img_all); 
        std::memset(b.data(), 0, N_img_all * sizeof(dtype_in));
    }

    std::array<buffer<float>, 8> update_img;
    size_t update_img_size = 16*latent_H*latent_W;
    // buffer<float> update_img(16*latent_H*latent_W);
    for (auto& b : update_img) {
        b = buffer<float>(update_img_size); 
        std::memset(b.data(), 0, (update_img_size) * sizeof(dtype_in));
    }

    {
        
        int64_t start_0 = cap_total_len + 1;
        int64_t size_arr[3]  = {F_tokens, H_tokens, W_tokens};
        int64_t start_arr[3] = {start_0, 0, 0};
        int ndim = 3;
        int64_t total_points = size_arr[0] * size_arr[1] * size_arr[2];
    
        int64_t numel = total_points * ndim;
        int32_t* image_ori_pos_ids = create_coordinate_grid_c(size_arr, start_arr, ndim);
        buffer<int32_t> ids_i32_x(image_ori_pos_ids, numel);
    
        int64_t size_dim0 = cap_total_len;
        int64_t start_arr_cap[3] = {1, 0, 0};
        int64_t size_arr_cap[3]  = {size_dim0, 1, 1};
    
        size_t num_elements = (size_t)(size_dim0 * 3);
        int32_t* ids_i32_cap_test = create_coordinate_grid_c(size_arr_cap, start_arr_cap, 3);
        buffer<int32_t> ids_i32_cap(ids_i32_cap_test, num_elements);
    
        int L_x   = img_seqlen;                // sequence length
        int L_cap   = cap_total_len;                // sequence length
        constexpr int ids_dim = 3;                   // 3 axes
        const int axes_dim[3] = {32, 48, 48};        // ROPE_AXES_DIMS
        const float theta     = 256.0f;              // ROPE_THETA
        std::vector<float> pos_0(L_x), pos_1(L_x), pos_2(L_x);
        for (int i = 0; i < L_x; ++i) {
            pos_0[i] = (float)ids_i32_x[i * 3 + 0];
            pos_1[i] = (float)ids_i32_x[i * 3 + 1];
            pos_2[i] = (float)ids_i32_x[i * 3 + 2];
        }
        std::vector<float> pos_0_cap(L_cap), pos_1_cap(L_cap), pos_2_cap(L_cap);
        for (int i = 0; i < L_cap; ++i) {
            pos_0_cap[i] = (float)ids_i32_cap[i * 3 + 0];
            pos_1_cap[i] = (float)ids_i32_cap[i * 3 + 1];
            pos_2_cap[i] = (float)ids_i32_cap[i * 3 + 2];
        }
        // ---------- 1) run rope_avx for each axis ----------
        int D_pairs0 = axes_dim[0] / 2;  // 16
        int D_pairs1 = axes_dim[1] / 2;  // 24
        int D_pairs2 = axes_dim[2] / 2;  // 24
    
        std::vector<float> pos_all_0(L_x * D_pairs0 * 4);
        std::vector<float> pos_all_1(L_x * D_pairs1 * 4);
        std::vector<float> pos_all_2(L_x * D_pairs2 * 4);
        std::vector<float> pos_all_0_cap(L_cap * D_pairs0 * 4);
        std::vector<float> pos_all_1_cap(L_cap * D_pairs1 * 4);
        std::vector<float> pos_all_2_cap(L_cap * D_pairs2 * 4);
    
        rope_avx(pos_0, L_x, axes_dim[0], theta, pos_all_0);
        rope_avx(pos_1, L_x, axes_dim[1], theta, pos_all_1);
        rope_avx(pos_2, L_x, axes_dim[2], theta, pos_all_2);
        rope_avx(pos_0_cap, L_cap, axes_dim[0], theta, pos_all_0_cap);
        rope_avx(pos_1_cap, L_cap, axes_dim[1], theta, pos_all_1_cap);
        rope_avx(pos_2_cap, L_cap, axes_dim[2], theta, pos_all_2_cap);
        // ---------- 2) merge axes into pos_all ----------
        int D_pairs = D_pairs0 + D_pairs1 + D_pairs2;   // 64
        int D_head  = 2 * D_pairs;                      // 128
        int stride_pos_all = D_pairs * 4;               // 4 floats per pair → 256
    
        buffer<float> pos_all(L_x * stride_pos_all);
        buffer<float> pos_all_cap(L_cap * stride_pos_all);
        for (int i = 0; i < L_x; ++i) {
            float *row = pos_all.data() + (size_t)i * stride_pos_all;
    
            std::memcpy(row,
                        &pos_all_0[i * D_pairs0 * 4],
                        (size_t)D_pairs0 * 4 * sizeof(float));
            std::memcpy(row + D_pairs0 * 4,
                        &pos_all_1[i * D_pairs1 * 4],
                        (size_t)D_pairs1 * 4 * sizeof(float));
            std::memcpy(row + (D_pairs0 + D_pairs1) * 4,
                        &pos_all_2[i * D_pairs2 * 4],
                        (size_t)D_pairs2 * 4 * sizeof(float));
        }
        for (int i = 0; i < L_cap; ++i) {
            float *row = pos_all_cap.data() + (size_t)i * stride_pos_all;
    
            std::memcpy(row,
                        &pos_all_0_cap[i * D_pairs0 * 4],
                        (size_t)D_pairs0 * 4 * sizeof(float));
            std::memcpy(row + D_pairs0 * 4,
                        &pos_all_1_cap[i * D_pairs1 * 4],
                        (size_t)D_pairs1 * 4 * sizeof(float));
            std::memcpy(row + (D_pairs0 + D_pairs1) * 4,
                        &pos_all_2_cap[i * D_pairs2 * 4],
                        (size_t)D_pairs2 * 4 * sizeof(float));
        }
        // ---------- 3) build pos_c_all / pos_s_all ----------
        buffer<float> pos_c_all;
        buffer<float> pos_s_all;
        build_pos_cs_from_pos_all(pos_all, pos_c_all, pos_s_all, L_x, D_head);
        buffer<float> pos_c_all_cap;
        buffer<float> pos_s_all_cap;
        build_pos_cs_from_pos_all(pos_all_cap, pos_c_all_cap, pos_s_all_cap, L_cap, D_head);
        int L_all = L_x + L_cap;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        buffer<float> pos_c_all_cat(D_head*L_all);
        buffer<float> pos_s_all_cat(D_head*L_all);
        for(int i = 0; i < L_x; i++){
            for(int j = 0; j < D_head; j++){
                pos_c_all_cat[i*D_head + j] = pos_c_all[i*D_head + j];
                pos_s_all_cat[i*D_head + j] = pos_s_all[i*D_head + j];
            }
        }
        for(int i = 0; i < L_cap; i++){
            for(int j = 0; j < D_head; j++){
                pos_c_all_cat[(L_x + i)*D_head + j] = pos_c_all_cap[i*D_head + j];
                pos_s_all_cat[(L_x + i)*D_head + j] = pos_s_all_cap[i*D_head + j];
            }
        }
    size_t Ntxt = cap_total_len * 4096;
    size_t Nimg = img_seqlen * 3840;

    std::array<buffer<dtype_in>, 3> cap_result_buffers;
    std::array<buffer<dtype_in>, 3> img_result_buffers;

    for (auto& b : cap_result_buffers) {
        b = buffer<dtype_in>(Ntxt); 
        std::memset(b.data(), 0, Ntxt * sizeof(dtype_in));
    }
    
    // 2. img_result_buffers
    for (auto& b : img_result_buffers) {
        b = buffer<dtype_in>(Nimg); 
        std::memset(b.data(), 0, Nimg * sizeof(dtype_in));
    }


    buffer<dtype_in> noise_qk_A_in = app_0.create_bo_buffer<dtype_in>(img_seqlen*4096, 3);
    buffer<dtype_in> noise_q_C_out = app_0.create_bo_buffer<dtype_in>(img_seqlen*4096, 5);
    buffer<dtype_in> noise_k_C_out = app_0.create_bo_buffer<dtype_in>(img_seqlen*4096, 5);
    buffer<dtype_in> noise_v_C_out = app_0.create_bo_buffer<dtype_in>(img_seqlen*4096, 5);
    buffer<dtype_in> noise_out_C_out = app_0.create_bo_buffer<dtype_in>(img_seqlen*4096, 5);

    buffer<dtype_out> noise_adaLN_in = app_adaLN.create_bo_buffer<dtype_out>(256*256, 3);
    buffer<dtype_in> noise_adaLN_dequan = app_adaLN.create_bo_buffer<dtype_in>(15360*256, 4);
    buffer<dtype_in> noise_adaLN_out = app_adaLN.create_bo_buffer<dtype_in>(15360*256, 5);

    buffer<dtype_in> noise_q_attn = app_attn_noise.create_bo_buffer<dtype_in>(32*128*128*32, 3);
    buffer<dtype_in> noise_kv_attn = app_attn_noise.create_bo_buffer<dtype_in>(64*128*64*32*2, 4);
    buffer<dtype_in> noise_out_attn = app_attn_noise.create_bo_buffer<dtype_in>(32*128*128*32, 5);
   
    buffer<dtype_in> noise_mlp0 = app_3.create_bo_buffer<dtype_in>(img_seqlen*4096, 3);
    buffer<dtype_in> noise_mlp0_out = app_3.create_bo_buffer<dtype_in>(img_seqlen*10240, 5);

    buffer<dtype_in> noise_mlp1 = app_1.create_bo_buffer<dtype_in>(img_seqlen*4096, 3);
    buffer<dtype_in> noise_mlp1_out = app_1.create_bo_buffer<dtype_in>(img_seqlen*10240, 5);
   
    buffer<dtype_in> noise_mlp2_in = app_2.create_bo_buffer<dtype_in>(img_seqlen*10240, 3);
    buffer<dtype_in> noise_mlp2_out = app_2.create_bo_buffer<dtype_in>(img_seqlen*4096, 5);  //7.81GB
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    buffer<dtype_in> cap_qk_A_in = app_cap_qkv.create_bo_buffer<dtype_in>(512*4096, 3);
    buffer<dtype_in> cap_q_C_out = app_cap_qkv.create_bo_buffer<dtype_in>(512*4096, 5);
    buffer<dtype_in> cap_k_C_out = app_cap_qkv.create_bo_buffer<dtype_in>(512*4096, 5);
    buffer<dtype_in> cap_v_C_out = app_cap_qkv.create_bo_buffer<dtype_in>(512*4096, 5);
    buffer<dtype_in> cap_out_C_out = app_cap_qkv.create_bo_buffer<dtype_in>(512*4096, 5);

    buffer<dtype_in> cap_q_attn = app_attn_cap.create_bo_buffer<dtype_in>(32*128*16*30, 3);
    buffer<dtype_in> cap_kv_attn = app_attn_cap.create_bo_buffer<dtype_in>(64*128*num_kv*30*2, 4);
    buffer<dtype_in> cap_out_attn = app_attn_cap.create_bo_buffer<dtype_in>(32*128*16*30, 5);
   
    buffer<dtype_in> cap_mlp0 = app_cap_mlp01.create_bo_buffer<dtype_in>(512*4096, 3);
    buffer<dtype_in> cap_mlp0_out = app_cap_mlp01.create_bo_buffer<dtype_in>(512*10240, 5);
    buffer<dtype_in> cap_mlp1_out = app_cap_mlp01.create_bo_buffer<dtype_in>(512*10240, 5);

    buffer<dtype_in> cap_mlp2_in = app_cap_mlp_out.create_bo_buffer<dtype_in>(512*10240, 3);
    buffer<dtype_in> cap_mlp2_out = app_cap_mlp_out.create_bo_buffer<dtype_in>(512*4096, 5); //

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    buffer<dtype_in> all_v_A_in = app_all_v.create_bo_buffer<dtype_in>(all_token*4096, 3);
    buffer<dtype_in> all_q_dequan = app_all_v.create_bo_buffer<dtype_in>((4096+0)*4096, 4);
    buffer<dtype_in> all_k_dequan = app_all_v.create_bo_buffer<dtype_in>((4096+0)*4096, 4);
    buffer<dtype_in> all_v_dequan = app_all_v.create_bo_buffer<dtype_in>(4096*4096, 4);
    buffer<dtype_in> all_out_dequan = app_all_v.create_bo_buffer<dtype_in>(4096*4096, 4);
    buffer<dtype_in> all_q_C_out = app_all_v.create_bo_buffer<dtype_in>(all_token*4096, 5);
    buffer<dtype_in> all_k_C_out = app_all_v.create_bo_buffer<dtype_in>(all_token*4096, 5);
    buffer<dtype_in> all_v_C_out = app_all_v.create_bo_buffer<dtype_in>(all_token*4096, 5);
    buffer<dtype_in> all_out_C_out = app_all_v.create_bo_buffer<dtype_in>(all_token*4096, 5);

    buffer<dtype_in> all_q_attn = app_attn_all.create_bo_buffer<dtype_in>(32*128*144*32, 3);
    buffer<dtype_in> all_kv_attn = app_attn_all.create_bo_buffer<dtype_in>(64*128*num_kv_all*32*2, 4);
    buffer<dtype_in> all_out_attn = app_attn_all.create_bo_buffer<dtype_in>(32*128*144*32, 5);

    buffer<dtype_in> all_mlp0 = app_all_mlp_silu.create_bo_buffer<dtype_in>(all_silu_token*4096, 3);
    buffer<dtype_in> all_mlp0_dequan = app_all_mlp_silu.create_bo_buffer<dtype_in>(4096*10240, 4);
    buffer<dtype_in> all_mlp0_out = app_all_mlp_silu.create_bo_buffer<dtype_in>(all_silu_token*10240, 5);

    buffer<dtype_in> all_mlp1 = app_all_mlp01.create_bo_buffer<dtype_in>(all_token*4096, 3);
    buffer<dtype_in> all_mlp1_dequan = app_all_mlp01.create_bo_buffer<dtype_in>(4096*10240, 4);
    buffer<dtype_in> all_mlp1_out = app_all_mlp01.create_bo_buffer<dtype_in>(all_token*10240, 5);

    buffer<dtype_in> all_mlp2_in = app_all_mlp_out.create_bo_buffer<dtype_in>(all_token*10240, 3);
    buffer<dtype_in> all_mlp2_dequan = app_all_mlp_out.create_bo_buffer<dtype_in>(4096*10240, 4);
    buffer<dtype_in> all_mlp2_out = app_all_mlp_out.create_bo_buffer<dtype_in>(all_token*4096, 5);/////7.66GB
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////load weights//////////////////////////////////////////////////////////////////////////
    struct LayerNorm {
        std::vector<uint16_t> noise_atten_norm1_u16;
        std::vector<uint16_t> noise_atten_norm2_u16;
        std::vector<uint16_t> noise_q_norm_weight_u16;
        std::vector<uint16_t> noise_k_norm_weight_u16;
        std::vector<uint16_t> noise_ffn_norm1_u16;
        std::vector<uint16_t> noise_ffn_norm2_u16;

        std::vector<uint16_t> cap_atten_norm1_u16;
        std::vector<uint16_t> cap_atten_norm2_u16;
        std::vector<uint16_t> cap_q_norm_weight_u16;
        std::vector<uint16_t> cap_k_norm_weight_u16;
        std::vector<uint16_t> cap_ffn_norm1_u16;
        std::vector<uint16_t> cap_ffn_norm2_u16;
    };
    constexpr int NUM_LAYERS = 2;
    std::vector<LayerNorm> noise_layer(NUM_LAYERS);


    struct LayerNorm_all{
        std::vector<uint16_t> all_atten_norm1_u16;
        std::vector<uint16_t> all_atten_norm2_u16;
        std::vector<uint16_t> all_q_norm_weight_u16;
        std::vector<uint16_t> all_k_norm_weight_u16;
        std::vector<uint16_t> all_ffn_norm1_u16;
        std::vector<uint16_t> all_ffn_norm2_u16;
    };
    std::vector<LayerNorm_all> all_layer(30);
    struct bo_noise_buffer_struct{
        buffer<dtype_in> noise_adaLN_weight;
        buffer<dtype_in> noise_adaLN_bias;
        std::array<buffer<dtype_in>, 2> B_in_qk;
        buffer<dtype_in> B_in_v;
        buffer<dtype_in> B_in_out;
        buffer<dtype_in> B_in_mlp0;
        buffer<dtype_in> B_in_mlp1;
        buffer<dtype_in> B_in_mlp2;

    };
    std::vector<bo_noise_buffer_struct> noise_buffers(2);
    struct bo_cap_buffer_struct{
        std::array<buffer<dtype_in>, 2> B_in_qk;
        buffer<dtype_in> B_in_v;
        buffer<dtype_in> B_in_out;
        buffer<dtype_in> B_in_mlp0;
        buffer<dtype_in> B_in_mlp1;
        buffer<dtype_in> B_in_mlp2;

    };
    std::vector<bo_cap_buffer_struct> cap_buffers(2);

    struct bo_all_buffer_struct{
        buffer<dtype_in> all_adaLN_weight;
        buffer<dtype_in> all_adaLN_bias;
        std::array<buffer<dtype_in>, 2> B_in_qk;
        buffer<dtype_in> B_in_v;
        buffer<dtype_in> B_in_out;
        buffer<dtype_in> B_in_mlp0;
        buffer<dtype_in> B_in_mlp1;
        buffer<dtype_in> B_in_mlp2;

    };
    std::vector<bo_all_buffer_struct> all_buffers(30);
    for (int i = 0; i < 2; i++){
        fs::path noise_adaLN_bias_path = make_numbered_file(base_dir, "noise_refiner_", i, "_adaLN_modulation_0_bias_bf16_u16.bin");
        fs::path noise_adaLN_weight_path = make_numbered_file(base_dir, "noise_refiner_", i, "_adaLN_modulation_0_weight_bf16_u16.bin");
        std::vector<uint16_t> noise_adaLN_weight_u16(15360*256);
        std::vector<uint16_t> noise_adaLN_bias_u16(15360);
        io::read_data_from_files(noise_adaLN_weight_path.c_str(), noise_adaLN_weight_u16, 15360*256);
        io::read_data_from_files(noise_adaLN_bias_path.c_str(), noise_adaLN_bias_u16, 15360);
        noise_buffers[i].noise_adaLN_weight = app_adaLN.create_bo_buffer<dtype_in>(15360*256, 4);
        noise_buffers[i].noise_adaLN_bias.resize(15360);
        copy_u16_to_bf16_avx512(noise_adaLN_weight_u16.data(), noise_buffers[i].noise_adaLN_weight.data(), 15360*256);
        copy_u16_to_bf16_avx512(noise_adaLN_bias_u16.data(), noise_buffers[i].noise_adaLN_bias.data(), 15360);

        fs::path noise_atten_norm1_path = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_norm1_weight_bf16_u16.bin");
        fs::path noise_atten_norm2_path = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_norm2_weight_bf16_u16.bin");
        noise_layer[i].noise_atten_norm1_u16.resize(3840);
        noise_layer[i].noise_atten_norm2_u16.resize(3840);
        io::read_data_from_files(noise_atten_norm1_path.c_str(), noise_layer[i].noise_atten_norm1_u16, 3840);
        io::read_data_from_files(noise_atten_norm2_path.c_str(), noise_layer[i].noise_atten_norm2_u16, 3840);
    
        fs::path noise_q_norm_weight = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_norm_q_weight_bf16_u16.bin");
        fs::path noise_k_norm_weight = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_norm_k_weight_bf16_u16.bin");
        noise_layer[i].noise_q_norm_weight_u16.resize(128);
        noise_layer[i].noise_k_norm_weight_u16.resize(128);
        io::read_data_from_files(noise_q_norm_weight.c_str(), noise_layer[i].noise_q_norm_weight_u16, 128);
        io::read_data_from_files(noise_k_norm_weight.c_str(), noise_layer[i].noise_k_norm_weight_u16, 128);
        
        fs::path noise_q_weight = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_to_q_weight_bf16_u16.bin");
        fs::path noise_k_weight = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_to_k_weight_bf16_u16.bin");
        fs::path noise_v_weight = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_to_v_weight_bf16_u16.bin");
        fs::path noise_out_weight = make_numbered_file(base_dir, "noise_refiner_", i, "_attention_to_out_0_weight_bf16_u16.bin");
        std::vector<uint16_t> noise_q_weight_u16(3840*3840);
        std::vector<uint16_t> noise_k_weight_u16(3840*3840);
        std::vector<uint16_t> noise_v_weight_u16(3840*3840);
        std::vector<uint16_t> noise_out_weight_u16(3840*3840);
        io::read_data_from_files(noise_q_weight.c_str(), noise_q_weight_u16, 3840*3840);
        io::read_data_from_files(noise_k_weight.c_str(), noise_k_weight_u16, 3840*3840);
        io::read_data_from_files(noise_v_weight.c_str(), noise_v_weight_u16, 3840*3840);
        io::read_data_from_files(noise_out_weight.c_str(), noise_out_weight_u16, 3840*3840);
        noise_buffers[i].B_in_qk[0] = app_0.create_bo_buffer<dtype_in>(4096*4096, 4);
        noise_buffers[i].B_in_qk[1] = app_0.create_bo_buffer<dtype_in>(4096*4096, 4);
        noise_buffers[i].B_in_v = app_0.create_bo_buffer<dtype_in>(4096*4096, 4);
        noise_buffers[i].B_in_out = app_0.create_bo_buffer<dtype_in>(4096*4096, 4);
        pad_square_bf16_3840_to_4096(noise_q_weight_u16, noise_buffers[i].B_in_qk[0], 0);
        pad_square_bf16_3840_to_4096(noise_k_weight_u16, noise_buffers[i].B_in_qk[1], 0);
        pad_square_bf16_3840_to_4096(noise_v_weight_u16, noise_buffers[i].B_in_v, 0);
        pad_square_bf16_3840_to_4096(noise_out_weight_u16, noise_buffers[i].B_in_out, 0);

        fs::path noise_ffn_norm1_path = make_numbered_file(base_dir, "noise_refiner_", i, "_ffn_norm1_weight_bf16_u16.bin");
        fs::path noise_ffn_norm2_path = make_numbered_file(base_dir, "noise_refiner_", i, "_ffn_norm2_weight_bf16_u16.bin");
        noise_layer[i].noise_ffn_norm1_u16.resize(3840);
        noise_layer[i].noise_ffn_norm2_u16.resize(3840);
        io::read_data_from_files(noise_ffn_norm1_path.c_str(), noise_layer[i].noise_ffn_norm1_u16, 3840);
        io::read_data_from_files(noise_ffn_norm2_path.c_str(), noise_layer[i].noise_ffn_norm2_u16, 3840);

        fs::path noise_ffn_weight1_path = make_numbered_file(base_dir, "noise_refiner_", i, "_feed_forward_w1_weight_bf16_u16.bin");
        fs::path noise_ffn_weight2_path = make_numbered_file(base_dir, "noise_refiner_", i, "_feed_forward_w3_weight_bf16_u16.bin");
        std::vector<uint16_t> noise_ffn_weight1_u16(3840*10240);
        std::vector<uint16_t> noise_ffn_weight2_u16(3840*10240);
        io::read_data_from_files(noise_ffn_weight1_path.c_str(), noise_ffn_weight1_u16, 3840*10240);
        io::read_data_from_files(noise_ffn_weight2_path.c_str(), noise_ffn_weight2_u16, 3840*10240);
        noise_buffers[i].B_in_mlp0 = app_3.create_bo_buffer<dtype_in>(4096*10240, 4);
        // noise_buffers[i].B_in_mlp0 = app_1.create_bo_buffer<dtype_in>(4096*10240, 4);
        noise_buffers[i].B_in_mlp1 = app_1.create_bo_buffer<dtype_in>(4096*10240, 4);
        copy_u16_to_bf16_avx512(noise_ffn_weight1_u16.data(), noise_buffers[i].B_in_mlp0.data(), 3840*10240);
        copy_u16_to_bf16_avx512(noise_ffn_weight2_u16.data(), noise_buffers[i].B_in_mlp1.data(), 3840*10240);

        fs::path noise_mlp_out_path = make_numbered_file(base_dir, "noise_refiner_", i, "_feed_forward_w2_weight_bf16_u16.bin");
        std::vector<uint16_t> noise_mlp_out_u16(3840*10240);
        io::read_data_from_files(noise_mlp_out_path.c_str(), noise_mlp_out_u16, 3840*10240);
        noise_buffers[i].B_in_mlp2 = app_2.create_bo_buffer<dtype_in>(4096*10240, 4);
        pad_bf16_10240x3840_to_10240x4096(noise_mlp_out_u16, noise_buffers[i].B_in_mlp2, 0);
        ////////////////////////////////////////////////////cap layer////////////////////////////////////////////////////
        fs::path cap_atten_norm1_path = make_numbered_file(base_dir, "context_refiner_", i, "_attention_norm1_weight_bf16_u16.bin");
        fs::path cap_atten_norm2_path = make_numbered_file(base_dir, "context_refiner_", i, "_attention_norm2_weight_bf16_u16.bin");
        noise_layer[i].cap_atten_norm1_u16.resize(3840);
        noise_layer[i].cap_atten_norm2_u16.resize(3840);
        io::read_data_from_files(cap_atten_norm1_path.c_str(), noise_layer[i].cap_atten_norm1_u16, 3840);
        io::read_data_from_files(cap_atten_norm2_path.c_str(), noise_layer[i].cap_atten_norm2_u16, 3840);
        fs::path cap_q_norm_weight = make_numbered_file(base_dir, "context_refiner_", i, "_attention_norm_q_weight_bf16_u16.bin");
        fs::path cap_k_norm_weight = make_numbered_file(base_dir, "context_refiner_", i, "_attention_norm_k_weight_bf16_u16.bin");
        noise_layer[i].cap_q_norm_weight_u16.resize(128);
        noise_layer[i].cap_k_norm_weight_u16.resize(128);
        io::read_data_from_files(cap_q_norm_weight.c_str(), noise_layer[i].cap_q_norm_weight_u16, 128);
        io::read_data_from_files(cap_k_norm_weight.c_str(), noise_layer[i].cap_k_norm_weight_u16, 128);
        fs::path cap_q_weight = make_numbered_file(base_dir, "context_refiner_", i, "_attention_to_q_weight_bf16_u16.bin");
        fs::path cap_k_weight = make_numbered_file(base_dir, "context_refiner_", i, "_attention_to_k_weight_bf16_u16.bin");
        fs::path cap_v_weight = make_numbered_file(base_dir, "context_refiner_", i, "_attention_to_v_weight_bf16_u16.bin");
        fs::path cap_out_weight = make_numbered_file(base_dir, "context_refiner_", i, "_attention_to_out_0_weight_bf16_u16.bin");
        std::vector<uint16_t> cap_q_weight_u16(3840*3840);
        std::vector<uint16_t> cap_k_weight_u16(3840*3840);
        std::vector<uint16_t> cap_v_weight_u16(3840*3840);
        std::vector<uint16_t> cap_out_weight_u16(3840*3840);
        io::read_data_from_files(cap_q_weight.c_str(), cap_q_weight_u16, 3840*3840);
        io::read_data_from_files(cap_k_weight.c_str(), cap_k_weight_u16, 3840*3840);
        io::read_data_from_files(cap_v_weight.c_str(), cap_v_weight_u16, 3840*3840);
        io::read_data_from_files(cap_out_weight.c_str(), cap_out_weight_u16, 3840*3840);
        cap_buffers[i].B_in_qk[0] = app_cap_qkv.create_bo_buffer<dtype_in>(4096*4096, 4);
        cap_buffers[i].B_in_qk[1] = app_cap_qkv.create_bo_buffer<dtype_in>(4096*4096, 4);
        cap_buffers[i].B_in_v = app_cap_qkv.create_bo_buffer<dtype_in>(4096*4096, 4);
        cap_buffers[i].B_in_out = app_cap_qkv.create_bo_buffer<dtype_in>(4096*4096, 4);
        pad_square_bf16_3840_to_4096(cap_q_weight_u16, cap_buffers[i].B_in_qk[0], 0);
        pad_square_bf16_3840_to_4096(cap_k_weight_u16, cap_buffers[i].B_in_qk[1], 0);
        pad_square_bf16_3840_to_4096(cap_v_weight_u16, cap_buffers[i].B_in_v, 0);
        pad_square_bf16_3840_to_4096(cap_out_weight_u16, cap_buffers[i].B_in_out, 0);

        fs::path cap_ffn_norm1_path = make_numbered_file(base_dir, "context_refiner_", i, "_ffn_norm1_weight_bf16_u16.bin");
        fs::path cap_ffn_norm2_path = make_numbered_file(base_dir, "context_refiner_", i, "_ffn_norm2_weight_bf16_u16.bin");
        noise_layer[i].cap_ffn_norm1_u16.resize(3840);
        noise_layer[i].cap_ffn_norm2_u16.resize(3840);
        io::read_data_from_files(cap_ffn_norm1_path.c_str(), noise_layer[i].cap_ffn_norm1_u16, 3840);
        io::read_data_from_files(cap_ffn_norm2_path.c_str(), noise_layer[i].cap_ffn_norm2_u16, 3840);
      
        fs::path cap_ffn_weight1_path = make_numbered_file(base_dir, "context_refiner_", i, "_feed_forward_w1_weight_bf16_u16.bin");
        fs::path cap_ffn_weight2_path = make_numbered_file(base_dir, "context_refiner_", i, "_feed_forward_w3_weight_bf16_u16.bin");
        std::vector<uint16_t> cap_ffn_weight1_u16(3840*10240);
        std::vector<uint16_t> cap_ffn_weight2_u16(3840*10240);
        io::read_data_from_files(cap_ffn_weight1_path.c_str(), cap_ffn_weight1_u16, 3840*10240);
        io::read_data_from_files(cap_ffn_weight2_path.c_str(), cap_ffn_weight2_u16, 3840*10240);
        cap_buffers[i].B_in_mlp0 = app_cap_mlp01.create_bo_buffer<dtype_in>(4096*10240, 4);
        cap_buffers[i].B_in_mlp1 = app_cap_mlp01.create_bo_buffer<dtype_in>(4096*10240, 4);

        copy_u16_to_bf16_avx512(cap_ffn_weight1_u16.data(), cap_buffers[i].B_in_mlp0.data(), 3840*10240);
        copy_u16_to_bf16_avx512(cap_ffn_weight2_u16.data(), cap_buffers[i].B_in_mlp1.data(), 3840*10240);

        fs::path cap_mlp_out_path = make_numbered_file(base_dir, "context_refiner_", i, "_feed_forward_w2_weight_bf16_u16.bin");
        std::vector<uint16_t> cap_mlp_out_u16(3840*10240);
        io::read_data_from_files(cap_mlp_out_path.c_str(), cap_mlp_out_u16, 3840*10240);
        cap_buffers[i].B_in_mlp2 = app_cap_mlp_out.create_bo_buffer<dtype_in>(4096*10240, 4);
        pad_bf16_10240x3840_to_10240x4096(cap_mlp_out_u16, cap_buffers[i].B_in_mlp2, 0);
      
    }
    
    for(int i = 0; i < 30; i++){
        fs::path all_adaLN_bias_path = make_numbered_file(base_dir, "layers_", i, "_adaLN_modulation_0_bias_bf16_u16.bin");
        std::vector<uint16_t> all_adaLN_bias_u16(15360);
        io::read_data_from_files(all_adaLN_bias_path.c_str(), all_adaLN_bias_u16, 15360);
        all_buffers[i].all_adaLN_bias.resize(15360);
        copy_u16_to_bf16_avx512(all_adaLN_bias_u16.data(), all_buffers[i].all_adaLN_bias.data(), 15360);

        fs::path all_atten_norm1_path = make_numbered_file(base_dir, "layers_", i, "_attention_norm1_weight_bf16_u16.bin");
        fs::path all_atten_norm2_path = make_numbered_file(base_dir, "layers_", i, "_attention_norm2_weight_bf16_u16.bin");
        all_layer[i].all_atten_norm1_u16.resize(3840);
        all_layer[i].all_atten_norm2_u16.resize(3840);
        io::read_data_from_files(all_atten_norm1_path.c_str(), all_layer[i].all_atten_norm1_u16, 3840);
        io::read_data_from_files(all_atten_norm2_path.c_str(), all_layer[i].all_atten_norm2_u16, 3840);
    
        fs::path all_q_norm_weight = make_numbered_file(base_dir, "layers_", i, "_attention_norm_q_weight_bf16_u16.bin");
        fs::path all_k_norm_weight = make_numbered_file(base_dir, "layers_", i, "_attention_norm_k_weight_bf16_u16.bin");
        all_layer[i].all_q_norm_weight_u16.resize(128);
        all_layer[i].all_k_norm_weight_u16.resize(128);
        io::read_data_from_files(all_q_norm_weight.c_str(), all_layer[i].all_q_norm_weight_u16, 128);
        io::read_data_from_files(all_k_norm_weight.c_str(), all_layer[i].all_k_norm_weight_u16, 128);
        
        fs::path all_ffn_norm1_path = make_numbered_file(base_dir, "layers_", i, "_ffn_norm1_weight_bf16_u16.bin");
        fs::path all_ffn_norm2_path = make_numbered_file(base_dir, "layers_", i, "_ffn_norm2_weight_bf16_u16.bin");
        all_layer[i].all_ffn_norm1_u16.resize(3840);
        all_layer[i].all_ffn_norm2_u16.resize(3840);
        io::read_data_from_files(all_ffn_norm1_path.c_str(), all_layer[i].all_ffn_norm1_u16, 3840);
        io::read_data_from_files(all_ffn_norm2_path.c_str(), all_layer[i].all_ffn_norm2_u16, 3840);      
    }
    
    fs::path first_mm_weight_path = base_dir / "all_x_embedder_2-1_weight_bf16_u16.bin";
    fs::path first_mm_bias_path = base_dir / "all_x_embedder_2-1_bias_bf16_u16.bin";
    std::vector<uint16_t> first_mm_weight_u16(3840*64);
    std::vector<uint16_t> first_mm_bias_u16(3840);
    io::read_data_from_files(first_mm_weight_path.c_str(), first_mm_weight_u16, 3840*64);
    io::read_data_from_files(first_mm_bias_path.c_str(), first_mm_bias_u16, 3840);
    buffer<dtype_in> first_mm_in = app_first_mm.create_bo_buffer<dtype_in>(4096*64, 3);
    buffer<dtype_in> first_mm_weight = app_first_mm.create_bo_buffer<dtype_in>(64*4096, 4);
    buffer<dtype_in> first_mm_out = app_first_mm.create_bo_buffer<dtype_in>(4096*4096, 5);
    buffer<dtype_in> first_mm_bias(3840);
    for (size_t i = 0; i < 64; ++i) {
        // source row: 3840 uint16_t values
        const uint16_t* src_row = (first_mm_weight_u16.data() + i * 3840);
        uint16_t* dst_row = reinterpret_cast<uint16_t*>(first_mm_weight.data() + i * 4096);
        std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
    }
    copy_u16_to_bf16_avx512(first_mm_bias_u16.data(), first_mm_bias.data(), 3840);

    fs::path final_adaLN_bias_path = base_dir / "all_final_layer_2-1_adaLN_modulation_1_bias_bf16_u16.bin";
    fs::path final_adaLN_weight_path = base_dir / "all_final_layer_2-1_adaLN_modulation_1_weight_bf16_u16.bin";
    std::vector<uint16_t> final_adaLN_weight_u16(3840*256);
    std::vector<uint16_t> final_adaLN_bias_u16(3840);
    io::read_data_from_files(final_adaLN_weight_path.c_str(), final_adaLN_weight_u16, 3840*256);
    io::read_data_from_files(final_adaLN_bias_path.c_str(), final_adaLN_bias_u16, 3840);
    buffer<dtype_in> final_adaLN_in = app_adaLN_final.create_bo_buffer<dtype_in>(256*256, 3);
    buffer<dtype_in> final_adaLN_weight = app_adaLN_final.create_bo_buffer<dtype_in>(4096*256, 4);
    buffer<dtype_in> final_adaLN_out = app_adaLN_final.create_bo_buffer<dtype_in>(256*4096, 5);
    buffer<dtype_in> final_adaLN_bias(3840);
    for (size_t i = 0; i < 256; ++i) {
        // source row: 3840 uint16_t values
        const uint16_t* src_row = (final_adaLN_weight_u16.data() + i * 3840);
        uint16_t* dst_row = reinterpret_cast<uint16_t*>(final_adaLN_weight.data() + i * 4096);
        std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
    }
    copy_u16_to_bf16_avx512(final_adaLN_bias_u16.data(), final_adaLN_bias.data(), 3840);

    fs::path final_mm_weight_path = base_dir / "all_final_layer_2-1_linear_weight_bf16_u16.bin";
    fs::path final_mm_bias_path = base_dir / "all_final_layer_2-1_linear_bias_bf16_u16.bin";
    std::vector<uint16_t> final_mm_weight_u16(3840*64);
    std::vector<uint16_t> final_mm_bias_u16(64);
    io::read_data_from_files(final_mm_weight_path.c_str(), final_mm_weight_u16, 3840*64);
    io::read_data_from_files(final_mm_bias_path.c_str(), final_mm_bias_u16, 64);
    buffer<dtype_in> final_mm_in = app_final_mm.create_bo_buffer<dtype_in>(4352*3840, 3);
    buffer<dtype_in> final_mm_weight = app_final_mm.create_bo_buffer<dtype_in>(3840*512, 4);
    buffer<dtype_in> final_mm_out = app_final_mm.create_bo_buffer<dtype_in>(4352*512, 5);
    buffer<dtype_in> final_adaLN_linear_bias(64);
    for (size_t i = 0; i < 3840; ++i) {
        // source row: 3840 uint16_t values
        const uint16_t* src_row = (final_mm_weight_u16.data() + i * 64);
        uint16_t* dst_row = reinterpret_cast<uint16_t*>(final_mm_weight.data() + i * 512);
        std::memcpy(dst_row, src_row, 64 * sizeof(uint16_t));
    }
    copy_u16_to_bf16_avx512(final_mm_bias_u16.data(), final_adaLN_linear_bias.data(), 64);
    
    ///////////////////////////////////////////////////10.65GB///////////////////////////////////////////////////////////////////////
    
    struct q4_1_buffer_struct{
        buffer<dequantize_params> dequan_adaLN_B;
        buffer<dequantize_params> dequan_qkv_B;
        buffer<dequantize_params> dequan_w3_B;
        buffer<dequantize_params> dequan_w1_B;
        buffer<dequantize_params> dequan_w2_B;
        buffer<dequantize_params> dequan_out_B;
    };
    std::vector<q4_1_buffer_struct> all_q41_buffers(30);

    
    int adaLN_group_size = 15360*256/128/32;
    int qkv_group_size = 12288*4096/128/32;
    int w3_group_size = 10240*4096/128/32;
    int out_group_size = 4096*4096/128/32;

    buffer<dtype_in> dequan_adaLN_out = app_adaLN_dequan.create_bo_buffer<dtype_in>(15360*256, 4);
    buffer<dtype_in> dequan_qkv_out = app_qkv_dequan.create_bo_buffer<dtype_in>(12288*4096, 4);
    buffer<dtype_in> dequan_w3_out = app_w3_dequan.create_bo_buffer<dtype_in>(10240*4096, 4);
    buffer<dtype_in> dequan_w1_out = app_w3_dequan.create_bo_buffer<dtype_in>(10240*4096, 4);
    buffer<dtype_in> dequan_w2_out = app_w2_dequan.create_bo_buffer<dtype_in>(10240*4096, 4);
    buffer<dtype_in> dequan_out = app_out_dequan.create_bo_buffer<dtype_in>(4096*4096, 4);

    for(int i = 0; i < 30; i++){       
        fs::path all_adaLN_scale_path = make_numbered_file(std::filesystem::path(weights_path)/"Z-Image-Turbo/denoising/q41_weights", "layer_", i, "_adaLN_packed_qx_scale_min.bin");
        all_q41_buffers[i].dequan_adaLN_B = app_adaLN_dequan.create_bo_buffer<dequantize_params>(adaLN_group_size, 3);
        load_params_bin_to_buffer(all_adaLN_scale_path.c_str(), all_q41_buffers[i].dequan_adaLN_B, adaLN_group_size);
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        fs::path all_qkv_scale_path = make_numbered_file(std::filesystem::path(weights_path)/"Z-Image-Turbo/denoising/q41_weights", "layer_", i, "_atten_qkv_packed_qx_scale_min.bin");
        all_q41_buffers[i].dequan_qkv_B = app_qkv_dequan.create_bo_buffer<dequantize_params>(qkv_group_size, 3);
        load_params_bin_to_buffer(all_qkv_scale_path.c_str(), all_q41_buffers[i].dequan_qkv_B, qkv_group_size);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        fs::path all_out_scale_path = make_numbered_file(std::filesystem::path(weights_path)/"Z-Image-Turbo/denoising/q41_weights", "layer_", i, "_atten_out_packed_qx_scale_min.bin");
        all_q41_buffers[i].dequan_out_B = app_out_dequan.create_bo_buffer<dequantize_params>(out_group_size, 3);
        load_params_bin_to_buffer(all_out_scale_path.c_str(), all_q41_buffers[i].dequan_out_B, out_group_size);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        fs::path all_w1_scale_path = make_numbered_file(std::filesystem::path(weights_path)/"Z-Image-Turbo/denoising/q41_weights", "layer_", i, "_feed_w1_packed_qx_scale_min.bin");
        all_q41_buffers[i].dequan_w1_B = app_w3_dequan.create_bo_buffer<dequantize_params>(w3_group_size, 3);
        load_params_bin_to_buffer(all_w1_scale_path.c_str(), all_q41_buffers[i].dequan_w1_B, w3_group_size);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        fs::path all_w3_scale_path = make_numbered_file(std::filesystem::path(weights_path)/"Z-Image-Turbo/denoising/q41_weights", "layer_", i, "_feed_w3_packed_qx_scale_min.bin");
        all_q41_buffers[i].dequan_w3_B = app_w3_dequan.create_bo_buffer<dequantize_params>(w3_group_size, 3);
        load_params_bin_to_buffer(all_w3_scale_path.c_str(), all_q41_buffers[i].dequan_w3_B, w3_group_size);
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
       
        fs::path all_w2_scale_path = make_numbered_file(std::filesystem::path(weights_path)/"Z-Image-Turbo/denoising/q41_weights", "layer_", i, "_feed_w2_packed_qx_scale_min.bin");
        all_q41_buffers[i].dequan_w2_B = app_w2_dequan.create_bo_buffer<dequantize_params>(w3_group_size, 3);
        load_params_bin_to_buffer(all_w2_scale_path.c_str(), all_q41_buffers[i].dequan_w2_B, w3_group_size);
    }

     //std::cout << " load q4 data test" << std::endl;  //////////4GB///////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////embedder data first/////////////////////////////////////////////////////////////////////////
   
    
    fs::path w1_path = base_dir / "t_embedder_mlp_0_weight_bf16_u16.bin";
    fs::path w2_path = base_dir / "t_embedder_mlp_2_weight_bf16_u16.bin";
    fs::path bias1_path = base_dir / "t_embedder_mlp_0_bias_bf16_u16.bin";
    fs::path bias2_path = base_dir / "t_embedder_mlp_2_bias_bf16_u16.bin";
    buffer<dtype_in> time_embed_in1 = app_time_emb.create_bo_buffer<dtype_in>(256*256, 3);
    buffer<dtype_in> time_embed_weight1 = app_time_emb.create_bo_buffer<dtype_in>(256*1024, 4);
    buffer<dtype_in> time_embed_out1 = app_time_emb.create_bo_buffer<dtype_in>(256*1024, 5);

    buffer<dtype_in> time_embed_in2 = app_time_emb2.create_bo_buffer<dtype_in>(256*1024, 3);
    buffer<dtype_in> time_embed_weight2 = app_time_emb2.create_bo_buffer<dtype_in>(512*1024, 4);
    buffer<dtype_in> time_embed_out2 = app_time_emb2.create_bo_buffer<dtype_in>(256*512, 5);

    const char* fname_W1 = w1_path.c_str();
    const char* fname_W2 = w2_path.c_str();
    const char* fname_bias1 = bias1_path.c_str();
    const char* fname_bias2 = bias2_path.c_str();
    std::vector<uint16_t> u16_W1(256*1024);
    std::vector<uint16_t> u16_W2(256*1024);
    std::vector<uint16_t> u16_bias1(1024);
    std::vector<uint16_t> u16_bias2(256);
    read_data_from_files(fname_W1, u16_W1);
    read_data_from_files(fname_W2, u16_W2);
    read_data_from_files(fname_bias1, u16_bias1);
    read_data_from_files(fname_bias2, u16_bias2);
    buffer<dtype_out> bias_1(1024);
    for (size_t i = 0; i < 1024; ++i) {
        bias_1[i] = std::bit_cast<std::bfloat16_t>(u16_bias1[i]);
    }   
    for (size_t i = 0; i < 1024; ++i) {
        // source row: 3840 uint16_t values
        const uint16_t* src_row = (u16_W2.data() + i * 256);
        uint16_t* dst_row = reinterpret_cast<uint16_t*>(time_embed_weight2.data() + i * 512);
        std::memcpy(dst_row, src_row, 256 * sizeof(uint16_t));
    }
    for(int i = 0; i < 256*1024; i++){
        time_embed_weight1[i] = std::bit_cast<std::bfloat16_t>(u16_W1[i]);
    }
    std::vector<float> t;
    if(step==4)
    {
        t = {0.0, 100, 250, 500};
    }
    else if(step==8)
    {
        t = {0.0, 45.45458984375, 100.0, 166.66668701171875, 250.0, 357.14288330078125, 500.0, 700.0};
    }
    else
    {
        t = {0.0, 45.45458984375, 100.0, 166.66668701171875, 250.0, 357.14288330078125, 500.0, 700.0};
    }
    // float t[step] = {0.0, 45.45458984375, 100.0, 166.66668701171875, 250.0, 357.14288330078125, 500.0, 700.0};
    
    float *emb = new float[step * 256];
    timestep_embedding_python_exact(t, step, 256, emb, 10000.0f);
    
    buffer<dtype_out> emb_buf(step*256);
    for (size_t i = 0; i < step; i++){
        for (size_t j = 0; j < 256; j++){
            time_embed_in1[i*256+j] = emb[i*256+j];
        }
    }
    time_embed_in1.sync_to_device();  //256*256
    time_embed_weight1.sync_to_device(); //256*1024
    app_time_emb(time_embed_in1, time_embed_weight1, time_embed_out1); 
    time_embed_out1.sync_from_device(); //256*1024
    for(int i = 0; i < step; i++){
        for(int k = 0; k<1024; k++){
            time_embed_out1[i*1024+k] = time_embed_out1[i*1024+k] + bias_1[k];
        }
    }
    silu_bf16_avx512_noexp(time_embed_out1, time_embed_in2, 1024*8);

    time_embed_in2.sync_to_device();
    time_embed_weight2.sync_to_device();
    app_time_emb2(time_embed_in2, time_embed_weight2, time_embed_out2);
    time_embed_out2.sync_from_device();

    buffer<dtype_out> adaln_input_data(step*256);
    for (size_t i = 0; i < step; ++i) {
        for(int k = 0; k<256; k++){
            adaln_input_data[i*256+k] = std::bit_cast<std::bfloat16_t>(u16_bias2[k]) + time_embed_out2[i*512+k];
        }
    } 
    ////////////////////////////////////////////////////////////////////////////////////////
    fs::path cap_w_path = base_dir / "cap_embedder_1_weight_bf16_u16.bin";
    fs::path cap_bias_path = base_dir / "cap_embedder_1_bias_bf16_u16.bin";
    fs::path cap_scale_path = base_dir / "cap_embedder_0_weight_bf16_u16.bin";

    buffer<dtype_in> cap_embed_in = app_cap_embedder.create_bo_buffer<dtype_in>(512*2560, 3);
    buffer<dtype_in> cap_embed_weight = app_cap_embedder.create_bo_buffer<dtype_in>(2560*4096, 4);
    buffer<dtype_in> cap_embed_out = app_cap_embedder.create_bo_buffer<dtype_in>(512*4096, 5);
    std::vector<uint16_t> u16_cap_w(2560*3840);
    std::vector<uint16_t> u16_cap_bias(3840);
    std::vector<uint16_t> u16_cap_scale(2560);
    read_data_from_files(cap_w_path.c_str(), u16_cap_w);
    read_data_from_files(cap_bias_path.c_str(), u16_cap_bias);
    read_data_from_files(cap_scale_path.c_str(), u16_cap_scale);
    buffer<dtype_out> cap_w(2560*3840);
    buffer<dtype_out> cap_bias(3840);
    buffer<dtype_out> cap_scale(2560);
    for (size_t i = 0; i < 2560; ++i) {
        // source row: 3840 uint16_t values
        const uint16_t* src_row = (u16_cap_w.data() + i * 3840);
        uint16_t* dst_row = reinterpret_cast<uint16_t*>(cap_embed_weight.data() + i * 4096);
        std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
    }
    for(int i = 0; i < 3840; i++) {
        cap_bias[i] = std::bit_cast<std::bfloat16_t>(u16_cap_bias[i]);
    }
    for(int i = 0; i < 2560; i++) {
        cap_scale[i] = std::bit_cast<std::bfloat16_t>(u16_cap_scale[i]);
    }
    // buffer<dtype_out> cap_feats_norm(128*2560);
    rmsnorm_mat_bf16_with_scale(cap_feats, u16_cap_scale.data(), cap_embed_in, cap_total_len, 2560);
    


    cap_embed_in.sync_to_device();
    cap_embed_weight.sync_to_device();
    app_cap_embedder(cap_embed_in, cap_embed_weight, cap_embed_out);
    cap_embed_out.sync_from_device();
    
    
    
    buffer<dtype_out> cap_embed_bias(cap_total_len*3840);

    for(int i = 0; i < num_true; i++){
        for(int j = 0; j < 3840; j++){
            cap_result_buffers[0][i*4096+j] = cap_embed_out[i*4096+j] + cap_bias[j];
        }
    } 
    fs::path cap_pad_path = base_dir/ "cap_pad_token_bf16_u16.bin";
    std::vector<uint16_t> cap_pad_data(1*3840);
    read_data_from_files(cap_pad_path.c_str(), cap_pad_data);
    for(int i = num_true; i < cap_total_len; i++){
        for(int j = 0; j < 3840; j++){
            cap_result_buffers[0][i*4096 + j] = std::bit_cast<std::bfloat16_t>(cap_pad_data[j]);
        }
    }

   
    /////////////////////////////////////////////////////////////////////////
    for(int i = 0; i < 2; i++){
        int cap_size = cap_total_len*3840;
        buffer<dtype_out> x_cap_in_norm1(cap_total_len*3840);
        rmsnorm_mat_bf16_with_scale_partial(cap_result_buffers[i], noise_layer[i].cap_atten_norm1_u16.data(), cap_qk_A_in, cap_total_len, 4096, 3840);
        
        time_utils::time_point end = time_utils::now();
        cap_qk_A_in.sync_to_device();
        cap_buffers[i].B_in_qk[0].sync_to_device();
        app_cap_qkv(cap_qk_A_in, cap_buffers[i].B_in_qk[0], cap_q_C_out);
        cap_q_C_out.sync_from_device();
    
        cap_qk_A_in.sync_to_device();
        cap_buffers[i].B_in_qk[1].sync_to_device();
        app_cap_qkv(cap_qk_A_in, cap_buffers[i].B_in_qk[1], cap_k_C_out);
        cap_k_C_out.sync_from_device();
    
        cap_qk_A_in.sync_to_device();
        cap_buffers[i].B_in_v.sync_to_device();
        app_cap_qkv(cap_qk_A_in, cap_buffers[i].B_in_v, cap_v_C_out);
        cap_v_C_out.sync_from_device();
        time_utils::time_point start_qk = time_utils::now();
        // std::cout << "qk mm no bias time: " << time_utils::duration_us(end, start_qk).first / 1000 << "ms" << std::endl;
        buffer<dtype_out> qf_cap_bf16(cap_size);
        buffer<dtype_out> kf_cap_bf16(cap_size);
        buffer<dtype_out> vf_cap_bf16(cap_size);
        time_utils::time_point start_qk_mm = time_utils::now();
        for (size_t i = 0; i < cap_total_len; ++i) {
            // source row: 3840 uint16_t values
            const uint16_t* src_row = reinterpret_cast<uint16_t*>(cap_q_C_out.data() + i * 4096);
            const uint16_t* src_row_kf = reinterpret_cast<uint16_t*>(cap_k_C_out.data() + i * 4096);
            const uint16_t* src_row_vf = reinterpret_cast<uint16_t*>(cap_v_C_out.data() + i * 4096);
            uint16_t* dst_row = reinterpret_cast<uint16_t*>(qf_cap_bf16.data() + i * 3840);
            uint16_t* dst_row_kf = reinterpret_cast<uint16_t*>(kf_cap_bf16.data() + i * 3840);
            uint16_t* dst_row_vf = reinterpret_cast<uint16_t*>(vf_cap_bf16.data() + i * 3840);
            std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
            std::memcpy(dst_row_kf, src_row_kf, 3840 * sizeof(uint16_t));
            std::memcpy(dst_row_vf, src_row_vf, 3840 * sizeof(uint16_t));
        }
        time_utils::time_point end_qk_mm = time_utils::now();
        // std::cout << "qk mm no bias time: " << time_utils::duration_us(start_qk_mm, end_qk_mm).first / 1000 << "ms" << std::endl;
        buffer<dtype_out> qf_norm_bf16(cap_size);
        buffer<dtype_out> kf_norm_bf16(cap_size);
        process_qk_end2end_bf16_new(qf_cap_bf16, kf_cap_bf16, noise_layer[i].cap_q_norm_weight_u16.data(),
         noise_layer[i].cap_k_norm_weight_u16.data(), qf_norm_bf16, kf_norm_bf16, 30, cap_total_len, 128);
    
        buffer<dtype_out> out_q_cap_rope(cap_size);
        buffer<dtype_out> out_k_cap_rope(cap_size);
        apply_rope_bf16_avx512_new(qf_norm_bf16, cap_q_attn, pos_c_all_cap, pos_s_all_cap, 1, 30, cap_total_len, 128);
        apply_rope_bf16_avx512_new(kf_norm_bf16, out_k_cap_rope, pos_c_all_cap, pos_s_all_cap, 1, 30, cap_total_len, 128);
        
        interleave_kv(out_k_cap_rope.data(), vf_cap_bf16.data(), cap_kv_attn.data(), cap_total_len, 3840); //row. col
        //////////////////////////////////////////////////////////////////////////////////////////////
        
        /////////////////////////////////////////////////////////////////////////
        time_utils::time_point start_attn = time_utils::now();
        cap_q_attn.sync_to_device();
        cap_kv_attn.sync_to_device();
        app_attn_cap(cap_q_attn, cap_kv_attn, cap_out_attn);
        cap_out_attn.sync_from_device();
        
        time_utils::time_point end_attn = time_utils::now();
        
        for (size_t i = 0; i < cap_total_len; ++i) {
            // source row: 3840 uint16_t values
            const uint16_t* src_row = reinterpret_cast<uint16_t*>(cap_out_attn.data() + i * 3840);  
            uint16_t* dst_row = reinterpret_cast<uint16_t*>(cap_qk_A_in.data() + i * 4096);
            std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
        }
        time_utils::time_point start_out = time_utils::now();
        // std::cout << "padding atten output time: " << time_utils::duration_us(end_attn, start_out).first / 1000 << "ms" << std::endl;
       
        cap_qk_A_in.sync_to_device();
        cap_buffers[i].B_in_out.sync_to_device();
        app_cap_qkv(cap_qk_A_in, cap_buffers[i].B_in_out, cap_out_C_out);
        cap_out_C_out.sync_from_device();
       
        buffer<dtype_out> out_cap_bf16(cap_size);  //4096*3840
        for (size_t i = 0; i < cap_total_len; ++i) {
            const uint16_t* src_row = reinterpret_cast<uint16_t*>(cap_out_C_out.data() + i * 4096);
            uint16_t* dst_row = reinterpret_cast<uint16_t*>(out_cap_bf16.data() + i * 3840);
            std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
        }
        
        buffer<dtype_out> x_cap_in_norm2(cap_total_len*3840);
        time_utils::time_point start_add = time_utils::now();
        rmsnorm_mat_bf16_with_scale(out_cap_bf16, noise_layer[i].cap_atten_norm2_u16.data(), x_cap_in_norm2, 128, 3840);
        buffer<dtype_out> x_cap_in_norm2_mlp_partial(cap_total_len*4096);
        
        std::memset(x_cap_in_norm2_mlp_partial.data(), 0, (cap_total_len*4096) * sizeof(dtype_in));
 
        for(int k = 0; k < cap_total_len; ++k) {
            for(int j = 0; j < 3840; ++j) {
                x_cap_in_norm2_mlp_partial[k*4096+j] = x_cap_in_norm2[k*3840+j] + cap_result_buffers[i][k*4096+j];  //L2 = 0.02
            }
        }
       
       
        buffer<dtype_out> x_cap_norm1_ff(128*3840);

        rmsnorm_mat_bf16_with_scale_partial(x_cap_in_norm2_mlp_partial, noise_layer[i].cap_ffn_norm1_u16.data(), cap_mlp0, 128, 4096, 3840);
       
        time_utils::time_point start_mlp0 = time_utils::now();
        cap_mlp0.sync_to_device();
        cap_buffers[i].B_in_mlp0.sync_to_device();
        app_cap_mlp01(cap_mlp0, cap_buffers[i].B_in_mlp0, cap_mlp0_out);
        cap_mlp0_out.sync_from_device();
    
        cap_mlp0.sync_to_device();
        cap_buffers[i].B_in_mlp1.sync_to_device();
        app_cap_mlp01(cap_mlp0, cap_buffers[i].B_in_mlp1, cap_mlp1_out);
        cap_mlp1_out.sync_from_device();
        
        buffer<dtype_out> x_cap_silu(cap_total_len*10240);
        silu_bf16_avx512_noexp(cap_mlp0_out, x_cap_silu, cap_total_len*10240);
        buffer<dtype_out> x_cap_silu_mlp(cap_total_len*10240);
    
        for(int i = 0; i < cap_total_len; ++i) {
            for(int j = 0; j < 10240; ++j) {
                cap_mlp2_in[i*10240+j] = x_cap_silu[i*10240+j] * cap_mlp1_out[i*10240+j];
            }
        }
        time_utils::time_point start_mlp2 = time_utils::now();
        cap_mlp2_in.sync_to_device();
        cap_buffers[i].B_in_mlp2.sync_to_device();
        app_cap_mlp_out(cap_mlp2_in, cap_buffers[i].B_in_mlp2, cap_mlp2_out);
        cap_mlp2_out.sync_from_device();
        time_utils::time_point end_mlp2 = time_utils::now();


    
        buffer<dtype_out> x_cap_out(cap_total_len*3840);
        buffer<dtype_out> x_cap_out_norm2(cap_total_len*4096);

        rmsnorm_mat_bf16_with_scale_partial(cap_mlp2_out, noise_layer[i].cap_ffn_norm2_u16.data(), x_cap_out_norm2, cap_total_len, 4096, 3840);
        for(int k = 0; k < cap_total_len; ++k) {
            for(int j = 0; j < 3840; ++j) {
                cap_result_buffers[i+1][k*4096+j] = x_cap_out_norm2[k*4096+j] + x_cap_in_norm2_mlp_partial[k*4096+j];
            }
        }   
        
      
    }
   
  
    buffer<dtype_out> cap_new(cap_total_len*3840);
    for (size_t i = 0; i < cap_total_len; ++i) {
        // source row: 3840 uint16_t values
        const uint16_t* src_row = reinterpret_cast<uint16_t*>(cap_result_buffers[2].data() + i * 4096);
        uint16_t* dst_row = reinterpret_cast<uint16_t*>(cap_new.data() + i * 3840);
        std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
    }
  
    time_utils::time_point start_iteration = time_utils::now(); 


    // Declare a dynamic vector instead of a raw C-array
    std::vector<float> timesteps;

    if(step == 4)
    {
        // You can safely assign an initializer list to a vector at any time
        timesteps = {1.0f, 0.9f, 0.75f, 0.5f, 0.0f};
    }
    else if(step == 8)
    {
        timesteps = {1.0f, 0.9545454f, 0.9f, 0.8333333f, 0.75f, 0.6428571f, 0.5f, 0.3f, 0.0f};
    }
    else
    {
        timesteps = {1.0f, 0.9545454f, 0.9f, 0.8333333f, 0.75f, 0.6428571f, 0.5f, 0.3f, 0.0f};
    }

    auto start_denoising = std::chrono::steady_clock::now();
    for(int iteration = 0; iteration < step; iteration++) {
        // std::cout << "##########################iteration: " << iteration << "##############################" << std::endl;
        // float timesteps[9] = {1,  0.9545454,  0.9,  0.8333333,  0.75,  0.6428571, 0.5,  0.3, 0};
        
        
        float t_curr = timesteps[iteration];  // t_curr is the sigma_next
        float t_prev = timesteps[iteration+1];
        // std::cout << "t_curr: " << t_curr << " t_prev: " << t_prev << std::endl;
        std::vector<float> noise_new_data(16*latent_H*latent_W);
    
        std::vector<float> tokens_new_data(16*latent_H*latent_W);
        if(iteration == 0){
            noise_new_data = get_noise_3d_cpp(16, latent_H, latent_W, random_seed);           
            patchify_fcHW_to_tokens_cpp<float>(
                noise_new_data.data(),
                tokens_new_data.data(), 
                1, 16, latent_H, latent_W, // Pass F first, then C!
                pF, pH, pW
            );
            
        }
        else{
  
            // fs::path fname_noise_float = make_numbered_file("outputs", "update_img_", iteration, ".bin");
            // buffer<float> noise_float = read_floats_from_bin(fname_noise_float.string(), (size_t)16*latent_H*latent_W);
            buffer<float> noise_float = update_img[iteration - 1];
            for(int i=0;i<img_seqlen*64; i++)
            {
                noise_new_data[i] = noise_float[i];
            }
            patchify_fcHW_to_tokens_cpp<float>(
                reinterpret_cast<const float*>(noise_float.data()), 
                tokens_new_data.data(), 
                1, 16, latent_H, latent_W, // Pass F first, then C!
                pF, pH, pW
            );
        }
      
        buffer<dtype_out> tokens_bf16(img_seqlen*64);
        tokens_bf16 = convert_float_vector_to_bf16_buffer(tokens_new_data);
        copy_bf16_to_bf16_avx512(tokens_bf16.data(), first_mm_in.data(), img_seqlen*64);
       
        first_mm_in.sync_to_device();
        first_mm_weight.sync_to_device();
        app_first_mm(first_mm_in, first_mm_weight, first_mm_out);
        first_mm_out.sync_from_device();
        buffer<dtype_out> test_data(img_seqlen*3840);


        for(int i = 0; i < img_seqlen; i++){
            for(int j = 0; j < 3840; j++){
                img_result_buffers[0][i * 3840 + j] = first_mm_out[i * 4096 + j] + first_mm_bias[j];
            }
        }
       

        float cosine_sim_final, rel_l1_final, rmse_final, rel_l2_final, avg_abs_O_final;

        for (size_t i = 0; i < 1; ++i) {
            const uint16_t* src_row = reinterpret_cast<uint16_t*>(adaln_input_data.data() + iteration * 256);
            uint16_t* dst_row = reinterpret_cast<uint16_t*>(noise_adaLN_in.data());
            std::memcpy(dst_row, src_row, 256 * sizeof(uint16_t));
        }   

        for(int i = 0; i < 2; i++)
        {   
            time_utils::time_point start_0 = time_utils::now();
            noise_adaLN_in.sync_to_device();
            noise_buffers[i].noise_adaLN_weight.sync_to_device();
            app_adaLN(noise_adaLN_in, noise_buffers[i].noise_adaLN_weight, noise_adaLN_out);
            noise_adaLN_out.sync_from_device();

           
            buffer<dtype_out> noise_adaLN_out_bf16(15360*1);
            add_bias_rows_bf16_avx512(noise_adaLN_out.data(), noise_buffers[i].noise_adaLN_bias.data(), noise_adaLN_out_bf16.data(), 1, 15360);

            buffer<dtype_out> scale_msa(3840);
            buffer<dtype_out> gate_msa(3840);
            buffer<dtype_out> scale_mlp(3840);
            buffer<dtype_out> gate_mlp(3840);
            for(int i = 0; i < 3840*1; i++){
                scale_msa[i] = noise_adaLN_out_bf16[i] + 1;
                gate_msa[i] = std::tanh(noise_adaLN_out_bf16[i+3840]);
                scale_mlp[i] = noise_adaLN_out_bf16[i+7680] + 1;
                gate_mlp[i] = std::tanh(noise_adaLN_out_bf16[i+11520]);
            }
            int noise_size = img_seqlen*3840;

            time_utils::time_point start_1 = time_utils::now();
            // std::cout << "adaLN time: " << time_utils::duration_us(start_0, start_1).first / 1000 << "ms" << std::endl;   
            buffer<dtype_out> x_noise_in_norm1(noise_size);
            rmsnorm_mat_bf16_with_two_scales(img_result_buffers[i], 
                noise_layer[i].noise_atten_norm1_u16.data(), scale_msa, x_noise_in_norm1, img_seqlen, 3840);

            time_utils::time_point start_2 = time_utils::now();
            // std::cout << "rmsnorm time: " << time_utils::duration_us(start_1, start_2).first / 1000 << "ms" << std::endl;   
            for (size_t i = 0; i < img_seqlen; ++i) {
        
                const uint16_t* src_row = reinterpret_cast<uint16_t*>(x_noise_in_norm1.data() + i * 3840);
                uint16_t* dst_row = reinterpret_cast<uint16_t*>(noise_qk_A_in.data() + i * 4096);
                std::memcpy(dst_row, src_row, 3840 * sizeof(uint16_t));
            }
            time_utils::time_point start_3 = time_utils::now();
            // std::cout << "all pading time: " << time_utils::duration_us(start_2, start_3).first / 1000 << "ms" << std::endl;                  

            noise_qk_A_in.sync_to_device();
            noise_buffers[i].B_in_qk[0].sync_to_device();
            app_0(noise_qk_A_in, noise_buffers[i].B_in_qk[0], noise_q_C_out);
            noise_q_C_out.sync_from_device();

            noise_qk_A_in.sync_to_device();
            noise_buffers[i].B_in_qk[1].sync_to_device();
            app_0(noise_qk_A_in, noise_buffers[i].B_in_qk[1], noise_k_C_out);
            noise_k_C_out.sync_from_device();

            noise_qk_A_in.sync_to_device();
            noise_buffers[i].B_in_v.sync_to_device();
            app_0(noise_qk_A_in, noise_buffers[i].B_in_v, noise_v_C_out);
            noise_v_C_out.sync_from_device();
            
            int weight_size = 4096;
            time_utils::time_point start_4 = time_utils::now();
         
            buffer<dtype_out> qf_norm_bf16(img_seqlen*weight_size);
            buffer<dtype_out> kf_norm_bf16(img_seqlen*weight_size);
            process_qk_end2end_bf16_new(noise_q_C_out, noise_k_C_out, noise_layer[i].noise_q_norm_weight_u16.data(),
             noise_layer[i].noise_k_norm_weight_u16.data(), qf_norm_bf16, kf_norm_bf16, 32, img_seqlen, 128);  /// H, L, D

            buffer<dtype_out> out_q_noise_rope(img_seqlen*weight_size);
            buffer<dtype_out> out_k_noise_rope(img_seqlen*weight_size);
            apply_rope_bf16_avx512_new(qf_norm_bf16, noise_q_attn, pos_c_all, pos_s_all, 1, 32, img_seqlen, 128);
            apply_rope_bf16_avx512_new(kf_norm_bf16, out_k_noise_rope, pos_c_all, pos_s_all, 1, 32, img_seqlen, 128);
            
            interleave_kv(out_k_noise_rope.data(), noise_v_C_out.data(), noise_kv_attn.data(), img_seqlen, 4096);
            time_utils::time_point start_5 = time_utils::now();
           
            noise_q_attn.sync_to_device();
            noise_kv_attn.sync_to_device();
            app_attn_noise(noise_q_attn, noise_kv_attn, noise_out_attn);
            noise_out_attn.sync_from_device();
            time_utils::time_point start_6 = time_utils::now();
            // std::cout << " noise attention time is " << time_utils::duration_us(start_5, start_6).first / 1000 << "ms" << std::endl;
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
           
            copy_bf16_to_bf16_avx512(noise_out_attn.data(), noise_qk_A_in.data(), img_seqlen*4096);
        
            noise_qk_A_in.sync_to_device();
            noise_buffers[i].B_in_out.sync_to_device();
            app_0(noise_qk_A_in, noise_buffers[i].B_in_out, noise_out_C_out);
            noise_out_C_out.sync_from_device();
            time_utils::time_point start_7 = time_utils::now();
            // std::cout << "noise out atten time: " << time_utils::duration_us(start_6, start_7).first / 1000 << "ms" << std::endl;   
            buffer<dtype_out> x_noise_in_norm2_mlp(img_seqlen*3840);  
           
            buffer<dtype_out> x_noise_norm1_ff(img_seqlen*3840);
   
            rmsnorm_mlp_fused_two_scales_exact(
                noise_out_C_out,          // [L,4096]
                img_result_buffers[i],               // [L,3840]
                noise_layer[i].noise_atten_norm2_u16.data(),
                gate_msa,
                noise_layer[i].noise_ffn_norm1_u16.data(),
                scale_mlp,
                x_noise_norm1_ff,
                x_noise_in_norm2_mlp,
                noise_mlp0,
                noise_mlp1,
                img_seqlen,
                3840);
            time_utils::time_point start_8 = time_utils::now();
            
            
        
            // std::cout << "all add operation time: " << time_utils::duration_us(start_7, start_8).first / 1000 << "ms" << std::endl; 
            noise_mlp1.sync_to_device();
            noise_buffers[i].B_in_mlp1.sync_to_device();
            app_1(noise_mlp1, noise_buffers[i].B_in_mlp1, noise_mlp1_out);
            noise_mlp1_out.sync_from_device();

            noise_mlp0.sync_to_device();
            noise_buffers[i].B_in_mlp0.sync_to_device();
            app_3(noise_mlp0, noise_buffers[i].B_in_mlp0, noise_mlp0_out);
            noise_mlp0_out.sync_from_device();
            
        
            bf16_mul_avx512(reinterpret_cast<uint16_t*>(noise_mlp2_in.data()),
             reinterpret_cast<uint16_t*>(noise_mlp0_out.data()),
             reinterpret_cast<uint16_t*>(noise_mlp1_out.data()), img_seqlen*10240);
   
                            
            noise_mlp2_in.sync_to_device();
            noise_buffers[i].B_in_mlp2.sync_to_device();
            app_2(noise_mlp2_in, noise_buffers[i].B_in_mlp2, noise_mlp2_out);
            noise_mlp2_out.sync_from_device();
            time_utils::time_point start_11 = time_utils::now();
     
            buffer<dtype_out> out_noise_bf16_mlp(img_seqlen*3840);  //4096*3840
            fused_norm2_add_4096x3840_avx512(
                reinterpret_cast<const uint16_t*>(noise_mlp2_out.data()),
                noise_layer[i].noise_ffn_norm2_u16.data(),
                reinterpret_cast<const uint16_t*>(gate_mlp.data()),
                reinterpret_cast<const uint16_t*>(x_noise_in_norm2_mlp.data()),
                reinterpret_cast<uint16_t*>(img_result_buffers[i + 1].data()),
                img_seqlen);
            
            time_utils::time_point start_12 = time_utils::now();
            
                
        }
        // std::cout << "##########################all layers##############################" << std::endl;
        time_utils::time_point start_all = time_utils::now();
        // std::cout << "all noise time: " << time_utils::duration_us(start_noise, start_all).first / 1000 << "ms" << std::endl;
        buffer<dtype_out> all_concat_out(img_cap_size*3840);
        concat_2d_bf16(img_result_buffers[2], cap_new, all_result_buffers[0], img_seqlen, cap_total_len, 3840);
      
        /////////////////////////////////////////////////////////////////////////
  
        /////////////////////////////////////////////////////////////////////////
        for(int i = 0; i < 30; ++i) {
            time_utils::time_point start_adaLN = time_utils::now();
            ///////////////////////////////////////////////////////////////////////////////////////////////////////   
            all_q41_buffers[i].dequan_adaLN_B.sync_to_device();
            app_adaLN_dequan(all_q41_buffers[i].dequan_adaLN_B, dequan_adaLN_out);
            dequan_adaLN_out.sync_from_device();

            all_q41_buffers[i].dequan_qkv_B.sync_to_device();
            app_qkv_dequan(all_q41_buffers[i].dequan_qkv_B, dequan_qkv_out);
            dequan_qkv_out.sync_from_device();

            all_q41_buffers[i].dequan_out_B.sync_to_device();
            app_out_dequan(all_q41_buffers[i].dequan_out_B, dequan_out);
            dequan_out.sync_from_device();

            all_q41_buffers[i].dequan_w1_B.sync_to_device();
            app_w3_dequan(all_q41_buffers[i].dequan_w1_B, dequan_w1_out);
            dequan_w1_out.sync_from_device();
          
            all_q41_buffers[i].dequan_w3_B.sync_to_device();
            app_w3_dequan(all_q41_buffers[i].dequan_w3_B, dequan_w3_out);
            dequan_w3_out.sync_from_device();

            all_q41_buffers[i].dequan_w2_B.sync_to_device();
            app_w2_dequan(all_q41_buffers[i].dequan_w2_B, dequan_w2_out);
            dequan_w2_out.sync_from_device();


            copy_bf16_to_bf16_avx512(dequan_adaLN_out.data(), noise_adaLN_dequan.data(), 15360*256);


            ///////////////////////////////////////////////////////////////////////////////////////////////////////
        
            noise_adaLN_in.sync_to_device();
            noise_adaLN_dequan.sync_to_device();
            app_adaLN(noise_adaLN_in, noise_adaLN_dequan, noise_adaLN_out);
            noise_adaLN_out.sync_from_device();
            
           
            buffer<dtype_out> all_adaLN_out_bf16(15360*1);
            add_bias_rows_bf16_avx512(noise_adaLN_out.data(), all_buffers[i].all_adaLN_bias.data(), all_adaLN_out_bf16.data(), 1, 15360);

            buffer<dtype_out> scale_msa(3840);
            buffer<dtype_out> gate_msa(3840);
            buffer<dtype_out> scale_mlp(3840);
            buffer<dtype_out> gate_mlp(3840);
            for(int i = 0; i < 3840*1; i++){
                scale_msa[i] = all_adaLN_out_bf16[i] + 1;
                gate_msa[i] = std::tanh(all_adaLN_out_bf16[i+3840]);
                scale_mlp[i] = all_adaLN_out_bf16[i+7680] + 1;
                gate_mlp[i] = std::tanh(all_adaLN_out_bf16[i+11520]);
            }

            buffer<dtype_out> x_all_in_norm1(img_cap_size*3840);

            time_utils::time_point start_norm1 = time_utils::now();
            // std::cout << "before norm1 time: " << time_utils::duration_us(start_adaLN, start_norm1).first / 1000 << "ms" << std::endl; 
            rmsnorm_mat_bf16_with_two_scales(all_result_buffers[i], all_layer[i].all_atten_norm1_u16.data(), scale_msa, x_all_in_norm1, img_cap_size, 3840);

            for (size_t i = 0; i < img_cap_size; ++i) {
                // source row: 3840 uint16_t values
                const uint16_t* src_row = reinterpret_cast<uint16_t*>(x_all_in_norm1.data() + i * 3840);  
                uint16_t* dst_row_v = reinterpret_cast<uint16_t*>(all_v_A_in.data() + i * 4096);
                std::memcpy(dst_row_v, src_row, 3840 * sizeof(uint16_t));
            }
            time_utils::time_point start_qkA = time_utils::now();
            /////////////////////////////////////////////////dequan qkv////////////////////////////////////////////////////////////        
            split_qkv_4096x12288_to_3x4096x4096_avx512_bf16(
                reinterpret_cast<const uint16_t*>(dequan_qkv_out.data()),
                reinterpret_cast<uint16_t*>(all_q_dequan.data()),
                reinterpret_cast<uint16_t*>(all_k_dequan.data()),
                reinterpret_cast<uint16_t*>(all_v_dequan.data())
            );
           
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////
            all_v_A_in.sync_to_device();
            all_q_dequan.sync_to_device();
            app_all_v(all_v_A_in, all_q_dequan, all_q_C_out);
            all_q_C_out.sync_from_device();
           
            all_v_A_in.sync_to_device();
            all_k_dequan.sync_to_device();
            app_all_v(all_v_A_in, all_k_dequan, all_k_C_out);
            all_k_C_out.sync_from_device();
            
            all_v_A_in.sync_to_device();
            all_v_dequan.sync_to_device();
            app_all_v(all_v_A_in, all_v_dequan, all_v_C_out);
            all_v_C_out.sync_from_device();
           
            time_utils::time_point start_qkv = time_utils::now();
            // std::cout << "all qkv time: " << time_utils::duration_us(start_qkA, start_qkv).first / 1000 << "ms" << std::endl; 
            buffer<dtype_out> qf_norm_bf16(img_cap_size*4096);
            buffer<dtype_out> kf_norm_bf16(img_cap_size*4096);
            process_qk_end2end_bf16_new(all_q_C_out, all_k_C_out, all_layer[i].all_q_norm_weight_u16.data(),
             all_layer[i].all_k_norm_weight_u16.data(), qf_norm_bf16, kf_norm_bf16, 32, img_cap_size, 128);

            buffer<dtype_out> out_q_all_rope(img_cap_size*4096);
            buffer<dtype_out> out_k_all_rope(img_cap_size*4096);
            time_utils::time_point start_rope = time_utils::now();
            // std::cout << "before rope time: " << time_utils::duration_us(start_qkv, start_rope).first / 1000 << "ms" << std::endl; 
    
            apply_rope_bf16_avx512_new(qf_norm_bf16, all_q_attn, pos_c_all_cat, pos_s_all_cat, 1, 32, img_cap_size, 128);
            apply_rope_bf16_avx512_new(kf_norm_bf16, out_k_all_rope, pos_c_all_cat, pos_s_all_cat, 1, 32, img_cap_size, 128);
            interleave_kv(out_k_all_rope.data(), all_v_C_out.data(), all_kv_attn.data(), img_cap_size, 4096);
            
            time_utils::time_point start_attn = time_utils::now();
            
            all_q_attn.sync_to_device();
            all_kv_attn.sync_to_device();
            app_attn_all(all_q_attn, all_kv_attn, all_out_attn);
            all_out_attn.sync_from_device();
            time_utils::time_point start_copy = time_utils::now();
      
 
            copy_bf16_to_bf16_avx512(all_out_attn.data(), all_v_A_in.data(), img_cap_size*4096);
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            
            copy_bf16_to_bf16_avx512(dequan_out.data(), all_out_dequan.data(), 4096*4096);
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            all_v_A_in.sync_to_device();
            all_out_dequan.sync_to_device();
            app_all_v(all_v_A_in, all_out_dequan, all_out_C_out);
            all_out_C_out.sync_from_device();
            
            time_utils::time_point start_out = time_utils::now();
            // std::cout << "all out time: " << time_utils::duration_us(start_copy, start_out).first / 1000 << "ms" << std::endl; 
            buffer<dtype_out> x_all_in_norm2_mlp(img_cap_size*3840);
            buffer<dtype_out> x_all_norm1_ff(img_cap_size*3840);
      
           
            rmsnorm_mlp_fused_two_scales_exact(
                all_out_C_out,          // [L,4096]
                all_result_buffers[i],               // [L,3840]
                all_layer[i].all_atten_norm2_u16.data(),
                gate_msa,
                all_layer[i].all_ffn_norm1_u16.data(),
                scale_mlp,
                x_all_norm1_ff,
                x_all_in_norm2_mlp,
                all_mlp0,
                all_mlp1,
                img_cap_size,
                3840);
            time_utils::time_point start_mlp0 = time_utils::now();
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
            copy_bf16_to_bf16_avx512(dequan_w1_out.data(), all_mlp0_dequan.data(), 10240*4096);


            copy_bf16_to_bf16_avx512(dequan_w3_out.data(), all_mlp1_dequan.data(), 10240*4096);
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
           
            all_mlp0.sync_to_device();
            all_mlp0_dequan.sync_to_device();
            app_all_mlp_silu(all_mlp0, all_mlp0_dequan, all_mlp0_out);
            all_mlp0_out.sync_from_device();
        
            all_mlp1.sync_to_device();
            all_mlp1_dequan.sync_to_device();
            app_all_mlp01(all_mlp1, all_mlp1_dequan, all_mlp1_out);
            all_mlp1_out.sync_from_device();
            time_utils::time_point end_mlp0 = time_utils::now();
            // std::cout << "mlp0 time: " << time_utils::duration_us(start_mlp0, end_mlp0).first / 1000 << "ms" << std::endl;
            /////////////////////////////////////////////////////////////////////////////////////////////
            bf16_mul_avx512(reinterpret_cast<uint16_t*>(all_mlp2_in.data()),
             reinterpret_cast<uint16_t*>(all_mlp0_out.data()), reinterpret_cast<uint16_t*>(all_mlp1_out.data()), img_cap_size*10240);
            time_utils::time_point start_mlp2 = time_utils::now();
            
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            copy_bf16_to_bf16_avx512(dequan_w2_out.data(), all_mlp2_dequan.data(), 10240*4096);
            /////////////////////////////////////////////////////////////////////////////////////////////////////
            all_mlp2_in.sync_to_device();
            all_mlp2_dequan.sync_to_device();
            app_all_mlp_out(all_mlp2_in, all_mlp2_dequan, all_mlp2_out);
            all_mlp2_out.sync_to_device();
            time_utils::time_point end_mlp2 = time_utils::now();
           
            fused_norm2_add_4096x3840_avx512(
                reinterpret_cast<const uint16_t*>(all_mlp2_out.data()),
                all_layer[i].all_ffn_norm2_u16.data(),
                reinterpret_cast<const uint16_t*>(gate_mlp.data()),
                reinterpret_cast<const uint16_t*>(x_all_in_norm2_mlp.data()),
                reinterpret_cast<uint16_t*>(all_result_buffers[i + 1].data()),
                img_cap_size);
            time_utils::time_point start_add = time_utils::now();
            // std::cout << "all add time: " << time_utils::duration_us(end_mlp2, start_add).first / 1000 << "ms" << std::endl;
                // std::cout << "total all time: " << time_utils::duration_us(start_adaLN, start_add).first / 1000 << "ms" << std::endl;
                            /////////////////////////////////////////////////////////////////////////////////////////////
            
        }
 
        //////////////////////////////////////////////////final layer//////////////////////////////////////////////////////////////////
        
        copy_bf16_to_bf16_avx512(noise_adaLN_in.data(), final_adaLN_in.data(), 1*256);
        silu_bf16_avx512_noexp(final_adaLN_in, final_adaLN_in, 1*256);
        final_adaLN_in.sync_to_device();
        final_adaLN_weight.sync_to_device();
        app_adaLN_final(final_adaLN_in, final_adaLN_weight, final_adaLN_out);
        final_adaLN_out.sync_from_device();
        buffer<dtype_out> final_adaLN_out_bias(3840);
        for(int i = 0; i < 3840; ++i){
            final_adaLN_out_bias[i] = final_adaLN_out[i] + final_adaLN_bias[i];
        }
    
        buffer<dtype_out> final_scale(3840);
        for(int i = 0; i < 3840; ++i){
            final_scale[i] = final_adaLN_out_bias[i] + 1.0f;
        }
    
      
        buffer<dtype_out> final_adaLN_out_norm1(img_cap_size*3840);
        layernorm_bf16_avx512_noaffine(all_result_buffers[30], final_adaLN_out_norm1, img_cap_size, 3840, 1e-6);    

        for(int i = 0; i < img_cap_size; ++i) {
            for(int j = 0; j < 3840; ++j) {
                final_mm_in[i*3840+j] = final_adaLN_out_norm1[i*3840+j] * final_scale[j];
            }
        }
        time_utils::time_point start_final_mm = time_utils::now();
        final_mm_in.sync_to_device();
        final_mm_weight.sync_to_device();
        app_final_mm(final_mm_in, final_mm_weight, final_mm_out);
        final_mm_out.sync_from_device();
        time_utils::time_point end_final_mm = time_utils::now();
  
        buffer<dtype_out> final_mm_out_bias(img_cap_size*64);
        add_bias_final_bf16_avx512(final_mm_out.data(), final_adaLN_linear_bias.data(), final_mm_out_bias.data(), img_cap_size, 512);
     
        /////////////////////////////////////unpatchify//////////////////////////////////////////////////////////////////
        buffer<dtype_out> unpatchify_out(16*latent_H*latent_W);
        unpatchify_bf16_no_vector(
            reinterpret_cast<const dtype_out*>(final_mm_out_bias.data()),
            /*L=*/img_cap_size,
            /*K=*/64,
            reinterpret_cast<dtype_out*>(unpatchify_out.data()),
            /*F=*/1, /*H=*/latent_H, /*W=*/latent_W,
            /*patch_size=*/2,
            /*f_patch_size=*/1,
            /*outC=*/16
        );

    
        buffer<float> noise_float(img_seqlen*64);
        for(int i = 0; i < img_seqlen*64; i++){
            noise_float[i] = noise_new_data[i];
        }
    
        for(int i = 0; i < 16*latent_H*latent_W; i++){
            update_img[iteration][i] = noise_float[i] + unpatchify_out[i] * (t_curr - t_prev);
        }
        print_progress_bar("Denoising", iteration, step, start_denoising);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    std::cout << std::endl;
   
    }
    /////////////////////////////////////////////////////////////////////////////////////vae decoder///////////////////////////////////
    vae_decoder_weights vae_weights;
    //////////load vae decoder weights ////////////////////////////////////////////////////////////////////
    fs::path conv_in_weight_path = vae_decoder_dir / "conv_in-weight_u16.bin";
    std::vector<uint16_t> conv_in_weight_u16(512*16*3*3);
    io::read_data_from_files(conv_in_weight_path.c_str(), vae_weights.conv_in_weight, 512*16*3*3);
  
    fs::path conv_in_bias_path = vae_decoder_dir / "conv_in-bias_u16.bin";
    std::vector<uint16_t> conv_in_bias_u16(512);
    io::read_data_from_files(conv_in_bias_path.c_str(), vae_weights.conv_in_bias, 512);

    ////////////////////////////////////////ResnetBlock2D_0 weights ////////////////////////////////////////////////////////////////// 
    fs::path mid_block_resnets_0_norm1_weight_path = vae_decoder_dir / "mid_block-resnets-0-norm1-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_norm1_weight_u16(512);
    io::read_data_from_files(mid_block_resnets_0_norm1_weight_path.c_str(), mid_block_resnets_0_norm1_weight_u16, 512);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_norm1_weight_u16.data(), vae_weights.mid_block_resnets_0_norm1_weight.data(), 512);
    fs::path mid_block_resnets_0_norm1_bias_path = vae_decoder_dir / "mid_block-resnets-0-norm1-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_norm1_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_0_norm1_bias_path.c_str(), mid_block_resnets_0_norm1_bias_u16, 512);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_norm1_bias_u16.data(), vae_weights.mid_block_resnets_0_norm1_bias.data(), 512);

    fs::path mid_block_resnets_0_weight_path = vae_decoder_dir / "mid_block-resnets-0-conv1-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_weight_u16(512*512*3*3);
    io::read_data_from_files(mid_block_resnets_0_weight_path.c_str(), mid_block_resnets_0_weight_u16, 512*512*3*3);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_weight_u16.data(), vae_weights.mid_block_resnets_0_weight.data(), 512*512*3*3);
    fs::path mid_block_resnets_0_bias_path = vae_decoder_dir / "mid_block-resnets-0-conv1-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_0_bias_path.c_str(), mid_block_resnets_0_bias_u16, 512);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_bias_u16.data(), vae_weights.mid_block_resnets_0_bias.data(), 512);

    fs::path mid_block_resnets_0_norm2_weight_path = vae_decoder_dir / "mid_block-resnets-0-norm2-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_norm2_weight_u16(512);
    io::read_data_from_files(mid_block_resnets_0_norm2_weight_path.c_str(), mid_block_resnets_0_norm2_weight_u16, 512);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_norm2_weight_u16.data(), vae_weights.mid_block_resnets_0_norm2_weight.data(), 512);
    fs::path mid_block_resnets_0_norm2_bias_path = vae_decoder_dir / "mid_block-resnets-0-norm2-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_norm2_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_0_norm2_bias_path.c_str(), mid_block_resnets_0_norm2_bias_u16, 512);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_norm2_bias_u16.data(), vae_weights.mid_block_resnets_0_norm2_bias.data(), 512);

    fs::path mid_block_resnets_0_conv2_weight_path = vae_decoder_dir / "mid_block-resnets-0-conv2-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_conv2_weight_u16(512*512*3*3);
    io::read_data_from_files(mid_block_resnets_0_conv2_weight_path.c_str(), mid_block_resnets_0_conv2_weight_u16, 512*512*3*3);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_conv2_weight_u16.data(), vae_weights.mid_block_resnets_0_conv2_weight.data(), 512*512*3*3);
    fs::path mid_block_resnets_0_conv2_bias_path = vae_decoder_dir / "mid_block-resnets-0-conv2-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_0_conv2_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_0_conv2_bias_path.c_str(), mid_block_resnets_0_conv2_bias_u16, 512);
    copy_u16_to_bf16_avx512(mid_block_resnets_0_conv2_bias_u16.data(), vae_weights.mid_block_resnets_0_conv2_bias.data(), 512);
    ////////////////////////////////////////ResnetBlock2D_1 weights ////////////////////////////////////////////////////////////////// 
    fs::path mid_block_resnets_1_norm1_weight_path = vae_decoder_dir / "mid_block-resnets-1-norm1-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_norm1_weight_u16(512);
    io::read_data_from_files(mid_block_resnets_1_norm1_weight_path.c_str(), mid_block_resnets_1_norm1_weight_u16, 512);
  
    fs::path mid_block_resnets_1_norm1_bias_path = vae_decoder_dir / "mid_block-resnets-1-norm1-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_norm1_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_1_norm1_bias_path.c_str(), mid_block_resnets_1_norm1_bias_u16, 512);
 

    fs::path mid_block_resnets_1_weight_path = vae_decoder_dir / "mid_block-resnets-1-conv1-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_weight_u16(512*512*3*3);
    io::read_data_from_files(mid_block_resnets_1_weight_path.c_str(), mid_block_resnets_1_weight_u16, 512*512*3*3);
  
    fs::path mid_block_resnets_1_bias_path = vae_decoder_dir / "mid_block-resnets-1-conv1-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_1_bias_path.c_str(), mid_block_resnets_1_bias_u16, 512);
  

    fs::path mid_block_resnets_1_norm2_weight_path = vae_decoder_dir / "mid_block-resnets-1-norm2-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_norm2_weight_u16(512);
    io::read_data_from_files(mid_block_resnets_1_norm2_weight_path.c_str(), mid_block_resnets_1_norm2_weight_u16, 512);
  
    fs::path mid_block_resnets_1_norm2_bias_path = vae_decoder_dir / "mid_block-resnets-1-norm2-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_norm2_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_1_norm2_bias_path.c_str(), mid_block_resnets_1_norm2_bias_u16, 512);
   

    fs::path mid_block_resnets_1_conv2_weight_path = vae_decoder_dir / "mid_block-resnets-1-conv2-weight_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_conv2_weight_u16(512*512*3*3);
    io::read_data_from_files(mid_block_resnets_1_conv2_weight_path.c_str(), mid_block_resnets_1_conv2_weight_u16, 512*512*3*3);
  
    fs::path mid_block_resnets_1_conv2_bias_path = vae_decoder_dir / "mid_block-resnets-1-conv2-bias_u16.bin";
    std::vector<uint16_t> mid_block_resnets_1_conv2_bias_u16(512);
    io::read_data_from_files(mid_block_resnets_1_conv2_bias_path.c_str(), mid_block_resnets_1_conv2_bias_u16, 512);
  

    ResNetBlockBF16Weights mid1 {
        .norm1_weight = mid_block_resnets_1_norm1_weight_u16.data(),
        .norm1_bias   = mid_block_resnets_1_norm1_bias_u16.data(),
    
        .conv1_weight = mid_block_resnets_1_weight_u16.data(),
        .conv1_bias   = mid_block_resnets_1_bias_u16.data(),
    
        .norm2_weight = mid_block_resnets_1_norm2_weight_u16.data(),
        .norm2_bias   = mid_block_resnets_1_norm2_bias_u16.data(),
    
        .conv2_weight = mid_block_resnets_1_conv2_weight_u16.data(),
        .conv2_bias   = mid_block_resnets_1_conv2_bias_u16.data(),
    };


    ////////////////////////////////////////updecoderBlock2d_0 weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder0_resnet_conv1_weights_path = vae_decoder_dir / "up_blocks-0-resnets-0-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_conv1_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_resnet_conv1_weights_path.c_str(), updecoder0_resnet_conv1_weights_u16, 512*512*3*3);
   
    fs::path updecoder0_resnet_conv1_biases_path = vae_decoder_dir / "up_blocks-0-resnets-0-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_conv1_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet_conv1_biases_path.c_str(), updecoder0_resnet_conv1_biases_u16, 512);
   
    fs::path updecoder0_resnet_conv2_weights_path = vae_decoder_dir / "up_blocks-0-resnets-0-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_conv2_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_resnet_conv2_weights_path.c_str(), updecoder0_resnet_conv2_weights_u16, 512*512*3*3);
    fs::path updecoder0_resnet_conv2_biases_path = vae_decoder_dir / "up_blocks-0-resnets-0-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_conv2_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet_conv2_biases_path.c_str(), updecoder0_resnet_conv2_biases_u16, 512);
  
    fs::path updecoder0_resnet_norm1_weights_path = vae_decoder_dir / "up_blocks-0-resnets-0-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_norm1_weights_u16(512);
    io::read_data_from_files(updecoder0_resnet_norm1_weights_path.c_str(), updecoder0_resnet_norm1_weights_u16, 512);
   
    fs::path updecoder0_resnet_norm1_biases_path = vae_decoder_dir / "up_blocks-0-resnets-0-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_norm1_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet_norm1_biases_path.c_str(), updecoder0_resnet_norm1_biases_u16, 512);
   
    fs::path updecoder0_resnet_norm2_weights_path = vae_decoder_dir / "up_blocks-0-resnets-0-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_norm2_weights_u16(512);
    io::read_data_from_files(updecoder0_resnet_norm2_weights_path.c_str(), updecoder0_resnet_norm2_weights_u16, 512);
   
    fs::path updecoder0_resnet_norm2_biases_path = vae_decoder_dir / "up_blocks-0-resnets-0-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet_norm2_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet_norm2_biases_path.c_str(), updecoder0_resnet_norm2_biases_u16, 512);
    

     ResNetBlockBF16Weights updecoder0_resnet0 {
        .norm1_weight = updecoder0_resnet_norm1_weights_u16.data(),
        .norm1_bias   = updecoder0_resnet_norm1_biases_u16.data(),
    
        .conv1_weight = updecoder0_resnet_conv1_weights_u16.data(),
        .conv1_bias   = updecoder0_resnet_conv1_biases_u16.data(),
    
        .norm2_weight = updecoder0_resnet_norm2_weights_u16.data(),
        .norm2_bias   = updecoder0_resnet_norm2_biases_u16.data(),
    
        .conv2_weight = updecoder0_resnet_conv2_weights_u16.data(),
        .conv2_bias   = updecoder0_resnet_conv2_biases_u16.data(),
    };
    ////////////////////////////////////////updecoderBlock2d0_1 weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder0_resnet1_conv1_weights_path = vae_decoder_dir / "up_blocks-0-resnets-1-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_conv1_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_resnet1_conv1_weights_path.c_str(), updecoder0_resnet1_conv1_weights_u16, 512*512*3*3);
   
    fs::path updecoder0_resnet1_conv1_biases_path = vae_decoder_dir / "up_blocks-0-resnets-1-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_conv1_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet1_conv1_biases_path.c_str(), updecoder0_resnet1_conv1_biases_u16, 512);

    fs::path updecoder0_resnet1_conv2_weights_path = vae_decoder_dir / "up_blocks-0-resnets-1-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_conv2_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_resnet1_conv2_weights_path.c_str(), updecoder0_resnet1_conv2_weights_u16, 512*512*3*3);
  
    fs::path updecoder0_resnet1_conv2_biases_path = vae_decoder_dir / "up_blocks-0-resnets-1-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_conv2_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet1_conv2_biases_path.c_str(), updecoder0_resnet1_conv2_biases_u16, 512);
   
    fs::path updecoder0_resnet1_norm1_weights_path = vae_decoder_dir / "up_blocks-0-resnets-1-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_norm1_weights_u16(512);
    io::read_data_from_files(updecoder0_resnet1_norm1_weights_path.c_str(), updecoder0_resnet1_norm1_weights_u16, 512);
 
    fs::path updecoder0_resnet1_norm1_biases_path = vae_decoder_dir / "up_blocks-0-resnets-1-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_norm1_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet1_norm1_biases_path.c_str(), updecoder0_resnet1_norm1_biases_u16, 512);
    
    fs::path updecoder0_resnet1_norm2_weights_path = vae_decoder_dir / "up_blocks-0-resnets-1-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_norm2_weights_u16(512);
    io::read_data_from_files(updecoder0_resnet1_norm2_weights_path.c_str(), updecoder0_resnet1_norm2_weights_u16, 512);
 
    fs::path updecoder0_resnet1_norm2_biases_path = vae_decoder_dir / "up_blocks-0-resnets-1-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet1_norm2_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet1_norm2_biases_path.c_str(), updecoder0_resnet1_norm2_biases_u16, 512);
    
    
    ResNetBlockBF16Weights updecoder0_resnet1 {
        .norm1_weight = updecoder0_resnet1_norm1_weights_u16.data(),
        .norm1_bias   = updecoder0_resnet1_norm1_biases_u16.data(),
    
        .conv1_weight = updecoder0_resnet1_conv1_weights_u16.data(),
        .conv1_bias   = updecoder0_resnet1_conv1_biases_u16.data(),
    
        .norm2_weight = updecoder0_resnet1_norm2_weights_u16.data(),
        .norm2_bias   = updecoder0_resnet1_norm2_biases_u16.data(),
    
        .conv2_weight = updecoder0_resnet1_conv2_weights_u16.data(),
        .conv2_bias   = updecoder0_resnet1_conv2_biases_u16.data(),
    };
    
    ////////////////////////////////////////updecoderBlock2d0_2 weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder0_resnet2_conv1_weights_path = vae_decoder_dir / "up_blocks-0-resnets-2-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_conv1_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_resnet2_conv1_weights_path.c_str(), updecoder0_resnet2_conv1_weights_u16, 512*512*3*3);
    
    fs::path updecoder0_resnet2_conv1_biases_path = vae_decoder_dir / "up_blocks-0-resnets-2-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_conv1_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet2_conv1_biases_path.c_str(), updecoder0_resnet2_conv1_biases_u16, 512);
   
    fs::path updecoder0_resnet2_conv2_weights_path = vae_decoder_dir / "up_blocks-0-resnets-2-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_conv2_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_resnet2_conv2_weights_path.c_str(), updecoder0_resnet2_conv2_weights_u16, 512*512*3*3);
    
    fs::path updecoder0_resnet2_conv2_biases_path = vae_decoder_dir / "up_blocks-0-resnets-2-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_conv2_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet2_conv2_biases_path.c_str(), updecoder0_resnet2_conv2_biases_u16, 512);
   
    fs::path updecoder0_resnet2_norm1_weights_path = vae_decoder_dir / "up_blocks-0-resnets-2-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_norm1_weights_u16(512);
    io::read_data_from_files(updecoder0_resnet2_norm1_weights_path.c_str(), updecoder0_resnet2_norm1_weights_u16, 512);
    fs::path updecoder0_resnet2_norm1_biases_path = vae_decoder_dir / "up_blocks-0-resnets-2-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_norm1_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet2_norm1_biases_path.c_str(), updecoder0_resnet2_norm1_biases_u16, 512);
  
    fs::path updecoder0_resnet2_norm2_weights_path = vae_decoder_dir / "up_blocks-0-resnets-2-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_norm2_weights_u16(512);
    io::read_data_from_files(updecoder0_resnet2_norm2_weights_path.c_str(), updecoder0_resnet2_norm2_weights_u16, 512);
   
    fs::path updecoder0_resnet2_norm2_biases_path = vae_decoder_dir / "up_blocks-0-resnets-2-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder0_resnet2_norm2_biases_u16(512);
    io::read_data_from_files(updecoder0_resnet2_norm2_biases_path.c_str(), updecoder0_resnet2_norm2_biases_u16, 512);
 


    ResNetBlockBF16Weights updecoder0_resnet2 {
        .norm1_weight = updecoder0_resnet2_norm1_weights_u16.data(),
        .norm1_bias   = updecoder0_resnet2_norm1_biases_u16.data(),
    
        .conv1_weight = updecoder0_resnet2_conv1_weights_u16.data(),
        .conv1_bias   = updecoder0_resnet2_conv1_biases_u16.data(),
    
        .norm2_weight = updecoder0_resnet2_norm2_weights_u16.data(),
        .norm2_bias   = updecoder0_resnet2_norm2_biases_u16.data(),
    
        .conv2_weight = updecoder0_resnet2_conv2_weights_u16.data(),
        .conv2_bias   = updecoder0_resnet2_conv2_biases_u16.data(),
    };
    ////////////////////////////////////////updecoder0_upsample weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder0_upsample_weights_path = vae_decoder_dir / "up_blocks-0-upsamplers-0-conv-weight_u16.bin";
    std::vector<uint16_t> updecoder0_upsample_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder0_upsample_weights_path.c_str(), updecoder0_upsample_weights_u16, 512*512*3*3);
    copy_u16_to_bf16_avx512(updecoder0_upsample_weights_u16.data(), vae_weights.updecoder0_upsample_weight.data(), 512*512*3*3);
    fs::path updecoder0_upsample_biases_path = vae_decoder_dir / "up_blocks-0-upsamplers-0-conv-bias_u16.bin";
    std::vector<uint16_t> updecoder0_upsample_biases_u16(512);
    io::read_data_from_files(updecoder0_upsample_biases_path.c_str(), updecoder0_upsample_biases_u16, 512);
    copy_u16_to_bf16_avx512(updecoder0_upsample_biases_u16.data(), vae_weights.updecoder0_upsample_bias.data(), 512);
    ////////////////////////////////////////updecoder1_upsample weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder1_resnet_conv1_weights_path = vae_decoder_dir / "up_blocks-1-resnets-0-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_conv1_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_resnet_conv1_weights_path.c_str(), updecoder1_resnet_conv1_weights_u16, 512*512*3*3);
   
    fs::path updecoder1_resnet_conv1_biases_path = vae_decoder_dir / "up_blocks-1-resnets-0-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_conv1_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet_conv1_biases_path.c_str(), updecoder1_resnet_conv1_biases_u16, 512);
  
    fs::path updecoder1_resnet_conv2_weights_path = vae_decoder_dir / "up_blocks-1-resnets-0-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_conv2_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_resnet_conv2_weights_path.c_str(), updecoder1_resnet_conv2_weights_u16, 512*512*3*3);
    
    fs::path updecoder1_resnet_conv2_biases_path = vae_decoder_dir / "up_blocks-1-resnets-0-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_conv2_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet_conv2_biases_path.c_str(), updecoder1_resnet_conv2_biases_u16, 512);
   
    fs::path updecoder1_resnet_norm1_weights_path = vae_decoder_dir / "up_blocks-1-resnets-0-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_norm1_weights_u16(512);
    io::read_data_from_files(updecoder1_resnet_norm1_weights_path.c_str(), updecoder1_resnet_norm1_weights_u16, 512);
  
    fs::path updecoder1_resnet_norm1_biases_path = vae_decoder_dir / "up_blocks-1-resnets-0-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_norm1_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet_norm1_biases_path.c_str(), updecoder1_resnet_norm1_biases_u16, 512);
  
    fs::path updecoder1_resnet_norm2_weights_path = vae_decoder_dir / "up_blocks-1-resnets-0-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_norm2_weights_u16(512);
    io::read_data_from_files(updecoder1_resnet_norm2_weights_path.c_str(), updecoder1_resnet_norm2_weights_u16, 512);
   
    fs::path updecoder1_resnet_norm2_biases_path = vae_decoder_dir / "up_blocks-1-resnets-0-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet_norm2_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet_norm2_biases_path.c_str(), updecoder1_resnet_norm2_biases_u16, 512);

    
    ResNetBlockBF16Weights updecoder1_resnet0 {
        .norm1_weight = updecoder1_resnet_norm1_weights_u16.data(),
        .norm1_bias   = updecoder1_resnet_norm1_biases_u16.data(),
    
        .conv1_weight = updecoder1_resnet_conv1_weights_u16.data(),
        .conv1_bias   = updecoder1_resnet_conv1_biases_u16.data(),
    
        .norm2_weight = updecoder1_resnet_norm2_weights_u16.data(),
        .norm2_bias   = updecoder1_resnet_norm2_biases_u16.data(),
    
        .conv2_weight = updecoder1_resnet_conv2_weights_u16.data(),
        .conv2_bias   = updecoder1_resnet_conv2_biases_u16.data(),
    };
    
    
    ////////////////////////////////////////updecoderBlock2d1_1 weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder1_resnet1_conv1_weights_path = vae_decoder_dir / "up_blocks-1-resnets-1-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_conv1_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_resnet1_conv1_weights_path.c_str(), updecoder1_resnet1_conv1_weights_u16, 512*512*3*3);
   
    fs::path updecoder1_resnet1_conv1_biases_path = vae_decoder_dir / "up_blocks-1-resnets-1-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_conv1_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet1_conv1_biases_path.c_str(), updecoder1_resnet1_conv1_biases_u16, 512);
  
    fs::path updecoder1_resnet1_conv2_weights_path = vae_decoder_dir / "up_blocks-1-resnets-1-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_conv2_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_resnet1_conv2_weights_path.c_str(), updecoder1_resnet1_conv2_weights_u16, 512*512*3*3);
   
    fs::path updecoder1_resnet1_conv2_biases_path = vae_decoder_dir / "up_blocks-1-resnets-1-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_conv2_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet1_conv2_biases_path.c_str(), updecoder1_resnet1_conv2_biases_u16, 512);

    fs::path updecoder1_resnet1_norm1_weights_path = vae_decoder_dir / "up_blocks-1-resnets-1-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_norm1_weights_u16(512);
    io::read_data_from_files(updecoder1_resnet1_norm1_weights_path.c_str(), updecoder1_resnet1_norm1_weights_u16, 512);
   
    fs::path updecoder1_resnet1_norm1_biases_path = vae_decoder_dir / "up_blocks-1-resnets-1-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_norm1_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet1_norm1_biases_path.c_str(), updecoder1_resnet1_norm1_biases_u16, 512);

    fs::path updecoder1_resnet1_norm2_weights_path = vae_decoder_dir / "up_blocks-1-resnets-1-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_norm2_weights_u16(512);
    io::read_data_from_files(updecoder1_resnet1_norm2_weights_path.c_str(), updecoder1_resnet1_norm2_weights_u16, 512);
    
    fs::path updecoder1_resnet1_norm2_biases_path = vae_decoder_dir / "up_blocks-1-resnets-1-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet1_norm2_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet1_norm2_biases_path.c_str(), updecoder1_resnet1_norm2_biases_u16, 512);
 
    
    
    ResNetBlockBF16Weights updecoder1_resnet1 {
        .norm1_weight = updecoder1_resnet1_norm1_weights_u16.data(),
        .norm1_bias   = updecoder1_resnet1_norm1_biases_u16.data(),
    
        .conv1_weight = updecoder1_resnet1_conv1_weights_u16.data(),
        .conv1_bias   = updecoder1_resnet1_conv1_biases_u16.data(),
    
        .norm2_weight = updecoder1_resnet1_norm2_weights_u16.data(),
        .norm2_bias   = updecoder1_resnet1_norm2_biases_u16.data(),
    
        .conv2_weight = updecoder1_resnet1_conv2_weights_u16.data(),
        .conv2_bias   = updecoder1_resnet1_conv2_biases_u16.data(),
    };
    
    
    
    ////////////////////////////////////////updecoderBlock2d1_2 weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder1_resnet2_conv1_weights_path = vae_decoder_dir / "up_blocks-1-resnets-2-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_conv1_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_resnet2_conv1_weights_path.c_str(), updecoder1_resnet2_conv1_weights_u16, 512*512*3*3);
   
    fs::path updecoder1_resnet2_conv1_biases_path = vae_decoder_dir / "up_blocks-1-resnets-2-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_conv1_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet2_conv1_biases_path.c_str(), updecoder1_resnet2_conv1_biases_u16, 512);
   
    fs::path updecoder1_resnet2_conv2_weights_path = vae_decoder_dir / "up_blocks-1-resnets-2-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_conv2_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_resnet2_conv2_weights_path.c_str(), updecoder1_resnet2_conv2_weights_u16, 512*512*3*3);
   
    fs::path updecoder1_resnet2_conv2_biases_path = vae_decoder_dir / "up_blocks-1-resnets-2-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_conv2_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet2_conv2_biases_path.c_str(), updecoder1_resnet2_conv2_biases_u16, 512);
  
    fs::path updecoder1_resnet2_norm1_weights_path = vae_decoder_dir / "up_blocks-1-resnets-2-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_norm1_weights_u16(512);
    io::read_data_from_files(updecoder1_resnet2_norm1_weights_path.c_str(), updecoder1_resnet2_norm1_weights_u16, 512);

    fs::path updecoder1_resnet2_norm1_biases_path = vae_decoder_dir / "up_blocks-1-resnets-2-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_norm1_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet2_norm1_biases_path.c_str(), updecoder1_resnet2_norm1_biases_u16, 512);

    fs::path updecoder1_resnet2_norm2_weights_path = vae_decoder_dir / "up_blocks-1-resnets-2-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_norm2_weights_u16(512);
    io::read_data_from_files(updecoder1_resnet2_norm2_weights_path.c_str(), updecoder1_resnet2_norm2_weights_u16, 512);
    
    fs::path updecoder1_resnet2_norm2_biases_path = vae_decoder_dir / "up_blocks-1-resnets-2-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder1_resnet2_norm2_biases_u16(512);
    io::read_data_from_files(updecoder1_resnet2_norm2_biases_path.c_str(), updecoder1_resnet2_norm2_biases_u16, 512);



    ResNetBlockBF16Weights updecoder1_resnet2 {
        .norm1_weight = updecoder1_resnet2_norm1_weights_u16.data(),
        .norm1_bias   = updecoder1_resnet2_norm1_biases_u16.data(),
    
        .conv1_weight = updecoder1_resnet2_conv1_weights_u16.data(),
        .conv1_bias   = updecoder1_resnet2_conv1_biases_u16.data(),
    
        .norm2_weight = updecoder1_resnet2_norm2_weights_u16.data(),
        .norm2_bias   = updecoder1_resnet2_norm2_biases_u16.data(),
    
        .conv2_weight = updecoder1_resnet2_conv2_weights_u16.data(),
        .conv2_bias   = updecoder1_resnet2_conv2_biases_u16.data(),
    };
    ////////////////////////////////////////updecoder1_upsample weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder1_upsample_weights_path = vae_decoder_dir / "up_blocks-1-upsamplers-0-conv-weight_u16.bin";
    std::vector<uint16_t> updecoder1_upsample_weights_u16(512*512*3*3);
    io::read_data_from_files(updecoder1_upsample_weights_path.c_str(), updecoder1_upsample_weights_u16, 512*512*3*3);
    copy_u16_to_bf16_avx512(updecoder1_upsample_weights_u16.data(), vae_weights.updecoder1_upsample_weight.data(), 512*512*3*3);
    fs::path updecoder1_upsample_biases_path = vae_decoder_dir / "up_blocks-1-upsamplers-0-conv-bias_u16.bin";
    std::vector<uint16_t> updecoder1_upsample_biases_u16(512);
    io::read_data_from_files(updecoder1_upsample_biases_path.c_str(), updecoder1_upsample_biases_u16, 512);
    copy_u16_to_bf16_avx512(updecoder1_upsample_biases_u16.data(), vae_weights.updecoder1_upsample_bias.data(), 512);
    ////////////////////////////////////////updecoder2_reset_input_conv weights ////////////////////////////////////////////////////////////////// 
   
        
    fs::path updecoder2_reset_conv_weights_path = vae_decoder_dir / "up_blocks-2-resnets-0-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset_conv_weights_u16(256*512*3*3);
    io::read_data_from_files(updecoder2_reset_conv_weights_path.c_str(), updecoder2_reset_conv_weights_u16, 256*512*3*3);
   
    fs::path updecoder2_reset_conv_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-0-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset_conv_biases_u16(256);
    io::read_data_from_files(updecoder2_reset_conv_biases_path.c_str(), updecoder2_reset_conv_biases_u16, 256);


    fs::path updecoder2_reset_conv2_weights_path = vae_decoder_dir / "up_blocks-2-resnets-0-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset_conv2_weights_u16(256*256*3*3);
    io::read_data_from_files(updecoder2_reset_conv2_weights_path.c_str(), updecoder2_reset_conv2_weights_u16, 256*256*3*3);
 
    fs::path updecoder2_reset_conv2_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-0-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset_conv2_biases_u16(256);
    io::read_data_from_files(updecoder2_reset_conv2_biases_path.c_str(), updecoder2_reset_conv2_biases_u16, 256);

    fs::path updecoder2_resnet_norm1_weights_path = vae_decoder_dir/ "up_blocks-2-resnets-0-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder2_resnet_norm1_weights_u16(512);
    io::read_data_from_files(updecoder2_resnet_norm1_weights_path.c_str(), updecoder2_resnet_norm1_weights_u16, 512);
  
    fs::path updecoder2_resnet_norm1_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-0-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder2_resnet_norm1_biases_u16(512);
    io::read_data_from_files(updecoder2_resnet_norm1_biases_path.c_str(), updecoder2_resnet_norm1_biases_u16, 512);
  
    fs::path updecoder2_resnet_norm2_weights_path = vae_decoder_dir/ "up_blocks-2-resnets-0-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder2_resnet_norm2_weights_u16(256);
    io::read_data_from_files(updecoder2_resnet_norm2_weights_path.c_str(), updecoder2_resnet_norm2_weights_u16, 256);
  
    fs::path updecoder2_resnet_norm2_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-0-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder2_resnet_norm2_biases_u16(256);
    io::read_data_from_files(updecoder2_resnet_norm2_biases_path.c_str(), updecoder2_resnet_norm2_biases_u16, 256);
   
  
    fs::path updecoder2_reset_input_conv_weights_path = vae_decoder_dir / "up_blocks-2-resnets-0-conv_shortcut-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset_input_conv_weights_u16(256*512*1*1);
    io::read_data_from_files(updecoder2_reset_input_conv_weights_path.c_str(), updecoder2_reset_input_conv_weights_u16, 256*512*1*1);
    
    fs::path updecoder2_reset_input_conv_biases_path = vae_decoder_dir / "up_blocks-2-resnets-0-conv_shortcut-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset_input_conv_biases_u16(256);
    io::read_data_from_files(updecoder2_reset_input_conv_biases_path.c_str(), updecoder2_reset_input_conv_biases_u16, 256);

    

    DecoderResNetBF16Weights updecoder2_resnet0 {
        updecoder2_resnet_norm1_weights_u16.data(),
        updecoder2_resnet_norm1_biases_u16.data(),
    
        updecoder2_reset_conv_weights_u16.data(),
        updecoder2_reset_conv_biases_u16.data(),
    
        updecoder2_resnet_norm2_weights_u16.data(),
        updecoder2_resnet_norm2_biases_u16.data(),
    
        updecoder2_reset_conv2_weights_u16.data(),
        updecoder2_reset_conv2_biases_u16.data(),
    
        updecoder2_reset_input_conv_weights_u16.data(),
        updecoder2_reset_input_conv_biases_u16.data()
    };
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fs::path updecoder2_reset1_conv_weights_path = vae_decoder_dir / "up_blocks-2-resnets-1-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset1_conv_weights_u16(256*256*3*3);
    io::read_data_from_files(updecoder2_reset1_conv_weights_path.c_str(), updecoder2_reset1_conv_weights_u16, 256*256*3*3);
   
    fs::path updecoder2_reset1_conv_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-1-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset1_conv_biases_u16(256);
    io::read_data_from_files(updecoder2_reset1_conv_biases_path.c_str(), updecoder2_reset1_conv_biases_u16, 256);


    fs::path updecoder2_reset1_conv2_weights_path = vae_decoder_dir / "up_blocks-2-resnets-1-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset1_conv2_weights_u16(256*256*3*3);
    io::read_data_from_files(updecoder2_reset1_conv2_weights_path.c_str(), updecoder2_reset1_conv2_weights_u16, 256*256*3*3);
   
    fs::path updecoder2_reset1_conv2_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-1-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset1_conv2_biases_u16(256);
    io::read_data_from_files(updecoder2_reset1_conv2_biases_path.c_str(), updecoder2_reset1_conv2_biases_u16, 256);
   
    fs::path updecoder2_resnet1_norm1_weights_path = vae_decoder_dir/ "up_blocks-2-resnets-1-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder2_resnet1_norm1_weights_u16(256);
    io::read_data_from_files(updecoder2_resnet1_norm1_weights_path.c_str(), updecoder2_resnet1_norm1_weights_u16, 256);
   
    fs::path updecoder2_resnet1_norm1_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-1-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder2_resnet1_norm1_biases_u16(256);
    io::read_data_from_files(updecoder2_resnet1_norm1_biases_path.c_str(), updecoder2_resnet1_norm1_biases_u16, 256);
   
    fs::path updecoder2_resnet1_norm2_weights_path = vae_decoder_dir/ "up_blocks-2-resnets-1-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder2_resnet1_norm2_weights_u16(256);
    io::read_data_from_files(updecoder2_resnet1_norm2_weights_path.c_str(), updecoder2_resnet1_norm2_weights_u16, 256);
    
    fs::path updecoder2_resnet1_norm2_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-1-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder2_resnet1_norm2_biases_u16(256);
    io::read_data_from_files(updecoder2_resnet1_norm2_biases_path.c_str(), updecoder2_resnet1_norm2_biases_u16, 256);
   
    
    
    ResNetBlockBF16Weights updecoder2_resnet1 {
        updecoder2_resnet1_norm1_weights_u16.data(),
        updecoder2_resnet1_norm1_biases_u16.data(),
    
        updecoder2_reset1_conv_weights_u16.data(),
        updecoder2_reset1_conv_biases_u16.data(),
    
        updecoder2_resnet1_norm2_weights_u16.data(),
        updecoder2_resnet1_norm2_biases_u16.data(),
    
        updecoder2_reset1_conv2_weights_u16.data(),
        updecoder2_reset1_conv2_biases_u16.data(),
    };
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fs::path updecoder2_reset2_conv_weights_path = vae_decoder_dir / "up_blocks-2-resnets-2-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset2_conv_weights_u16(256*256*3*3);
    io::read_data_from_files(updecoder2_reset2_conv_weights_path.c_str(), updecoder2_reset2_conv_weights_u16, 256*256*3*3);
  
    fs::path updecoder2_reset2_conv_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-2-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset2_conv_biases_u16(256);
    io::read_data_from_files(updecoder2_reset2_conv_biases_path.c_str(), updecoder2_reset2_conv_biases_u16, 256);


    fs::path updecoder2_reset2_conv2_weights_path = vae_decoder_dir / "up_blocks-2-resnets-2-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder2_reset2_conv2_weights_u16(256*256*3*3);
    io::read_data_from_files(updecoder2_reset2_conv2_weights_path.c_str(), updecoder2_reset2_conv2_weights_u16, 256*256*3*3);
    fs::path updecoder2_reset2_conv2_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-2-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder2_reset2_conv2_biases_u16(256);
    io::read_data_from_files(updecoder2_reset2_conv2_biases_path.c_str(), updecoder2_reset2_conv2_biases_u16, 256);

    fs::path updecoder2_resnet2_norm1_weights_path = vae_decoder_dir/ "up_blocks-2-resnets-2-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder2_resnet2_norm1_weights_u16(256);
    io::read_data_from_files(updecoder2_resnet2_norm1_weights_path.c_str(), updecoder2_resnet2_norm1_weights_u16, 256);
    fs::path updecoder2_resnet2_norm1_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-2-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder2_resnet2_norm1_biases_u16(256);
    io::read_data_from_files(updecoder2_resnet2_norm1_biases_path.c_str(), updecoder2_resnet2_norm1_biases_u16, 256);
    fs::path updecoder2_resnet2_norm2_weights_path = vae_decoder_dir/ "up_blocks-2-resnets-2-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder2_resnet2_norm2_weights_u16(256);
    io::read_data_from_files(updecoder2_resnet2_norm2_weights_path.c_str(), updecoder2_resnet2_norm2_weights_u16, 256);
    fs::path updecoder2_resnet2_norm2_biases_path = vae_decoder_dir/ "up_blocks-2-resnets-2-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder2_resnet2_norm2_biases_u16(256);
    io::read_data_from_files(updecoder2_resnet2_norm2_biases_path.c_str(), updecoder2_resnet2_norm2_biases_u16, 256);
   
    ResNetBlockBF16Weights updecoder2_resnet2 {
        updecoder2_resnet2_norm1_weights_u16.data(),
        updecoder2_resnet2_norm1_biases_u16.data(),
    
        updecoder2_reset2_conv_weights_u16.data(),
        updecoder2_reset2_conv_biases_u16.data(),
    
        updecoder2_resnet2_norm2_weights_u16.data(),
        updecoder2_resnet2_norm2_biases_u16.data(),
    
        updecoder2_reset2_conv2_weights_u16.data(),
        updecoder2_reset2_conv2_biases_u16.data(),
    };
   
   
    //////////////////////////////////////updecoder2_reset_1weights ////////////////////////////////////////////////////////////////// 
    fs::path updecoder2_upsample_weights_path = vae_decoder_dir / "up_blocks-2-upsamplers-0-conv-weight_u16.bin";
    std::vector<uint16_t> updecoder2_upsample_weights_u16(256*256*3*3);
    io::read_data_from_files(updecoder2_upsample_weights_path.c_str(), updecoder2_upsample_weights_u16, 256*256*3*3);
    copy_u16_to_bf16_avx512(updecoder2_upsample_weights_u16.data(), vae_weights.updecoder2_upsample_weight.data(), 256*256*3*3);
    fs::path updecoder2_upsample_biases_path = vae_decoder_dir / "up_blocks-2-upsamplers-0-conv-bias_u16.bin";
    std::vector<uint16_t> updecoder2_upsample_biases_u16(256);
    io::read_data_from_files(updecoder2_upsample_biases_path.c_str(), updecoder2_upsample_biases_u16, 256);
    copy_u16_to_bf16_avx512(updecoder2_upsample_biases_u16.data(), vae_weights.updecoder2_upsample_bias.data(), 256);
    //////////////////////////////////////updecoder3_reset_input_conv weights ////////////////////////////////////////////////////////////////// 
   
        
    fs::path updecoder3_reset_conv_weights_path = vae_decoder_dir / "up_blocks-3-resnets-0-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset_conv_weights_u16(128*256*3*3);
    io::read_data_from_files(updecoder3_reset_conv_weights_path.c_str(), updecoder3_reset_conv_weights_u16, 128*256*3*3);
    fs::path updecoder3_reset_conv_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-0-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset_conv_biases_u16(128);
    io::read_data_from_files(updecoder3_reset_conv_biases_path.c_str(), updecoder3_reset_conv_biases_u16, 128);
   
    fs::path updecoder3_reset_conv2_weights_path = vae_decoder_dir / "up_blocks-3-resnets-0-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset_conv2_weights_u16(128*128*3*3);
    io::read_data_from_files(updecoder3_reset_conv2_weights_path.c_str(), updecoder3_reset_conv2_weights_u16, 128*128*3*3);
   
    fs::path updecoder3_reset_conv2_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-0-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset_conv2_biases_u16(128);
    io::read_data_from_files(updecoder3_reset_conv2_biases_path.c_str(), updecoder3_reset_conv2_biases_u16, 128);
    fs::path updecoder3_resnet_norm1_weights_path = vae_decoder_dir/ "up_blocks-3-resnets-0-norm1-weight_u16.bin";
   
    std::vector<uint16_t> updecoder3_resnet_norm1_weights_u16(256);
    io::read_data_from_files(updecoder3_resnet_norm1_weights_path.c_str(), updecoder3_resnet_norm1_weights_u16, 256);
    fs::path updecoder3_resnet_norm1_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-0-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder3_resnet_norm1_biases_u16(256);
    io::read_data_from_files(updecoder3_resnet_norm1_biases_path.c_str(), updecoder3_resnet_norm1_biases_u16, 256);
    fs::path updecoder3_resnet_norm2_weights_path = vae_decoder_dir/ "up_blocks-3-resnets-0-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder3_resnet_norm2_weights_u16(128);
    io::read_data_from_files(updecoder3_resnet_norm2_weights_path.c_str(), updecoder3_resnet_norm2_weights_u16, 128);

    fs::path updecoder3_resnet_norm2_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-0-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder3_resnet_norm2_biases_u16(128);
    io::read_data_from_files(updecoder3_resnet_norm2_biases_path.c_str(), updecoder3_resnet_norm2_biases_u16, 128);
    fs::path updecoder3_reset_input_conv_weights_path = vae_decoder_dir / "up_blocks-3-resnets-0-conv_shortcut-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset_input_conv_weights_u16(128*256*1*1);
    io::read_data_from_files(updecoder3_reset_input_conv_weights_path.c_str(), updecoder3_reset_input_conv_weights_u16, 128*256*1*1);
    fs::path updecoder3_reset_input_conv_biases_path = vae_decoder_dir / "up_blocks-3-resnets-0-conv_shortcut-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset_input_conv_biases_u16(128);
    io::read_data_from_files(updecoder3_reset_input_conv_biases_path.c_str(), updecoder3_reset_input_conv_biases_u16, 128);
    
    DecoderResNetBF16Weights updecoder3_resnet0 {
        updecoder3_resnet_norm1_weights_u16.data(),
        updecoder3_resnet_norm1_biases_u16.data(),
    
        updecoder3_reset_conv_weights_u16.data(),
        updecoder3_reset_conv_biases_u16.data(),
    
        updecoder3_resnet_norm2_weights_u16.data(),
        updecoder3_resnet_norm2_biases_u16.data(),
    
        updecoder3_reset_conv2_weights_u16.data(),
        updecoder3_reset_conv2_biases_u16.data(),
    
        updecoder3_reset_input_conv_weights_u16.data(),
        updecoder3_reset_input_conv_biases_u16.data()
    };
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fs::path updecoder3_reset1_conv_weights_path = vae_decoder_dir / "up_blocks-3-resnets-1-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset1_conv_weights_u16(128*128*3*3);
    io::read_data_from_files(updecoder3_reset1_conv_weights_path.c_str(), updecoder3_reset1_conv_weights_u16, 128*128*3*3);
    fs::path updecoder3_reset1_conv_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-1-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset1_conv_biases_u16(128);
    io::read_data_from_files(updecoder3_reset1_conv_biases_path.c_str(), updecoder3_reset1_conv_biases_u16, 128);

    fs::path updecoder3_reset1_conv2_weights_path = vae_decoder_dir / "up_blocks-3-resnets-1-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset1_conv2_weights_u16(128*128*3*3);
    io::read_data_from_files(updecoder3_reset1_conv2_weights_path.c_str(), updecoder3_reset1_conv2_weights_u16, 128*128*3*3);
    fs::path updecoder3_reset1_conv2_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-1-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset1_conv2_biases_u16(128);
    io::read_data_from_files(updecoder3_reset1_conv2_biases_path.c_str(), updecoder3_reset1_conv2_biases_u16, 128);
    fs::path updecoder3_resnet1_norm1_weights_path = vae_decoder_dir/ "up_blocks-3-resnets-1-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder3_resnet1_norm1_weights_u16(128);
    io::read_data_from_files(updecoder3_resnet1_norm1_weights_path.c_str(), updecoder3_resnet1_norm1_weights_u16, 128);
    fs::path updecoder3_resnet1_norm1_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-1-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder3_resnet1_norm1_biases_u16(128);
    io::read_data_from_files(updecoder3_resnet1_norm1_biases_path.c_str(), updecoder3_resnet1_norm1_biases_u16, 128);
  
    fs::path updecoder3_resnet1_norm2_weights_path = vae_decoder_dir/ "up_blocks-3-resnets-1-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder3_resnet1_norm2_weights_u16(128);
    io::read_data_from_files(updecoder3_resnet1_norm2_weights_path.c_str(), updecoder3_resnet1_norm2_weights_u16, 128);
    fs::path updecoder3_resnet1_norm2_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-1-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder3_resnet1_norm2_biases_u16(128);
    io::read_data_from_files(updecoder3_resnet1_norm2_biases_path.c_str(), updecoder3_resnet1_norm2_biases_u16, 128);

    ResNetBlockBF16Weights updecoder3_resnet1 {
        updecoder3_resnet1_norm1_weights_u16.data(),
        updecoder3_resnet1_norm1_biases_u16.data(),
    
        updecoder3_reset1_conv_weights_u16.data(),
        updecoder3_reset1_conv_biases_u16.data(),

        updecoder3_resnet1_norm2_weights_u16.data(),
        updecoder3_resnet1_norm2_biases_u16.data(),

        updecoder3_reset1_conv2_weights_u16.data(),
        updecoder3_reset1_conv2_biases_u16.data(),
    
        
    };
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    fs::path updecoder3_reset2_conv_weights_path = vae_decoder_dir / "up_blocks-3-resnets-2-conv1-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset2_conv_weights_u16(128*128*3*3);
    io::read_data_from_files(updecoder3_reset2_conv_weights_path.c_str(), updecoder3_reset2_conv_weights_u16, 128*128*3*3);
    fs::path updecoder3_reset2_conv_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-2-conv1-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset2_conv_biases_u16(128);
    io::read_data_from_files(updecoder3_reset2_conv_biases_path.c_str(), updecoder3_reset2_conv_biases_u16, 128);

    fs::path updecoder3_reset2_conv2_weights_path = vae_decoder_dir / "up_blocks-3-resnets-2-conv2-weight_u16.bin";
    std::vector<uint16_t> updecoder3_reset2_conv2_weights_u16(128*128*3*3);
    io::read_data_from_files(updecoder3_reset2_conv2_weights_path.c_str(), updecoder3_reset2_conv2_weights_u16, 128*128*3*3);
    fs::path updecoder3_reset2_conv2_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-2-conv2-bias_u16.bin";
    std::vector<uint16_t> updecoder3_reset2_conv2_biases_u16(128);
    io::read_data_from_files(updecoder3_reset2_conv2_biases_path.c_str(), updecoder3_reset2_conv2_biases_u16, 128);

    fs::path updecoder3_resnet2_norm1_weights_path = vae_decoder_dir/ "up_blocks-3-resnets-2-norm1-weight_u16.bin";
    std::vector<uint16_t> updecoder3_resnet2_norm1_weights_u16(128);
    io::read_data_from_files(updecoder3_resnet2_norm1_weights_path.c_str(), updecoder3_resnet2_norm1_weights_u16, 128);
    fs::path updecoder3_resnet2_norm1_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-2-norm1-bias_u16.bin";
    std::vector<uint16_t> updecoder3_resnet2_norm1_biases_u16(128);
    io::read_data_from_files(updecoder3_resnet2_norm1_biases_path.c_str(), updecoder3_resnet2_norm1_biases_u16, 128);
    fs::path updecoder3_resnet2_norm2_weights_path = vae_decoder_dir/ "up_blocks-3-resnets-2-norm2-weight_u16.bin";
    std::vector<uint16_t> updecoder3_resnet2_norm2_weights_u16(128);
    io::read_data_from_files(updecoder3_resnet2_norm2_weights_path.c_str(), updecoder3_resnet2_norm2_weights_u16, 128);
    fs::path updecoder3_resnet2_norm2_biases_path = vae_decoder_dir/ "up_blocks-3-resnets-2-norm2-bias_u16.bin";
    std::vector<uint16_t> updecoder3_resnet2_norm2_biases_u16(128);
    io::read_data_from_files(updecoder3_resnet2_norm2_biases_path.c_str(), updecoder3_resnet2_norm2_biases_u16, 128);
    ResNetBlockBF16Weights updecoder3_resnet2 {
        updecoder3_resnet2_norm1_weights_u16.data(),
        updecoder3_resnet2_norm1_biases_u16.data(),
    
        updecoder3_reset2_conv_weights_u16.data(),
        updecoder3_reset2_conv_biases_u16.data(),

        updecoder3_resnet2_norm2_weights_u16.data(),
        updecoder3_resnet2_norm2_biases_u16.data(),

        updecoder3_reset2_conv2_weights_u16.data(),
        updecoder3_reset2_conv2_biases_u16.data(),
    
        
    };
    //////////////////////////////////////updecoder2_reset_1weights ////////////////////////////////////////////////////////////////// 
    fs::path conv_norm_out_weights_path = vae_decoder_dir/ "conv_norm_out-weight_u16.bin";
    std::vector<uint16_t> conv_norm_out_weights_u16(128);
    io::read_data_from_files(conv_norm_out_weights_path.c_str(), conv_norm_out_weights_u16, 128);
    copy_u16_to_bf16_avx512(conv_norm_out_weights_u16.data(), vae_weights.final_conv_norm_out_weight.data(), 128);
    fs::path conv_norm_out_biases_path = vae_decoder_dir/ "conv_norm_out-bias_u16.bin";
    std::vector<uint16_t> conv_norm_out_biases_u16(128);
    io::read_data_from_files(conv_norm_out_biases_path.c_str(), conv_norm_out_biases_u16, 128);
    copy_u16_to_bf16_avx512(conv_norm_out_biases_u16.data(), vae_weights.final_conv_norm_out_bias.data(), 128);
    fs::path conv_out_weights_path = vae_decoder_dir/ "conv_out-weight_u16.bin";
    std::vector<uint16_t> conv_out_weights_u16(3*128*3*3);
    io::read_data_from_files(conv_out_weights_path.c_str(), conv_out_weights_u16, 3*128*3*3);
    copy_u16_to_bf16_avx512(conv_out_weights_u16.data(), vae_weights.final_conv_out_weight.data(), 3*128*3*3);
    fs::path conv_out_biases_path = vae_decoder_dir/ "conv_out-bias_u16.bin";
    std::vector<uint16_t> conv_out_biases_u16(3);
    io::read_data_from_files(conv_out_biases_path.c_str(), conv_out_biases_u16, 3);
    copy_u16_to_bf16_avx512(conv_out_biases_u16.data(), vae_weights.final_conv_out_bias.data(), 3);

   
    //////////////////////////////////////////////load_weights////////////////////////////////////////////////
    std::string xclbin_name_vae = npu_files_path +"/xclbins/vae_128x512.xclbin";
    npu_app_desc accel_desc_vae;
    accel_desc_vae.xclbin_name = xclbin_name_vae;
    accel_desc_vae.app_name = "vae_128x512";
    npu_app app_vae = npu_instance.create_app(accel_desc_vae);
    app_vae.instr_seq->from_file(npu_files_path +"/insts/vae_128x512.txt");
    buffer<float> seq_vae = app_vae.instr_seq->dump().cast_to<float>();


    std::string xclbin_name_vae_256x512 = npu_files_path + "/xclbins/vae_256x512.xclbin";
    npu_app_desc accel_desc_vae_256x512;
    accel_desc_vae_256x512.xclbin_name = xclbin_name_vae_256x512;
    accel_desc_vae_256x512.app_name = "vae_256x512";
    npu_app app_vae_256x512 = npu_instance.create_app(accel_desc_vae_256x512);
    app_vae_256x512.instr_seq->from_file(npu_files_path +"/insts/vae_256x512.txt");
    buffer<float> seq_vae_256x512 = app_vae_256x512.instr_seq->dump().cast_to<float>();


    std::string xclbin_name_vae_512x512 = npu_files_path +"/xclbins/vae_512x512_new.xclbin";
    npu_app_desc accel_desc_vae_512x512, accel_desc_vae_512_256, accel_desc_vae_512_256_all;
    accel_desc_vae_512x512.xclbin_name = xclbin_name_vae_512x512;
    accel_desc_vae_512x512.app_name = "vae_512x512";
    accel_desc_vae_512_256.xclbin_name = xclbin_name_vae_512x512;
    accel_desc_vae_512_256.app_name = "vae_512x512_256";
    accel_desc_vae_512_256_all.xclbin_name = xclbin_name_vae_512x512;
    accel_desc_vae_512_256_all.app_name = "vae_512x512_256_all";
    npu_app app_vae_512x512 = npu_instance.create_app(accel_desc_vae_512x512);
    npu_app app_vae_512x512_256 = npu_instance.create_app(accel_desc_vae_512_256);
    npu_app app_vae_512x512_256_all = npu_instance.create_app(accel_desc_vae_512_256_all);
    app_vae_512x512.instr_seq->from_file(npu_files_path +"/insts/vae_512x512.txt");
    app_vae_512x512_256.instr_seq->from_file(npu_files_path +"/insts/vae_512_256.txt");
    app_vae_512x512_256_all.instr_seq->from_file(npu_files_path +"/insts/vae_512_256_all.txt");
    buffer<float> seq_vae_512x512 = app_vae_512x512.instr_seq->dump().cast_to<float>();
    buffer<float> seq_vae_512x512_256 = app_vae_512x512_256.instr_seq->dump().cast_to<float>();
    buffer<float> seq_vae_512x512_256_all = app_vae_512x512_256_all.instr_seq->dump().cast_to<float>();
  
    std::string xclbin_name_vae_1024x256_256_128 = npu_files_path +"/xclbins/vae_1024_256_128.xclbin";
    npu_app_desc accel_desc_vae_1024x256, accel_desc_vae_1024_256_128, accel_desc_vae_1024_128;
    accel_desc_vae_1024x256.xclbin_name = xclbin_name_vae_1024x256_256_128;
    accel_desc_vae_1024x256.app_name = "vae_1024x256";
    accel_desc_vae_1024_256_128.xclbin_name = xclbin_name_vae_1024x256_256_128;
    accel_desc_vae_1024_256_128.app_name = "vae_1024x256_256_128";
    accel_desc_vae_1024_128.xclbin_name = xclbin_name_vae_1024x256_256_128;
    accel_desc_vae_1024_128.app_name = "vae_1024x256_128";
    npu_app app_vae_1024x256 = npu_instance.create_app(accel_desc_vae_1024x256);
    npu_app app_vae_1024x256_256_128 = npu_instance.create_app(accel_desc_vae_1024_256_128);
    npu_app app_vae_1024x256_128 = npu_instance.create_app(accel_desc_vae_1024_128);
    app_vae_1024x256.instr_seq->from_file(npu_files_path +"/insts/vae_1024_256.txt");
    app_vae_1024x256_256_128.instr_seq->from_file(npu_files_path +"/insts/vae_1024_256_128.txt");
    app_vae_1024x256_128.instr_seq->from_file(npu_files_path +"/insts/vae_1024_128.txt");
    buffer<float> seq_vae_1024x256 = app_vae_1024x256.instr_seq->dump().cast_to<float>();
    buffer<float> seq_vae_1024x256_256_128 = app_vae_1024x256_256_128.instr_seq->dump().cast_to<float>();
    buffer<float> seq_vae_1024x256_128 = app_vae_1024x256_128.instr_seq->dump().cast_to<float>();

    buffer<dtype_in> x_conv_in_vae_1024x256 = app_vae_1024x256.create_bo_buffer<dtype_in>(1026*1024*256, 3);
    buffer<dtype_in> x_conv_kernel_vae_1024x256 = app_vae_1024x256.create_bo_buffer<dtype_in>(256*256*16, 4);
    buffer<dtype_out> x_conv_out_vae_1024x256 = app_vae_1024x256.create_bo_buffer<dtype_out>(1024*1024*256, 5);

    buffer<dtype_in> x_conv_in_vae_1024x256_256_128 = app_vae_1024x256_256_128.create_bo_buffer<dtype_in>(1026*1024*256, 3);
    buffer<dtype_in> x_conv_kernel_vae_1024x256_256_128 = app_vae_1024x256_256_128.create_bo_buffer<dtype_in>(128*256*16, 4);
    buffer<dtype_out> x_conv_out_vae_1024x256_256_128 = app_vae_1024x256_256_128.create_bo_buffer<dtype_out>(1024*1024*128, 5);

    buffer<dtype_in> x_conv_in_vae_1024x256_128 = app_vae_1024x256_128.create_bo_buffer<dtype_in>(1026*1024*128, 3);
    buffer<dtype_in> x_conv_kernel_vae_1024x256_128 = app_vae_1024x256_128.create_bo_buffer<dtype_in>(128*256*16, 4);
    buffer<dtype_out> x_conv_out_vae_1024x256_128 = app_vae_1024x256_128.create_bo_buffer<dtype_out>(1024*1024*128, 5);

   
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string xclbin_name_vae_512_elem = npu_files_path +"/xclbins/conv_512_elem.xclbin";
    npu_app_desc accel_desc_vae_512_elem;
    accel_desc_vae_512_elem.xclbin_name = xclbin_name_vae_512_elem;
    accel_desc_vae_512_elem.app_name = "vae_512_elem";
    npu_app app_vae_512_elem = npu_instance.create_app(accel_desc_vae_512_elem);
    app_vae_512_elem.instr_seq->from_file(npu_files_path +"/insts/conv_512_elem.txt");
    buffer<float> seq_vae_512_elem = app_vae_512_elem.instr_seq->dump().cast_to<float>();

    buffer<dtype_in> x_conv_in_vae_512_elem = app_vae_512_elem.create_bo_buffer<dtype_in>(512*512*512, 3);
    buffer<dtype_in> x_conv_kernel_vae_512_elem = app_vae_512_elem.create_bo_buffer<dtype_in>(512*256*8, 4);
    buffer<dtype_out> x_conv_out_vae_512_elem = app_vae_512_elem.create_bo_buffer<dtype_out>(512*512*256, 5);

    std::string xclbin_name_vae_1024_elem = npu_files_path +"/xclbins/conv_1024_elem.xclbin";
    npu_app_desc accel_desc_vae_1024_elem;
    accel_desc_vae_1024_elem.xclbin_name = xclbin_name_vae_1024_elem;
    accel_desc_vae_1024_elem.app_name = "vae_1024_elem";
    npu_app app_vae_1024_elem = npu_instance.create_app(accel_desc_vae_1024_elem);
    app_vae_1024_elem.instr_seq->from_file(npu_files_path +"/insts/conv_1024_elem.txt");
    buffer<float> seq_vae_1024_elem = app_vae_1024_elem.instr_seq->dump().cast_to<float>();

    buffer<dtype_in> x_conv_in_vae_1024_elem = app_vae_1024_elem.create_bo_buffer<dtype_in>(1026*1024*256, 3);
    buffer<dtype_in> x_conv_kernel_vae_1024_elem = app_vae_1024_elem.create_bo_buffer<dtype_in>(128*256*8, 4);
    buffer<dtype_out> x_conv_out_vae_1024_elem = app_vae_1024_elem.create_bo_buffer<dtype_out>(1024*1024*128, 5);
   
    ////////load latent bin ////////////////////////////////////////////////////////////////////
    int vae_steps = 1; 
    auto start_vae = std::chrono::steady_clock::now();
    buffer<float> latent_input(16*latent_H*latent_W);
    for(int i = 0; i < 16*latent_H*latent_W; i++){
      latent_input[i] = update_img[step-1][i]/0.3611 + 0.1159;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    buffer<dtype_in> x_conv_in(latent_H*latent_W*512);
  
    conv2d_nchw_native_padded(
        latent_input.data(), 
       (vae_weights.conv_in_weight.data()), 
        (vae_weights.conv_in_bias.data()), 
        reinterpret_cast<uint16_t*>(x_conv_in.data()),
        latent_H, latent_W, 16, 512, 3, 3, 1, 1 // <-- Pass PAD=1 here
    );
 
  
    ///////////////////////////////////////////////ResnetBlock2D_0 modules ////////////////////////////////////////////////////////
    buffer<dtype_in> x_updecoder0_resnet_same_2(latent_H*latent_W*512);
   {
    buffer<dtype_in> x_conv_in_vae = app_vae.create_bo_buffer<dtype_in>(130*128*512, 3);
    buffer<dtype_in> x_conv_kernel_vae = app_vae.create_bo_buffer<dtype_in>(512*512*16, 4);
    buffer<dtype_out> x_conv_out_vae = app_vae.create_bo_buffer<dtype_out>(128*128*512, 5);

    buffer<dtype_in> x_conv_in_norm1(latent_H*latent_W*512);
    groupnorm_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_in.data()),
    reinterpret_cast<bfloat16*>(x_conv_in_norm1.data()), 
    reinterpret_cast<const bfloat16*>(vae_weights.mid_block_resnets_0_norm1_weight.data()), 
    reinterpret_cast<const bfloat16*>(vae_weights.mid_block_resnets_0_norm1_bias.data()),
     1, 512, latent_H, latent_W, 32, 1e-6);
    swish_avx512_bf16_pad_hw_chw(reinterpret_cast<const uint16_t*>(x_conv_in_norm1.data()),
    reinterpret_cast<uint16_t*>(x_conv_in_vae.data()), 512, latent_H, latent_W, 130, 128);
       
  
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      for(int i = 0; i < 512*512; i++){
          for(int j = 0; j < 9; j++){
              x_conv_kernel_vae[i*16 + j] = vae_weights.mid_block_resnets_0_weight[i*9 + j];
          }
      }
      for (int n = 0; n < 512; n++) {
          int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
          x_conv_kernel_vae[idx] = vae_weights.mid_block_resnets_0_bias[n];        // example value
      }
      x_conv_in_vae.sync_to_device();
      x_conv_kernel_vae.sync_to_device();
      app_vae(x_conv_in_vae, x_conv_kernel_vae, x_conv_out_vae);
      x_conv_out_vae.sync_from_device();

  
    //////////////////////////////////////////////////////////////////////////////////
   
    ////////////////////////////////////////////////////////////////////////////////////
    buffer<dtype_in> x_conv_in_norm2(latent_H*latent_W*512);
    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(x_conv_out_vae.data()),
        reinterpret_cast<bfloat16*>(x_conv_in_norm2.data()), 
        reinterpret_cast<const bfloat16*>(vae_weights.mid_block_resnets_0_norm2_weight.data()), 
        reinterpret_cast<const bfloat16*>(vae_weights.mid_block_resnets_0_norm2_bias.data()),
        1, 512, latent_H, latent_W, 128, 128, 0, 0, 32, 1e-6);
    buffer<dtype_in> x_norm2_swish_conv2(128*130*512);
    swish_avx512_bf16_pad_hw_chw(reinterpret_cast<const uint16_t*>(x_conv_in_norm2.data()),
        reinterpret_cast<uint16_t*>(x_conv_in_vae.data()), 512, latent_H, latent_W, 130, 128);
    time_utils::time_point one_swish_bf16 = time_utils::now();
   
    for(int i = 0; i < 512*512; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae[i*16 + j] = vae_weights.mid_block_resnets_0_conv2_weight[i*9 + j];
        }
    }
    for (int n = 0; n < 512; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae[idx] = vae_weights.mid_block_resnets_0_conv2_bias[n];        // example value
    }


    x_conv_in_vae.sync_to_device();
    x_conv_kernel_vae.sync_to_device();
    app_vae(x_conv_in_vae, x_conv_kernel_vae, x_conv_out_vae);
    x_conv_out_vae.sync_from_device();
   

    buffer<dtype_in> x_resnet_same_0(latent_H*latent_W*512);
    add_general_avx512_bf16(
        reinterpret_cast<const bfloat16*>(x_conv_out_vae.data()), //input1
        reinterpret_cast<const bfloat16*>(x_conv_in.data()),      //input2
        reinterpret_cast<bfloat16*>(x_resnet_same_0.data()),      //output
        512, latent_H, latent_W, 
        128, 128,
        latent_H, latent_W
    );
    

    ///////////////////////////////////////////////ResnetBlock2D_1 modules //////////////////////////////////////////////////////// 
    buffer<dtype_in> x_resnet_same_1(latent_H*latent_W*512);

   
    resnet_block_bf16_128x128_npu(
        reinterpret_cast<const bfloat16*>(x_resnet_same_0.data()),
        reinterpret_cast<bfloat16*>(x_resnet_same_1.data()),
         mid1, app_vae, x_conv_in_vae, x_conv_kernel_vae,
          x_conv_out_vae, latent_H, latent_W, 512, 130, 128, 128);
       ///////////////////////////////////////////////updecoderBlock2d_0 image ////////////////////////////////////////////////////////
    buffer<dtype_in> x_updecoder0_resnet_same_0(latent_H*latent_W*512);

   
    resnet_block_bf16_128x128_npu(
        reinterpret_cast<const bfloat16*>(x_resnet_same_1.data()),
        reinterpret_cast<bfloat16*>(x_updecoder0_resnet_same_0.data()), 
        updecoder0_resnet0, app_vae, x_conv_in_vae, x_conv_kernel_vae,
         x_conv_out_vae, latent_H, latent_W, 512, 130, 128, 128);

    buffer<dtype_in> x_updecoder0_resnet_same_1(latent_H*latent_W*512);
   
  
    resnet_block_bf16_128x128_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder0_resnet_same_0.data()),
        reinterpret_cast<bfloat16*>(x_updecoder0_resnet_same_1.data()),
         updecoder0_resnet1, app_vae, x_conv_in_vae, x_conv_kernel_vae, 
         x_conv_out_vae, latent_H, latent_W, 512, 130, 128, 128);

  

   
    resnet_block_bf16_128x128_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder0_resnet_same_1.data()),
        reinterpret_cast<bfloat16*>(x_updecoder0_resnet_same_2.data()),
         updecoder0_resnet2, app_vae, x_conv_in_vae, x_conv_kernel_vae, 
         x_conv_out_vae, latent_H, latent_W, 512, 130, 128, 128);
    
    }

    buffer<dtype_in> x_updecoder1_resnet_same_2(double_H*double_W*512);
  
    {
    buffer<dtype_in> x_conv_in_vae_256x512 = app_vae_256x512.create_bo_buffer<dtype_in>(258*256*512, 3);
    buffer<dtype_in> x_conv_kernel_vae_256x512 = app_vae_256x512.create_bo_buffer<dtype_in>(512*512*16, 4);
    buffer<dtype_out> x_conv_out_vae_256x512 = app_vae_256x512.create_bo_buffer<dtype_out>(256*256*512, 5);

    buffer<dtype_in> x_updecoder0_upsample_input(256*256*512);
    nearest_neighbor_pad_avx512_bf16(
        reinterpret_cast<const uint16_t*>(x_updecoder0_resnet_same_2.data()),
        reinterpret_cast<uint16_t*>(x_updecoder0_upsample_input.data()),
        512, latent_H, latent_W, 256, 256
    );

   
    for (int c = 0; c < 512; ++c)
        for (int h = 0; h < 256; ++h) {
            size_t in_base  = ((size_t)c * 256 + h) * 256;
            size_t out_base = ((size_t)c * 258 + (h + 1)) * 256;
            for (int w = 0; w < 256; ++w) {
            x_conv_in_vae_256x512[out_base + w] = (x_updecoder0_upsample_input[in_base + w]); // if cast supported
            }
    }
    for(int i = 0; i < 512*512; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae_256x512[i*16 + j] = vae_weights.updecoder0_upsample_weight[i*9 + j];
        }
    }
    for (int n = 0; n < 512; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae_256x512[idx] = vae_weights.updecoder0_upsample_bias[n];        // example value
    }
    x_conv_in_vae_256x512.sync_to_device();
    x_conv_kernel_vae_256x512.sync_to_device();
    app_vae_256x512(x_conv_in_vae_256x512, x_conv_kernel_vae_256x512, x_conv_out_vae_256x512);
    x_conv_out_vae_256x512.sync_from_device();

    // //////////////////////////////////////////////////////////////////////////////////////////
    
    // ////////////////////////////////////////////////////////////////////////////////////////
   
    //    ///////////////////////////////////////////////updecoderBlock2d_1 image ////////////////////////////////////////////////////////
    buffer<dtype_in> x_updecoder1_resnet_same_0(double_H*double_W*512);
    resnet_block_bf16_256x256_npu(
        reinterpret_cast<const bfloat16*>(x_conv_out_vae_256x512.data()),
        reinterpret_cast<bfloat16*>(x_updecoder1_resnet_same_0.data()), updecoder1_resnet0, 
        app_vae_256x512, x_conv_in_vae_256x512, x_conv_kernel_vae_256x512, 
        x_conv_out_vae_256x512, double_H, double_W, 512, 258, 256, 256);
    
    buffer<dtype_in> x_updecoder1_resnet_same_1(256*double_W*512);

    resnet_block_bf16_128x128_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder1_resnet_same_0.data()),
        reinterpret_cast<bfloat16*>(x_updecoder1_resnet_same_1.data()), updecoder1_resnet1, 
        app_vae_256x512, x_conv_in_vae_256x512, x_conv_kernel_vae_256x512,
         x_conv_out_vae_256x512, double_H, double_W, 512, 258, 256, 256);
  
   
  
  
    resnet_block_bf16_128x128_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder1_resnet_same_1.data()),
        reinterpret_cast<bfloat16*>(x_updecoder1_resnet_same_2.data()), updecoder1_resnet2, 
        app_vae_256x512, x_conv_in_vae_256x512, x_conv_kernel_vae_256x512, 
        x_conv_out_vae_256x512, double_H, double_W, 512, 258, 256, 256);
    }

    buffer<dtype_in> x_updecoder2_reset_2(256*512*512);
    {
        buffer<dtype_in> x_conv_in_vae_512x512 = app_vae_512x512.create_bo_buffer<dtype_in>(514*512*512, 3);
        buffer<dtype_in> x_conv_kernel_vae_512x512 = app_vae_512x512.create_bo_buffer<dtype_in>(512*512*16, 4);
        buffer<dtype_out> x_conv_out_vae_512x512 = app_vae_512x512.create_bo_buffer<dtype_out>(512*512*512, 5);
    
        buffer<dtype_in> x_conv_in_vae_512x512_256 = app_vae_512x512_256.create_bo_buffer<dtype_in>(514*512*512, 3);
        buffer<dtype_in> x_conv_kernel_vae_512x512_256 = app_vae_512x512_256.create_bo_buffer<dtype_in>(256*512*16, 4);
        buffer<dtype_out> x_conv_out_vae_512x512_256 = app_vae_512x512_256.create_bo_buffer<dtype_out>(512*512*256, 5);
    
        buffer<dtype_in> x_conv_in_vae_512x512_256_all = app_vae_512x512_256_all.create_bo_buffer<dtype_in>(514*512*256, 3);
        buffer<dtype_in> x_conv_kernel_vae_512x512_256_all = app_vae_512x512_256_all.create_bo_buffer<dtype_in>(512*256*16, 4);
        buffer<dtype_out> x_conv_out_vae_512x512_256_all = app_vae_512x512_256_all.create_bo_buffer<dtype_out>(512*512*256, 5);
    buffer<dtype_in> x_updecoder1_upsample_input(512*512*512);
    nearest_neighbor_pad_avx512_bf16(
        reinterpret_cast<const uint16_t*>(x_updecoder1_resnet_same_2.data()),
        reinterpret_cast<uint16_t*>(x_updecoder1_upsample_input.data()),
        512, double_H, double_W, 512, 512
    );
   
    for (int c = 0; c < 512; ++c)
    for (int h = 0; h < 512; ++h) {
        size_t in_base  = ((size_t)c * 512 + h) * 512;
        size_t out_base = ((size_t)c * 514 + (h + 1)) * 512;
        for (int w = 0; w < 512; ++w) {
        x_conv_in_vae_512x512[out_base + w] = (x_updecoder1_upsample_input[in_base + w]); // if cast supported
        }
    }
    for(int i = 0; i < 512*512; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae_512x512[i*16 + j] = vae_weights.updecoder1_upsample_weight[i*9 + j];
        }
    }
    for (int n = 0; n < 512; n++) {
        int idx = ((n * 512) + 511) * 16 + 9;   // [n][511][9]
        x_conv_kernel_vae_512x512[idx] = vae_weights.updecoder1_upsample_bias[n];        // example value
    }
    x_conv_in_vae_512x512.sync_to_device();
    x_conv_kernel_vae_512x512.sync_to_device();
    app_vae_512x512(x_conv_in_vae_512x512, x_conv_kernel_vae_512x512, x_conv_out_vae_512x512);
    x_conv_out_vae_512x512.sync_from_device();
    // ///////////////////////////////////////////////updecoderBlock2d_2_0 image ////////////////////////////////////////////////////////
  
    buffer<dtype_in> x_updecoder2_reset_0(256*512*512);

    decoder_resnet_block_bf16_512_256_npu(
        reinterpret_cast<const bfloat16*>(x_conv_out_vae_512x512.data()),
        reinterpret_cast<bfloat16*>(x_updecoder2_reset_0.data()),
        updecoder2_resnet0,
        tri_H, tri_W,   // H, W
        512, 256,    // Cin, Cout,
        app_vae_512x512_256_all, x_conv_in_vae_512x512_256_all, x_conv_kernel_vae_512x512_256_all, x_conv_out_vae_512x512_256_all,
        app_vae_512x512_256, x_conv_in_vae_512x512_256, x_conv_kernel_vae_512x512_256, x_conv_out_vae_512x512_256,
        app_vae_512_elem, x_conv_in_vae_512_elem, x_conv_kernel_vae_512_elem, x_conv_out_vae_512_elem,
        514, 512, 512
    );
   
    // ///////////////////////////////////////////////updecoderBlock2d_2_1 image ////////////////////////////////////////////////////////
 
    buffer<dtype_in> x_updecoder2_reset_1(256*512*512);
    
    
    resnet_block_512_256_all_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder2_reset_0.data()),
        reinterpret_cast<bfloat16*>(x_updecoder2_reset_1.data()), updecoder2_resnet1, 
        app_vae_512x512_256_all, x_conv_in_vae_512x512_256_all, x_conv_kernel_vae_512x512_256_all,
         x_conv_out_vae_512x512_256_all, tri_H, tri_W, 256, 514, 512, 512);

  
//     // ///////////////////////////////////////////////updecoderBlock2d_2_2 image ////////////////////////////////////////////////////////


   

    resnet_block_512_256_all_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder2_reset_1.data()),
        reinterpret_cast<bfloat16*>(x_updecoder2_reset_2.data()), updecoder2_resnet2, 
        app_vae_512x512_256_all, x_conv_in_vae_512x512_256_all, 
        x_conv_kernel_vae_512x512_256_all, x_conv_out_vae_512x512_256_all, 
        tri_H, tri_W, 256, 514, 512, 512);
    }
    buffer<dtype_in> x_updecoder2_upsample_input(256*1024*1024);
    nearest_neighbor_512x128x128_avx512(
        reinterpret_cast<const uint16_t*>(x_updecoder2_reset_2.data()),
        reinterpret_cast<uint16_t*>(x_updecoder2_upsample_input.data()),
        256, 512, 512
    );
    
    
 
    buffer<dtype_in> x_updecoder2_upsample(256*1024*1024);
  
  
    for (int c = 0; c < 256; ++c)
    for (int h = 0; h < 1024; ++h) {
        size_t in_base  = ((size_t)c * 1024 + h) * 1024;
        size_t out_base = ((size_t)c * 1026 + (h + 1)) * 1024;
        for (int w = 0; w < 1024; ++w) {
        x_conv_in_vae_1024x256[out_base + w] = (x_updecoder2_upsample_input[in_base + w]); // if cast supported
        }
    }
    for(int i = 0; i < 256*256; i++){
        for(int j = 0; j < 9; j++){
            x_conv_kernel_vae_1024x256[i*16 + j] = vae_weights.updecoder2_upsample_weight[i*9 + j];
        }
    }
    for (int n = 0; n < 256; n++) {
        int idx = ((n * 256) + 255) * 16 + 9;   // [n][255][9]
        x_conv_kernel_vae_1024x256[idx] = vae_weights.updecoder2_upsample_bias[n];        // example value
    }
    x_conv_in_vae_1024x256.sync_to_device();
    x_conv_kernel_vae_1024x256.sync_to_device();
    app_vae_1024x256(x_conv_in_vae_1024x256, x_conv_kernel_vae_1024x256, x_conv_out_vae_1024x256);
    x_conv_out_vae_1024x256.sync_from_device();
//     /////////////////////////////////////////////////////////////////////
 
//     ///////////////////////////////////////////////////////////////////////////////////
//     ///////////////////////////////////////////////////upblock3//////////////////////////////////////////////////////////////////////////////

    buffer<dtype_in> x_updecoder3_reset_0(128*1024*1024);
    
    decoder_resnet_block_bf16_1024_128_npu(
        reinterpret_cast<const bfloat16*>(x_conv_out_vae_1024x256.data()),
        reinterpret_cast<bfloat16*>(x_updecoder3_reset_0.data()),
        updecoder3_resnet0,
        1024, 1024,   // H, W
        256, 128,    // Cin, Cout,
        app_vae_1024x256_128, x_conv_in_vae_1024x256_128, x_conv_kernel_vae_1024x256_128, x_conv_out_vae_1024x256_128,
        app_vae_1024x256_256_128, x_conv_in_vae_1024x256_256_128, x_conv_kernel_vae_1024x256_256_128, x_conv_out_vae_1024x256_256_128,
        app_vae_1024_elem, x_conv_in_vae_1024_elem, x_conv_kernel_vae_1024_elem, 
        x_conv_out_vae_1024_elem, 1026, 1024, 1024
        
    );
  
//     // /////////////////////////////////////////////updecoderBlock2d_3_1 image ////////////////////////////////////////////////////////

    buffer<dtype_in> x_updecoder3_reset_1(128*1024*1024);
   
   
    resnet_block_1024_128_all_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder3_reset_0.data()),
        reinterpret_cast<bfloat16*>(x_updecoder3_reset_1.data()), updecoder3_resnet1,
        app_vae_1024x256_128, x_conv_in_vae_1024x256_128, x_conv_kernel_vae_1024x256_128, x_conv_out_vae_1024x256_128,
        1024, 1024, 128, 1026, 1024, 1024
    );
    
    // /////////////////////////////////////////////updecoderBlock2d_3_2 image ////////////////////////////////////////////////////////

    buffer<dtype_in> x_updecoder3_reset_2(128*1024*1024);
 
    resnet_block_1024_128_all_npu(
        reinterpret_cast<const bfloat16*>(x_updecoder3_reset_1.data()),
        reinterpret_cast<bfloat16*>(x_updecoder3_reset_2.data()), updecoder3_resnet2,
        app_vae_1024x256_128, x_conv_in_vae_1024x256_128, x_conv_kernel_vae_1024x256_128, x_conv_out_vae_1024x256_128,
        1024, 1024, 128, 1026, 1024, 1024
    );
   
    // ///////////////////////////////////////////////////conv_norm_out//////////////////////////////////////////////////////////////////////////////
    buffer<dtype_in> x_conv_norm_out(128*image_H*image_W);
    groupnorm_unpad_avx512_bf16(reinterpret_cast<const bfloat16*>(x_updecoder3_reset_2.data()),
    reinterpret_cast<bfloat16*>(x_conv_norm_out.data()), 
    reinterpret_cast<const bfloat16*>(vae_weights.final_conv_norm_out_weight.data()), 
    reinterpret_cast<const bfloat16*>(vae_weights.final_conv_norm_out_bias.data()),
    1, 128, image_H, image_W,1024, 1024, 0, 0, 32, 1e-6);
    
    buffer<dtype_in> x_silu_conv_in(128*image_H*image_W);
    silu_bf16_avx512(reinterpret_cast<const uint16_t*>(x_conv_norm_out.data()), 
    reinterpret_cast<uint16_t*>(x_silu_conv_in.data()), image_H*image_W*128);
    
    buffer<dtype_in> x_vae_output(3*image_H*image_W);
   
    conv2d_bf16_avx512_optimized(
        reinterpret_cast<const uint16_t*>(x_silu_conv_in.data()), 
        reinterpret_cast<const uint16_t*>(vae_weights.final_conv_out_weight.data()), 
        reinterpret_cast<const uint16_t*>(vae_weights.final_conv_out_bias.data()), 
        reinterpret_cast<uint16_t*>(x_vae_output.data()),
        image_H, image_W, 128, 3); //(H, W, Cin, Cout, K_H, K_W, STRIDE, PAD)

    time_utils::time_point end_conv2d_cnn = time_utils::now();
   
 
   
    uint8_t* x_vae_output_uint8 = new uint8_t[image_H*image_W*3];
    ProcessVaeOutputBF16(reinterpret_cast<const uint16_t*>(
        x_vae_output.data()), x_vae_output_uint8, image_H, image_W, 3);
    save_rgb_png(output_path.c_str(),  x_vae_output_uint8, image_W, image_H);
    delete[] x_vae_output_uint8;
    print_progress_bar("VAE Decoder", 0, vae_steps, start_vae);
    std::cout << std::endl;
    time_utils::time_point end_begin = time_utils::now();
    std::cout << "----------------------------------------------------whole time: " << time_utils::duration_us(start_begin, end_begin).first / 1000 << "ms" << std::endl;
   
    ////////////////////////////////////////////////////////////////////////////////
    return 0;
}

