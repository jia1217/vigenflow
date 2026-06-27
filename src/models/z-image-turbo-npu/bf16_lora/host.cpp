#ifndef HOST_FLUX_LIBRARY_IMPL

extern "C" int zimage_bf16_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return zimage_bf16_host_main(argc, argv);
}

#else


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
#include "npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
#include <unordered_map>
#include <cstring>
#include <thread> 
#include <iostream>
#include <malloc.h>

namespace po = boost::program_options;
namespace fs = std::filesystem;


#include "fused_lora.hpp"
#include "host_utils.hpp"
#include "load_weights.hpp"
#include "mvm_sequence.hpp"
#include "lora_hf_export.hpp"
#include "qwen_lib.hpp"
#include "copy_data.hpp"
#include "denoise_lib.hpp"
#include "vae_cnn.hpp"
#include "npu_setup_runtime.hpp"
extern "C" int zimage_bf16_host_main(int argc, const char *argv[]) {
   
    std::string weights_path   = "/model_weights";
    std::string npu_files_path = "/vigenflow/npu_files/z-image-turbo";
    std::string lora_source;
    std::string lora_dir;
    std::string lora_file;
    int lora_rank              = 32;
    float lora_scale             = 1.0f;
    int seed                   = 42;
    int image_H                = 1024;
    int image_W                = 1024;
    int step                   = 4;
    std::string my_prompt = "Pixel art style, Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
    "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
    "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
    "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";

    std::string output_path    = "./images/output.png";

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Show this help message and exit")
            ("output,o", po::value<std::string>(&output_path)->default_value("./output.png"),
                "Path to save the generated image")
            ("weights_path,w", po::value<std::string>(&weights_path),
                "Path to model weights directly")
            ("npu_files_path,n", po::value<std::string>(&npu_files_path)->default_value("/opt/vigenflow/npu_files/z-image-turbo"),
                "Path to NPU files directory")
            ("seed", po::value<int>(&seed)->default_value(42), "Random seed")
            ("steps", po::value<int>(&step)->default_value(4), "Number of sample steps")
            ("height,H", po::value<int>(&image_H)->default_value(1024),
                "Image height, in pixel space")
            ("width,W", po::value<int>(&image_W)->default_value(1024),
                "Image width, in pixel space")
            ("prompt,p", po::value<std::string>(&my_prompt), "Text prompt for image generation")
            ("lora_source", po::value<std::string>(&lora_source),
                "Hugging Face LoRA repo id or file URL (e.g. Kelsey1217/Z-Image-Turbo-npu)")
            ("lora_dir", po::value<std::string>(&lora_dir),
                "Directory for generated LoRA-only bf16 bin folders")
            ("lora_rank", po::value<int>(&lora_rank),
                "LoRA rank used when fusing weights into base matrices")
            ("lora_file", po::value<std::string>(&lora_file),
                "LoRA file used when fusing weights into base matrices")
            ("lora_scale", po::value<float>(&lora_scale), "Scale factor for LoRA weights");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << "usage: ./run.exe [options]\n\n" << desc << "\n";
            return 0;
        }

    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Run with --help for usage information.\n";
        return 1;
    }
    std::cout << "H = " << image_H << "\n";
    std::cout << "W = " << image_W << "\n";
    std::cout << "Prompt = " << my_prompt << "\n";
    std::cout << "LoRA rank size = " << lora_rank << "\n";
  
    ////////////////////////////////////////////////////////////////////////////////
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }
    /////////////////////////////////////////////////////////////////////////
    std::string qwen_qkv_mm_xclbin =  npu_files_path + "/xclbins/MM_128_round.xclbin";
    time_utils::time_with_unit npu_time = {0.0, "us"};
    #include "common_inls/host_config.inl"
    #include "host_lora_weights_bf16.inl"
    #include "common_inls/host_text_encoder.inl" 
    #include "host_denoising_bf16_lora.inl"
    #include "common_inls/host_vae_decoder.inl"
    ////////////////////////////////////////////////////////////////////////////////
    return 0;
}

#endif
