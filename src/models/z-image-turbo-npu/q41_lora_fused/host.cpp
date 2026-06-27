#ifndef HOST_FLUX_LIBRARY_IMPL

extern "C" int zimage_bf16_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return zimage_bf16_host_main(argc, argv);
}

#else


#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
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
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
#include <unordered_map>
#include <cstring>
#include <thread> 
#include <iostream>
#include <malloc.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace po = boost::program_options;
namespace fs = std::filesystem;

#include "host_utils.hpp"
#include "load_weights.hpp"
#include "copy_data.hpp"
#include "denoise_lib.hpp"
#include "vae_cnn.hpp"
#include "load_lora.hpp"
#include "lora_hf_export.hpp"
#include "mvm_sequence.hpp"
#include "qwen_lib.hpp"
#include "q4_1_dequant.h"

extern "C" int zimage_bf16_host_main(int argc, const char *argv[]) {
    std::string weights_path   = "/model_weights";
    std::string npu_files_path = "/vigenflow/npu_files/z-image-turbo";
    std::string lora_source;
    std::string lora_dir;
    int seed                   = 42;
    int image_H                = 1024;
    int image_W                = 1024;
    int step                   = 4;
    int lora_rank              = 32;
    std::string lora_file;
    
    std::string my_prompt = "Pixel art style. anime image of bat_eared_fox_kemono_friends sitting in a cafe with a steaming cup of coffee on the table in front of her. she is looking to the side at the viewer with a light smile.";
    std::string output_path    = "./output.png";
    std::string gguf_path_arg;
    std::string gguf_hf_repo;
    std::string gguf_file;
    std::string gguf_revision = "main";
    std::string gguf_hf_token;
    bool force_gguf_download = false;

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
            ("gguf_path", po::value<std::string>(&gguf_path_arg),
                "Local GGUF file to parse and pack; omit to use existing packed weights")
            ("gguf_repo", po::value<std::string>(&gguf_hf_repo),
                "Hugging Face repo name, for example unsloth/Z-Image-Turbo-GGUF")
            ("gguf_file", po::value<std::string>(&gguf_file),
                "GGUF filename in the Hugging Face repo, for example z-image-turbo-Q4_1.gguf")
            ("gguf_revision", po::value<std::string>(&gguf_revision)->default_value("main"),
                "Hugging Face revision, branch, or commit for the GGUF file")
            ("gguf_hf_token", po::value<std::string>(&gguf_hf_token),
                "Hugging Face access token; defaults to HF_TOKEN")
            ("force_gguf_download", po::bool_switch(&force_gguf_download),
                "Download the GGUF again when --gguf_repo and --gguf_file are provided")
            ("lora_source", po::value<std::string>(&lora_source),
                "Hugging Face LoRA repo id or file URL (e.g. Kelsey1217/Z-Image-Turbo-npu)")
                ("lora_dir", po::value<std::string>(&lora_dir),
                "Directory for generated LoRA-only bf16 bin folders")
            ("lora_rank", po::value<int>(&lora_rank),
                "LoRA rank used when fusing weights into base matrices")
            ("lora_file", po::value<std::string>(&lora_file),
                "LoRA file used when fusing weights into base matrices");

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
    std::cout << "LoRA rank = " << lora_rank << "\n";

    /////////////////////////////////////////////////////////////////////////////////////////////
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }
    /////////////////////////////////////////////////////////////////////////
    #include "common_inls/host_load_gguf.inl"
    std::string qwen_qkv_mm_xclbin = npu_files_path + "/xclbins/MM_128_round.xclbin";
    time_utils::time_with_unit npu_time = {0.0, "us"};
    #include "common_inls/host_config.inl"
    #include "host_lora_weights.inl"
    #include "common_inls/host_text_encoder.inl"
    #include "host_denoising_q41_lora.inl"
    #include "common_inls/host_vae_decoder.inl"
    /////////////////////////////////////////////////////////////////////////////////////vae decoder///////////////////////////////////

    return 0;
}

#endif
