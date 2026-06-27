#ifndef HOST_FLUX_LIBRARY_IMPL

extern "C" int flux_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return flux_host_main(argc, argv);
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
#define STB_IMAGE_IMPLEMENTATION 
#include "stb_image.h"
#include <malloc.h>
#include <nlohmann/json.hpp>
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "qwen_lib.hpp"
#include "denoise_lib.hpp"
#include "lodepng.h"
#include "load_weights.hpp"
#include "klein_lib.hpp"
#include "typedef.hpp"
#include "mvm_sequence.hpp"
#include "encoder_img.hpp"
#include "npu_setup_runtime.hpp"
#include "host_utils.hpp"
#include "pad_qk.hpp"
#include "copy_data.hpp"
#include "vae_cnn_text.hpp"
#include "vae_encoder.hpp"
#include "fused_lora.hpp"
#include "lora_hf_export.hpp"
namespace po = boost::program_options;
namespace fs = std::filesystem;
using host_bf16  = dtype_out;      // your type, e.g. __bf16

using json = nlohmann::json;


extern "C" int flux_host_main(int argc, const char *argv[]) {
    std::string weights_path   = "/home/kelsey/NPU_projects/model_weights";
    std::string npu_files_path = "/home/kelsey/NPU_projects/NPU_repo/host_flux_klein_bf16_img_lora_test/src/npu_files/flux.2-klein-4B";
    int seed                   = 43;
    int image_H                = 1024;
    int image_W                = 1024;
    int step                   = 4;
    int round_mode_num         = 0;
    std::string lora_source;
    std::string lora_dir;
    std::string lora_file;
    int lora_rank              = 0;
    float lora_scale            = 1.0f;

    std::string my_prompt = "2x2 sprite sheet";
    std::string output_path    = "./images/output.png";
    std::vector<std::string> input_images;

    try {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show this help message and exit")
        ("output,o", po::value<std::string>(&output_path)->default_value("./output.png"),
            "Path to save the generated image")
        ("input_image", po::value<std::vector<std::string>>(&input_images)->composing(),
            "Path to an input reference image (can be used multiple times)")
        ("weights_path,w", po::value<std::string>(&weights_path),
            "Path to model weights directly")
        ("npu_files_path,n", po::value<std::string>(&npu_files_path),
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
        ("lora_rank", po::value<int>(&lora_rank)->default_value(32),
            "LoRA rank used when fusing weights into base matrices")
        ("lora_file", po::value<std::string>(&lora_file),
            "LoRA file used when fusing weights into base matrices")
        ("lora_scale", po::value<float>(&lora_scale)->default_value(1.0f),
            "LoRA scaling factor used when fusing weights into base matrices");
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
    if (input_images.empty()) {
        input_images.push_back("../../models/flux.2-klein-4B-edit-npu/edit_test.png");
    }
    std::cout << "H = " << image_H << "\n";
    std::cout << "W = " << image_W << "\n";
    std::cout << "Prompt = " << my_prompt << "\n";
   
    
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }
    //////////////////////////////////////////encoder img////////////////////////////////////////
    #include "host_flux2_config_edit.inl"
    #include "host_input_img.inl"
    #include "host_flux2_klein_load_lora.inl"
    #include "host_flux2_vae_encoder.inl"
    #include "host_flux2_text_encoder.inl"
    #include "host_flux2_denoising_edit.inl"
    {
    #include "host_flux2_vae_decoder.inl"
    }
 

    return 0;
}

#endif
