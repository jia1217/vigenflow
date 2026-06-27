#pragma once

#include <string>
#include <vector>

namespace server {

struct GenParams {
    int H = 1024;
    int W = 1024;
    int steps = 4;
    int seed = 42;
    std::string model;
    std::string prompt =
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup...";
    std::string lora_dir;
    std::string lora_source;
    std::string lora_file;
    std::string lora_revision;
    std::string lora_cache_dir;
    std::string lora_export_dir;
    bool lora_force = false;
    bool lora_keep_cache = false;
    int lora_rank = 0;
    float lora_scale = 0.0f;
    std::string gguf_path;
    std::string gguf_repo;
    std::string gguf_file;
    std::string gguf_revision;
    std::string gguf_hf_token;
    bool force_gguf_download = false;
    std::vector<std::string> input_images;
};

struct WorkerPaths {
    std::string npu_files;
    std::string exe;
};

}  // namespace server
