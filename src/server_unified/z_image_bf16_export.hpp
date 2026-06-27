#pragma once

#include <filesystem>
#include <string>

namespace server {

struct ZImageBf16ExportOptions {
    std::string repo_id;
    std::string revision;
    std::string hf_token;
    std::filesystem::path cache_dir;
    std::filesystem::path output_root;
    bool force = false;
};

void prepare_z_image_bf16_weights_cpp(const ZImageBf16ExportOptions& options);

}  // namespace server
