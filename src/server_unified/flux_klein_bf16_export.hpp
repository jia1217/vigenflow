#pragma once

#include <filesystem>
#include <string>

namespace server {

struct FluxKleinBf16DownloadOptions {
    std::string repo_id;
    std::string revision;
    std::string hf_token;
    std::filesystem::path cache_dir;
    std::filesystem::path output_root;
    bool force = false;
};

void prepare_flux_klein_bf16_weights_cpp(const FluxKleinBf16DownloadOptions& options);

}  // namespace server
