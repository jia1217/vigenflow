#pragma once

#include "server_config.hpp"

#include <optional>
#include <string>
#include <vector>

namespace server {

// Per-request inputs forwarded to the selected model worker.
struct GenParams {
    const config::ModelConfig* model = &config::default_model();
    int seed = 42;
    int steps = 4;
    int width = config::kImageWidth;
    int height = config::kImageHeight;
    std::string prompt;  // Omitted prompt keeps the worker's default.
    std::string input_image;  // Edit models: prepared 1024x1024 image path.
    // Names from the worker's loras.jsonc, passed as --lora (empty: --lora none);
    // nullopt when the worker has no LoRA list.
    std::optional<std::vector<std::string>> loras;
};

}  // namespace server
