#pragma once

#include "server_config.hpp"

#include <string>

namespace server {

// Per-request inputs forwarded to the selected model worker.
struct GenParams {
    const config::ModelConfig* model = &config::default_model();
    int seed = 42;
    int steps = 4;
    std::string prompt;  // Omitted prompt keeps the worker's default.
    std::string input_image;  // Edit models: prepared 1024x1024 image path.
};

}  // namespace server
