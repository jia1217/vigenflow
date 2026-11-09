#pragma once

#include "server_config.hpp"

#include <optional>
#include <string>
#include <vector>

namespace server {

// One entry of a worker's loras.jsonc, read the way its run.exe --list_loras reads it.
struct LoraInfo {
    std::string name;
    std::string file;  // file name only; clients do not need the local folder
    float scale = 1.0f;
    std::string trigger, license;
    bool enabled = false;  // used when a request does not choose LoRAs
    std::string status;    // "ok", "download" (fetched from its source on first use) or "missing"
};

// The LoRA list beside the model's run.exe; nullopt when the worker has none.
// Throws std::runtime_error when the file cannot be parsed.
std::optional<std::vector<LoraInfo>> read_loras(const config::ModelConfig& model);

}  // namespace server
