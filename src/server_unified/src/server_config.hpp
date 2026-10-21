#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace server::config {

// All compiled NPU pipelines currently produce 1024x1024 images.
inline constexpr int kImageHeight = 1024;
inline constexpr int kImageWidth = 1024;
inline constexpr std::size_t kMemoryImageLimit = 32;

// Fallback for POST /v1/images/edits when the server default is not an edit model.
inline constexpr const char* kEditModelId = "flux2-klein-4B-edit";

struct ModelConfig {
    const char* id;
    // worker: ../<directory>/run.exe, else exe_models/<directory>/run.exe; on Windows
    // ../<directory>/build-windows/Release/run.exe, else exe_models/<directory>/run.exe
    // (older Windows packages: host.exe).
    const char* directory;
    const char* weights_directory;  // ../../all_model_weights/<weights_directory> (Windows: else ../all_model_weights/...)
    const char* npu_directory;      // ../all_npu_files/<npu_directory>
    std::vector<int> supported_steps;
    int default_seed;
    bool edits_image = false;  // worker needs --input_image
    int default_steps = 4;
    bool writes_png = false;  // worker saves the final PNG itself; the others save a BMP the server converts
    int fixed_steps = 0;  // nonzero: ignore client/CLI steps and always use this schedule
    std::string worker_exe;
    std::string weights_path;
    std::string npu_files_path;
    bool available = true;  // false: no worker found (Windows only); requests return 400
};

// Configured at startup, then read-only while serving requests.
extern std::array<ModelConfig, 7> models;
extern std::string default_model_id;
extern std::string output_dir;
extern unsigned short port;
extern bool keep_images;  // false: delete temporary files and serve image URLs from memory

ModelConfig* find_model(const std::string& id);
ModelConfig& default_model();
void initialize_default_paths(const char* argv0);
void make_paths_absolute();
bool is_worker(const std::string& path);
bool is_supported_steps(const ModelConfig& model, int steps);
std::string supported_steps_text(const ModelConfig& model);

}  // namespace server::config
