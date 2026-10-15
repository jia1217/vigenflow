#include "server_config.hpp"

#include <algorithm>
#include <filesystem>

namespace server::config {
namespace fs = std::filesystem;

#ifdef _WIN32
constexpr const char* kWorkerExeName = "host.exe";
#else
constexpr const char* kWorkerExeName = "run.exe";
#endif

std::array<ModelConfig, 5> models = {{
    {"z-image-turbo", "z_image_turbo", "z_image_turbo", "z_image_turbo", {4, 8}, 42},
    {"flux1-schnell", "flux1_schnell", "flux1_schnell", "flux1_schnell", {4}, 0},
    {"flux2-klein-4B", "flux2_klein_4B", "flux2_klein_4B", "flux2_klein_4B/base", {4}, 42},
    // Distilled for 8 steps, so 8 is also its default. Its worker writes PNG whatever the file name.
    {"krea2-turbo", "krea2_turbo", "krea2_turbo", "krea2_turbo", {8}, 42, false, 8, true},
    // Shares the Flux2 weights; the worker reads base/ and edit/ under its NPU folder.
    {kEditModelId, "flux2_klein_4B_edit", "flux2_klein_4B", "flux2_klein_4B", {4}, 42, true},
}};
std::string default_model_id = "z-image-turbo";
std::string output_dir = "./images";
unsigned short port = 11283;
bool keep_images = false;  // Set true, or pass -k true, to save generated PNGs on disk.

ModelConfig* find_model(const std::string& id) {
    for (auto& model : models) {
        if (id == model.id || id == model.directory) return &model;
    }
    return nullptr;
}

ModelConfig& default_model() {
    return *find_model(default_model_id);  // Validated by parse_cli before serving.
}

namespace {
// NPU files are one level above vgf-serve; model weights are two levels above.
fs::path server_root;
}  // namespace

void initialize_default_paths(const char* argv0) {
    const fs::path server_dir = fs::absolute(argv0 && *argv0 ? argv0 : "vgf-serve").parent_path();
    server_root = (server_dir / "..").lexically_normal();
    for (auto& model : models) {
        // Development tree: ../<model>/run.exe. Packaged: exe_models/<model>/run.exe next to vgf-serve.
        const fs::path dev = server_dir / ".." / model.directory / kWorkerExeName;
        const fs::path packaged = server_dir / "exe_models" / model.directory / kWorkerExeName;
        model.worker_exe = (fs::exists(dev) || !fs::exists(packaged) ? dev : packaged).lexically_normal().string();
    }
}

void make_paths_absolute() {
    for (auto& model : models) {
        model.worker_exe = fs::absolute(model.worker_exe).lexically_normal().string();
        const auto& root = server_root;
        if (model.weights_path.empty()) {
            model.weights_path = (root / ".." / "all_model_weights" / model.weights_directory).lexically_normal().string();
        }
        if (model.npu_files_path.empty()) {
            model.npu_files_path = (root / "all_npu_files" / model.npu_directory).string();
        }
        model.weights_path = fs::absolute(model.weights_path).lexically_normal().string();
        model.npu_files_path = fs::absolute(model.npu_files_path).lexically_normal().string();
    }
    output_dir = fs::absolute(output_dir).lexically_normal().string();
}

bool is_supported_steps(const ModelConfig& model, int steps) {
    return std::find(model.supported_steps.begin(), model.supported_steps.end(), steps)
           != model.supported_steps.end();
}

std::string supported_steps_text(const ModelConfig& model) {
    std::string result;
    for (int steps : model.supported_steps) {
        result += (result.empty() ? "" : ", ") + std::to_string(steps);
    }
    return result;
}

}  // namespace server::config
