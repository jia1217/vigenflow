#include "server_config.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace server::config {
namespace fs = std::filesystem;

#ifdef _WIN32
// Development tree: each host's CMake build.
constexpr const char* kDevWorker = "build-windows/Release/run.exe";
#else
constexpr const char* kDevWorker = "run.exe";
#endif
constexpr const char* kPackagedWorker = "run.exe";

std::array<ModelConfig, 7> models = {{
    {"z-image-turbo", "z_image_turbo", "z_image_turbo", "z_image_turbo", {4, 8}, 42},
    {"flux1-schnell", "flux1_schnell", "flux1_schnell", "flux1_schnell", {4}, 0},
    {"flux2-klein-4B", "flux2_klein_4B", "flux2_klein_4B", "flux2_klein_4B/base", {4}, 42},
    // Distilled for 8 steps, so 8 is also its default. Its worker writes PNG whatever the file name.
    {"krea2-turbo", "krea2_turbo", "krea2_turbo", "krea2_turbo", {8}, 42, false, 8, true},
    {"qwen-image-2.1-turbo", "qwen_image_2_1_turbo", "qwen_image_2_1_turbo", "qwen_image/base", {8}, 42, false, 8, true, 8},
    // Shares the Flux2 weights; the worker reads base/ and edit/ under its NPU folder.
    {kEditModelId, "flux2_klein_4B_edit", "flux2_klein_4B", "flux2_klein_4B", {4}, 42, true},
    // Both Qwen tasks share all five weight stages; each host receives its own bundle root.
    {"qwen-image-2.1-turbo-edit", "qwen_image_2_1_turbo_edit", "qwen_image_2_1_turbo", "qwen_image/edit", {8}, 42, true, 8, true, 8},
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
        // Development tree: ../<model>/run.exe (Windows: its build-windows/Release/run.exe).
        // Packaged: exe_models/<model>/run.exe next to vgf-serve.
        const fs::path dev = server_dir / ".." / model.directory / kDevWorker;
        const fs::path package_dir = server_dir / "exe_models" / model.directory;
        fs::path packaged = package_dir / kPackagedWorker;
#ifdef _WIN32
        // Older Windows packages use host.exe. Prefer a native run.exe when present,
        // and ignore any Linux run.exe left beside the Windows worker.
        const fs::path legacy = package_dir / "host.exe";
        if (!is_worker(packaged.string()) && is_worker(legacy.string())) packaged = legacy;
#endif
        model.worker_exe = (is_worker(dev.string()) || !is_worker(packaged.string()) ? dev : packaged)
                               .lexically_normal().string();
    }
}

bool is_worker(const std::string& path) {
    if (!fs::is_regular_file(path)) return false;
#ifdef _WIN32
    // Only Windows executables: the Ubuntu run.exe files in the projects and exe_models are ELF.
    std::ifstream in(fs::path(path), std::ios::binary);
    char magic[2] = {};
    return in.read(magic, 2) && magic[0] == 'M' && magic[1] == 'Z';
#else
    return true;
#endif
}

void make_paths_absolute() {
    for (auto& model : models) {
        model.worker_exe = fs::absolute(model.worker_exe).lexically_normal().string();
        const auto& root = server_root;
        if (model.weights_path.empty()) {
            const fs::path shared = root / ".." / "all_model_weights" / model.weights_directory;
#ifdef _WIN32
            // Weights are beside the package folder, where prepare_base.exe puts them so that every package
            // on the machine shares one copy. A model found only in the package's own all_model_weights is
            // read from there: weights installed by 0.2.0-beta.1, which kept them inside the package.
            const fs::path local = root / "all_model_weights" / model.weights_directory;
            std::error_code ec;  // an unreadable folder counts as missing
            model.weights_path = (!fs::is_directory(shared, ec) && fs::is_directory(local, ec) ? local : shared).string();
#else
            model.weights_path = shared.lexically_normal().string();
#endif
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
