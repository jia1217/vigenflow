#include "server_config.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace server::config {

namespace {

namespace fs = std::filesystem;

bool looks_like_install_root(const fs::path& path) {
    std::error_code ec;
    return fs::is_directory(path / "exe_models", ec) &&
           fs::is_directory(path / "npu_files", ec);
}

std::vector<fs::path> path_and_parents(fs::path path) {
    std::vector<fs::path> out;
    if (path.empty()) {
        return out;
    }

    std::error_code ec;
    path = fs::absolute(path, ec);
    if (ec) {
        return out;
    }

    for (;;) {
        out.push_back(path);
        const fs::path parent = path.parent_path();
        if (parent.empty() || parent == path) {
            break;
        }
        path = parent;
    }
    return out;
}

fs::path discover_install_root(const char* argv0) {
    std::vector<fs::path> search_roots;

    std::error_code ec;
    search_roots.push_back(fs::current_path(ec));

    if (argv0 != nullptr && *argv0 != '\0') {
        fs::path exe_path(argv0);
        if (exe_path.is_relative()) {
            exe_path = fs::absolute(exe_path, ec);
        }
        if (!ec) {
            search_roots.push_back(exe_path.parent_path());
        }
    }

    for (const auto& root : search_roots) {
        for (const auto& candidate : path_and_parents(root)) {
            if (looks_like_install_root(candidate)) {
                return candidate;
            }

            const fs::path vigenflow_win = candidate / "vigenflow_win";
            if (looks_like_install_root(vigenflow_win)) {
                return vigenflow_win;
            }
        }
    }

    return {};
}

std::string normalized_string(const fs::path& path) {
    std::error_code ec;
    const fs::path absolute = fs::absolute(path, ec).lexically_normal();
    return ec ? path.lexically_normal().string() : absolute.string();
}

}  // namespace

// const std::string kInstallDir = "/opt/vigenflow";
const std::string kInstallDir = ".";

std::string model_id = "flux.2-klein-4B";
std::string served_model_id;
std::string weights_path = kInstallDir + "/model_weights";
std::string npu_base_dir = kInstallDir + "/npu_files";
std::string exe_base_dir = kInstallDir + "/exe_models";
std::string custom_exe;
std::string workdir = kInstallDir;

std::string lora_dir;
std::string lora_catalog_dir;
std::string lora_model_list;
std::string lora_model_list_q41;
std::string lora_model_list_bf16;
std::string lora_model_list_flux;
std::string lora_model_list_flux_edit;
std::string lora_source;
std::string lora_file;
std::string lora_revision = "main";
std::string lora_cache_dir;
std::string lora_export_dir;
bool lora_force = false;
bool lora_keep_cache = false;
int lora_rank = 0;

std::string gguf_path;
std::string gguf_repo;
std::string gguf_file;
std::string gguf_revision;
std::string gguf_hf_token;
bool force_gguf_download = false;

bool auto_download_z_image_bf16 = true;
bool force_z_image_bf16_download = false;
std::string z_image_bf16_repo = "Tongyi-MAI/Z-Image-Turbo";
std::string z_image_bf16_revision = "main";
std::string z_image_bf16_cache_dir;
std::string z_image_bf16_hf_token;

bool auto_download_flux_klein_bf16 = true;
bool force_flux_klein_bf16_download = false;
std::string flux_klein_bf16_repo = "Kelsey1217/FLUX.2-klein-4B-npu";
std::string flux_klein_bf16_revision = "main";
std::string flux_klein_bf16_cache_dir;
std::string flux_klein_bf16_hf_token;

unsigned short port = 11283;
std::string public_base_url = "http://127.0.0.1:" + std::to_string(port);
std::string output_dir = "../images";

bool keep_images = false;

int default_seed = 42;
int default_steps = 4;
int default_height = 1024;
int default_width = 1024;
std::string default_prompt =
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup...";

void initialize_default_paths(const char* argv0) {
    const fs::path install_root = discover_install_root(argv0);
    if (install_root.empty()) {
        return;
    }

    const std::string root = normalized_string(install_root);
    workdir = root;
    weights_path = normalized_string(install_root / "model_weights");
    npu_base_dir = normalized_string(install_root / "npu_files");
    exe_base_dir = normalized_string(install_root / "exe_models");
}

}  // namespace server::config
