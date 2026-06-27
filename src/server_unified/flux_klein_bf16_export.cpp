#include "flux_klein_bf16_export.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace server {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr const char* kDefaultRawRepo = "black-forest-labs/FLUX.2-klein-4B";
constexpr const char* kDefaultNpuRepo = "Kelsey1217/FLUX.2-klein-4B-npu";

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

std::string trim_copy(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string shell_quote(std::string_view value) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
#else
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
#endif
}

bool is_url_unreserved(unsigned char c) {
    return std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~';
}

std::string url_encode(std::string_view value, bool keep_slash) {
    std::ostringstream out;
    out << std::uppercase << std::hex;
    for (unsigned char c : value) {
        if (is_url_unreserved(c) || (keep_slash && c == '/')) {
            out << static_cast<char>(c);
        } else {
            out << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(c);
        }
    }
    return out.str();
}

std::string safe_name(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    bool last_was_sep = false;
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '-' || c == '_') {
            out.push_back(static_cast<char>(c));
            last_was_sep = false;
        } else if (!last_was_sep) {
            out.push_back('_');
            last_was_sep = true;
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "cache" : out;
}

std::string hf_cache_repo_dir_name(std::string repo_id) {
    std::string out = "models--";
    for (char c : repo_id) {
        if (c == '/') {
            out += "--";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string hf_endpoint() {
    const char* env = std::getenv("HF_ENDPOINT");
    std::string endpoint = (env != nullptr && *env != '\0')
        ? trim_copy(env)
        : std::string("https://huggingface.co");
    while (!endpoint.empty() && (endpoint.back() == '/' || endpoint.back() == '\\')) {
        endpoint.pop_back();
    }
    return endpoint.empty() ? std::string("https://huggingface.co") : endpoint;
}

std::string effective_token(const std::string& explicit_token) {
    if (!explicit_token.empty()) return explicit_token;
    const char* env = std::getenv("HF_TOKEN");
    return env ? std::string(env) : std::string();
}

fs::path default_hf_home() {
    if (const char* hf_home = std::getenv("HF_HOME")) {
        return fs::path(hf_home);
    }
    if (const char* home = std::getenv("HOME")) {
        return fs::path(home) / ".cache" / "huggingface";
    }
    return fs::path(".cache") / "huggingface";
}

fs::path default_cache_dir(const fs::path& output_root) {
    return output_root.parent_path() / ".flux_klein_bf16_hf_cache";
}

fs::path download_cache_file(
    const fs::path& cache_dir,
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename) {
    return cache_dir / safe_name(repo_id) / safe_name(revision) / fs::path(filename);
}

std::string hf_resolve_url(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& revision) {
    return hf_endpoint() + "/" + url_encode(repo_id, true) + "/resolve/" +
           url_encode(revision, false) + "/" + url_encode(filename, true);
}

std::string hf_api_url(const std::string& repo_id, const std::string& revision) {
    return hf_endpoint() + "/api/models/" + url_encode(repo_id, true) +
           "/revision/" + url_encode(revision, false);
}

std::string hf_download_env_prefix() {
#ifdef _WIN32
    return "set HF_HUB_DISABLE_XET=1&& set HF_HUB_ENABLE_HF_TRANSFER=0&& ";
#else
    return "HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 ";
#endif
}

bool file_is_present(const fs::path& path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec) && fs::file_size(path, ec) > 0;
}

const char* path_environment() {
#ifdef _WIN32
    const char* path = std::getenv("PATH");
    if (path == nullptr || *path == '\0') {
        path = std::getenv("Path");
    }
    return path;
#else
    return std::getenv("PATH");
#endif
}

std::vector<std::string> executable_names(std::string_view name) {
#ifdef _WIN32
    std::string base(name);
    if (fs::path(base).has_extension()) {
        return {base};
    }
    return {base, base + ".exe", base + ".cmd", base + ".bat"};
#else
    return {std::string(name)};
#endif
}

std::optional<fs::path> find_executable_on_path(std::string_view name) {
    const char* env = path_environment();
    if (env == nullptr || *env == '\0') {
        return std::nullopt;
    }

#ifdef _WIN32
    constexpr char kPathSeparator = ';';
#else
    constexpr char kPathSeparator = ':';
#endif

    const std::vector<std::string> names = executable_names(name);
    const std::string path_list(env);
    std::size_t start = 0;
    for (;;) {
        const std::size_t end = path_list.find(kPathSeparator, start);
        std::string dir = trim_copy(path_list.substr(
            start,
            end == std::string::npos ? std::string::npos : end - start));
        if (dir.size() >= 2 && dir.front() == '"' && dir.back() == '"') {
            dir = dir.substr(1, dir.size() - 2);
        }

        if (!dir.empty()) {
            for (const auto& candidate_name : names) {
                fs::path candidate = fs::path(dir) / candidate_name;
                if (file_is_present(candidate)) {
                    return candidate;
                }
            }
        }

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return std::nullopt;
}

std::string silent_command_redirect() {
#ifdef _WIN32
    return " >NUL 2>&1";
#else
    return " >/dev/null 2>&1";
#endif
}

bool hf_cli_is_usable(const fs::path& exe) {
    const std::string command =
        shell_quote(exe.string()) + " download --help" + silent_command_redirect();
    return std::system(command.c_str()) == 0;
}

std::optional<std::string> hf_download_command() {
    static bool checked = false;
    static std::optional<std::string> cached_command;
    if (checked) {
        return cached_command;
    }
    checked = true;

    if (auto hf = find_executable_on_path("hf")) {
        if (hf_cli_is_usable(*hf)) {
            cached_command = shell_quote(hf->string()) + " download";
            return cached_command;
        }
        std::cout << "[INFO] Hugging Face CLI at " << hf->string()
                  << " is not usable; using curl instead\n";
    }
    if (auto legacy = find_executable_on_path("huggingface-cli")) {
        if (hf_cli_is_usable(*legacy)) {
            cached_command = shell_quote(legacy->string()) + " download";
            return cached_command;
        }
        std::cout << "[INFO] Hugging Face CLI at " << legacy->string()
                  << " is not usable; using curl instead\n";
    }
    return std::nullopt;
}

void curl_download_to_file(
    const std::string& url,
    const fs::path& output_path,
    const std::string& token,
    bool force,
    bool show_progress = false,
    const std::string& display_name = {}) {
    if (file_is_present(output_path) && !force) {
        return;
    }

    fs::create_directories(output_path.parent_path());
    const fs::path tmp_path = output_path.string() + ".partial";
    std::error_code ec;

    bool resume = false;
    std::uintmax_t partial_bytes = 0;
    if (force) {
        fs::remove(tmp_path, ec);
    } else if (fs::is_regular_file(tmp_path, ec)) {
        partial_bytes = fs::file_size(tmp_path, ec);
        resume = !ec && partial_bytes > 0;
    }

    const std::string name = display_name.empty() ? output_path.filename().string() : display_name;
    if (show_progress) {
        if (resume) {
            std::cout << "[INFO] Resuming " << name << " from "
                      << (partial_bytes / (1024 * 1024)) << " MiB\n";
        } else {
            std::cout << "[INFO] Downloading " << name << "\n";
        }
    }

    auto build_command = [&](bool allow_resume) {
        std::string command = "curl -L --fail --show-error --retry 5 --retry-delay 2 --retry-all-errors";
        command += show_progress ? " --progress-bar" : " --silent";
        if (allow_resume) {
            command += " -C -";
        }
        if (!token.empty()) {
            command += " -H " + shell_quote("Authorization: Bearer " + token);
        }
        command += " -o " + shell_quote(tmp_path.string()) + " " + shell_quote(url);
        return command;
    };

    int rc = std::system(build_command(resume).c_str());
    if (rc != 0 && resume) {
        std::cerr << "[WARN] Resume failed for " << name
                  << "; restarting the download\n";
        fs::remove(tmp_path, ec);
        rc = std::system(build_command(false).c_str());
    }
    if (rc != 0) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("curl failed while downloading: " + url);
    }

    fs::remove(output_path, ec);
    fs::rename(tmp_path, output_path);
}

bool hf_cli_download_to_file(
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename,
    const fs::path& local_dir,
    const fs::path& output_path,
    const std::string& token,
    bool force,
    bool show_progress) {
    fs::create_directories(local_dir);

    const auto download_command = hf_download_command();
    if (!download_command) {
        if (show_progress) {
            std::cout << "[INFO] Hugging Face CLI not available; using curl for "
                      << repo_id << "/" << filename << "\n";
        }
        return false;
    }

    if (show_progress) {
        std::cout << "[INFO] Downloading " << repo_id << "/" << filename
                  << " with Hugging Face downloader\n";
    }

    std::string command = hf_download_env_prefix() + *download_command;
    command += " " + shell_quote(repo_id);
    command += " " + shell_quote(filename);
    command += " --revision " + shell_quote(revision);
    command += " --local-dir " + shell_quote(local_dir.string());
    command += " --max-workers 8";
    if (force) {
        command += " --force-download";
    }
    if (!show_progress) {
        command += " --quiet";
    }
    if (!token.empty()) {
        command += " --token " + shell_quote(token);
    }

    const int rc = std::system(command.c_str());
    if (rc != 0 || !file_is_present(output_path)) {
        if (show_progress) {
            std::cerr << "[WARN] Hugging Face downloader failed for "
                      << repo_id << "/" << filename
                      << "; falling back to curl\n";
        }
        return false;
    }
    return true;
}

void curl_download_range_to_file(
    const std::string& url,
    const fs::path& output_path,
    const std::string& token,
    std::uint64_t start,
    std::uint64_t end,
    bool force) {
    if (end < start) {
        throw std::runtime_error("Invalid HTTP range requested");
    }

    const std::uintmax_t expected_size = static_cast<std::uintmax_t>(end - start + 1);
    std::error_code ec;
    if (!force && fs::is_regular_file(output_path, ec) && fs::file_size(output_path, ec) == expected_size) {
        return;
    }

    fs::create_directories(output_path.parent_path());
    const fs::path tmp_path = output_path.string() + ".partial";
    fs::remove(tmp_path, ec);

    std::string command = "curl -L --fail --silent --show-error";
    command += " --range " + std::to_string(start) + "-" + std::to_string(end);
    command += " -H " + shell_quote("Accept-Encoding: identity");
    if (!token.empty()) {
        command += " -H " + shell_quote("Authorization: Bearer " + token);
    }
    command += " -o " + shell_quote(tmp_path.string()) + " " + shell_quote(url);

    const int rc = std::system(command.c_str());
    if (rc != 0) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("curl failed while downloading range from: " + url);
    }

    const auto actual_size = fs::file_size(tmp_path, ec);
    if (ec || actual_size != expected_size) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error(
            "HTTP range download size mismatch for " + url + ": expected " +
            std::to_string(expected_size) + " bytes");
    }

    fs::remove(output_path, ec);
    fs::rename(tmp_path, output_path);
}

std::vector<unsigned char> read_binary_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("Could not determine file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);

    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    if (!data.empty()) {
        input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (input.gcount() != static_cast<std::streamsize>(data.size())) {
            throw std::runtime_error("Could not read file: " + path.string());
        }
    }
    return data;
}

std::optional<fs::path> local_hf_snapshot_dir(
    const std::string& repo_id,
    const std::string& revision) {
    fs::path repo_dir = default_hf_home() / "hub" / hf_cache_repo_dir_name(repo_id);
    std::error_code ec;
    if (!fs::is_directory(repo_dir, ec)) {
        return std::nullopt;
    }

    std::string snapshot = revision.empty() ? "main" : revision;
    const fs::path ref_path = repo_dir / "refs" / snapshot;
    if (fs::is_regular_file(ref_path, ec)) {
        std::ifstream ref_in(ref_path);
        std::string resolved;
        std::getline(ref_in, resolved);
        resolved = trim_copy(resolved);
        if (!resolved.empty()) {
            snapshot = resolved;
        }
    }

    const fs::path snapshot_dir = repo_dir / "snapshots" / snapshot;
    if (fs::is_directory(snapshot_dir, ec)) {
        return snapshot_dir;
    }
    return std::nullopt;
}

std::vector<std::string> cached_snapshot_files(
    const std::string& repo_id,
    const std::string& revision) {
    std::vector<std::string> files;
    const auto snapshot_dir = local_hf_snapshot_dir(repo_id, revision);
    if (!snapshot_dir) {
        return files;
    }

    std::error_code ec;
    for (fs::recursive_directory_iterator it(*snapshot_dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (it->is_directory(entry_ec)) {
            continue;
        }
        fs::path relative_path = it->path().lexically_relative(*snapshot_dir);
        if (!relative_path.empty()) {
            files.push_back(relative_path.generic_string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::optional<fs::path> cached_snapshot_file(
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename) {
    const auto snapshot_dir = local_hf_snapshot_dir(repo_id, revision);
    if (!snapshot_dir) {
        return std::nullopt;
    }
    const fs::path path = *snapshot_dir / fs::path(filename);
    if (file_is_present(path)) {
        return path;
    }
    return std::nullopt;
}

std::vector<std::string> remote_repo_files(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    const fs::path api_path =
        cache_dir / safe_name(repo_id) / ("model_info_" + safe_name(revision) + ".json");
    curl_download_to_file(hf_api_url(repo_id, revision), api_path, token, force);

    std::ifstream input(api_path);
    if (!input) {
        throw std::runtime_error("Could not open Hugging Face model info: " + api_path.string());
    }

    const json info = json::parse(input);
    std::vector<std::string> files;
    const json siblings = info.value("siblings", json::array());
    for (const auto& sibling : siblings) {
        if (!sibling.is_object()) continue;
        const std::string filename = sibling.value("rfilename", "");
        if (!filename.empty()) {
            files.push_back(filename);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> repo_files(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    if (!force) {
        auto files = cached_snapshot_files(repo_id, revision);
        if (!files.empty()) {
            std::cout << "[INFO] Using local Hugging Face snapshot for " << repo_id << "\n";
            return files;
        }
    }
    return remote_repo_files(repo_id, revision, cache_dir, token, force);
}

fs::path repo_file_path(
    const std::string& repo_id,
    const std::string& revision,
    const std::string& filename,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    bool show_progress = true) {
    if (!force) {
        if (auto path = cached_snapshot_file(repo_id, revision, filename)) {
            return *path;
        }
    }

    const fs::path local_dir = cache_dir / safe_name(repo_id) / safe_name(revision);
    const fs::path output_path = local_dir / fs::path(filename);
    const bool large_model_file = ends_with(filename, ".safetensors") || ends_with(filename, ".gguf");
    if (large_model_file && hf_cli_download_to_file(
            repo_id,
            revision,
            filename,
            local_dir,
            output_path,
            token,
            force,
            show_progress)) {
        return output_path;
    }

    curl_download_to_file(
        hf_resolve_url(repo_id, filename, revision),
        output_path,
        token,
        force,
        show_progress,
        repo_id + "/" + filename);
    return output_path;
}

std::vector<std::string> unique_sorted(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<std::string> files_from_index(const fs::path& index_path) {
    std::ifstream input(index_path);
    if (!input) {
        throw std::runtime_error("Could not open safetensors index: " + index_path.string());
    }
    const json index = json::parse(input);
    std::vector<std::string> files;
    const json weight_map = index.value("weight_map", json::object());
    for (const auto& item : weight_map.items()) {
        if (item.value().is_string()) {
            files.push_back(item.value().get<std::string>());
        }
    }
    return unique_sorted(std::move(files));
}

std::vector<std::string> component_safetensor_files(
    const std::vector<std::string>& files,
    const std::string& component,
    const std::string& index_name,
    const std::string& single_name,
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    const std::string index_path = component + "/" + index_name;
    if (std::find(files.begin(), files.end(), index_path) != files.end()) {
        const fs::path local_index =
            repo_file_path(repo_id, revision, index_path, cache_dir, token, force);
        std::vector<std::string> indexed_files = files_from_index(local_index);
        for (auto& file : indexed_files) {
            if (file.find('/') == std::string::npos) {
                file = component + "/" + file;
            }
        }
        return unique_sorted(std::move(indexed_files));
    }

    const std::string single_path = component + "/" + single_name;
    if (std::find(files.begin(), files.end(), single_path) != files.end()) {
        return {single_path};
    }

    std::vector<std::string> matches;
    const std::string prefix = component + "/";
    for (const auto& file : files) {
        if (starts_with(file, prefix) && ends_with(file, ".safetensors")) {
            matches.push_back(file);
        }
    }
    if (matches.empty()) {
        throw std::runtime_error("No safetensors files found for component: " + component);
    }
    return unique_sorted(std::move(matches));
}

std::vector<fs::path> local_paths_for_repo_files(
    const std::vector<std::string>& files,
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force) {
    std::vector<fs::path> paths;
    paths.reserve(files.size());
    for (const auto& file : files) {
        paths.push_back(repo_file_path(repo_id, revision, file, cache_dir, token, force));
    }
    return paths;
}

std::uint64_t read_le_u64(const char* data) {
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | static_cast<unsigned char>(data[i]);
    }
    return value;
}

std::uint32_t read_le_u32(const unsigned char* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

std::uint16_t read_le_u16(const unsigned char* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint16_t f32_bits_to_bf16(std::uint32_t bits) {
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16u);
}

std::uint32_t f16_bits_to_f32_bits(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
    std::uint32_t exp = (h >> 10u) & 0x1Fu;
    std::uint32_t mant = h & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) {
            return sign;
        }
        exp = 1;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1u;
            --exp;
        }
        mant &= 0x03FFu;
        const std::uint32_t fexp = exp + (127u - 15u);
        return sign | (fexp << 23u) | (mant << 13u);
    }
    if (exp == 0x1Fu) {
        return sign | 0x7F800000u | (mant << 13u);
    }

    const std::uint32_t fexp = exp + (127u - 15u);
    return sign | (fexp << 23u) | (mant << 13u);
}

std::size_t product(const std::vector<std::size_t>& dims) {
    std::size_t count = 1;
    for (std::size_t dim : dims) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("Tensor shape is too large");
        }
        count *= dim;
    }
    return count;
}

struct TensorPayload {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> dims;
    std::vector<unsigned char> raw;
};

template <typename ShouldReadFn, typename Fn>
void for_each_safetensor_selected(const fs::path& safetensors_path, ShouldReadFn&& should_read, Fn&& fn) {
    std::ifstream input(safetensors_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open safetensors file: " + safetensors_path.string());
    }

    char len_bytes[8]{};
    input.read(len_bytes, sizeof(len_bytes));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(len_bytes))) {
        throw std::runtime_error("File is too small to be safetensors: " + safetensors_path.string());
    }

    const std::uint64_t header_len = read_le_u64(len_bytes);
    if (header_len == 0 || header_len > 512ull * 1024ull * 1024ull) {
        throw std::runtime_error("Unexpected safetensors header size");
    }

    std::string header_text(header_len, '\0');
    input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    if (input.gcount() != static_cast<std::streamsize>(header_text.size())) {
        throw std::runtime_error("Could not read safetensors header");
    }

    const json header = json::parse(header_text);
    const std::uint64_t data_start = 8 + header_len;

    for (const auto& item : header.items()) {
        const std::string tensor_name = item.key();
        if (tensor_name == "__metadata__") continue;
        const json& info = item.value();
        if (!info.is_object()) continue;

        const std::string dtype = info.value("dtype", "");
        const auto shape_json = info.value("shape", json::array());
        const auto offsets_json = info.value("data_offsets", json::array());
        if (dtype.empty() || !shape_json.is_array() || offsets_json.size() != 2) {
            continue;
        }

        std::vector<std::size_t> dims;
        dims.reserve(shape_json.size());
        for (const auto& dim : shape_json) {
            dims.push_back(dim.get<std::size_t>());
        }

        const std::uint64_t start = offsets_json[0].get<std::uint64_t>();
        const std::uint64_t end = offsets_json[1].get<std::uint64_t>();
        if (end < start) {
            throw std::runtime_error("Bad safetensors offsets for tensor: " + tensor_name);
        }

        if (!should_read(tensor_name)) {
            continue;
        }

        std::vector<unsigned char> raw(static_cast<std::size_t>(end - start));
        input.seekg(static_cast<std::streamoff>(data_start + start), std::ios::beg);
        input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (input.gcount() != static_cast<std::streamsize>(raw.size())) {
            throw std::runtime_error("Could not read tensor payload: " + tensor_name);
        }

        fn(TensorPayload{tensor_name, dtype, std::move(dims), std::move(raw)});
    }
}

template <typename Fn>
void for_each_safetensor(const fs::path& safetensors_path, Fn&& fn) {
    for_each_safetensor_selected(
        safetensors_path,
        [](const std::string&) { return true; },
        std::forward<Fn>(fn));
}

std::uint16_t tensor_element_as_bf16(const TensorPayload& payload, std::size_t index) {
    if (payload.dtype == "BF16") {
        return read_le_u16(payload.raw.data() + index * 2);
    }
    if (payload.dtype == "F16") {
        return f32_bits_to_bf16(f16_bits_to_f32_bits(read_le_u16(payload.raw.data() + index * 2)));
    }
    if (payload.dtype == "F32") {
        return f32_bits_to_bf16(read_le_u32(payload.raw.data() + index * 4));
    }
    throw std::runtime_error("Unsupported safetensors dtype: " + payload.dtype);
}

void validate_tensor_payload(const TensorPayload& payload) {
    const std::size_t count = product(payload.dims);
    std::size_t expected_bytes = 0;
    if (payload.dtype == "BF16" || payload.dtype == "F16") {
        expected_bytes = count * 2;
    } else if (payload.dtype == "F32") {
        expected_bytes = count * 4;
    } else {
        throw std::runtime_error("Unsupported safetensors dtype: " + payload.dtype);
    }
    if (payload.raw.size() != expected_bytes) {
        throw std::runtime_error("Tensor byte size does not match shape: " + payload.name);
    }
}

void write_bf16_tensor_file(
    const fs::path& path,
    const TensorPayload& payload,
    bool transpose_2d,
    bool force,
    std::size_t row_begin = 0,
    std::size_t row_count = 0) {
    if (file_is_present(path) && !force) {
        return;
    }
    validate_tensor_payload(payload);

    const bool sliced = row_count != 0;
    if (sliced && (!transpose_2d || payload.dims.size() != 2)) {
        throw std::runtime_error("Tensor slicing requires a 2D transposed tensor: " + payload.name);
    }

    fs::create_directories(path.parent_path());
    const fs::path tmp_path = path.string() + ".partial";
    std::error_code ec;
    fs::remove(tmp_path, ec);

    std::ofstream output(tmp_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + tmp_path.string());
    }

    std::vector<std::uint16_t> buffer;
    buffer.reserve(1u << 20u);
    const auto flush = [&]() {
        if (buffer.empty()) return;
        output.write(
            reinterpret_cast<const char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size() * sizeof(std::uint16_t)));
        if (!output) {
            throw std::runtime_error("Failed writing output file: " + tmp_path.string());
        }
        buffer.clear();
    };
    const auto emit = [&](std::uint16_t value) {
        buffer.push_back(value);
        if (buffer.size() == buffer.capacity()) {
            flush();
        }
    };

    std::uintmax_t expected_bytes = 0;
    if (transpose_2d && payload.dims.size() == 2) {
        const std::size_t rows = payload.dims[0];
        const std::size_t cols = payload.dims[1];
        const std::size_t begin = sliced ? row_begin : 0;
        const std::size_t count = sliced ? row_count : rows;
        if (begin > rows || count > rows - begin) {
            throw std::runtime_error("Tensor row slice is outside shape: " + payload.name);
        }
        expected_bytes = static_cast<std::uintmax_t>(count) * cols * sizeof(std::uint16_t);
        for (std::size_t col = 0; col < cols; ++col) {
            for (std::size_t row = begin; row < begin + count; ++row) {
                emit(tensor_element_as_bf16(payload, row * cols + col));
            }
        }
    } else {
        if (sliced) {
            throw std::runtime_error("Tensor row slice was requested for a non-transposed tensor: " + payload.name);
        }
        const std::size_t count = product(payload.dims);
        expected_bytes = static_cast<std::uintmax_t>(count) * sizeof(std::uint16_t);
        for (std::size_t i = 0; i < count; ++i) {
            emit(tensor_element_as_bf16(payload, i));
        }
    }
    flush();
    output.close();
    if (!output) {
        throw std::runtime_error("Failed closing output file: " + tmp_path.string());
    }

    const auto actual_bytes = fs::file_size(tmp_path, ec);
    if (ec || actual_bytes != expected_bytes) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("Output file size check failed for " + path.string());
    }
    fs::remove(path, ec);
    fs::rename(tmp_path, path);
}

void replace_all(std::string& value, std::string_view from, std::string_view to) {
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string strip_prefix(std::string value, std::string_view prefix) {
    if (starts_with(value, prefix)) {
        value.erase(0, prefix.size());
    }
    return value;
}

std::string normalize_bin_name(std::string name) {
    replace_all(name, ".", "_");
    replace_all(name, "/", "_");
    replace_all(name, "\\", "_");
    return name + "_bf16_u16.bin";
}

std::optional<std::string> transformer_bin_name(std::string tensor_name) {
    if (starts_with(tensor_name, "transformer_blocks.") &&
        (ends_with(tensor_name, ".attn.to_q.weight") ||
         ends_with(tensor_name, ".attn.to_k.weight"))) {
        return std::nullopt;
    }

    replace_all(tensor_name, ".attn.norm_k.weight", ".attn.to_knorm.weight");
    replace_all(tensor_name, ".attn.norm_q.weight", ".attn.to_qnorm.weight");
    return normalize_bin_name(std::move(tensor_name));
}

std::string text_encoder_bin_name(std::string tensor_name) {
    tensor_name = strip_prefix(std::move(tensor_name), "model.");
    return normalize_bin_name(std::move(tensor_name));
}

std::optional<int> single_block_qkv_tensor_index(const std::string& tensor_name) {
    constexpr std::string_view prefix = "single_transformer_blocks.";
    constexpr std::string_view suffix = ".attn.to_qkv_mlp_proj.weight";
    if (!starts_with(tensor_name, prefix) || !ends_with(tensor_name, suffix)) {
        return std::nullopt;
    }
    const std::size_t begin = prefix.size();
    const std::size_t end = tensor_name.size() - suffix.size();
    if (begin >= end) {
        return std::nullopt;
    }
    return std::stoi(tensor_name.substr(begin, end - begin));
}

std::size_t export_transformer_safetensors(
    const std::vector<fs::path>& safetensors_paths,
    const fs::path& output_dir,
    bool force) {
    fs::create_directories(output_dir);
    std::size_t exported = 0;
    for (const auto& path : safetensors_paths) {
        std::cout << "[INFO] Exporting FLUX transformer " << path << " -> " << output_dir << "\n";
        for_each_safetensor(path, [&](const TensorPayload& payload) {
            if (const auto block = single_block_qkv_tensor_index(payload.name)) {
                const std::string base =
                    "single_transformer_blocks_" + std::to_string(*block) + "_attn_";
                write_bf16_tensor_file(
                    output_dir / (base + "to_v_weight_bf16_u16.bin"),
                    payload,
                    true,
                    force,
                    6144,
                    3072);
                write_bf16_tensor_file(
                    output_dir / (base + "to_qkv_mlp_proj_weight_bf16_u16.bin"),
                    payload,
                    true,
                    force,
                    9216,
                    18432);
                exported += 2;
                return;
            }

            const auto out_name = transformer_bin_name(payload.name);
            if (!out_name) {
                return;
            }
            write_bf16_tensor_file(output_dir / *out_name, payload, true, force);
            ++exported;
        });
    }
    return exported;
}

std::size_t export_final_layer_safetensors(
    const fs::path& safetensors_path,
    const fs::path& output_dir,
    bool force) {
    fs::create_directories(output_dir);
    std::cout << "[INFO] Exporting FLUX final_layer tensors "
              << safetensors_path << " -> " << output_dir << "\n";

    std::size_t exported = 0;
    for_each_safetensor_selected(
        safetensors_path,
        [](const std::string& name) {
            return name == "final_layer.adaLN_modulation.1.weight" ||
                   name == "final_layer.linear.weight";
        },
        [&](const TensorPayload& payload) {
            write_bf16_tensor_file(
                output_dir / normalize_bin_name(payload.name),
                payload,
                true,
                force);
            ++exported;
        });

    if (exported != 2) {
        throw std::runtime_error(
            "Expected 2 final_layer tensors in " + safetensors_path.string() +
            ", exported " + std::to_string(exported));
    }
    return exported;
}

std::size_t export_final_layer_safetensors_from_repo(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_dir) {
    constexpr const char* kTopLevelFluxFile = "flux-2-klein-4b.safetensors";

    if (!force) {
        if (auto path = cached_snapshot_file(repo_id, revision, kTopLevelFluxFile)) {
            return export_final_layer_safetensors(*path, output_dir, force);
        }
        const fs::path cached_full_file = download_cache_file(
            cache_dir, repo_id, revision, kTopLevelFluxFile);
        if (file_is_present(cached_full_file)) {
            return export_final_layer_safetensors(cached_full_file, output_dir, force);
        }
    }

    fs::create_directories(output_dir);
    const std::string url = hf_resolve_url(repo_id, kTopLevelFluxFile, revision);
    const fs::path range_dir = cache_dir / safe_name(repo_id) / safe_name(revision) /
        ".ranges" / safe_name(kTopLevelFluxFile);

    std::cout << "[INFO] Fetching FLUX final_layer tensor ranges from "
              << repo_id << "/" << kTopLevelFluxFile << "\n";

    const fs::path len_path = range_dir / "header_length.bin";
    curl_download_range_to_file(url, len_path, token, 0, 7, force);
    const auto len_bytes = read_binary_file(len_path);
    if (len_bytes.size() != 8) {
        throw std::runtime_error("Could not read safetensors header length");
    }

    const std::uint64_t header_len = read_le_u64(reinterpret_cast<const char*>(len_bytes.data()));
    if (header_len == 0 || header_len > 512ull * 1024ull * 1024ull) {
        throw std::runtime_error("Unexpected safetensors header size");
    }

    const fs::path header_path = range_dir / ("header_" + std::to_string(header_len) + ".json");
    curl_download_range_to_file(url, header_path, token, 8, 7 + header_len, force);
    const auto header_bytes = read_binary_file(header_path);
    if (header_bytes.size() != header_len) {
        throw std::runtime_error("Could not read safetensors header bytes");
    }

    const std::string header_text(
        reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
    const json header = json::parse(header_text);
    const std::uint64_t data_start = 8 + header_len;

    std::size_t exported = 0;
    for (const auto& item : header.items()) {
        const std::string tensor_name = item.key();
        if (tensor_name != "final_layer.adaLN_modulation.1.weight" &&
            tensor_name != "final_layer.linear.weight") {
            continue;
        }

        const json& info = item.value();
        const std::string dtype = info.value("dtype", "");
        const auto shape_json = info.value("shape", json::array());
        const auto offsets_json = info.value("data_offsets", json::array());
        if (dtype.empty() || !shape_json.is_array() || offsets_json.size() != 2) {
            throw std::runtime_error("Bad safetensors metadata for tensor: " + tensor_name);
        }

        std::vector<std::size_t> dims;
        dims.reserve(shape_json.size());
        for (const auto& dim : shape_json) {
            dims.push_back(dim.get<std::size_t>());
        }

        const std::uint64_t start = offsets_json[0].get<std::uint64_t>();
        const std::uint64_t end = offsets_json[1].get<std::uint64_t>();
        if (end <= start) {
            throw std::runtime_error("Bad safetensors offsets for tensor: " + tensor_name);
        }

        const std::uint64_t byte_start = data_start + start;
        const std::uint64_t byte_end = data_start + end - 1;
        const fs::path tensor_path = range_dir / (normalize_bin_name(tensor_name) + ".range");
        curl_download_range_to_file(url, tensor_path, token, byte_start, byte_end, force);

        auto raw = read_binary_file(tensor_path);
        if (raw.size() != static_cast<std::size_t>(end - start)) {
            throw std::runtime_error("Could not read tensor payload range: " + tensor_name);
        }

        write_bf16_tensor_file(
            output_dir / normalize_bin_name(tensor_name),
            TensorPayload{tensor_name, dtype, std::move(dims), std::move(raw)},
            true,
            force);
        ++exported;
    }

    if (exported != 2) {
        throw std::runtime_error(
            "Expected 2 final_layer tensors in " + std::string(kTopLevelFluxFile) +
            ", exported " + std::to_string(exported));
    }
    return exported;
}

std::size_t export_text_encoder_safetensors(
    const std::vector<fs::path>& safetensors_paths,
    const fs::path& output_dir,
    bool force) {
    fs::create_directories(output_dir);
    std::size_t exported = 0;
    for (const auto& path : safetensors_paths) {
        std::cout << "[INFO] Exporting FLUX text encoder " << path << " -> " << output_dir << "\n";
        for_each_safetensor(path, [&](const TensorPayload& payload) {
            write_bf16_tensor_file(
                output_dir / text_encoder_bin_name(payload.name),
                payload,
                true,
                force);
            ++exported;
        });
    }
    return exported;
}

void copy_file_to_output(const fs::path& src, const fs::path& dst, bool force) {
    if (file_is_present(dst) && !force) {
        return;
    }

    fs::create_directories(dst.parent_path());
    std::error_code ec;
    if (force || fs::exists(dst, ec) || fs::is_symlink(dst, ec)) {
        fs::remove(dst, ec);
    }

    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error(
            "Could not copy FLUX BF16 weight " + src.string() + " to " +
            dst.string() + ": " + ec.message());
    }
}

void copy_tokenizer_json(
    const std::vector<std::string>& files,
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_dir) {
    const std::string tokenizer = "tokenizer/tokenizer.json";
    if (std::find(files.begin(), files.end(), tokenizer) == files.end()) {
        throw std::runtime_error("tokenizer/tokenizer.json was not found in " + repo_id);
    }
    const fs::path src = repo_file_path(repo_id, revision, tokenizer, cache_dir, token, force);
    copy_file_to_output(src, output_dir / "tokenizer.json", force);
}

void copy_raw_tokenizer_json(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_dir) {
    const fs::path src = repo_file_path(
        repo_id, revision, "tokenizer/tokenizer.json", cache_dir, token, force);
    copy_file_to_output(src, output_dir / "tokenizer.json", force);
}

std::string packed_flux_bin_filename(
    const std::string& component,
    const std::string& tensor_name) {
    if (tensor_name.empty() || tensor_name.find('/') != std::string::npos ||
        tensor_name.find('\\') != std::string::npos) {
        throw std::runtime_error("Unsafe packed tensor name: " + tensor_name);
    }
    if (ends_with(tensor_name, ".bin")) {
        return tensor_name;
    }
    if (ends_with(tensor_name, "_bf16_u16")) {
        return tensor_name + ".bin";
    }
    if (component == "vae" && ends_with(tensor_name, "_weights_transposed")) {
        return tensor_name + ".bin";
    }
    return tensor_name + "_bf16_u16.bin";
}

bool write_raw_bf16_tensor_file(
    const fs::path& path,
    const TensorPayload& payload,
    bool force) {
    if (file_is_present(path) && !force) {
        return false;
    }
    if (payload.dtype != "BF16") {
        throw std::runtime_error(
            "Packed FLUX tensor is not BF16: " + payload.name + " dtype=" + payload.dtype);
    }
    validate_tensor_payload(payload);

    fs::create_directories(path.parent_path());
    const fs::path tmp_path = path.string() + ".partial";
    std::error_code ec;
    fs::remove(tmp_path, ec);

    std::ofstream output(tmp_path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + tmp_path.string());
    }
    output.write(
        reinterpret_cast<const char*>(payload.raw.data()),
        static_cast<std::streamsize>(payload.raw.size()));
    output.close();
    if (!output) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("Failed writing output file: " + tmp_path.string());
    }

    const auto actual_bytes = fs::file_size(tmp_path, ec);
    if (ec || actual_bytes != payload.raw.size()) {
        fs::remove(tmp_path, ec);
        throw std::runtime_error("Output file size check failed for " + path.string());
    }
    fs::remove(path, ec);
    fs::rename(tmp_path, path);
    return true;
}

std::size_t unpack_packed_flux_safetensors(
    const fs::path& safetensors_path,
    const std::string& component,
    const fs::path& output_root,
    bool force) {
    const fs::path output_dir = output_root / component;
    std::size_t written = 0;
    std::size_t seen = 0;
    std::cout << "[INFO] Unpacking packed FLUX " << component << " weights from "
              << safetensors_path << "\n";
    for_each_safetensor(safetensors_path, [&](const TensorPayload& payload) {
        ++seen;
        const fs::path out_path = output_dir / packed_flux_bin_filename(component, payload.name);
        if (write_raw_bf16_tensor_file(out_path, payload, force)) {
            ++written;
        }
    });
    std::cout << "[INFO] Unpacked " << written << " / " << seen
              << " FLUX " << component << " tensors\n";
    return written;
}

std::size_t unpack_packed_flux_repo(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_root,
    bool need_denoising,
    bool need_text_weights,
    bool need_vae) {
    struct ComponentPack {
        const char* component;
        const char* filename;
        bool needed;
    };

    const ComponentPack packs[] = {
        {"denoising", "denoising.safetensors", need_denoising},
        {"text_encoder", "text_encoder.safetensors", need_text_weights},
        {"vae", "vae.safetensors", need_vae},
    };

    std::size_t written = 0;
    for (const auto& pack : packs) {
        if (!pack.needed) {
            continue;
        }
        const fs::path src = repo_file_path(
            repo_id, revision, pack.filename, cache_dir, token, force);
        written += unpack_packed_flux_safetensors(
            src, pack.component, output_root, force);
    }
    return written;
}

std::size_t mirror_vae_from_npu_repo(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_root) {
    const auto files = repo_files(repo_id, revision, cache_dir, token, force);
    std::size_t copied = 0;
    for (const auto& file : files) {
        if (!starts_with(file, "vae/")) {
            continue;
        }
        const fs::path src = repo_file_path(repo_id, revision, file, cache_dir, token, force);
        copy_file_to_output(src, output_root / fs::path(file), force);
        ++copied;
    }
    if (copied == 0) {
        throw std::runtime_error("No VAE files were found in Hugging Face repo: " + repo_id);
    }
    return copied;
}

std::optional<fs::path> flux_npu_relative_path(const std::string& file) {
    constexpr std::string_view prefixes[] = {
        "",
        "FLUX.2-klein-4B/",
        "model_weights/FLUX.2-klein-4B/",
    };
    constexpr std::string_view component_prefixes[] = {
        "denoising/",
        "text_encoder/",
        "vae/",
    };

    for (std::string_view prefix : prefixes) {
        if (!starts_with(file, prefix)) {
            continue;
        }
        const std::string_view relative(file.data() + prefix.size(), file.size() - prefix.size());
        for (std::string_view component_prefix : component_prefixes) {
            if (starts_with(relative, component_prefix)) {
                return fs::path(std::string(relative));
            }
        }
    }
    return std::nullopt;
}

std::size_t mirror_flux_npu_repo(
    const std::string& repo_id,
    const std::string& revision,
    const fs::path& cache_dir,
    const std::string& token,
    bool force,
    const fs::path& output_root) {
    const auto files = repo_files(repo_id, revision, cache_dir, token, force);
    std::size_t copied = 0;
    for (const auto& file : files) {
        const auto relative = flux_npu_relative_path(file);
        if (!relative) {
            continue;
        }
        const fs::path dst = output_root / *relative;
        if (file_is_present(dst) && !force) {
            continue;
        }
        const fs::path src = repo_file_path(repo_id, revision, file, cache_dir, token, force);
        copy_file_to_output(src, dst, force);
        ++copied;
    }
    return copied;
}

std::vector<fs::path> transformer_required_files(const fs::path& output_root) {
    return {
        output_root / "denoising" / "context_embedder_weight_bf16_u16.bin",
        output_root / "denoising" / "transformer_blocks_0_attn_add_q_proj_weight_bf16_u16.bin",
        output_root / "denoising" / "single_transformer_blocks_0_attn_to_qnorm_weight_bf16_u16.bin",
        output_root / "denoising" / "single_transformer_blocks_0_attn_to_v_weight_bf16_u16.bin",
        output_root / "denoising" / "single_transformer_blocks_0_attn_to_qkv_mlp_proj_weight_bf16_u16.bin",
    };
}

std::vector<fs::path> final_layer_required_files(const fs::path& output_root) {
    return {
        output_root / "denoising" / "final_layer_adaLN_modulation_1_weight_bf16_u16.bin",
        output_root / "denoising" / "final_layer_linear_weight_bf16_u16.bin",
    };
}

std::vector<fs::path> denoising_required_files(const fs::path& output_root) {
    auto files = transformer_required_files(output_root);
    auto final_layer = final_layer_required_files(output_root);
    files.insert(files.end(), final_layer.begin(), final_layer.end());
    return files;
}

std::vector<fs::path> text_encoder_weight_required_files(const fs::path& output_root) {
    return {
        output_root / "text_encoder" / "embed_tokens_weight_bf16_u16.bin",
        output_root / "text_encoder" / "layers_0_self_attn_q_proj_weight_bf16_u16.bin",
        output_root / "text_encoder" / "layers_35_self_attn_v_proj_weight_bf16_u16.bin",
        output_root / "text_encoder" / "norm_weight_bf16_u16.bin",
    };
}

std::vector<fs::path> text_encoder_required_files(const fs::path& output_root) {
    auto files = text_encoder_weight_required_files(output_root);
    files.insert(files.begin(), output_root / "text_encoder" / "tokenizer.json");
    return files;
}

std::vector<fs::path> vae_required_files(const fs::path& output_root) {
    return {
        output_root / "vae" / "decoder_conv_in_weight_bf16_u16.bin",
        output_root / "vae" / "vae_mid_b1_attn1_q_weights_transposed.bin",
    };
}

std::vector<fs::path> required_files(const fs::path& output_root) {
    auto files = denoising_required_files(output_root);
    auto text = text_encoder_required_files(output_root);
    auto vae = vae_required_files(output_root);
    files.insert(files.end(), text.begin(), text.end());
    files.insert(files.end(), vae.begin(), vae.end());
    return files;
}

std::vector<fs::path> missing_files(const std::vector<fs::path>& paths) {
    std::vector<fs::path> missing;
    for (const auto& path : paths) {
        if (!file_is_present(path)) {
            missing.push_back(path);
        }
    }
    return missing;
}

std::vector<fs::path> missing_required_files(const fs::path& output_root) {
    return missing_files(required_files(output_root));
}

bool group_ready(const std::vector<fs::path>& paths) {
    return missing_files(paths).empty();
}

std::uintmax_t remove_tree_if_present(const fs::path& path, const std::string& label) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return 0;
    }

    const std::uintmax_t removed = fs::remove_all(path, ec);
    if (ec) {
        std::cerr << "[WARN] Could not remove " << label << " " << path
                  << ": " << ec.message() << "\n";
        return 0;
    }
    return removed;
}

std::uintmax_t remove_file_if_present(const fs::path& path, const std::string& label) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return 0;
    }

    const bool removed = fs::remove(path, ec);
    if (ec) {
        std::cerr << "[WARN] Could not remove " << label << " " << path
                  << ": " << ec.message() << "\n";
        return 0;
    }
    return removed ? 1 : 0;
}

void remove_empty_dir_if_present(const fs::path& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

std::uintmax_t cleanup_repo_download_cache(
    const fs::path& cache_dir,
    const std::string& repo_id,
    const std::string& revision) {
    const fs::path repo_cache_dir = cache_dir / safe_name(repo_id);
    std::uintmax_t removed = 0;
    removed += remove_tree_if_present(
        repo_cache_dir / safe_name(revision),
        "FLUX downloaded weight cache");
    removed += remove_file_if_present(
        repo_cache_dir / ("model_info_" + safe_name(revision) + ".json"),
        "FLUX model info cache");

    remove_empty_dir_if_present(repo_cache_dir);
    remove_empty_dir_if_present(cache_dir);
    return removed;
}

bool is_safetensors_artifact(const fs::path& path) {
    const std::string name = path.filename().string();
    return ends_with(name, ".safetensors") ||
           name.find(".safetensors.") != std::string::npos;
}

std::uintmax_t cleanup_output_safetensors(const fs::path& output_root) {
    std::error_code ec;
    if (!fs::is_directory(output_root, ec)) {
        return 0;
    }

    std::vector<fs::path> artifacts;
    for (fs::recursive_directory_iterator it(output_root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (it->is_regular_file(entry_ec) && is_safetensors_artifact(it->path())) {
            artifacts.push_back(it->path());
        }
    }
    if (ec) {
        std::cerr << "[WARN] Could not finish scanning FLUX output for safetensors cleanup: "
                  << ec.message() << "\n";
    }

    std::uintmax_t removed = 0;
    for (const auto& artifact : artifacts) {
        removed += remove_file_if_present(artifact, "FLUX safetensors artifact");
    }
    return removed;
}

void cleanup_flux_download_artifacts(
    const fs::path& cache_dir,
    const fs::path& output_root,
    const std::string& raw_repo_id,
    const std::string& revision,
    const std::string& npu_repo_id,
    const std::string& npu_revision) {
    std::uintmax_t cache_removed = cleanup_repo_download_cache(
        cache_dir, npu_repo_id, npu_revision);
    if (raw_repo_id != npu_repo_id || revision != npu_revision) {
        cache_removed += cleanup_repo_download_cache(cache_dir, raw_repo_id, revision);
    }

    const std::uintmax_t output_removed = cleanup_output_safetensors(output_root);
    if (cache_removed > 0 || output_removed > 0) {
        std::cout << "[INFO] Cleaned FLUX download artifacts: "
                  << cache_removed << " cache entries, "
                  << output_removed << " safetensors files\n";
    }
}

std::string normalize_raw_repo_id(std::string repo_id) {
    if (repo_id.empty() || repo_id == kDefaultNpuRepo ||
        repo_id == "Kelsey1217/Kelsey1217/FLUX.2-klein-4B-npu") {
        return kDefaultRawRepo;
    }
    return repo_id;
}

bool is_default_npu_repo_id(const std::string& repo_id) {
    return repo_id.empty() || repo_id == kDefaultNpuRepo ||
           repo_id == "Kelsey1217/Kelsey1217/FLUX.2-klein-4B-npu";
}

}  // namespace

void prepare_flux_klein_bf16_weights_cpp(const FluxKleinBf16DownloadOptions& options) {
    const std::string configured_repo_id = options.repo_id.empty() ? kDefaultNpuRepo : options.repo_id;
    const std::string raw_repo_id = normalize_raw_repo_id(configured_repo_id);
    const std::string npu_repo_id = kDefaultNpuRepo;
    const std::string revision = options.revision.empty() ? "main" : options.revision;
    const std::string npu_revision = "main";
    const fs::path output_root = options.output_root.empty()
        ? fs::path("./model_weights/FLUX.2-klein-4B")
        : options.output_root;
    const fs::path cache_dir = options.cache_dir.empty()
        ? default_cache_dir(output_root)
        : options.cache_dir;
    const std::string token = effective_token(options.hf_token);

    fs::create_directories(output_root);

    bool need_transformer = options.force || !group_ready(transformer_required_files(output_root));
    bool need_final_layer = options.force || !group_ready(final_layer_required_files(output_root));
    bool need_denoising = need_transformer || need_final_layer;
    bool need_text_weights = options.force || !group_ready(text_encoder_weight_required_files(output_root));
    bool need_tokenizer = options.force || !file_is_present(output_root / "text_encoder" / "tokenizer.json");
    bool need_text = need_text_weights || need_tokenizer;
    bool need_vae = options.force || !group_ready(vae_required_files(output_root));

    if (!need_denoising && !need_text && !need_vae) {
        std::cout << "[INFO] FLUX.2-klein-4B BF16 weights are already ready under "
                  << output_root << "\n";
        return;
    }

    std::cout << "[INFO] Preparing FLUX.2-klein-4B BF16 weights into "
              << output_root << "\n";
    std::cout << "[INFO] Model weights source: " << npu_repo_id
              << " (" << npu_revision << ")\n";
    std::cout << "[INFO] Raw denoising/text source: " << raw_repo_id
              << " (" << revision << ")\n";

    if (is_default_npu_repo_id(configured_repo_id)) {
        try {
            const std::size_t unpacked = unpack_packed_flux_repo(
                npu_repo_id,
                npu_revision,
                cache_dir,
                token,
                options.force,
                output_root,
                need_denoising,
                need_text_weights,
                need_vae);
            if (unpacked > 0) {
                std::cout << "[INFO] Unpacked " << unpacked
                          << " FLUX tensors from packed safetensors repo "
                          << npu_repo_id << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Could not use packed FLUX safetensors from "
                      << npu_repo_id << ": " << e.what() << "\n";
            const std::size_t copied = mirror_flux_npu_repo(
                npu_repo_id,
                npu_revision,
                cache_dir,
                token,
                options.force,
                output_root);
            if (copied > 0) {
                std::cout << "[INFO] Copied " << copied << " legacy FLUX NPU files from "
                          << npu_repo_id << "\n";
            }
        }

        if (need_tokenizer) {
            copy_raw_tokenizer_json(
                raw_repo_id,
                revision,
                cache_dir,
                token,
                options.force,
                output_root / "text_encoder");
        }

        need_transformer = !group_ready(transformer_required_files(output_root));
        need_final_layer = !group_ready(final_layer_required_files(output_root));
        need_denoising = need_transformer || need_final_layer;
        need_text_weights = !group_ready(text_encoder_weight_required_files(output_root));
        need_tokenizer = !file_is_present(output_root / "text_encoder" / "tokenizer.json");
        need_text = need_text_weights || need_tokenizer;
        need_vae = !group_ready(vae_required_files(output_root));
    }

    if (need_transformer || need_text_weights) {
        const auto files = repo_files(raw_repo_id, revision, cache_dir, token, options.force);
        if (need_transformer) {
            const auto transformer_files = component_safetensor_files(
                files,
                "transformer",
                "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model.safetensors",
                raw_repo_id,
                revision,
                cache_dir,
                token,
                options.force);
            const auto transformer_paths = local_paths_for_repo_files(
                transformer_files, raw_repo_id, revision, cache_dir, token, options.force);
            const std::size_t count = export_transformer_safetensors(
                transformer_paths, output_root / "denoising", options.force);
            std::cout << "[INFO] Exported " << count << " FLUX transformer tensors\n";
        }
        if (need_text_weights) {
            const auto text_files = component_safetensor_files(
                files,
                "text_encoder",
                "model.safetensors.index.json",
                "model.safetensors",
                raw_repo_id,
                revision,
                cache_dir,
                token,
                options.force);
            const auto text_paths = local_paths_for_repo_files(
                text_files, raw_repo_id, revision, cache_dir, token, options.force);
            const std::size_t count = export_text_encoder_safetensors(
                text_paths, output_root / "text_encoder", options.force);
            std::cout << "[INFO] Exported " << count << " FLUX text encoder tensors\n";
        }
    }

    if (need_tokenizer) {
        copy_raw_tokenizer_json(
            raw_repo_id,
            revision,
            cache_dir,
            token,
            options.force,
            output_root / "text_encoder");
    }

    if (need_final_layer) {
        const std::size_t count = export_final_layer_safetensors_from_repo(
            raw_repo_id,
            revision,
            cache_dir,
            token,
            options.force,
            output_root / "denoising");
        std::cout << "[INFO] Exported " << count << " FLUX final_layer tensors\n";
    }

    if (need_vae) {
        const std::size_t copied = mirror_vae_from_npu_repo(
            npu_repo_id,
            npu_revision,
            cache_dir,
            token,
            options.force,
            output_root);
        std::cout << "[INFO] Copied " << copied << " FLUX VAE files from NPU repo\n";
    }

    const auto missing = missing_required_files(output_root);
    if (!missing.empty()) {
        std::ostringstream out;
        out << "FLUX BF16 preparation finished, but required files are missing:";
        for (const auto& path : missing) {
            out << "\n  - " << path.string();
        }
        throw std::runtime_error(out.str());
    }

    cleanup_flux_download_artifacts(
        cache_dir,
        output_root,
        raw_repo_id,
        revision,
        npu_repo_id,
        npu_revision);

    std::cout << "[INFO] FLUX.2-klein-4B BF16 weights are ready under "
              << output_root << "\n";
}

}  // namespace server
