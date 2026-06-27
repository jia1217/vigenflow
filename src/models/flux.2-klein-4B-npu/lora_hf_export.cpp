#include "lora_hf_export.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

namespace host {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr std::array<std::string_view, 5> kWeightExts{
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt"};
constexpr std::array<std::string_view, 4> kSkipTokens{
    "optimizer", "scheduler", "trainer_state", "training_args"};

struct HfSource {
    std::string repo_id;
    std::string filename;
    std::string revision;
};

bool starts_with(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::string> split(std::string_view value, char delimiter)
{
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t pos = value.find(delimiter, start);
        const std::size_t end = pos == std::string_view::npos ? value.size() : pos;
        if (end > start) {
            parts.emplace_back(value.substr(start, end - start));
        }
        if (pos == std::string_view::npos) {
            break;
        }
        start = pos + 1;
    }
    return parts;
}

int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

std::string url_decode(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '%' && i + 2 < value.size()) {
            const int hi = hex_value(value[i + 1]);
            const int lo = hex_value(value[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(static_cast<char>(value[i]));
    }
    return out;
}

bool is_url_unreserved(unsigned char c)
{
    return std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~';
}

std::string url_encode(std::string_view value, bool keep_slash)
{
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

std::string shell_quote(std::string_view value)
{
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
}

std::string safe_name(std::string_view value)
{
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
    return out.empty() ? "lora" : out;
}

std::string safe_repo_dir(std::string_view repo_id)
{
    std::string value(repo_id);
    std::replace(value.begin(), value.end(), '/', '_');
    return safe_name(value);
}

fs::path repo_cache_dir(const fs::path& cache_dir, const std::string& repo_id)
{
    return cache_dir / safe_repo_dir(repo_id);
}

fs::path model_info_cache_file(const fs::path& cache_dir,
                               const std::string& repo_id,
                               const std::string& revision)
{
    return repo_cache_dir(cache_dir, repo_id) / ("model_info_" + safe_name(revision) + ".json");
}

bool path_is_within(const fs::path& path, const fs::path& root)
{
    const fs::path normalized_path = path.lexically_normal();
    const fs::path normalized_root = root.lexically_normal();
    auto path_it = normalized_path.begin();
    auto root_it = normalized_root.begin();
    for (; root_it != normalized_root.end(); ++root_it, ++path_it) {
        if (path_it == normalized_path.end() || *path_it != *root_it) {
            return false;
        }
    }
    return true;
}

void remove_cache_file_if_present(const fs::path& path, const fs::path& cache_root)
{
    if (!path_is_within(path, cache_root)) {
        std::cerr << "Skipping LoRA cache cleanup outside cache dir: " << path << "\n";
        return;
    }

    std::error_code ec;
    const bool exists = fs::exists(path, ec);
    if (ec) {
        std::cerr << "Could not inspect LoRA cache file " << path << ": " << ec.message() << "\n";
        return;
    }
    const bool is_link = fs::is_symlink(path, ec);
    if (ec) {
        std::cerr << "Could not inspect LoRA cache file " << path << ": " << ec.message() << "\n";
        return;
    }
    if (!exists && !is_link) {
        return;
    }

    fs::remove(path, ec);
    if (ec) {
        std::cerr << "Could not remove LoRA cache file " << path << ": " << ec.message() << "\n";
        return;
    }
    std::cout << "Removed LoRA cache file: " << path << "\n";
}

void remove_empty_cache_dirs(fs::path dir, const fs::path& cache_root)
{
    const fs::path root = cache_root.lexically_normal();
    dir = dir.lexically_normal();
    while (dir != root && path_is_within(dir, root) && dir.has_parent_path()) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec) || ec) {
            return;
        }
        if (!fs::is_empty(dir, ec) || ec) {
            return;
        }
        fs::remove(dir, ec);
        if (ec) {
            std::cerr << "Could not remove empty LoRA cache dir " << dir << ": " << ec.message() << "\n";
            return;
        }
        std::cout << "Removed empty LoRA cache dir: " << dir << "\n";
        dir = dir.parent_path().lexically_normal();
    }
}

bool is_weight_file(std::string_view filename)
{
    const std::string lower = to_lower(std::string(filename));
    bool has_weight_ext = false;
    for (std::string_view ext : kWeightExts) {
        if (ends_with(lower, ext)) {
            has_weight_ext = true;
            break;
        }
    }
    if (!has_weight_ext) return false;
    for (std::string_view token : kSkipTokens) {
        if (lower.find(token) != std::string::npos) {
            return false;
        }
    }
    return true;
}

std::string strip_query_fragment(std::string value)
{
    const std::size_t query = value.find('?');
    const std::size_t fragment = value.find('#');
    std::size_t cut = std::min(query == std::string::npos ? value.size() : query,
                               fragment == std::string::npos ? value.size() : fragment);
    value.resize(cut);
    return value;
}

HfSource parse_hf_source(const std::string& source, const std::string& default_revision)
{
    if (source.empty()) {
        throw std::runtime_error("LoRA source is empty");
    }

    const std::string hf_prefix = "https://huggingface.co/";
    const std::string hf_http_prefix = "http://huggingface.co/";
    if (starts_with(source, hf_prefix) || starts_with(source, hf_http_prefix)) {
        std::string path = starts_with(source, hf_prefix)
            ? source.substr(hf_prefix.size())
            : source.substr(hf_http_prefix.size());
        path = strip_query_fragment(path);
        const auto parts = split(path, '/');
        if (parts.size() < 2) {
            throw std::runtime_error("Could not find Hugging Face repo id in URL: " + source);
        }

        HfSource parsed;
        parsed.repo_id = url_decode(parts[0]) + "/" + url_decode(parts[1]);
        parsed.revision = default_revision.empty() ? "main" : default_revision;
        if (parts.size() >= 5 && (parts[2] == "blob" || parts[2] == "resolve")) {
            parsed.revision = url_decode(parts[3]);
            std::vector<std::string> file_parts;
            for (std::size_t i = 4; i < parts.size(); ++i) {
                file_parts.push_back(url_decode(parts[i]));
            }
            for (std::size_t i = 0; i < file_parts.size(); ++i) {
                if (i != 0) parsed.filename.push_back('/');
                parsed.filename += file_parts[i];
            }
        }
        return parsed;
    }

    const auto parts = split(source, '/');
    if (parts.size() < 2) {
        throw std::runtime_error("Use a Hugging Face repo id like owner/repo, or a full file URL");
    }

    HfSource parsed;
    parsed.repo_id = parts[0] + "/" + parts[1];
    parsed.revision = default_revision.empty() ? "main" : default_revision;
    if (parts.size() > 2 && is_weight_file(source)) {
        for (std::size_t i = 2; i < parts.size(); ++i) {
            if (i != 2) parsed.filename.push_back('/');
            parsed.filename += parts[i];
        }
    }
    return parsed;
}

std::string hf_resolve_url(const std::string& repo_id, const std::string& filename, const std::string& revision)
{
    return "https://huggingface.co/" + url_encode(repo_id, true) + "/resolve/" +
           url_encode(revision, false) + "/" + url_encode(filename, true);
}

std::string hf_api_url(const std::string& repo_id, const std::string& revision)
{
    return "https://huggingface.co/api/models/" + url_encode(repo_id, true) +
           "/revision/" + url_encode(revision, false);
}

std::string effective_token(const std::string& explicit_token)
{
    if (!explicit_token.empty()) return explicit_token;
    const char* env = std::getenv("HF_TOKEN");
    return env ? std::string(env) : std::string();
}

void curl_download_to_file(const std::string& url,
                           const fs::path& output_path,
                           const std::string& token,
                           bool force)
{
    if (fs::exists(output_path) && !force) {
        return;
    }

    fs::create_directories(output_path.parent_path());
    const fs::path tmp_path = output_path.string() + ".partial";
    fs::remove(tmp_path);

    std::string command = "curl -L --fail --silent --show-error";
    if (!token.empty()) {
        command += " -H " + shell_quote("Authorization: Bearer " + token);
    }
    command += " -o " + shell_quote(tmp_path.string()) + " " + shell_quote(url);

    const int rc = std::system(command.c_str());
    if (rc != 0) {
        fs::remove(tmp_path);
        throw std::runtime_error("curl failed while downloading: " + url);
    }
    fs::rename(tmp_path, output_path);
}

std::string choose_weight_file(const std::string& repo_id,
                               const std::string& requested_file,
                               const std::string& revision,
                               const fs::path& cache_dir,
                               const std::string& token,
                               bool force)
{
    if (!requested_file.empty()) {
        return requested_file;
    }

    const fs::path api_path = model_info_cache_file(cache_dir, repo_id, revision);
    curl_download_to_file(hf_api_url(repo_id, revision), api_path, token, force);

    std::ifstream input(api_path);
    if (!input) {
        throw std::runtime_error("Could not open Hugging Face model info: " + api_path.string());
    }
    const json info = json::parse(input);

    std::vector<std::string> filenames;
    for (const auto& sibling : info.value("siblings", json::array())) {
        if (!sibling.is_object()) continue;
        const std::string filename = sibling.value("rfilename", "");
        if (!filename.empty() && is_weight_file(filename)) {
            filenames.push_back(filename);
        }
    }

    if (filenames.empty()) {
        throw std::runtime_error("No LoRA weight file found in " + repo_id);
    }
    if (filenames.size() == 1) {
        return filenames.front();
    }

    std::ostringstream message;
    message << "Multiple weight files found in " << repo_id << ". Pass --lora_file with one of:";
    for (const auto& filename : filenames) {
        message << "\n  - " << filename;
    }
    throw std::runtime_error(message.str());
}

fs::path download_weight_file(const std::string& repo_id,
                              const std::string& filename,
                              const std::string& revision,
                              const fs::path& cache_dir,
                              const std::string& token,
                              bool force)
{
    const fs::path output_path = repo_cache_dir(cache_dir, repo_id) / filename;
    curl_download_to_file(hf_resolve_url(repo_id, filename, revision), output_path, token, force);
    return output_path;
}

std::uint64_t read_le_u64(const char* data)
{
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | static_cast<unsigned char>(data[i]);
    }
    return value;
}

std::uint32_t read_le_u32(const unsigned char* data)
{
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

std::uint16_t read_le_u16(const unsigned char* data)
{
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint16_t f32_bits_to_bf16(std::uint32_t bits)
{
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16u);
}

std::uint32_t f16_bits_to_f32_bits(std::uint16_t h)
{
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

std::size_t product(const std::vector<std::size_t>& dims)
{
    std::size_t count = 1;
    for (std::size_t dim : dims) {
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("Tensor shape is too large");
        }
        count *= dim;
    }
    return count;
}

std::vector<std::uint16_t> raw_tensor_to_bf16(const std::vector<unsigned char>& raw,
                                             const std::string& dtype,
                                             std::size_t count)
{
    std::vector<std::uint16_t> out(count);
    if (dtype == "BF16") {
        if (raw.size() != count * 2) {
            throw std::runtime_error("BF16 tensor byte size does not match shape");
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = read_le_u16(raw.data() + i * 2);
        }
        return out;
    }
    if (dtype == "F32") {
        if (raw.size() != count * 4) {
            throw std::runtime_error("F32 tensor byte size does not match shape");
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = f32_bits_to_bf16(read_le_u32(raw.data() + i * 4));
        }
        return out;
    }
    if (dtype == "F16") {
        if (raw.size() != count * 2) {
            throw std::runtime_error("F16 tensor byte size does not match shape");
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = f32_bits_to_bf16(f16_bits_to_f32_bits(read_le_u16(raw.data() + i * 2)));
        }
        return out;
    }

    throw std::runtime_error("Unsupported safetensors dtype for LoRA export: " + dtype);
}

std::vector<std::uint16_t> maybe_transpose_2d(const std::vector<std::uint16_t>& src,
                                             std::vector<std::size_t>& dims,
                                             bool transpose_2d)
{
    if (!transpose_2d || dims.size() != 2) {
        return src;
    }

    const std::size_t rows = dims[0];
    const std::size_t cols = dims[1];
    std::vector<std::uint16_t> dst(src.size());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            dst[col * rows + row] = src[row * cols + col];
        }
    }
    dims = {cols, rows};
    return dst;
}

void replace_all(std::string& value, std::string_view from, std::string_view to)
{
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string normalized_lora_bin_name(std::string tensor_name)
{
    for (std::string_view prefix : {"diffusion_model.", "transformer.", "model.diffusion_model."}) {
        if (starts_with(tensor_name, prefix)) {
            tensor_name.erase(0, prefix.size());
            break;
        }
    }

    replace_all(tensor_name, ".lora_A.weight", ".lora_A.default_0.weight");
    replace_all(tensor_name, ".lora_B.weight", ".lora_B.default_0.weight");
    replace_all(tensor_name, ".", "_");
    replace_all(tensor_name, "/", "_");
    replace_all(tensor_name, "\\", "_");

    tensor_name = safe_name(tensor_name);
    return tensor_name + "_bf16_u16.bin";
}

void write_u16_file(const fs::path& path, const std::vector<std::uint16_t>& values)
{
    fs::create_directories(path.parent_path());
    if (fs::exists(path) || fs::is_symlink(path)) {
        fs::remove(path);
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
    if (!output) {
        throw std::runtime_error("Failed writing output file: " + path.string());
    }
}

void export_safetensors_lora_to_bins(const fs::path& safetensors_path,
                                     const fs::path& output_dir,
                                     bool transpose_2d)
{
    std::ifstream input(safetensors_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open LoRA safetensors file: " + safetensors_path.string());
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
    json shapes = json::object();
    std::size_t tensor_count = 0;

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

        std::vector<unsigned char> raw(static_cast<std::size_t>(end - start));
        input.seekg(static_cast<std::streamoff>(data_start + start), std::ios::beg);
        input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (input.gcount() != static_cast<std::streamsize>(raw.size())) {
            throw std::runtime_error("Could not read tensor payload: " + tensor_name);
        }

        const std::size_t count = product(dims);
        auto values = raw_tensor_to_bf16(raw, dtype, count);
        values = maybe_transpose_2d(values, dims, transpose_2d);

        const std::string out_name = normalized_lora_bin_name(tensor_name);
        if (out_name.find("_lora_A_") == std::string::npos &&
            out_name.find("_lora_B_") == std::string::npos) {
            continue;
        }

        write_u16_file(output_dir / out_name, values);
        shapes[out_name] = dims;
        ++tensor_count;
    }

    if (tensor_count == 0) {
        throw std::runtime_error("No LoRA tensors were exported from: " + safetensors_path.string());
    }

    const fs::path shapes_path = output_dir / "shapes.json";
    if (fs::exists(shapes_path) || fs::is_symlink(shapes_path)) {
        fs::remove(shapes_path);
    }
    std::ofstream shapes_out(shapes_path);
    shapes_out << std::setw(2) << shapes << "\n";

    std::cout << "Exported " << tensor_count << " LoRA tensors to " << output_dir << "\n";
}

void remove_lora_bins(const fs::path& lora_dir)
{
    if (!fs::is_directory(lora_dir)) {
        return;
    }

    for (const auto& entry : fs::recursive_directory_iterator(lora_dir)) {
        const fs::path path = entry.path();
        const std::string name = path.filename().string();
        if (name.find("_lora_A_") != std::string::npos ||
            name.find("_lora_B_") != std::string::npos) {
            fs::remove(path);
        }
    }
}

fs::path default_lora_dir(const fs::path& export_root,
                          const std::string& repo_id,
                          const std::string& filename)
{
    const std::string stem = safe_name(repo_id + "__" + fs::path(filename).stem().string());
    return export_root / stem;
}

bool directory_contains_lora_bins(const fs::path& lora_dir)
{
    std::error_code ec;
    if (!fs::is_directory(lora_dir, ec)) {
        return false;
    }

    bool has_lora_a = false;
    bool has_lora_b = false;
    for (fs::directory_iterator it(lora_dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_regular_file(entry_ec)) {
            continue;
        }

        const std::string name = it->path().filename().string();
        has_lora_a = has_lora_a || name.find("_lora_A_") != std::string::npos;
        has_lora_b = has_lora_b || name.find("_lora_B_") != std::string::npos;
        if (has_lora_a && has_lora_b) {
            return true;
        }
    }
    return false;
}

fs::path existing_lora_bin_dir(const fs::path& lora_dir)
{
    return directory_contains_lora_bins(lora_dir) ? lora_dir : fs::path();
}

fs::path find_existing_lora_bin_dir(const fs::path& export_root,
                                    const std::string& repo_id,
                                    const std::string& filename)
{
    if (!filename.empty()) {
        return existing_lora_bin_dir(default_lora_dir(export_root, repo_id, filename));
    }

    std::error_code ec;
    if (!fs::is_directory(export_root, ec)) {
        return {};
    }

    const std::string prefix = safe_repo_dir(repo_id) + "__";
    std::vector<fs::path> matches;
    for (fs::directory_iterator it(export_root, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_directory(entry_ec)) {
            continue;
        }
        const std::string dirname = it->path().filename().string();
        if (starts_with(dirname, prefix) && directory_contains_lora_bins(it->path())) {
            matches.push_back(it->path());
        }
    }

    return matches.size() == 1 ? matches.front() : fs::path();
}

} // namespace

fs::path prepare_huggingface_lora_dir(const LoraPrepareOptions& options)
{
    if (options.source.empty()) {
        throw std::runtime_error("--lora_source is empty");
    }

    const std::string token = effective_token(options.token);
    const HfSource parsed = parse_hf_source(options.source, options.revision.empty() ? "main" : options.revision);
    const std::string filename_hint = options.file.empty() ? parsed.filename : options.file;
    if (!options.force) {
        const fs::path existing_dir =
            find_existing_lora_bin_dir(options.export_root, parsed.repo_id, filename_hint);
        if (!existing_dir.empty()) {
            std::cout << "Using existing LoRA bin dir: " << existing_dir << "\n";
            return existing_dir;
        }
    }

    const std::string filename = choose_weight_file(parsed.repo_id,
                                                     filename_hint,
                                                     parsed.revision,
                                                     options.cache_dir,
                                                     token,
                                                     options.force);
    if (!options.force) {
        const fs::path existing_dir =
            find_existing_lora_bin_dir(options.export_root, parsed.repo_id, filename);
        if (!existing_dir.empty()) {
            std::cout << "Using existing LoRA bin dir: " << existing_dir << "\n";
            return existing_dir;
        }
    }

    const fs::path lora_path = download_weight_file(parsed.repo_id,
                                                    filename,
                                                    parsed.revision,
                                                    options.cache_dir,
                                                    token,
                                                    options.force);
    if (lora_path.extension() != ".safetensors") {
        throw std::runtime_error("Only .safetensors LoRA export is currently supported in C++: " + lora_path.string());
    }

    const fs::path lora_dir = default_lora_dir(options.export_root, parsed.repo_id, filename);
    std::cout << "Using Hugging Face LoRA: " << parsed.repo_id << " / " << filename << "\n";
    std::cout << "LoRA cache file: " << lora_path << "\n";
    std::cout << "LoRA bin dir: " << lora_dir << "\n";

    fs::create_directories(lora_dir);
    remove_lora_bins(lora_dir);
    export_safetensors_lora_to_bins(lora_path, lora_dir, true);
    if (!options.keep_cache) {
        const fs::path api_path = model_info_cache_file(options.cache_dir, parsed.repo_id, parsed.revision);
        remove_cache_file_if_present(lora_path, options.cache_dir);
        remove_cache_file_if_present(api_path, options.cache_dir);
        remove_empty_cache_dirs(lora_path.parent_path(), options.cache_dir);
        remove_empty_cache_dirs(api_path.parent_path(), options.cache_dir);
    }
    return lora_dir;
}

} // namespace host
