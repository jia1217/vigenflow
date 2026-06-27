#include "request_parser.hpp"

#include "file_utils.hpp"
#include "server_config.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace server {

namespace {

void apply_cli_defaults(GenParams& params) {
    params.seed = config::default_seed;
    params.steps = config::default_steps;
    params.H = config::default_height;
    params.W = config::default_width;
    params.prompt = config::default_prompt;
}

void apply_size_preset(GenParams& params, const std::string& size) {
    if (size == "3:4" || size == "768x1024") {
        params.W = 768;
        params.H = 1024;
    } else if (size == "4:3" || size == "1024x768") {
        params.W = 1024;
        params.H = 768;
    } else if (size == "9:16" || size == "576x1024") {
        params.W = 576;
        params.H = 1024;
    } else if (size == "16:9" || size == "1024x576") {
        params.W = 1024;
        params.H = 576;
    } else if (size == "1:1" || size == "1024x1024") {
        params.W = 1024;
        params.H = 1024;
    } 
}

void apply_size_from_body_fallback(GenParams& params, const std::string& body) {
    if (body.find("3:4") != std::string::npos) {
        params.H = 1024;
        params.W = 768;
    } else if (body.find("4:3") != std::string::npos) {
        params.H = 768;
        params.W = 1024;
    } else if (body.find("9:16") != std::string::npos) {
        params.H = 1024;
        params.W = 576;
    } else if (body.find("16:9") != std::string::npos) {
        params.H = 576;
        params.W = 1024;
    } else if (body.find("1:1") != std::string::npos) {
        params.H = 1024;
        params.W = 1024;
    }
}

bool parse_bool_field(const std::string& value) {
    return value == "true" || value == "1" || value == "yes" || value == "on";
}

std::pair<std::size_t, std::uint64_t> image_part_signature(const std::string& data) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char ch : data) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return {data.size(), hash};
}

void apply_lora_json_options(GenParams& params, const json& j) {
    if (j.contains("lora_dir") && j["lora_dir"].is_string()) {
        params.lora_dir = j["lora_dir"].get<std::string>();
    }
    if (j.contains("lora_source") && j["lora_source"].is_string()) {
        params.lora_source = j["lora_source"].get<std::string>();
    }
    if (j.contains("lora_file") && j["lora_file"].is_string()) {
        params.lora_file = j["lora_file"].get<std::string>();
    }
    if (j.contains("lora_revision") && j["lora_revision"].is_string()) {
        params.lora_revision = j["lora_revision"].get<std::string>();
    }
    if (j.contains("lora_cache_dir") && j["lora_cache_dir"].is_string()) {
        params.lora_cache_dir = j["lora_cache_dir"].get<std::string>();
    }
    if (j.contains("lora_export_dir") && j["lora_export_dir"].is_string()) {
        params.lora_export_dir = j["lora_export_dir"].get<std::string>();
    }
    if (j.contains("lora_force") && j["lora_force"].is_boolean()) {
        params.lora_force = j["lora_force"].get<bool>();
    }
    if (j.contains("lora_keep_cache") && j["lora_keep_cache"].is_boolean()) {
        params.lora_keep_cache = j["lora_keep_cache"].get<bool>();
    }
    if (j.contains("lora_rank") && j["lora_rank"].is_number_integer()) {
        params.lora_rank = j["lora_rank"].get<int>();
    }
    if (j.contains("lora_scale") && j["lora_scale"].is_number()) {
        params.lora_scale = j["lora_scale"].get<float>();
    }
}

void apply_gguf_json_options(GenParams& params, const json& j) {
    if (j.contains("gguf_path") && j["gguf_path"].is_string()) {
        params.gguf_path = j["gguf_path"].get<std::string>();
    }
    if (j.contains("gguf_repo") && j["gguf_repo"].is_string()) {
        params.gguf_repo = j["gguf_repo"].get<std::string>();
    }
    if (j.contains("gguf_file") && j["gguf_file"].is_string()) {
        params.gguf_file = j["gguf_file"].get<std::string>();
    }
    if (j.contains("gguf_revision") && j["gguf_revision"].is_string()) {
        params.gguf_revision = j["gguf_revision"].get<std::string>();
    }
    if (j.contains("gguf_hf_token") && j["gguf_hf_token"].is_string()) {
        params.gguf_hf_token = j["gguf_hf_token"].get<std::string>();
    }
    if (j.contains("force_gguf_download") && j["force_gguf_download"].is_boolean()) {
        params.force_gguf_download = j["force_gguf_download"].get<bool>();
    }
}

}  // namespace

std::string get_response_format(const std::string& body) {
    try {
        json j = json::parse(body);
        if (j.contains("response_format") && j["response_format"].is_string()) {
            return j["response_format"].get<std::string>();
        }
    } catch (...) {
    }
    return "b64_json";
}

GenParams parse_request(const std::string& body) {
    std::cout << "========== RAW HTTP BODY ==========\n";
    std::cout << body << "\n";
    std::cout << "===================================\n";

    GenParams params;
    params.model = config::model_id;
    apply_cli_defaults(params);

    try {
        json j = json::parse(body);

        if (j.contains("model") && j["model"].is_string()) {
            params.model = j["model"].get<std::string>();
        }
        if (j.contains("prompt") && j["prompt"].is_string()) {
            params.prompt = j["prompt"].get<std::string>();
        }
        if (j.contains("steps") && j["steps"].is_number_integer()) {
            params.steps = j["steps"].get<int>();
        }
        if (j.contains("seed") && j["seed"].is_number_integer()) {
            params.seed = j["seed"].get<int>();
        }
        if (j.contains("size") && j["size"].is_string()) {
            apply_size_preset(params, j["size"].get<std::string>());
        }
        apply_lora_json_options(params, j);
        apply_gguf_json_options(params, j);
        return params;
    } catch (...) {
    }

    apply_size_from_body_fallback(params, body);

    std::regex steps_regex(R"("steps"\s*:\s*(\d+))");
    std::smatch match_steps;
    if (std::regex_search(body, match_steps, steps_regex)) {
        params.steps = std::stoi(match_steps[1].str());
    }

    std::regex seed_regex(R"("seed"\s*:\s*(\d+))");
    std::smatch match_seed;
    if (std::regex_search(body, match_seed, seed_regex)) {
        params.seed = std::stoi(match_seed[1].str());
    }

    const std::string prompt_key = "\"prompt\":";
    const size_t p_key_pos = body.find(prompt_key);
    if (p_key_pos != std::string::npos) {
        const size_t v_start = body.find("\"", p_key_pos + prompt_key.length());
        const size_t v_end = body.find("\"", v_start + 1);
        if (v_start != std::string::npos && v_end != std::string::npos) {
            params.prompt = body.substr(v_start + 1, v_end - v_start - 1);
        }
    }

    return params;
}

GenParams parse_multipart_request(const http::request<http::string_body>& req) {
    GenParams params;
    params.model = config::model_id;
    apply_cli_defaults(params);
    ensure_output_dir();

    const std::string content_type(req[http::field::content_type]);
    std::string boundary;
    const auto pos = content_type.find("boundary=");
    if (pos != std::string::npos) {
        boundary = "--" + content_type.substr(pos + 9);
    } else {
        throw std::runtime_error("Missing multipart boundary");
    }

    const std::string& body = req.body();
    size_t start_pos = 0;
    std::set<std::pair<std::size_t, std::uint64_t>> saved_image_parts;

    while ((start_pos = body.find(boundary, start_pos)) != std::string::npos) {
        start_pos += boundary.length();
        if (body.compare(start_pos, 2, "--") == 0) break;
        start_pos += 2;

        const size_t header_end = body.find("\r\n\r\n", start_pos);
        if (header_end == std::string::npos) break;

        const std::string headers = body.substr(start_pos, header_end - start_pos);
        const size_t body_start = header_end + 4;
        const size_t next_boundary = body.find(boundary, body_start);
        if (next_boundary == std::string::npos) break;

        const size_t data_len =
            (next_boundary > body_start + 2) ? (next_boundary - body_start - 2) : 0;
        const std::string part_data = body.substr(body_start, data_len);

        std::smatch match;
        if (std::regex_search(headers, match, std::regex(R"re(name="([^"]+)")re"))) {
            const std::string name = match[1].str();

            if (name == "prompt") {
                params.prompt = part_data;
            } else if (name == "size") {
                apply_size_preset(params, part_data);
            } else if (name == "model") {
                params.model = part_data;
            } else if (name == "lora_dir") {
                params.lora_dir = part_data;
            } else if (name == "lora_source") {
                params.lora_source = part_data;
            } else if (name == "lora_file") {
                params.lora_file = part_data;
            } else if (name == "lora_revision") {
                params.lora_revision = part_data;
            } else if (name == "lora_cache_dir") {
                params.lora_cache_dir = part_data;
            } else if (name == "lora_export_dir") {
                params.lora_export_dir = part_data;
            } else if (name == "lora_force") {
                params.lora_force = parse_bool_field(part_data);
            } else if (name == "lora_keep_cache") {
                params.lora_keep_cache = parse_bool_field(part_data);
            } else if (name == "lora_rank") {
                params.lora_rank = std::stoi(part_data);
            } else if (name == "lora_scale") {
                params.lora_scale = std::stof(part_data);
            } else if (name == "gguf_path") {
                params.gguf_path = part_data;
            } else if (name == "gguf_repo") {
                params.gguf_repo = part_data;
            } else if (name == "gguf_file") {
                params.gguf_file = part_data;
            } else if (name == "gguf_revision") {
                params.gguf_revision = part_data;
            } else if (name == "gguf_hf_token") {
                params.gguf_hf_token = part_data;
            } else if (name == "force_gguf_download") {
                params.force_gguf_download = parse_bool_field(part_data);
            } else if (name == "image[]" || name == "image") {
                if (!params.input_images.empty()) {
                    std::cout << "[INFO] Ignoring extra uploaded image part: "
                              << name << "\n";
                    start_pos = next_boundary;
                    continue;
                }

                const auto signature = image_part_signature(part_data);
                if (!saved_image_parts.insert(signature).second) {
                    std::cout << "[INFO] Ignoring duplicate uploaded image part: "
                              << name << "\n";
                    start_pos = next_boundary;
                    continue;
                }

                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();

                const std::string tmp_filename =
                    "upload_" + std::to_string(ms) + "_" +
                    std::to_string(params.input_images.size()) + ".png";
                const std::string save_path =
                    (fs::path(config::output_dir) / tmp_filename).string();

                std::ofstream out(save_path, std::ios::binary);
                out.write(part_data.data(), static_cast<std::streamsize>(part_data.size()));

                std::cout << "[INFO] Received and saved input image: " << save_path << "\n";
                params.input_images.push_back(save_path);
            }
        }
        start_pos = next_boundary;
    }

    return params;
}

}  // namespace server
