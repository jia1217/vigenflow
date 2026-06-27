#include "http_handlers.hpp"

#include "file_utils.hpp"
#include "request_parser.hpp"
#include "server_config.hpp"
#include "worker.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace server {

namespace {

std::mutex g_infer_mutex;
std::mutex g_openwebui_model_mutex;
std::mutex g_automatic1111_model_mutex;
std::mutex g_last_generation_info_mutex;
std::string g_last_openwebui_model;
std::string g_automatic1111_model;
json g_last_generation_usage = json::object();
std::uint64_t g_openwebui_model_generation = 0;
std::chrono::steady_clock::time_point g_last_openwebui_model_at{};

constexpr auto kRecentOpenWebUIModelWindow = std::chrono::seconds(20);

struct RememberedOpenWebUIModel {
    std::optional<std::string> model;
    std::uint64_t generation = 0;
    std::chrono::steady_clock::time_point remembered_at{};
};

http::response<http::string_body> make_json_response(
    http::status status, unsigned version, bool keep_alive, const json& body) {
    http::response<http::string_body> res{status, version};
    res.set(http::field::content_type, "application/json");
    res.keep_alive(keep_alive);
    res.body() = body.dump();
    res.prepare_payload();
    return res;
}

http::response<http::vector_body<char>> make_image_response(
    unsigned version, bool keep_alive, std::vector<char>&& data, const std::string& mime) {
    http::response<http::vector_body<char>> res{http::status::ok, version};
    res.set(http::field::content_type, mime);
    res.keep_alive(keep_alive);
    res.body() = std::move(data);
    res.prepare_payload();
    return res;
}

std::string format_seconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << "s";
    return out.str();
}

std::string format_whole_seconds(double seconds) {
    const auto rounded = static_cast<long long>(seconds + 0.5);
    return std::to_string(rounded) + "s";
}

nlohmann::ordered_json build_openwebui_generation_info(
    const GenParams& params,
    const GenerationTimingInfo& timing) {
    nlohmann::ordered_json info;
    info["1_Image_H"] = params.H;
    info["2_Image_W"] = params.W;
    info["3_Denoising_steps"] = timing.denoising_steps.value_or(params.steps);

    if (timing.vae_encoder_seconds) {
        info["4_Vae_encoder"] = format_seconds(*timing.vae_encoder_seconds);
    }
    if (timing.text_encoder_seconds) {
        info["5_Text_encoder"] = format_seconds(*timing.text_encoder_seconds);
    }
    if (timing.denoising_seconds_per_step) {
        const double total_seconds = timing.denoising_total_seconds.value_or(
            *timing.denoising_seconds_per_step *
            static_cast<double>(timing.denoising_steps.value_or(params.steps)));
        info["6_Denoising"] =
            format_seconds(*timing.denoising_seconds_per_step) + "/step, " +
            format_whole_seconds(total_seconds);
    }
    if (timing.vae_decoder_seconds) {
        info["7_Vae_decoder"] = format_seconds(*timing.vae_decoder_seconds);
    }
    if (timing.e2e_runtime_seconds) {
        info["8_E2E_runtime"] = format_seconds(*timing.e2e_runtime_seconds);
    }

    return info;
}

json generation_info_as_json(const nlohmann::ordered_json& info) {
    return json::parse(info.dump());
}

void remember_generation_usage(const json& usage) {
    std::lock_guard<std::mutex> lock(g_last_generation_info_mutex);
    g_last_generation_usage = usage;
}

json last_generation_usage_or_default() {
    std::lock_guard<std::mutex> lock(g_last_generation_info_mutex);
    if (!g_last_generation_usage.empty()) {
        return g_last_generation_usage;
    }
    return {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}};
}

json build_image_response_body(const GenParams& params, const WorkerResult& result) {
    const auto data = read_binary_file(result.image_path);
    const auto ordered_info = build_openwebui_generation_info(params, result.timing);
    const std::string info_text = ordered_info.dump();
    const json metadata = generation_info_as_json(ordered_info);
    remember_generation_usage(metadata);

    const json item = {
        {"b64_json", base64_encode(data)},
        {"revised_prompt", params.prompt},
        {"info", info_text},
        {"metadata", metadata}};
    return {
        {"created", static_cast<long long>(std::time(nullptr))},
        {"data", json::array({item})},
        {"info", info_text},
        {"metadata", metadata}};
}

constexpr const char* kZImageTurboQ41ModelId = "z-image-turbo-Q4_1-GGUF";
constexpr const char* kZImageTurboQ41LoraModelId = "z-image-turbo-Q4_1-lora";
constexpr const char* kZImageTurboQ41GgufLoraModelId = "z-image-turbo-Q4_1-GGUF-lora";
constexpr const char* kZImageTurboBf16ModelId = "z-image-turbo-BF16";
constexpr const char* kZImageTurboBf16LoraModelId = "z-image-turbo-BF16-lora";

std::string configured_served_model_id() {
    const std::string model_id = !config::served_model_id.empty()
        ? config::served_model_id
        : config::model_id;

    if (model_id == "flux.2-klein-4B (lora)" || model_id == "flux.2-klein-4B(lora)" ||
        model_id == "flux.2-klein-lora" || model_id == "flux.2-klein (lora)" ||
        model_id == "flux.2-klein(lora)") {
        return "flux.2-klein-4B-lora";
    }
    if (model_id == "flux.2-klein-4B-edit (lora)" ||
        model_id == "flux.2-klein-4B-edit(lora)" ||
        model_id == "flux.2-klein-edit-lora" ||
        model_id == "flux.2-klein-edit (lora)" ||
        model_id == "flux.2-klein-edit(lora)") {
        return "flux.2-klein-4B-edit-lora";
    }
    if (model_id == "z-image-turbo" || model_id == "z-image-turbo-q41") {
        return kZImageTurboQ41ModelId;
    }
    if (model_id == "z-image-turbo-q41-lora" || model_id == "z-image-turbo-q41_lora" ||
        model_id == "z-image-turbo-lora" || model_id == "z-image-turbo-lora-q41" ||
        model_id == kZImageTurboQ41GgufLoraModelId) {
        return kZImageTurboQ41LoraModelId;
    }
    if (model_id == "z-image-turbo-bf16") {
        return kZImageTurboBf16ModelId;
    }
    if (model_id == "z-image-turbo-bf16 (lora)" || model_id == "z-image-turbo-bf16(lora)" ||
        model_id == "z-image-turbo-bf16-lora") {
        return kZImageTurboBf16LoraModelId;
    }

    return model_id;
}

std::string current_automatic1111_model_id() {
    std::lock_guard<std::mutex> lock(g_automatic1111_model_mutex);
    return g_automatic1111_model.empty() ? configured_served_model_id() : g_automatic1111_model;
}

void set_current_automatic1111_model_id(const std::string& model_id) {
    if (model_id.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_automatic1111_model_mutex);
    g_automatic1111_model = model_id;
    std::cout << "[INFO] Selected Automatic1111/OpenWebUI model: "
              << g_automatic1111_model << "\n";
}

bool is_flux_klein_model(const std::string& model_id) {
    return model_id == "flux.2-klein-4B" || model_id == "flux.2-klein" ||
           model_id == "flux.2-klein-4B (lora)" || model_id == "flux.2-klein-4B(lora)" ||
           model_id == "flux.2-klein-4B-lora" || model_id == "flux.2-klein (lora)" ||
           model_id == "flux.2-klein(lora)" || model_id == "flux.2-klein-lora" ||
           model_id == "flux.2-klein-4B-edit" || model_id == "flux.2-klein-edit" ||
           model_id == "flux.2-klein-4B-edit (lora)" ||
           model_id == "flux.2-klein-4B-edit(lora)" ||
           model_id == "flux.2-klein-4B-edit-lora" ||
           model_id == "flux.2-klein-edit (lora)" ||
           model_id == "flux.2-klein-edit(lora)" ||
           model_id == "flux.2-klein-edit-lora";
}

bool is_flux_klein_lora_model(const std::string& model_id) {
    return model_id == "flux.2-klein-4B (lora)" || model_id == "flux.2-klein-4B(lora)" ||
           model_id == "flux.2-klein-4B-lora" || model_id == "flux.2-klein (lora)" ||
           model_id == "flux.2-klein(lora)" || model_id == "flux.2-klein-lora" ||
           model_id == "flux.2-klein-4B-edit (lora)" ||
           model_id == "flux.2-klein-4B-edit(lora)" ||
           model_id == "flux.2-klein-4B-edit-lora" ||
           model_id == "flux.2-klein-edit (lora)" ||
           model_id == "flux.2-klein-edit(lora)" ||
           model_id == "flux.2-klein-edit-lora";
}

bool is_z_image_bf16_model(const std::string& model_id) {
    return model_id == "z-image-turbo-bf16" || model_id == kZImageTurboBf16ModelId ||
           model_id == "z-image-turbo-bf16 (lora)" || model_id == "z-image-turbo-bf16(lora)" ||
           model_id == "z-image-turbo-bf16-lora" || model_id == kZImageTurboBf16LoraModelId;
}

bool is_z_image_bf16_lora_model(const std::string& model_id) {
    return model_id == "z-image-turbo-bf16 (lora)" || model_id == "z-image-turbo-bf16(lora)" ||
           model_id == "z-image-turbo-bf16-lora" || model_id == kZImageTurboBf16LoraModelId;
}

bool is_z_image_q41_lora_model(const std::string& model_id) {
    return model_id == "z-image-turbo-q41-lora" ||
           model_id == "z-image-turbo-q41_lora" ||
           model_id == "z-image-turbo-lora" ||
           model_id == "z-image-turbo-lora-q41" ||
           model_id == kZImageTurboQ41LoraModelId ||
           model_id == kZImageTurboQ41GgufLoraModelId;
}

bool is_z_image_lora_model(const std::string& model_id) {
    return is_z_image_q41_lora_model(model_id) ||
           is_z_image_bf16_lora_model(model_id);
}

bool is_z_image_q41_model(const std::string& model_id) {
    return model_id == "z-image-turbo" || model_id == "z-image-turbo-q41" ||
           model_id == kZImageTurboQ41ModelId ||
           is_z_image_q41_lora_model(model_id);
}

bool serving_z_image_bf16(const std::string& served_model_id) {
    return is_z_image_bf16_model(served_model_id) || is_z_image_bf16_model(config::model_id);
}

bool serving_z_image_q41(const std::string& served_model_id) {
    return is_z_image_q41_model(served_model_id) || is_z_image_q41_model(config::model_id);
}

void add_model_entry(json& data, std::set<std::string>& seen, const std::string& model_id) {
    if (model_id.empty() || !seen.insert(model_id).second) {
        return;
    }

    data.push_back(
        {{"id", model_id},
         {"object", "model"},
         {"created", 0},
         {"owned_by", "local"}});
}

void add_sd_model_entry(json& data, std::set<std::string>& seen, const std::string& model_id) {
    if (model_id.empty() || !seen.insert(model_id).second) {
        return;
    }

    data.push_back(
        {{"title", model_id},
         {"model_name", model_id},
         {"hash", nullptr},
         {"sha256", nullptr},
         {"filename", model_id},
         {"config", nullptr}});
}


bool is_local_model_alias(const std::string& model_id) {
    return model_id.empty() ||
           model_id == "local-model" ||
           model_id == "local_model" ||
           model_id == "local-model-1" ||
           model_id == "local_model_1" ||
           model_id == "local-model-npu" ||
           model_id == "local_model_npu" ||
           model_id == "local-model-npu-1" ||
           model_id == "local_model_npu_1" ||
           model_id == "edit_model";
}

std::optional<std::string> canonical_lora_model_id(const std::string& model_id) {
    const auto entry = resolve_lora_model_entry(model_id);
    if (!entry) {
        return std::nullopt;
    }
    return entry->model_id;
}

bool is_generic_flux_klein_lora_model(const std::string& model_id) {
    if (canonical_lora_model_id(model_id)) {
        return false;
    }
    if (is_flux_klein_lora_model(model_id)) {
        return true;
    }
    if (is_local_model_alias(model_id)) {
        if (is_flux_klein_lora_model(config::model_id) &&
            !canonical_lora_model_id(config::model_id)) {
            return true;
        }
        return !config::served_model_id.empty() &&
               is_flux_klein_lora_model(config::served_model_id) &&
               !canonical_lora_model_id(config::served_model_id);
    }
    return false;
}

bool is_generic_z_image_lora_model(const std::string& model_id) {
    if (canonical_lora_model_id(model_id)) {
        return false;
    }
    if (is_z_image_lora_model(model_id)) {
        return true;
    }
    if (is_local_model_alias(model_id)) {
        if (is_z_image_lora_model(config::model_id) &&
            !canonical_lora_model_id(config::model_id)) {
            return true;
        }
        return !config::served_model_id.empty() &&
               is_z_image_lora_model(config::served_model_id) &&
               !canonical_lora_model_id(config::served_model_id);
    }
    return false;
}

bool is_generic_lora_model(const std::string& model_id) {
    return is_generic_flux_klein_lora_model(model_id) ||
           is_generic_z_image_lora_model(model_id);
}

bool should_apply_remembered_model(
    const std::string& requested_model,
    const std::string& remembered_model) {
    if (is_local_model_alias(requested_model)) {
        return !is_local_model_alias(remembered_model);
    }
    if (canonical_lora_model_id(requested_model)) {
        return false;
    }
    if (requested_model == "flux.2-klein-4B" ||
        requested_model == "flux.2-klein" ||
        requested_model == "flux.2-klein-4B-edit" ||
        requested_model == "flux.2-klein-edit") {
        return false;
    }
    return normalize_model_target(requested_model) == normalize_model_target(remembered_model);
}


bool contains_string(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

std::vector<std::string> lora_prompt_match_tokens(std::string value) {
    std::vector<std::string> tokens;
    std::string token;

    auto flush_token = [&]() {
        if (token.size() < 3) {
            token.clear();
            return;
        }
        static const std::vector<std::string> ignored = {
            "flux", "klein", "lora", "edit", "model", "safetensors", "fal", "the", "and", "with"
        };
        if (!contains_string(ignored, token) && !contains_string(tokens, token)) {
            tokens.push_back(token);
        }
        token.clear();
    };

    for (unsigned char ch : value) {
        if (std::isalnum(ch)) {
            token.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            flush_token();
        }
    }
    flush_token();
    return tokens;
}

int lora_prompt_match_score(const LoraModelEntry& entry, const std::string& prompt) {
    const auto prompt_tokens = lora_prompt_match_tokens(prompt);
    if (prompt_tokens.empty()) {
        return 0;
    }

    const auto entry_tokens = lora_prompt_match_tokens(
        entry.model_id + " " + entry.lora_source + " " + entry.lora_file);
    int score = 0;
    for (const auto& entry_token : entry_tokens) {
        for (const auto& prompt_token : prompt_tokens) {
            if (entry_token == prompt_token ||
                (entry_token.size() >= 4 && prompt_token.find(entry_token) != std::string::npos) ||
                (prompt_token.size() >= 4 && entry_token.find(prompt_token) != std::string::npos)) {
                ++score;
                break;
            }
        }
    }
    return score;
}

std::optional<std::string> prompt_matched_lora_model_id(
    const std::string& requested_model,
    const std::string& prompt) {
    const std::string requested_target = normalize_model_target(requested_model);
    std::optional<LoraModelEntry> best_entry;
    int best_score = 0;
    bool tied = false;

    for (const auto& entry : list_lora_model_entries()) {
        if (normalize_model_target(entry.model_id) != requested_target) {
            continue;
        }
        const int score = lora_prompt_match_score(entry, prompt);
        if (score > best_score) {
            best_score = score;
            best_entry = entry;
            tied = false;
        } else if (score > 0 && score == best_score) {
            tied = true;
        }
    }

    if (best_score > 0 && best_entry && !tied) {
        return best_entry->model_id;
    }
    return std::nullopt;
}

std::optional<std::string> request_model_from_json_body(const std::string& body) {
    try {
        const json parsed = json::parse(body);
        if (parsed.contains("model") && parsed["model"].is_string()) {
            return parsed["model"].get<std::string>();
        }
    } catch (...) {
    }
    return std::nullopt;
}

void remember_openwebui_model(const std::string& body) {
    const auto model = request_model_from_json_body(body);
    if (!model || is_local_model_alias(*model)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_openwebui_model_mutex);
    const auto canonical_model = canonical_lora_model_id(*model);
    const std::string selected_model = canonical_model ? *canonical_model : *model;
    if (g_last_openwebui_model != selected_model) {
        std::cout << "[INFO] Remembered OpenWebUI model: "
                  << selected_model << "\n";
    }
    g_last_openwebui_model = selected_model;
    ++g_openwebui_model_generation;
    g_last_openwebui_model_at = std::chrono::steady_clock::now();
}

RememberedOpenWebUIModel remembered_openwebui_model() {
    std::lock_guard<std::mutex> lock(g_openwebui_model_mutex);
    if (g_last_openwebui_model.empty()) {
        return {std::nullopt, g_openwebui_model_generation, {}};
    }
    return {g_last_openwebui_model, g_openwebui_model_generation, g_last_openwebui_model_at};
}

bool is_recent_openwebui_model(const RememberedOpenWebUIModel& remembered_model) {
    return remembered_model.model &&
           remembered_model.remembered_at != std::chrono::steady_clock::time_point{} &&
           std::chrono::steady_clock::now() - remembered_model.remembered_at <=
               kRecentOpenWebUIModelWindow;
}

std::optional<RememberedOpenWebUIModel> wait_for_openwebui_model_update(
    const std::string& requested_model,
    std::uint64_t after_generation) {
    for (int attempt = 0; attempt < 60; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto remembered_model = remembered_openwebui_model();
        if (remembered_model.generation == after_generation) {
            continue;
        }
        if (remembered_model.model &&
            should_apply_remembered_model(requested_model, *remembered_model.model)) {
            return remembered_model;
        }
        return remembered_model;
    }
    return std::nullopt;
}

void apply_openwebui_image_model_alias(GenParams& params) {
    if (const auto canonical_model = canonical_lora_model_id(params.model)) {
        if (*canonical_model != params.model) {
            std::cout << "[INFO] OpenWebUI sent image model " << params.model
                      << "; using canonical LoRA model " << *canonical_model << "\n";
            params.model = *canonical_model;
        }
        {
            std::lock_guard<std::mutex> lock(g_openwebui_model_mutex);
            if (g_last_openwebui_model != *canonical_model) {
                std::cout << "[INFO] Remembered OpenWebUI image LoRA model: "
                          << *canonical_model << "\n";
            }
            g_last_openwebui_model = *canonical_model;
            ++g_openwebui_model_generation;
            g_last_openwebui_model_at = std::chrono::steady_clock::now();
        }
        return;
    }

    const bool requested_local_model_alias = is_local_model_alias(params.model);
    RememberedOpenWebUIModel remembered_model = remembered_openwebui_model();
    if (requested_local_model_alias) {
        if (is_recent_openwebui_model(remembered_model) &&
            should_apply_remembered_model(params.model, *remembered_model.model)) {
            std::cout << "[INFO] OpenWebUI sent image model " << params.model
                      << "; using recent OpenWebUI model selected before image edit "
                      << *remembered_model.model << "\n";
            params.model = *remembered_model.model;
            return;
        }

        if (const auto fresh_model =
                wait_for_openwebui_model_update(params.model, remembered_model.generation)) {
            remembered_model = *fresh_model;
            if (remembered_model.model &&
                should_apply_remembered_model(params.model, *remembered_model.model)) {
                std::cout << "[INFO] OpenWebUI sent image model " << params.model
                          << "; using fresh OpenWebUI model "
                          << *remembered_model.model << "\n";
                params.model = *remembered_model.model;
                return;
            }
        }

        std::cout << "[INFO] OpenWebUI sent image model " << params.model
                  << "; no fresh OpenWebUI model update arrived, using configured model "
                  << config::model_id << "\n";
        params.model = config::model_id;
    }

    if (is_generic_lora_model(params.model)) {
        if (!requested_local_model_alias) {
            if (const auto fresh_model =
                    wait_for_openwebui_model_update(params.model, remembered_model.generation)) {
                remembered_model = *fresh_model;
                if (remembered_model.model &&
                    should_apply_remembered_model(params.model, *remembered_model.model)) {
                    std::cout << "[INFO] OpenWebUI sent image model " << params.model
                              << "; using fresh LoRA model " << *remembered_model.model << "\n";
                    params.model = *remembered_model.model;
                }
                return;
            }
        }
        if (const auto prompt_model = prompt_matched_lora_model_id(params.model, params.prompt)) {
            std::cout << "[INFO] OpenWebUI sent image model " << params.model
                      << "; using prompt-matched LoRA model " << *prompt_model << "\n";
            params.model = *prompt_model;
            return;
        }
        if (!requested_local_model_alias &&
            remembered_model.model &&
            should_apply_remembered_model(params.model, *remembered_model.model)) {
            std::cout << "[INFO] OpenWebUI sent generic image model " << params.model
                      << "; using remembered LoRA model "
                      << *remembered_model.model << "\n";
            params.model = *remembered_model.model;
            return;
        }
        return;
    }

    if (remembered_model.model) {
        std::cout << "[INFO] OpenWebUI sent image model " << params.model
                  << "; ignoring stale remembered OpenWebUI model "
                  << *remembered_model.model << "\n";
    }
}

GenParams parse_automatic1111_txt2img_request(const std::string& body) {
    GenParams params;
    params.model = current_automatic1111_model_id();
    params.seed = config::default_seed;
    params.steps = config::default_steps;
    params.H = config::default_height;
    params.W = config::default_width;
    params.prompt = config::default_prompt;

    try {
        const json j = json::parse(body);
        if (j.contains("prompt") && j["prompt"].is_string()) {
            params.prompt = j["prompt"].get<std::string>();
        }
        if (j.contains("steps") && j["steps"].is_number_integer()) {
            params.steps = j["steps"].get<int>();
        }
        if (j.contains("seed") && j["seed"].is_number_integer()) {
            params.seed = j["seed"].get<int>();
        }
        if (j.contains("height") && j["height"].is_number_integer()) {
            params.H = j["height"].get<int>();
        }
        if (j.contains("width") && j["width"].is_number_integer()) {
            params.W = j["width"].get<int>();
        }
        if (j.contains("model") && j["model"].is_string()) {
            params.model = j["model"].get<std::string>();
        }
        apply_openwebui_image_model_alias(params);
    } catch (...) {
    }

    return params;
}

json request_parameters_or_empty_object(const std::string& body) {
    try {
        const json parsed = json::parse(body);
        if (parsed.is_object()) {
            return parsed;
        }
    } catch (...) {
    }
    return json::object();
}

json build_automatic1111_response_body(
    const GenParams& params,
    const WorkerResult& result,
    const std::string& request_body) {
    const auto data = read_binary_file(result.image_path);
    const auto ordered_info = build_openwebui_generation_info(params, result.timing);
    const json metadata = generation_info_as_json(ordered_info);
    remember_generation_usage(metadata);

    return {
        {"images", json::array({base64_encode(data)})},
        {"parameters", request_parameters_or_empty_object(request_body)},
        {"info", ordered_info.dump()}};
}

http::message_generator handle_get_models(unsigned version, bool keep_alive) {
    // Advertise base models. Catalog entries below add selectable per-LoRA model IDs.
    json data = json::array();
    std::set<std::string> seen;

    add_model_entry(data, seen, "flux.2-klein-4B");
    add_model_entry(data, seen, "flux.2-klein-4B-edit");
    add_model_entry(data, seen, kZImageTurboQ41ModelId);
    add_model_entry(data, seen, kZImageTurboBf16ModelId);

    for (const auto& entry : list_lora_model_entries()) {
        add_model_entry(data, seen, entry.model_id);
    }

    const json body = {{"object", "list"}, {"data", data}};
    return make_json_response(http::status::ok, version, keep_alive, body);
}

http::message_generator handle_get_sd_models(unsigned version, bool keep_alive) {
    json data = json::array();
    std::set<std::string> seen;

    add_sd_model_entry(data, seen, "flux.2-klein-4B");
    add_sd_model_entry(data, seen, "flux.2-klein-4B-edit");
    add_sd_model_entry(data, seen, kZImageTurboQ41ModelId);
    add_sd_model_entry(data, seen, kZImageTurboBf16ModelId);

    for (const auto& entry : list_lora_model_entries()) {
        add_sd_model_entry(data, seen, entry.model_id);
    }

    return make_json_response(http::status::ok, version, keep_alive, data);
}

http::message_generator handle_get_sd_options(unsigned version, bool keep_alive) {
    return make_json_response(
        http::status::ok,
        version,
        keep_alive,
        json{{"sd_model_checkpoint", current_automatic1111_model_id()}});
}

http::message_generator handle_post_sd_options(
    const std::string& request_body,
    unsigned version,
    bool keep_alive) {
    try {
        const json body = json::parse(request_body);
        if (body.contains("sd_model_checkpoint") && body["sd_model_checkpoint"].is_string()) {
            set_current_automatic1111_model_id(body["sd_model_checkpoint"].get<std::string>());
        }
    } catch (...) {
    }

    return handle_get_sd_options(version, keep_alive);
}

http::message_generator handle_get_image_file(
    const std::string& target, unsigned version, bool keep_alive) {
    const std::string filename = target.substr(8);
    if (!is_safe_filename(filename)) {
        return make_json_response(
            http::status::bad_request, version, keep_alive, json{{"error", "invalid filename"}});
    }

    const std::string fullpath = config::output_dir + "/" + filename;
    if (!fs::exists(fullpath)) {
        return make_json_response(
            http::status::not_found, version, keep_alive, json{{"error", "image not found"}});
    }

    auto data = read_binary_file(fullpath);
    const std::string ext = fs::path(fullpath).extension().string();
    std::string mime = "image/png";
    if (ext == ".jpg" || ext == ".jpeg") {
        mime = "image/jpeg";
    } else if (ext == ".webp") {
        mime = "image/webp";
    }

    return make_image_response(version, keep_alive, std::move(data), mime);
}

http::message_generator handle_image_generation(
    GenParams params,
    const std::string& request_body,
    unsigned version,
    bool keep_alive) {
    const std::string response_format = get_response_format(request_body);
    apply_openwebui_image_model_alias(params);
    WorkerResult result;
    {
        std::lock_guard<std::mutex> lock(g_infer_mutex);
        result = run_worker(params);
    }

    const json body = build_image_response_body(params, result);
    const bool keep_output = config::keep_images || response_format == "url";
    cleanup_generated_image(result.image_path, keep_output);

    return make_json_response(http::status::ok, version, keep_alive, body);
}

http::message_generator handle_automatic1111_txt2img(
    GenParams params,
    const std::string& request_body,
    unsigned version,
    bool keep_alive) {
    WorkerResult result;
    {
        std::lock_guard<std::mutex> lock(g_infer_mutex);
        result = run_worker(params);
    }

    const json body = build_automatic1111_response_body(params, result, request_body);
    cleanup_generated_image(result.image_path, config::keep_images);

    return make_json_response(http::status::ok, version, keep_alive, body);
}

http::message_generator handle_image_edit(
    const http::request<http::string_body>& req, unsigned version, bool keep_alive) {
    if (req[http::field::content_type].find("multipart/form-data") == std::string::npos) {
        return make_json_response(
            http::status::bad_request,
            version,
            keep_alive,
            json{{"error", "Content-Type must be multipart/form-data"}});
    }

    GenParams params = parse_multipart_request(req);
    apply_openwebui_image_model_alias(params);
    WorkerResult result;
    try {
        std::lock_guard<std::mutex> lock(g_infer_mutex);
        result = run_worker(params);
    } catch (...) {
        cleanup_uploaded_images(params.input_images);
        throw;
    }

    const json body = build_image_response_body(params, result);
    cleanup_generated_image(result.image_path, config::keep_images);
    cleanup_uploaded_images(params.input_images);

    return make_json_response(http::status::ok, version, keep_alive, body);
}

}  // namespace

http::message_generator handle_request(http::request<http::string_body>&& req) {
    const bool keep_alive = req.keep_alive();
    const std::string target = std::string(req.target());
    std::cout << "[REQ] " << req.method_string() << " " << target << "\n";

    try {
        if (req.method() == http::verb::get && target == "/health") {
            return make_json_response(
                http::status::ok, req.version(), keep_alive, json{{"status", "ok"}});
        }

        if (req.method() == http::verb::get && target == "/v1/models") {
            return handle_get_models(req.version(), keep_alive);
        }

        if (req.method() == http::verb::get && target == "/sdapi/v1/sd-models") {
            return handle_get_sd_models(req.version(), keep_alive);
        }

        if (req.method() == http::verb::get && target == "/sdapi/v1/options") {
            return handle_get_sd_options(req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/sdapi/v1/options") {
            return handle_post_sd_options(req.body(), req.version(), keep_alive);
        }

        if (req.method() == http::verb::get && target.rfind("/images/", 0) == 0) {
            return handle_get_image_file(target, req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/v1/images/generations") {
            return handle_image_generation(
                parse_request(req.body()), req.body(), req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/sdapi/v1/txt2img") {
            return handle_automatic1111_txt2img(
                parse_automatic1111_txt2img_request(req.body()),
                req.body(),
                req.version(),
                keep_alive);
        }

        if (req.method() == http::verb::post && target == "/v1/images/edits") {
            return handle_image_edit(req, req.version(), keep_alive);
        }

        if (req.method() == http::verb::post && target == "/v1/chat/completions") {
            remember_openwebui_model(req.body());

            const json usage = last_generation_usage_or_default();
            const json body = {
                {"id", "chatcmpl-local"},
                {"object", "chat.completion"},
                {"created", static_cast<long long>(std::time(nullptr))},
                {"model", configured_served_model_id()},
                {"choices",
                 json::array(
                     {{{"index", 0},
                       {"message", {{"role", "assistant"}, {"content", " "}}},
                       {"finish_reason", "stop"}}})},
                {"usage", usage}};
            return make_json_response(http::status::ok, req.version(), keep_alive, body);
        }

        if (target == "/v1/models" || target == "/v1/images/generations" ||
            target == "/sdapi/v1/sd-models" || target == "/sdapi/v1/options" ||
            target == "/sdapi/v1/txt2img" || target == "/health" ||
            target.rfind("/images/", 0) == 0) {
            return make_json_response(
                http::status::method_not_allowed,
                req.version(),
                keep_alive,
                json{{"error", "method not allowed"}});
        }

        return make_json_response(
            http::status::not_found, req.version(), keep_alive, json{{"error", "not found"}});
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << req.method_string() << " " << target
                  << ": " << e.what() << "\n";
        return make_json_response(
            http::status::internal_server_error,
            req.version(),
            keep_alive,
            json{{"error", e.what()}});
    }
}

}  // namespace server
