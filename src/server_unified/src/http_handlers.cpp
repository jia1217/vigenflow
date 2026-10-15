#include "http_handlers.hpp"

#include "file_utils.hpp"
#include "input_image.hpp"
#include "request_parser.hpp"
#include "server_config.hpp"
#include "worker.hpp"

#include <atomic>
#include <cctype>
#include <ctime>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace server {

namespace {

// ponytail: all models share one NPU; use per-device locks if more NPUs are added.
std::mutex g_infer_mutex;

// Keep only the latest kMemoryImageLimit image URLs; -k true preserves older chats on disk.
std::mutex g_images_mutex;
std::deque<std::pair<std::string, std::vector<char>>> g_images;

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
    res.set(http::field::content_disposition, "inline");
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

nlohmann::ordered_json build_generation_info(
    const GenParams& params,
    const GenerationTimingInfo& timing) {
    nlohmann::ordered_json info;
    info["0_Model"] = params.model->id;
    info["1_Image_H"] = config::kImageHeight;
    info["2_Image_W"] = config::kImageWidth;
    info["3_Denoising_steps"] = params.steps;
    info["4_Seed"] = params.seed;

    if (timing.text_encoder_seconds) {
        info["5_Text_encoder"] = format_seconds(*timing.text_encoder_seconds);
    }
    if (timing.denoising_seconds_per_step) {
        const double total_seconds = timing.denoising_total_seconds.value_or(
            *timing.denoising_seconds_per_step * params.steps);
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

    info["9_Worker_wall_time"] = format_seconds(timing.worker_wall_seconds);
    return info;
}

// Base URL as the client reached us, so returned image URLs work off-machine too.
std::string request_base_url(const http::request<http::string_body>& req) {
    const auto host = req[http::field::host];
    if (!host.empty()) {
        return "http://" + std::string(host);
    }
    return "http://127.0.0.1:" + std::to_string(config::port);
}

// Both chat and image API URLs obey the same storage setting.
std::string publish_image(const http::request<http::string_body>& req, const std::string& path) {
    const auto filename = fs::path(path).filename().string();
    if (!config::keep_images) {
        auto data = read_binary_file(path);
        cleanup_generated_image(path, false);
        std::lock_guard<std::mutex> lock(g_images_mutex);
        g_images.emplace_back(filename, std::move(data));
        while (g_images.size() > config::kMemoryImageLimit) g_images.pop_front();
    }
    return request_base_url(req) + "/images/" + filename;
}

http::message_generator handle_get_models(unsigned version, bool keep_alive) {
    json data = json::array();
    for (const auto& model : config::models) {
        if (!model.available) continue;
        data.push_back({{"id", model.id}, {"object", "model"}, {"created", 0}, {"owned_by", "local"}});
    }
    return make_json_response(http::status::ok, version, keep_alive,
                              json{{"object", "list"}, {"data", data}});
}

http::message_generator handle_get_image_file(
    const std::string& target, unsigned version, bool keep_alive) {
    const std::string filename = target.substr(8);
    if (!is_safe_filename(filename)) {
        return make_json_response(
            http::status::bad_request, version, keep_alive, json{{"error", "invalid filename"}});
    }

    {
        std::lock_guard<std::mutex> lock(g_images_mutex);
        for (const auto& image : g_images) {
            if (image.first == filename) {
                auto res = make_image_response(version, keep_alive, std::vector<char>(image.second), "image/png");
                res.set(http::field::cache_control, "no-store");
                return res;
            }
        }
    }

    const std::string fullpath = (fs::path(config::output_dir) / filename).string();
    if (!fs::exists(fullpath)) {
        return make_json_response(
            http::status::not_found, version, keep_alive, json{{"error", "image not found"}});
    }

    auto data = read_binary_file(fullpath);
    const std::string ext = fs::path(fullpath).extension().string();
    const std::string mime = ext == ".bmp" ? "image/bmp" : "image/png";
    return make_image_response(version, keep_alive, std::move(data), mime);
}

// A PNG this server returned earlier, from the memory cache or the output directory.
std::optional<std::vector<char>> find_published_image(const std::string& filename) {
    if (!is_safe_filename(filename)) return std::nullopt;
    {
        std::lock_guard<std::mutex> lock(g_images_mutex);
        for (const auto& image : g_images) {
            if (image.first == filename) return image.second;
        }
    }
    const auto path = fs::path(config::output_dir) / filename;
    if (!fs::is_regular_file(path)) return std::nullopt;
    return read_binary_file(path.string());
}

// Edit inputs are decoded before waiting for the NPU, so bad images fail fast.
WorkerResult run_generation(GenerationRequest& request) {
    std::optional<WorkerInputImage> input;
    if (!request.image.empty()) {
        input.emplace(request.image);
        request.params.input_image = input->path();
    }
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    return run_worker(request.params);
}

http::message_generator make_generation_response(
    const http::request<http::string_body>& req, const GenerationRequest& request,
    const WorkerResult& result, unsigned version, bool keep_alive) {
    if (req[http::field::accept] == "image/png") {
        auto data = read_binary_file(result.image_path);
        cleanup_generated_image(result.image_path, config::keep_images);
        auto res = make_image_response(version, keep_alive, std::move(data), "image/png");
        std::string timing = "worker;dur=" + std::to_string(result.timing.worker_wall_seconds * 1000.0);
        if (result.timing.e2e_runtime_seconds) {
            timing += ", inference;dur=" + std::to_string(*result.timing.e2e_runtime_seconds * 1000.0);
        }
        res.set("Server-Timing", timing);
        res.set("X-Image-Model", request.params.model->id);
        return res;
    }

    const bool return_url = request.response_format == "url";
    json item;
    if (return_url) {
        item["url"] = publish_image(req, result.image_path);
    } else {
        item["b64_json"] = base64_encode(read_binary_file(result.image_path));
    }

    const auto info = build_generation_info(request.params, result.timing);
    const json body = {
        {"created", static_cast<long long>(std::time(nullptr))},
        {"model", request.params.model->id},
        {"data", json::array({item})},
        {"info", info.dump()},
        {"metadata", json::parse(info.dump())}};

    if (!return_url) cleanup_generated_image(result.image_path, config::keep_images);

    return make_json_response(http::status::ok, version, keep_alive, body);
}

http::message_generator handle_image_generation(
    const http::request<http::string_body>& req, unsigned version, bool keep_alive) {
    GenerationRequest request = parse_generation_request(req.body());
    const WorkerResult result = run_generation(request);
    return make_generation_response(req, request, result, version, keep_alive);
}

http::message_generator handle_image_edit(
    const http::request<http::string_body>& req, unsigned version, bool keep_alive) {
    GenerationRequest request =
        parse_edit_request(std::string(req[http::field::content_type]), req.body());
    const WorkerResult result = run_generation(request);
    return make_generation_response(req, request, result, version, keep_alive);
}

bool is_image_tool(const json& name) {
    return name == "generate_image" || name == "edit_image";
}

// Resolve an image reference from a chat message to encoded image bytes.
std::string load_chat_image(const std::string& url) {
    if (url.rfind("data:", 0) == 0) return decode_image_data(url);
    const auto at = url.rfind("/images/");
    if (at != std::string::npos) {
        const auto filename = url.substr(at + 8, url.find_first_of("?#", at) - (at + 8));
        if (auto image = find_published_image(filename)) return std::string(image->begin(), image->end());
        throw BadRequest("the image to edit has expired; attach it again, or start the server "
                         "with -k true to keep images");
    }
    throw BadRequest("cannot read image " + url.substr(0, 200) + "; attach the image to your message");
}

void add_markdown_images(const std::string& text, std::vector<std::string>& urls) {
    for (auto pos = text.find("!["); pos != std::string::npos; pos = text.find("![", pos + 2)) {
        const auto close = text.find("](", pos + 2);
        if (close == std::string::npos) return;
        const auto end = text.find_first_of(") \t\r\n", close + 2);
        if (end != std::string::npos && text[end] == ')') {
            urls.push_back(text.substr(close + 2, end - close - 2));
        }
    }
}

bool is_tag_name_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == ':';
}

// Index just past the '>' that closes a tag, ignoring '>' inside quoted attributes.
std::size_t find_tag_end(const std::string& text, std::size_t pos) {
    char quote = 0;
    for (; pos < text.size(); ++pos) {
        const char c = text[pos];
        if (quote) {
            if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '>') {
            return pos + 1;
        }
    }
    return std::string::npos;
}

// Remove complete <tag ...>...</tag> and <tag .../> elements, or with keep_inner only
// their tags. A '<' that does not start such an element stays, e.g. "I <3 cats".
std::string strip_markup(const std::string& text, bool keep_inner) {
    std::string out;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto lt = text.find('<', pos);
        if (lt == std::string::npos) {
            out.append(text, pos, std::string::npos);
            break;
        }
        out.append(text, pos, lt - pos);
        std::size_t name_end = lt + 1;
        while (name_end < text.size() && is_tag_name_char(text[name_end])) ++name_end;
        const std::string name = text.substr(lt + 1, name_end - lt - 1);
        const bool name_ok = !name.empty() && std::isalpha(static_cast<unsigned char>(name[0])) &&
                             name_end < text.size() &&
                             std::string(" \t\r\n/>").find(text[name_end]) != std::string::npos;
        const auto open_end = name_ok ? find_tag_end(text, name_end) : std::string::npos;
        if (open_end != std::string::npos && text[open_end - 2] == '/') {
            pos = open_end;  // <tag .../>
            continue;
        }
        const std::string close_tag = "</" + name + ">";
        const auto close = open_end == std::string::npos ? std::string::npos : text.find(close_tag, open_end);
        if (close == std::string::npos) {
            out.push_back('<');
            pos = lt + 1;
            continue;
        }
        if (keep_inner) out += strip_markup(text.substr(open_end, close - open_end), true);
        pos = close + close_tag.size();
    }
    return out;
}

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// Open WebUI puts markup such as <attached_files>...</attached_files> in the user's
// message; the text encoder should see only what the user typed. If everything was
// markup, keep the text inside the tags, e.g. <user_query>a cat</user_query>.
std::string clean_chat_prompt(const std::string& text) {
    std::string prompt = trim(strip_markup(text, false));
    if (prompt.empty()) prompt = trim(strip_markup(text, true));
    return prompt;
}

// Newest image in the chat: an attached image, or a Markdown image this server
// returned earlier, so follow-up messages keep editing the latest result.
std::optional<std::string> find_chat_image(const json& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        const json role = it->value("role", json());
        if (role != "user" && role != "assistant") continue;
        const json content = it->value("content", json());
        std::vector<std::string> urls;
        if (content.is_string()) {
            add_markdown_images(content.get<std::string>(), urls);
        } else if (content.is_array()) {
            for (const auto& part : content) {
                if (!part.is_object()) continue;
                const json type = part.value("type", json());
                if (type == "text" && part.contains("text") && part["text"].is_string()) {
                    add_markdown_images(part["text"].get<std::string>(), urls);
                } else if (type == "image_url" && part.contains("image_url")) {
                    // Open WebUI sends either {"url": ...} or a plain string.
                    const json& image_url = part["image_url"];
                    if (image_url.is_string()) {
                        urls.push_back(image_url.get<std::string>());
                    } else if (image_url.is_object() && image_url.contains("url") &&
                               image_url["url"].is_string()) {
                        urls.push_back(image_url["url"].get<std::string>());
                    }
                }
            }
        }
        if (!urls.empty()) return load_chat_image(urls.back());
    }
    return std::nullopt;
}


// The image tool in Native mode signals that the user enabled Image.
// Generate here so the chat's model selects the worker, independent of WebUI's
// global image-engine model. Background chat calls must not start another worker.
http::message_generator handle_chat_completion(
    const http::request<http::string_body>& req, unsigned version, bool keep_alive) {
    const json body = json::parse(req.body(), nullptr, false);
    if (!body.is_object() || !body.contains("messages") ||
        !body["messages"].is_array() || body["messages"].empty()) {
        throw BadRequest("messages must be a non-empty array");
    }
    for (const auto& message : body["messages"]) {
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string()) {
            throw BadRequest("each message must have a string role");
        }
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) {
        throw BadRequest("stream must be a boolean");
    }
    const auto& model = parse_model(body);
    const json tools = body.value("tools", json::array());
    if (!tools.is_null() && !tools.is_array()) {
        throw BadRequest("tools must be an array");
    }

    static std::atomic<unsigned long long> counter{0};
    const std::string id = "chatcmpl-" + std::to_string(std::time(nullptr)) + "-" +
                           std::to_string(++counter);
    json message = {{"role", "assistant"}, {"content", " "}};
    json metadata;
    const auto& last = body["messages"].back();
    const json choice = body.value("tool_choice", json("auto"));
    if (choice.is_object()) {
        if (choice.value("type", json()) != "function" || !choice.contains("function") ||
            !choice["function"].is_object() ||
            !is_image_tool(choice["function"].value("name", json()))) {
            throw BadRequest("only the generate_image and edit_image tools are supported");
        }
    } else if (!choice.is_null() && choice != "auto" && choice != "none" && choice != "required") {
        throw BadRequest("invalid tool_choice");
    }
    const bool allow_image = choice != "none";
    if (last["role"] == "user" && allow_image) {
        for (const auto& tool : tools) {
            if (!tool.is_object() || tool.value("type", json()) != "function" ||
                !tool.contains("function") || !tool["function"].is_object() ||
                !is_image_tool(tool["function"].value("name", json()))) {
                continue;
            }
            std::string prompt;
            const json content = last.value("content", json());
            if (content.is_string()) {
                prompt = content.get<std::string>();
            } else if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.is_object() && part.value("type", json()) == "text" &&
                        part.contains("text") && part["text"].is_string()) {
                        if (!prompt.empty()) prompt += "\n";
                        prompt += part["text"].get<std::string>();
                    }
                }
            }
            prompt = clean_chat_prompt(prompt);
            if (prompt.empty()) {
                throw BadRequest("image generation requires a text message");
            }
            json image_body = {{"model", model.id}, {"prompt", prompt}};
            for (const char* field : {"seed", "steps"}) {
                if (body.contains(field)) image_body[field] = body[field];
            }
            GenerationRequest request;
            if (model.edits_image) {
                auto image = find_chat_image(body["messages"]);
                if (!image) {
                    throw BadRequest(std::string(model.id) + " edits an image: attach one to your "
                                     "message, or generate one earlier in this chat");
                }
                request = parse_edit_request(image_body, std::move(*image));
            } else {
                request = parse_generation_request(image_body.dump());
            }
            const WorkerResult result = run_generation(request);
            message["content"] = std::string(model.edits_image ? "![Edited image](" : "![Generated image](") +
                                 publish_image(req, result.image_path) + ")";
            metadata = build_generation_info(request.params, result.timing);
            break;
        }
    }

    json reply = {{"id", id}, {"object", "chat.completion"},
                  {"created", static_cast<long long>(std::time(nullptr))},
                  {"model", model.id},
                  {"choices", json::array({{{"index", 0}, {"message", message},
                                             {"finish_reason", "stop"}}})}};
    if (!metadata.is_null()) reply["metadata"] = metadata;
    if (!body.value("stream", false)) {
        return make_json_response(http::status::ok, version, keep_alive, reply);
    }

    reply["object"] = "chat.completion.chunk";
    http::response<http::string_body> res{http::status::ok, version};
    res.set(http::field::content_type, "text/event-stream");
    res.set(http::field::cache_control, "no-cache");
    res.keep_alive(keep_alive);
    // Send the complete image Markdown in one small event, with no Base64 text fragments.
    reply["choices"] = json::array({{{"index", 0}, {"delta", message}, {"finish_reason", nullptr}}});
    res.body() = "data: " + reply.dump() + "\n\n";
    reply["choices"] = json::array({{{"index", 0}, {"delta", json::object()},
                                    {"finish_reason", "stop"}}});
    res.body() += "data: " + reply.dump() + "\n\ndata: [DONE]\n\n";
    res.prepare_payload();
    return res;
}

}  // namespace

http::message_generator handle_request(http::request<http::string_body>&& req) {
    const bool keep_alive = req.keep_alive();
    const std::string target = std::string(req.target());
    std::cout << "[REQ] " << req.method_string() << " " << target << "\n";

    try {
        const bool is_get = req.method() == http::verb::get;
        const bool is_post = req.method() == http::verb::post;

        if (target == "/health") {
            if (is_get) {
                return make_json_response(
                    http::status::ok, req.version(), keep_alive, json{{"status", "ok"}});
            }
        } else if (target == "/v1/models") {
            if (is_get) {
                return handle_get_models(req.version(), keep_alive);
            }
        } else if (target == "/v1/chat/completions") {
            if (is_post) {
                return handle_chat_completion(req, req.version(), keep_alive);
            }
        } else if (target == "/v1/images/generations") {
            if (is_post) {
                return handle_image_generation(req, req.version(), keep_alive);
            }
        } else if (target == "/v1/images/edits") {
            if (is_post) {
                return handle_image_edit(req, req.version(), keep_alive);
            }
        } else if (target.rfind("/images/", 0) == 0) {
            if (is_get) {
                return handle_get_image_file(target, req.version(), keep_alive);
            }
        } else {
            return make_json_response(
                http::status::not_found, req.version(), keep_alive, json{{"error", "not found"}});
        }

        return make_json_response(
            http::status::method_not_allowed,
            req.version(),
            keep_alive,
            json{{"error", "method not allowed"}});
    } catch (const BadRequest& e) {
        std::cerr << "[ERROR] " << req.method_string() << " " << target
                  << ": " << e.what() << "\n";
        return make_json_response(
            http::status::bad_request, req.version(), keep_alive, json{{"error", e.what()}});
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
