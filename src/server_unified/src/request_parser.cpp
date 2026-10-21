#include "request_parser.hpp"

#include "file_utils.hpp"
#include "server_config.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <vector>

using json = nlohmann::json;

namespace server {

namespace {

bool is_blank(const std::string& body) {
    return body.find_first_not_of(" \t\r\n") == std::string::npos;
}

std::string to_lower(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

int get_int_field(const json& j, const char* key, int fallback) {
    if (!j.contains(key)) {
        return fallback;
    }
    if (!j[key].is_number_integer()) {
        throw BadRequest(std::string(key) + " must be an integer");
    }
    const double value = j[key].get<double>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw BadRequest(std::string(key) + " is outside the supported integer range");
    }
    return j[key].get<int>();
}

json parse_json_object(const std::string& body) {
    // An empty body means "use the server defaults".
    json j = is_blank(body) ? json::object() : json::parse(body, nullptr, false);
    if (!j.is_object()) throw BadRequest("request body must be a JSON object");
    return j;
}

const config::ModelConfig& require_available(const config::ModelConfig& model) {
    if (!model.available) {
        throw BadRequest(std::string(model.id) + " is not served here (worker not built); see GET /v1/models");
    }
    return model;
}

// An explicitly selected edit default applies to /edits too; otherwise keep the Flux2 fallback.
const config::ModelConfig& request_model(const json& j, bool edit) {
    const auto& edit_default = config::default_model().edits_image ? config::default_model()
                                                                 : *config::find_model(config::kEditModelId);
    return edit && !j.contains("model") ? require_available(edit_default) : parse_model(j);
}

GenerationRequest parse_fields(const json& j, bool edit) {
    GenerationRequest request;
    const auto& model = request_model(j, edit);
    if (model.edits_image != edit) {
        throw BadRequest(edit ? std::string(model.id) + " does not edit images; select an image-edit model"
                              : std::string(model.id) +
                                    " edits an input image; send it to POST /v1/images/edits");
    }
    request.params.model = &model;
    request.params.seed = model.default_seed;
    request.params.steps = model.default_steps;

    if (j.contains("prompt")) {
        if (!j["prompt"].is_string()) {
            throw BadRequest("prompt must be a string");
        }
        request.params.prompt = j["prompt"].get<std::string>();
        if (is_blank(request.params.prompt) || request.params.prompt.find('\0') != std::string::npos) {
            throw BadRequest("prompt must be non-empty and contain no NUL characters");
        }
        if (request.params.prompt.size() > 16 * 1024) {
            throw BadRequest("prompt must be at most 16384 UTF-8 bytes");
        }
    } else if (edit) {
        throw BadRequest("prompt is required for image edits");
    }

    request.params.seed = get_int_field(j, "seed", request.params.seed);
    // Qwen has a fixed eight-step schedule; even a malformed client steps value is ignored.
    request.params.steps = model.fixed_steps ? model.fixed_steps : get_int_field(j, "steps", request.params.steps);
    if (j.contains("response_format")) {
        if (!j["response_format"].is_string()) throw BadRequest("response_format must be a string");
        request.response_format = j["response_format"].get<std::string>();
    }
    if (get_int_field(j, "n", 1) != 1) throw BadRequest("n must be 1");
    if (j.contains("size") && j["size"] != "1024x1024" && !(edit && j["size"] == "auto")) {
        throw BadRequest("size must be 1024x1024");
    }

    std::string ignored;
    for (const auto& item : j.items()) {
        if (item.key() != "seed" && item.key() != "steps" &&
            item.key() != "response_format" && item.key() != "prompt" &&
            item.key() != "model" && item.key() != "n" && item.key() != "size") {
            ignored += (ignored.empty() ? "" : ", ") + item.key();
        }
    }
    if (!ignored.empty()) {
        std::cout << "[INFO] Ignoring request fields not supported by run.exe: "
                  << ignored << "\n";
    }

    if (!config::is_supported_steps(model, request.params.steps)) {
        throw BadRequest(std::string("steps for ") + model.id + " must be one of: " +
                         config::supported_steps_text(model));
    }
    if (request.response_format != "b64_json" && request.response_format != "url") {
        throw BadRequest("response_format must be b64_json or url");
    }
    return request;
}

struct FormPart {
    std::string name;
    std::string data;
};

// Value of `key` in a header such as: form-data; name="image"; filename="a.png"
std::optional<std::string> header_param(std::string_view header, std::string_view key) {
    std::size_t pos = header.find(';');
    while (pos != std::string_view::npos && pos < header.size()) {
        ++pos;
        while (pos < header.size() && header[pos] == ' ') ++pos;
        const std::size_t eq = header.find('=', pos);
        if (eq == std::string_view::npos) return std::nullopt;
        const std::string name = to_lower(header.substr(pos, eq - pos));
        std::string value;
        pos = eq + 1;
        if (pos < header.size() && header[pos] == '"') {
            for (++pos; pos < header.size() && header[pos] != '"'; ++pos) {
                if (header[pos] == '\\' && pos + 1 < header.size()) ++pos;
                value.push_back(header[pos]);
            }
            pos = header.find(';', pos);
        } else {
            const std::size_t end = header.find(';', pos);
            value = std::string(header.substr(pos, end == std::string_view::npos ? end : end - pos));
            pos = end;
        }
        if (name == key) return value;
    }
    return std::nullopt;
}

std::vector<FormPart> parse_multipart(const std::string& content_type, const std::string& body) {
    auto boundary = header_param(content_type, "boundary");
    if (!boundary || boundary->empty()) throw BadRequest("multipart body needs a boundary");
    const std::string delimiter = "--" + *boundary;
    const auto malformed = [] { return BadRequest("malformed multipart/form-data body"); };

    std::vector<FormPart> parts;
    std::size_t pos = body.find(delimiter);
    if (pos == std::string::npos) throw malformed();
    for (;;) {
        pos += delimiter.size();
        if (body.compare(pos, 2, "--") == 0) return parts;
        if (body.compare(pos, 2, "\r\n") != 0) throw malformed();
        pos += 2;
        const std::size_t header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) throw malformed();
        const std::size_t next = body.find("\r\n" + delimiter, header_end + 4);
        if (next == std::string::npos) throw malformed();

        FormPart part;
        const std::string_view headers(body.data() + pos, header_end - pos);
        for (std::size_t line_start = 0; line_start < headers.size();) {
            std::size_t line_end = headers.find("\r\n", line_start);
            if (line_end == std::string_view::npos) line_end = headers.size();
            const auto line = headers.substr(line_start, line_end - line_start);
            if (to_lower(line.substr(0, 20)) == "content-disposition:") {
                part.name = header_param(line, "name").value_or("");
            }
            line_start = line_end + 2;
        }
        part.data = body.substr(header_end + 4, next - header_end - 4);
        parts.push_back(std::move(part));
        pos = next + 2;
    }
}

// Multipart fields arrive as text; convert the integer ones for parse_fields.
json form_value(const std::string& key, const std::string& value) {
    if (key != "n" && key != "seed" && key != "steps") return value;
    try {
        std::size_t used = 0;
        const long long number = std::stoll(value, &used);
        if (used == value.size()) return number;
    } catch (const std::exception&) {
    }
    throw BadRequest(key + " must be an integer");
}

}  // namespace

const config::ModelConfig& parse_model(const json& body) {
    if (!body.contains("model")) return config::default_model();
    if (!body["model"].is_string()) throw BadRequest("model must be a string");
    const auto id = body["model"].get<std::string>();
    const auto* model = config::find_model(id);
    if (!model) throw BadRequest("unknown model: " + id + "; see GET /v1/models");
    return require_available(*model);
}

std::string decode_image_data(const std::string& text) {
    std::string_view data(text);
    if (data.rfind("data:", 0) == 0) {
        const auto comma = data.find(',');
        if (comma == std::string_view::npos ||
            to_lower(data.substr(0, comma)).find(";base64") == std::string::npos) {
            throw BadRequest("image data URLs must be base64 encoded");
        }
        data.remove_prefix(comma + 1);
    }
    auto image = base64_decode(data);
    if (!image || image->empty()) {
        throw BadRequest("image must be base64 or a data:image/...;base64 URL");
    }
    return std::move(*image);
}

GenerationRequest parse_generation_request(const std::string& body) {
    return parse_fields(parse_json_object(body), false);
}

GenerationRequest parse_edit_request(const std::string& content_type, const std::string& body) {
    json fields = json::object();
    std::string image;
    int image_count = 0;
    if (to_lower(content_type).rfind("multipart/form-data", 0) == 0) {
        for (auto& part : parse_multipart(content_type, body)) {
            if (part.name == "image" || part.name == "image[]") {
                // Open WebUI lists the newest image first; the NPU pipeline takes one.
                if (image_count++ == 0) image = std::move(part.data);
            } else if (!part.name.empty()) {
                // Resolve the model after reading the whole form, including fields after steps.
                fields[part.name] = part.name == "steps" ? json(part.data) : form_value(part.name, part.data);
            }
        }
        if (fields.contains("steps") && !request_model(fields, true).fixed_steps) {
            fields["steps"] = form_value("steps", fields["steps"].get<std::string>());
        }
    } else {
        fields = parse_json_object(body);
        if (fields.contains("image")) {
            json value = fields["image"];
            if (value.is_array() && !value.empty()) {
                image_count = static_cast<int>(value.size());
                value = value[0];
            } else {
                image_count = 1;
            }
            if (!value.is_string()) throw BadRequest("image must be a base64 string or data URL");
            image = decode_image_data(value.get<std::string>());
            fields.erase("image");
        }
    }
    if (image_count == 0) throw BadRequest("image edits need an input image");
    if (image_count > 1) {
        std::cout << "[INFO] Received " << image_count << " input images; editing the first\n";
    }
    return parse_edit_request(fields, std::move(image));
}

GenerationRequest parse_edit_request(const json& body, std::string image) {
    if (image.empty()) throw BadRequest("input image is empty");
    GenerationRequest request = parse_fields(body, true);
    request.image = std::move(image);
    return request;
}

}  // namespace server
