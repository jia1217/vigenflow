#pragma once

#include "gen_params.hpp"

#include <nlohmann/json_fwd.hpp>
#include <stdexcept>
#include <string>

namespace server {

struct GenerationRequest {
    GenParams params;
    std::string response_format = "b64_json";  // or "url"
    std::string image;  // Edit requests: encoded input image bytes.
};

// Thrown for request bodies the server should answer with 400 Bad Request.
class BadRequest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Select a configured model; folder names are accepted as aliases.
const config::ModelConfig& parse_model(const nlohmann::json& body);

// Decode a "data:image/...;base64,..." URL or a bare base64 string.
std::string decode_image_data(const std::string& text);

// POST /v1/images/generations: parse and validate inputs for a text-to-image worker.
GenerationRequest parse_generation_request(const std::string& body);

// POST /v1/images/edits: an OpenAI multipart form ("image" or "image[]" file) or a
// JSON body whose "image" is base64 or a data URL.
GenerationRequest parse_edit_request(const std::string& content_type, const std::string& body);

// Chat edits: request fields plus the already decoded input image.
GenerationRequest parse_edit_request(const nlohmann::json& body, std::string image);

}  // namespace server
