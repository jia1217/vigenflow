#pragma once

#include "gen_params.hpp"

#include <boost/beast/http.hpp>
#include <string>

namespace server {

namespace http = boost::beast::http;

// Parse JSON request bodies for /v1/images/generations.
GenParams parse_request(const std::string& body);

// Parse multipart/form-data requests for /v1/images/edits, including uploads.
GenParams parse_multipart_request(const http::request<http::string_body>& req);

// Read response_format from JSON; defaults to b64_json.
std::string get_response_format(const std::string& body);

}  // namespace server
