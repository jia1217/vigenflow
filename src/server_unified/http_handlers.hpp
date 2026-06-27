#pragma once

#include <boost/beast/http.hpp>

namespace server {

namespace http = boost::beast::http;

// Route a single HTTP request and return a Beast response generator.
http::message_generator handle_request(http::request<http::string_body>&& req);

}  // namespace server
