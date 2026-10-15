#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace server {

void ensure_output_dir();
bool is_safe_filename(const std::string& name);
std::vector<char> read_binary_file(const std::string& path);
std::string base64_encode(const std::vector<char>& data);
// Returns nullopt for characters outside the standard alphabet; whitespace is skipped.
std::optional<std::string> base64_decode(std::string_view text);

}  // namespace server
