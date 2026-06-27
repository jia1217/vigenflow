#pragma once

#include <string>
#include <vector>

namespace server {

void ensure_output_dir();
bool is_safe_filename(const std::string& name);
std::vector<char> read_binary_file(const std::string& path);
std::string base64_encode(const std::vector<char>& data);

}  // namespace server
