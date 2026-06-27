#include "file_utils.hpp"

#include "server_config.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace server {

void ensure_output_dir() {
    std::error_code ec;
    fs::create_directories(config::output_dir, ec);
    if (ec) {
        throw std::runtime_error(
            "Failed to create output dir: " + config::output_dir + " : " + ec.message());
    }
}

bool is_safe_filename(const std::string& name) {
    if (name.empty()) return false;
    if (name.find("..") != std::string::npos) return false;
    if (name.find('/') != std::string::npos) return false;
    if (name.find('\\') != std::string::npos) return false;
    return true;
}

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<size_t>(size));
    if (size > 0 && !file.read(data.data(), size)) {
        throw std::runtime_error("failed to read file: " + path);
    }
    return data;
}

std::string base64_encode(const std::vector<char>& data) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    const auto* bytes = reinterpret_cast<const unsigned char*>(data.data());
    const size_t len = data.size();

    while (i + 2 < len) {
        unsigned int n = (static_cast<unsigned int>(bytes[i]) << 16) |
                         (static_cast<unsigned int>(bytes[i + 1]) << 8) |
                         static_cast<unsigned int>(bytes[i + 2]);
        out.push_back(table[(n >> 18) & 63]);
        out.push_back(table[(n >> 12) & 63]);
        out.push_back(table[(n >> 6) & 63]);
        out.push_back(table[n & 63]);
        i += 3;
    }

    if (i < len) {
        unsigned int n = static_cast<unsigned int>(bytes[i]) << 16;
        out.push_back(table[(n >> 18) & 63]);
        if (i + 1 < len) {
            n |= static_cast<unsigned int>(bytes[i + 1]) << 8;
            out.push_back(table[(n >> 12) & 63]);
            out.push_back(table[(n >> 6) & 63]);
            out.push_back('=');
        } else {
            out.push_back(table[(n >> 12) & 63]);
            out.push_back('=');
            out.push_back('=');
        }
    }

    return out;
}

}  // namespace server
