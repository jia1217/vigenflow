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

std::optional<std::string> base64_decode(std::string_view text) {
    std::string out;
    out.reserve(text.size() / 4 * 3);
    unsigned int bits = 0;
    int bit_count = 0;
    bool padding = false;
    for (const char c : text) {
        int value = -1;
        if (c >= 'A' && c <= 'Z') value = c - 'A';
        else if (c >= 'a' && c <= 'z') value = c - 'a' + 26;
        else if (c >= '0' && c <= '9') value = c - '0' + 52;
        else if (c == '+') value = 62;
        else if (c == '/') value = 63;
        else if (c == '=') { padding = true; continue; }
        else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        if (value < 0 || padding) return std::nullopt;
        bits = (bits << 6) | static_cast<unsigned int>(value);
        bit_count += 6;
        if (bit_count >= 8) {
            bit_count -= 8;
            out.push_back(static_cast<char>((bits >> bit_count) & 0xFF));
        }
    }
    return out;
}

}  // namespace server
