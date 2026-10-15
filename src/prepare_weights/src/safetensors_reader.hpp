#pragma once
// Minimal safetensors reader: one tensor index over all shards of a component, and reads of a whole
// tensor or a range of its rows, converted to BF16. Only the requested bytes are read.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace prep {
namespace fs = std::filesystem;

struct TensorInfo {
    fs::path file;
    std::string dtype;  // BF16, F16 or F32 (others are indexed but cannot be read)
    std::vector<size_t> shape;
    uint64_t begin = 0, end = 0;  // absolute byte range in file
    size_t numel() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }
    size_t rows() const { return shape.empty() ? 1 : shape[0]; }
    size_t row_elems() const { return shape.empty() ? 1 : numel() / shape[0]; }
};

inline uint16_t f32_bits_to_bf16(uint32_t bits) {
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x7FFFFFu)) return uint16_t((bits >> 16) | 0x40);  // quiet NaN
    return uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);  // round to nearest even, as torch does
}

inline uint32_t f16_bits_to_f32_bits(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    if (exp == 0x1F) return sign | 0x7F800000u | (man << 13);
    if (exp == 0) {
        if (man == 0) return sign;
        exp = 127 - 15 + 1;
        while (!(man & 0x400u)) man <<= 1, --exp;
        return sign | (exp << 23) | ((man & 0x3FFu) << 13);
    }
    return sign | ((exp + 127 - 15) << 23) | (man << 13);
}

class SafetensorsIndex {
public:
    void add_file(const fs::path& file) {
        std::ifstream f(file, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + file.string());
        uint64_t header_len = 0;
        f.read(reinterpret_cast<char*>(&header_len), 8);
        if (!f || header_len == 0 || header_len > (512ull << 20)) throw std::runtime_error("not a safetensors file: " + file.string());
        std::string header(header_len, '\0');
        f.read(header.data(), std::streamsize(header_len));
        if (!f) throw std::runtime_error("truncated safetensors header: " + file.string());
        const auto j = nlohmann::json::parse(header);
        const uint64_t data_start = 8 + header_len;
        for (const auto& [name, v] : j.items()) {
            if (name == "__metadata__") continue;
            TensorInfo t;
            t.file = file;
            t.dtype = v.at("dtype").get<std::string>();
            t.shape = v.at("shape").get<std::vector<size_t>>();
            t.begin = data_start + v.at("data_offsets")[0].get<uint64_t>();
            t.end = data_start + v.at("data_offsets")[1].get<uint64_t>();
            if (!tensors_.emplace(name, std::move(t)).second) throw std::runtime_error("tensor " + name + " is in more than one shard");
        }
    }

    const std::map<std::string, TensorInfo>& all() const { return tensors_; }
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorInfo& info(const std::string& name) const {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }

    // Rows [row0, row1) of the tensor viewed as [rows, row_elems], as BF16 bits.
    std::vector<uint16_t> read_bf16(const std::string& name, size_t row0 = 0,
                                    size_t row1 = std::numeric_limits<size_t>::max()) const {
        const TensorInfo& t = info(name);
        const size_t elem = t.dtype == "F32" ? 4 : (t.dtype == "BF16" || t.dtype == "F16") ? 2 : 0;
        if (!elem) throw std::runtime_error(name + ": unsupported dtype " + t.dtype);
        if (t.end - t.begin != uint64_t(t.numel()) * elem) throw std::runtime_error(name + ": size does not match its shape");
        row1 = std::min(row1, t.rows());
        if (row0 > row1) throw std::runtime_error(name + ": bad row range");
        const size_t n = (row1 - row0) * t.row_elems();
        std::vector<char> raw(n * elem);
        std::ifstream f(t.file, std::ios::binary);
        f.seekg(std::streamoff(t.begin + uint64_t(row0) * t.row_elems() * elem));
        f.read(raw.data(), std::streamsize(raw.size()));
        if (!f) throw std::runtime_error("short read of " + name + " from " + t.file.string());
        std::vector<uint16_t> out(n);
        if (t.dtype == "BF16") {
            std::memcpy(out.data(), raw.data(), raw.size());
        } else if (t.dtype == "F16") {
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, raw.data() + 2 * i, 2);
                out[i] = f32_bits_to_bf16(f16_bits_to_f32_bits(h));
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                uint32_t b;
                std::memcpy(&b, raw.data() + 4 * i, 4);
                out[i] = f32_bits_to_bf16(b);
            }
        }
        return out;
    }

private:
    std::map<std::string, TensorInfo> tensors_;
};

}  // namespace prep
