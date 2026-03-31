#ifndef LOAD_WEIGHTS_HPP
#define LOAD_WEIGHTS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>

// Replaces both read_data_from_files and copy_u16_to_bf16_avx512
inline bool load_directly_to_tensor(const std::filesystem::path& file,
                                    dtype_out* dst,
                                    std::size_t expected_elems = 0)
{
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(file, ec);
    if (ec) {
        std::cerr << "file_size failed for " << file << ": " << ec.message() << "\n";
        return false;
    }

    if (bytes % sizeof(uint16_t) != 0) {
        std::cerr << "Bad element alignment in " << file << "\n";
        return false;
    }

    std::size_t elems = static_cast<std::size_t>(bytes / sizeof(uint16_t));
    if (expected_elems && elems != expected_elems) {
        std::cerr << "Size mismatch for " << file 
                  << " (expected " << expected_elems << ", got " << elems << ")\n";
        return false;
    }

    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << file << "\n";
        return false;
    }

    // Read directly into the destination buffer pointer
    const auto want = static_cast<std::streamsize>(elems * sizeof(uint16_t));
    in.read(reinterpret_cast<char*>(dst), want);
    
    if (!in) {
        std::cerr << "Short read from " << file << "\n";
        return false;
    }
    return true;
}


#endif

