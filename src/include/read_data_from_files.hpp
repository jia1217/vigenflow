#ifndef READ_DATA_FROM_FILES_HPP
#define READ_DATA_FROM_FILES_HPP

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
constexpr std::size_t GROUP_BYTES = 2560; // 2048(qx) + 512(meta)
namespace fs = std::filesystem;
static inline void read_exact(std::ifstream& fin, void* dst, std::size_t nbytes) {
  fin.read(reinterpret_cast<char*>(dst), nbytes);
  if (!fin) throw std::runtime_error("Unexpected EOF while reading params bin");
}

void load_params_bin_to_buffer(const std::string& bin_path,
    buffer<dequantize_params>& w_0, 
    std::size_t bo_groups)   // how many groups you allocated in BO
{
const std::uint64_t fsz = std::filesystem::file_size(bin_path);

if (fsz % GROUP_BYTES != 0) {
throw std::runtime_error("Bin size is not a multiple of 5120 bytes (corrupt layout?)");
}

const std::size_t file_groups = static_cast<std::size_t>(fsz / GROUP_BYTES);
const std::size_t n_groups = std::min(bo_groups, file_groups);

std::ifstream fin(bin_path, std::ios::binary);
if (!fin) throw std::runtime_error("Failed to open: " + bin_path);

// helpful debug print
// std::cerr << "bin bytes=" << fsz
// << " file_groups=" << file_groups
// << " bo_groups=" << bo_groups
// << " reading_groups=" << n_groups << "\n";

for (std::size_t g = 0; g < n_groups; ++g) {
auto& p = w_0[g]; // assumes operator[] gives host reference
read_exact(fin, p.qx,    128 * 16);
read_exact(fin, p.sx_min,128 * 2 * sizeof(std::uint16_t));
}

if (file_groups < bo_groups) {
throw std::runtime_error("BO expects more groups than file provides (file too small)");
}
}

inline void fill_scale_24x128_avx512(dtype_out* scale,
    const uint16_t* u16_scale)
{
constexpr size_t ROWS      = 24;
constexpr size_t COLS      = 128;
constexpr size_t VEC_ELEMS = 32;      // 512 bits / 16 bits

dtype_out* dst = scale;

// safety (optional)


// --- 1) First row: convert 128×u16 -> 128×bf16 using AVX-512 ---
dtype_out* row0 = dst;  // first row

    for (size_t j = 0; j < COLS; j += VEC_ELEMS) {
        // load 32 × u16
        __m512i v = _mm512_loadu_si512((const void*)(u16_scale + j));
        // store as 32 × bf16 (bit-identical, just a different type)
        _mm512_store_si512((__m512i*)(row0 + j), v);
    }

// --- 2) Replicate row0 into the remaining 23 rows ---
    for (size_t i = 1; i < ROWS; ++i) {
        dtype_out* row_i = dst + i * COLS;
        std::memcpy(row_i, row0, COLS * sizeof(dtype_out));
    }
}

static inline void read_u16_file_avx512(const char* path, std::vector<uint16_t>& out) {
    out.clear();

    // open
    int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        perror("open");
        throw std::runtime_error(std::string("open failed: ") + path);
    }

    // size
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        int e = errno; ::close(fd);
        throw std::runtime_error("fstat failed: " + std::to_string(e));
    }
    size_t bytes = static_cast<size_t>(st.st_size);
    if (bytes % sizeof(uint16_t) != 0) {
        ::close(fd);
        throw std::runtime_error("file size not multiple of 2 bytes");
    }
    size_t n = bytes / sizeof(uint16_t);
    if (n == 0) { ::close(fd); return; }

    // mmap
    void* map = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        // Fallback: buffered read
        out.resize(n);
        uint8_t* dstB = reinterpret_cast<uint8_t*>(out.data());
        size_t left = bytes;
        while (left > 0) {
            ssize_t got = ::read(fd, dstB, left);
            if (got <= 0) { ::close(fd); throw std::runtime_error("read failed"); }
            dstB += got; left -= size_t(got);
        }
        ::close(fd);
        return;
    }

    // advise (optional)
#ifdef POSIX_MADV_SEQUENTIAL
    (void)::posix_madvise(map, bytes, POSIX_MADV_SEQUENTIAL);
#endif

    out.resize(n);

    // AVX-512 copy: 64B per iter => 32 uint16_t
    const uint8_t* srcB = static_cast<const uint8_t*>(map);
    uint16_t* dst = out.data();

    size_t vecBytes = bytes & ~size_t(63);         // multiple of 64
    size_t i = 0;
    for (; i < vecBytes; i += 64) {
        __m512i v = _mm512_loadu_si512((const void*)(srcB + i));
        _mm512_storeu_si512((void*)(dst + (i >> 1)), v); // i/2 = elements offset
    }
    // tail
    if (i < bytes) {
        std::memcpy(dst + (i >> 1), srcB + i, bytes - i);
    }

    ::munmap(map, bytes);
    ::close(fd);
}
template <typename dtype_out>
buffer<dtype_out> read_bf16_from_u16bin(const fs::path& path, size_t count) {
    static_assert(sizeof(dtype_out) == 2, "dtype_out type must be 16-bit");
    static_assert(std::is_trivially_copyable_v<dtype_out>, "dtype_out type must be trivially copyable");

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open " + path.string());
    }

    // Read raw u16 bits
    buffer<std::uint16_t> bits(count);
    in.read(reinterpret_cast<char*>(bits.data()), bits.size() * sizeof(std::uint16_t));
    if (!in) {
        throw std::runtime_error("Failed to read enough data from " + path.string());
    }

    // Bit-cast u16 -> BF16
    buffer<dtype_out> out(count);
    for (size_t i = 0; i < count; ++i) {
        out.data()[i] = std::bit_cast<dtype_out>(bits.data()[i]);
    }
    return out;
}

////////////////////////////////////////////////////////////////////////////////
void read_data_from_files(const char* fname_W, std::vector<uint16_t> &u16_W){
    std::ifstream in_W(fname_W, std::ios::binary|std::ios::ate);
    if(!in_W){
        std::cerr << "Cannot open " << fname_W << "\n";
    }
    size_t bytes_W = in_W.tellg();
    in_W.seekg(0, std::ios::beg);
    size_t N_W = bytes_W / sizeof(uint16_t);
    // std::cout << "N_W = " << N_W << "\n";
    u16_W.resize(N_W);
    in_W.read(reinterpret_cast<char*>(u16_W.data()), bytes_W);
    in_W.close();
}
namespace fs = std::filesystem;
buffer<float> read_floats_from_bin(const fs::path &path, size_t count){
    buffer<float> buf(count);
    std::ifstream in(path, std::ios::binary);
    if(!in){
        throw std::runtime_error("Failed to open " + path.string());
    }
    // Read exactly count * sizeof(float) bytes
    in.read(reinterpret_cast<char*>(buf.data()), buf.size()*sizeof(float));
    if(!in){
        throw std::runtime_error("Failed to read enough data from " + path.string());
    }
    return buf;
}

buffer<int32_t> read_ints_from_bin(const fs::path &path, size_t count){
    buffer<int32_t> buf(count);
    std::ifstream in(path, std::ios::binary);
    if(!in){
        throw std::runtime_error("Failed to open " + path.string());
    }
    // Read exactly count * sizeof(float) bytes
    in.read(reinterpret_cast<char*>(buf.data()), buf.size()*sizeof(int32_t));
    if(!in){
        throw std::runtime_error("Failed to read enough data from " + path.string());
    }
    return buf;
}

////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <type_traits>

namespace io {

// Generic binary loader (works for uint16_t, float, etc.)
template<class T>
inline bool load_binary(const std::filesystem::path& file,
                        std::vector<T>& dst,
                        std::size_t expected_elems = 0)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");

    std::error_code ec;
    const auto bytes = std::filesystem::file_size(file, ec);
    if (ec) {
        std::cerr << "file_size failed for " << file << ": "
                  << ec.message() << "\n";
        return false;
    }
    if (bytes % sizeof(T) != 0) {
        std::cerr << "Bad element alignment in " << file
                  << " (bytes=" << bytes
                  << ", elem=" << sizeof(T) << ")\n";
        return false;
    }

    std::size_t elems = static_cast<std::size_t>(bytes / sizeof(T));
    if (expected_elems && elems != expected_elems) {
        std::cerr << "Size mismatch for " << file
                  << " (expected " << expected_elems
                  << ", got " << elems << " elements)\n";
        return false;
    }

    dst.resize(expected_elems ? expected_elems : elems);

    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << file << "\n";
        return false;
    }

    const auto want = static_cast<std::streamsize>(dst.size() * sizeof(T));
    in.read(reinterpret_cast<char*>(dst.data()), want);
    if (!in) {
        std::cerr << "Short read from " << file
                  << " (got " << in.gcount() << " of " << want << " bytes)\n";
        return false;
    }
    return true;
}

// Drop-in replacement for your old function (uint16_t specialization)
inline bool read_data_from_files(const char* fname,
                                 std::vector<uint16_t>& out,
                                 std::size_t expected_elems = 0)
{
    return load_binary<uint16_t>(std::filesystem::path(fname), out, expected_elems);
}

namespace fs = std::filesystem;

template<typename T, typename Buffer>
bool load_binary_to_buffer(const fs::path& path,
                           Buffer& buf,
                           std::size_t expected_elems = 0)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << path << "\n";
        return false;
    }

    std::size_t n = expected_elems ? expected_elems : buf.size();
    if (buf.size() < n) {
        std::cerr << "Buffer too small: buf.size()=" << buf.size()
                  << " expected=" << n << "\n";
        return false;
    }

    in.read(reinterpret_cast<char*>(buf.data()), n * sizeof(T));
    if (!in) {
        std::cerr << "Failed to read " << n << " elements from " << path << "\n";
        return false;
    }

    return true;
}

inline bool read_data_from_files_uint8(const char* fname,
    buffer<uint8_t>& out,
    std::size_t expected_elems = 0)
{
return load_binary_to_buffer<uint8_t>(fs::path(fname), out, expected_elems);
}


} 

#endif