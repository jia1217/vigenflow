#include "input_image.hpp"

#include "request_parser.hpp"
#include "server_config.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#if __has_include(<stb/stb_image.h>)
#include <stb/stb_image.h>  // Ubuntu libstb-dev
#else
#include <stb_image.h>  // vcpkg stb
#endif

#include <png.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace server {

namespace {

constexpr long long kMaxInputPixels = 40'000'000;

struct Taps {
    int first = 0;
    std::vector<float> weights;
};

// Triangle filter widened by the downscale factor (bilinear with antialiasing).
std::vector<Taps> make_taps(int src_size, int dst_size) {
    const double scale = static_cast<double>(src_size) / dst_size;
    const double support = std::max(scale, 1.0);
    std::vector<Taps> taps(dst_size);
    for (int i = 0; i < dst_size; ++i) {
        const double center = (i + 0.5) * scale;
        const int lo = std::max(0, static_cast<int>(std::floor(center - support)));
        const int hi = std::min(src_size, static_cast<int>(std::ceil(center + support)));
        double total = 0.0;
        taps[i].first = lo;
        for (int s = lo; s < hi; ++s) {
            const double w = std::max(0.0, 1.0 - std::abs(s + 0.5 - center) / support);
            taps[i].weights.push_back(static_cast<float>(w));
            total += w;
        }
        for (auto& w : taps[i].weights) w = static_cast<float>(w / total);
    }
    return taps;
}

// Resize the centered square of an RGB image to size x size.
std::vector<std::uint8_t> center_crop_resize(const std::uint8_t* rgb, int width, int height, int size) {
    const int side = std::min(width, height);
    const int x0 = (width - side) / 2;
    const int y0 = (height - side) / 2;
    const auto taps = make_taps(side, size);

    std::vector<float> rows(static_cast<std::size_t>(side) * size * 3);
    for (int y = 0; y < side; ++y) {
        const std::uint8_t* src = rgb + (static_cast<std::size_t>(y0 + y) * width + x0) * 3;
        float* dst = rows.data() + static_cast<std::size_t>(y) * size * 3;
        for (int x = 0; x < size; ++x) {
            float sum[3] = {0.0f, 0.0f, 0.0f};
            for (std::size_t k = 0; k < taps[x].weights.size(); ++k) {
                const std::uint8_t* p = src + (taps[x].first + k) * 3;
                for (int c = 0; c < 3; ++c) sum[c] += taps[x].weights[k] * p[c];
            }
            for (int c = 0; c < 3; ++c) dst[x * 3 + c] = sum[c];
        }
    }

    std::vector<std::uint8_t> out(static_cast<std::size_t>(size) * size * 3);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size * 3; ++x) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < taps[y].weights.size(); ++k) {
                sum += taps[y].weights[k] * rows[(taps[y].first + k) * size * 3 + x];
            }
            out[static_cast<std::size_t>(y) * size * 3 + x] =
                static_cast<std::uint8_t>(std::clamp(std::lround(sum), 0L, 255L));
        }
    }
    return out;
}

void write_png(const std::string& path, const std::vector<std::uint8_t>& rgb, int size) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = size;
    image.height = size;
    image.format = PNG_FORMAT_RGB;
    // Positive stride keeps the top-down RGB layout produced by resizing.
    const bool written = png_image_write_to_file(&image, path.c_str(), 0, rgb.data(), size * 3, nullptr);
    const std::string error = image.message;
    png_image_free(&image);
    if (!written) {
        throw std::runtime_error("failed to write input PNG: " + path + ": " + error);
    }
}

}  // namespace

WorkerInputImage::WorkerInputImage(const std::string& encoded) {
    const auto* bytes = reinterpret_cast<const stbi_uc*>(encoded.data());
    const int length = static_cast<int>(std::min<std::size_t>(encoded.size(), INT32_MAX));
    int width = 0, height = 0, channels = 0;
    if (!stbi_info_from_memory(bytes, length, &width, &height, &channels)) {
        throw BadRequest("input image must be a PNG, JPEG, BMP or GIF file");
    }
    if (static_cast<long long>(width) * height > kMaxInputPixels) {
        throw BadRequest("input image must be at most 40 megapixels");
    }
    std::unique_ptr<stbi_uc, void (*)(void*)> rgb(
        stbi_load_from_memory(bytes, length, &width, &height, &channels, 3), stbi_image_free);
    if (!rgb) {
        throw BadRequest(std::string("cannot decode input image: ") + stbi_failure_reason());
    }

    const int size = config::kImageWidth;
    const auto resized = center_crop_resize(rgb.get(), width, height, size);

    static std::atomic<unsigned long long> counter{0};
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
    path_ = (fs::temp_directory_path() /
             ("vgf-edit-input-" + std::to_string(ms) + "_" + std::to_string(++counter) + ".png"))
                .string();
    try {
        write_png(path_, resized, size);
    } catch (...) {
        std::error_code ec;
        fs::remove(path_, ec);
        throw;
    }
    std::cout << "[INFO] Received input image: " << width << "x" << height;
    if (width != size || height != size) {
        std::cout << " (center-cropped and resized to " << size << "x" << size << ")";
    }
    std::cout << "\n";
}

WorkerInputImage::~WorkerInputImage() {
    std::error_code ec;
    fs::remove(path_, ec);
}

}  // namespace server
