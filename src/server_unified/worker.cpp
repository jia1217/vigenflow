#include "worker.hpp"

#include "file_utils.hpp"
#include "server_config.hpp"

// Boost 1.86+ moved the classic process API under boost/process/v1/.
#if __has_include(<boost/process/v1/child.hpp>)
#include <boost/process/v1/args.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/io.hpp>
#include <boost/process/v1/pipe.hpp>
#include <boost/process/v1/start_dir.hpp>
namespace bp = boost::process::v1;
#else
#include <boost/process/args.hpp>
#include <boost/process/child.hpp>
#include <boost/process/io.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/start_dir.hpp>
namespace bp = boost::process;
#endif

#include <boost/gil/extension/io/bmp.hpp>
#include <boost/gil/extension/io/png.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace server {

namespace {

struct ProgressTiming {
    double elapsed_seconds = 0.0;
    double seconds_per_step = 0.0;
};

// Parse the last "<label>:  NN%|....| i/N [mm:ss<mm:ss, X.XXs/it]" line printed by run.exe.
std::optional<ProgressTiming> parse_progress_timing(
    const std::string& output,
    const std::string& label) {
    const std::regex re(
        label +
        R"(:\s+\d+%\|[^\r\n]*\|\s+\d+/\d+\s+\[(\d+):(\d+)<[^\]\r\n]*,\s*([0-9]+\.?[0-9]*)s/it\])");

    std::optional<ProgressTiming> result;
    for (std::sregex_iterator it(output.begin(), output.end(), re), end; it != end; ++it) {
        ProgressTiming timing;
        timing.elapsed_seconds =
            std::stoi((*it)[1].str()) * 60.0 + std::stoi((*it)[2].str());
        timing.seconds_per_step = std::stod((*it)[3].str());
        result = timing;
    }
    return result;
}

// Parse run.exe's "E2E runtime: <value><unit>" line.
std::optional<double> parse_e2e_runtime_seconds(const std::string& output) {
    static const std::regex re(R"(E2E\s+runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)?)");

    std::optional<double> result;
    for (std::sregex_iterator it(output.begin(), output.end(), re), end; it != end; ++it) {
        double value = std::stod((*it)[1].str());
        const std::string unit = (*it)[2].matched ? (*it)[2].str() : "ms";
        if (unit == "us") {
            value /= 1000000.0;
        } else if (unit == "ms") {
            value /= 1000.0;
        }
        result = value;
    }
    return result;
}

GenerationTimingInfo parse_generation_timing_info(const std::string& output) {
    GenerationTimingInfo timing;

    if (const auto text_timing = parse_progress_timing(output, "Text Encoder")) {
        timing.text_encoder_seconds = text_timing->seconds_per_step;
    }
    if (const auto denoising_timing = parse_progress_timing(output, "Denoising")) {
        timing.denoising_seconds_per_step = denoising_timing->seconds_per_step;
        timing.denoising_total_seconds = denoising_timing->elapsed_seconds;
    }
    if (const auto vae_timing = parse_progress_timing(output, "VAE Decoder")) {
        timing.vae_decoder_seconds = vae_timing->seconds_per_step;
    }
    timing.e2e_runtime_seconds = parse_e2e_runtime_seconds(output);

    return timing;
}

}  // namespace

WorkerResult run_worker(const GenParams& params) {
    const auto& model = *params.model;
    ensure_output_dir();

    static std::atomic<std::uint64_t> job_counter{0};
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

    // Most workers write BMP output, encoded as PNG below; a writes_png worker's file is the response.
    const std::string filename = "output_" + std::to_string(ms) + "_" + std::to_string(++job_counter) +
                                 (model.writes_png ? ".png" : ".bmp");
    const std::string fullpath = (fs::path(config::output_dir) / filename).string();

    std::vector<std::string> exec_args;
    if (!model.weights_path.empty()) {
        exec_args.insert(exec_args.end(), {"--weights_path", model.weights_path});
    }
    if (!model.npu_files_path.empty()) {
        exec_args.insert(exec_args.end(), {"--npu_files_path", model.npu_files_path});
    }
    if (!params.prompt.empty()) {
        // One argv element preserves quotes, newlines and leading option-like text.
        exec_args.push_back("--prompt=" + params.prompt);
    }
    if (model.edits_image) {
        if (params.input_image.empty()) throw std::logic_error("edit request without an input image");
        exec_args.insert(exec_args.end(), {"--input_image", params.input_image});
    }
    exec_args.insert(exec_args.end(), {
        "--seed", std::to_string(params.seed),
        "--steps", std::to_string(params.steps),
        "--output", fullpath});

    std::cout << "[INFO] Model: " << model.id << "\n";
    std::cout << "[INFO] Starting " << model.worker_exe << " (seed=" << params.seed
              << ", steps=" << params.steps << ")\n";
    std::cout << "[INFO] Output path: " << fullpath << "\n";

    const auto start_time = std::chrono::steady_clock::now();

    bp::ipstream child_output;
    std::string worker_output;

    // Start run.exe in its own folder so its relative default paths resolve.
    bp::child child(
        model.worker_exe,
        bp::args(exec_args),
        (bp::std_out & bp::std_err) > child_output,
        bp::start_dir = fs::path(model.worker_exe).parent_path().string());

    std::thread output_reader([&]() {
        char ch = '\0';
        while (child_output.get(ch)) {
            worker_output.push_back(ch);
            std::cout.put(ch);
            if (ch == '\n' || ch == '\r') {
                std::cout.flush();
            }
        }
    });

    child.wait();
    if (output_reader.joinable()) {
        output_reader.join();
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start_time)
                                .count();

    if (child.exit_code() != 0) {
        cleanup_generated_image(fullpath, false);
        std::cerr << "[ERROR] run.exe failed with exit code " << child.exit_code() << "\n";
        throw std::runtime_error(
            "run.exe failed with exit code " + std::to_string(child.exit_code()));
    }
    if (!fs::exists(fullpath)) {
        throw std::runtime_error("run.exe finished but output image was not created");
    }

    GenerationTimingInfo timing = parse_generation_timing_info(worker_output);
    timing.worker_wall_seconds = static_cast<double>(elapsed_ms) / 1000.0;
    if (timing.e2e_runtime_seconds) {
        std::cout << "[INFO] Model inference time: " << *timing.e2e_runtime_seconds << " s\n";
    }
    std::cout << "[INFO] Worker wall time: " << timing.worker_wall_seconds
              << " s (includes startup, loading, inference and shutdown)\n";

    if (model.writes_png) {
        // Served unchanged, so only check that the worker left a readable PNG.
        try {
            boost::gil::read_image_info(fullpath, boost::gil::png_tag{});
        } catch (...) {
            cleanup_generated_image(fullpath, false);
            throw;
        }
        std::cout << "[INFO] Image ready: " << fullpath << "\n";
        std::cout << "[INFO] Waiting for next request\n";
        return {fullpath, timing};
    }

    // Encode real PNG bytes for clients that display the image inline.
    const std::string png_path = fs::path(fullpath).replace_extension(".png").string();
    try {
        const auto bmp = boost::gil::read_image_info(fullpath, boost::gil::bmp_tag{});
        boost::gil::rgb8_image_t image;
        boost::gil::read_image(fullpath, image, boost::gil::bmp_tag{});
        // ponytail: Boost.GIL ignores top-down BMP orientation; remove this flip when its reader is fixed.
        if (bmp._info._top_down) {
            boost::gil::write_view(png_path, boost::gil::flipped_up_down_view(boost::gil::const_view(image)),
                                  boost::gil::png_tag{});
        } else {
            boost::gil::write_view(png_path, boost::gil::const_view(image), boost::gil::png_tag{});
        }
    } catch (...) {
        cleanup_generated_image(fullpath, false);
        cleanup_generated_image(png_path, false);
        throw;
    }
    cleanup_generated_image(fullpath, false);
    std::cout << "[INFO] Image ready: " << png_path << "\n";
    std::cout << "[INFO] Waiting for next request\n";

    return {png_path, timing};
}

void cleanup_generated_image(const std::string& image_path, bool keep_on_disk) {
    if (keep_on_disk) return;

    std::error_code ec;
    fs::remove(image_path, ec);
}

}  // namespace server
