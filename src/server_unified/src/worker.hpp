#pragma once

#include "gen_params.hpp"

#include <optional>
#include <string>

namespace server {

// Model stage timings plus the separately measured worker process duration.
struct GenerationTimingInfo {
    std::optional<double> text_encoder_seconds;
    std::optional<double> denoising_seconds_per_step;
    std::optional<double> denoising_total_seconds;
    std::optional<double> vae_decoder_seconds;
    std::optional<double> e2e_runtime_seconds;
    double worker_wall_seconds = 0.0;
};

struct WorkerResult {
    std::string image_path;
    GenerationTimingInfo timing;
};

// Run the selected worker once and return the generated image path plus timings.
WorkerResult run_worker(const GenParams& params);

// Remove a generated image after the response when it should not be kept.
void cleanup_generated_image(const std::string& image_path, bool keep_on_disk);

}  // namespace server
