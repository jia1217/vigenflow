#pragma once

#include "gen_params.hpp"

#include <optional>
#include <string>
#include <vector>

namespace server {

struct LoraModelEntry {
    std::string model_id;
    std::string lora_dir;
    std::string lora_source;
    std::string lora_file;
    int lora_rank = 0;
    float lora_scale = 0.0f;
};

struct GenerationTimingInfo {
    std::optional<double> vae_encoder_seconds;
    std::optional<double> text_encoder_seconds;
    std::optional<double> denoising_seconds_per_step;
    std::optional<double> denoising_total_seconds;
    std::optional<int> denoising_steps;
    std::optional<double> vae_decoder_seconds;
    std::optional<double> e2e_runtime_seconds;
};

struct WorkerResult {
    std::string image_path;
    GenerationTimingInfo timing;
};

// List configured LoRA catalog entries as OpenAI-compatible model IDs.
std::vector<LoraModelEntry> list_lora_model_entries();

// Resolve a request model ID or UI spelling to its canonical LoRA catalog entry.
std::optional<LoraModelEntry> resolve_lora_model_entry(const std::string& model_id);

// Accept UI aliases such as local-model-1 and normalize edit model suffixes.
std::string normalize_model_target(std::string raw_target);

// Download/export model weights needed by the selected target when they are missing.
void ensure_model_weights_ready(const std::string& model_id);

// Convert a model id into the NPU file directory and worker executable path.
WorkerPaths resolve_worker_paths(const std::string& raw_target);

// Run the external model executable and return the generated image path plus timings.
WorkerResult run_worker(const GenParams& params);

// Remove generated or uploaded images after a request when configured to do so.
void cleanup_generated_image(const std::string& image_path, bool keep_on_disk);
void cleanup_uploaded_images(const std::vector<std::string>& paths);

}  // namespace server
