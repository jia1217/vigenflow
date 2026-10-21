#include "cli.hpp"

#include "server_config.hpp"

#include <iostream>
#include <optional>
#include <string>

namespace server {

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [edit] [options]\n\n"
              << "Serves five text-to-image workers through POST /v1/images/generations and\n"
              << "Flux2 and Qwen image-edit workers through POST /v1/images/edits.\n"
              << "`edit` is short for --model " << config::kEditModelId << ".\n\n"
              << "Options:\n"
              << "  -m, --model <id>               Default model (default: z-image-turbo)\n"
              << "                                 IDs: z-image-turbo, flux1-schnell, flux2-klein-4B,\n"
              << "                                 krea2-turbo, qwen-image-2.1-turbo,\n"
              << "                                 " << config::kEditModelId << ", qwen-image-2.1-turbo-edit\n"
              << "  -e, --exe-path <path>          Override the default model worker\n"
              << "  -w, --weights-path <path>      Override the default model weights directory\n"
              << "  -n, --npu-files-path <path>    Override the default model NPU files directory\n"
              << "  -o, --output-dir <path>        Directory for generated images (default: ./images)\n"
              << "  -p, --port <port>              Server port (default: 11283)\n"
              << "  -k <true|false>                Save generated images (default: false)\n"
              << "      --seed <int>               Default model seed (Flux1: 0, others: 42)\n"
              << "      --steps <int>              Default model steps (default: 4; Krea2 and Qwen: 8)\n"
              << "                                 Qwen always uses 8; supplied steps are ignored.\n"
              << "  -h, --help                     Show this help message\n";
}

// Returns: 0 = show help and exit, 1 = ok, -1 = error.
int parse_cli(int argc, char* argv[]) {
    std::optional<std::string> model_id, worker_exe, weights_path, npu_files_path, steps;
    std::optional<int> seed;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "edit") {
            model_id = config::kEditModelId;
            continue;
        }
        if (!has_value) {
            std::cerr << "Missing value for " << arg << "\n\n";
            print_usage(argv[0]);
            return -1;
        }

        if (arg == "-k") {
            const std::string value = argv[++i];
            if (value != "true" && value != "false") {
                std::cerr << "-k must be true or false\n";
                return -1;
            }
            config::keep_images = value == "true";
        } else if (arg == "-m" || arg == "--model") {
            model_id = argv[++i];
        } else if (arg == "-e" || arg == "--exe-path") {
            worker_exe = argv[++i];
        } else if (arg == "-w" || arg == "--weights-path") {
            weights_path = argv[++i];
        } else if (arg == "-n" || arg == "--npu-files-path") {
            npu_files_path = argv[++i];
        } else if (arg == "-o" || arg == "--output-dir") {
            config::output_dir = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            config::port = static_cast<unsigned short>(std::stoi(argv[++i]));
        } else if (arg == "--seed") {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--steps") {
            steps = argv[++i];  // interpret only after the selected model's step policy is known
        } else {
            std::cerr << "Unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return -1;
        }
    }

    if (model_id) config::default_model_id = *model_id;
    auto* model = config::find_model(config::default_model_id);
    if (!model) {
        std::cerr << "Unknown model: " << config::default_model_id << "\n";
        return -1;
    }
    config::default_model_id = model->id;
    // Overrides apply to --model regardless of argument order.
    if (worker_exe) model->worker_exe = *worker_exe;
    if (weights_path) model->weights_path = *weights_path;
    if (npu_files_path) model->npu_files_path = *npu_files_path;
    if (seed) model->default_seed = *seed;
    if (model->fixed_steps) model->default_steps = model->fixed_steps;
    else if (steps) model->default_steps = std::stoi(*steps);
    if (!config::is_supported_steps(*model, model->default_steps)) {
        std::cerr << "--steps for " << model->id << " must be one of: "
                  << config::supported_steps_text(*model) << "\n";
        return -1;
    }
    return 1;
}

}  // namespace server
