#include "cli.hpp"

#include "server_config.hpp"

#include <iostream>
#include <string>

namespace server {

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [model_name] [options]\n\n"
              << "Available Models (examples):\n"
              << "  flux.2-klein-4B\n"
              << "  flux.2-klein-4B-lora\n"
              << "  flux.2-klein-4B-edit\n"
              << "  flux.2-klein-4B-edit-lora\n"
              << "  z-image-turbo-Q4_1-GGUF\n"
              << "  z-image-turbo-Q4_1-GGUF-lora\n"
              << "  z-image-turbo-BF16\n"
              << "  z-image-turbo-BF16-lora\n"
              << "Options:\n"
              << "  -w, --weights-path <path>      Path to model weights (default: auto-detected install root/model_weights)\n"
              << "  -n, --npu-files-path <path>    Base directory for NPU files\n"
              << "  -o, --output-dir <path>        Output image directory\n"
              << "  -m, --model-id <id>            Served/API model ID label\n"
              << "  -p, --port <port>              Server port\n"
              << "  -e, --exe-path <path>          Worker executable path (Overrides automatic path)\n"
              << "      --exe-base-dir <path>      Base directory for model executables\n"
              << "  -d, --workdir <path>           Working directory\n"
              << "      --lora-dir <path>          Existing LoRA-only bf16 bin directory\n"
              << "      --lora-catalog-dir <path>  Parent folder whose LoRA children appear in /v1/models\n"
              << "                                  (Default: same as --weights-path)\n"
            //   << "      --lora-model-list <file>   JSON/JSONC list of LoRA model IDs and Hugging Face sources\n"
              << "      --lora-model-list-q41 <file>   JSON/JSONC list for z-image-turbo-Q4_1-lora IDs\n"
              << "                                      (Default: ./z_image_q41_lora_models.jsonc, then .json)\n"
              << "      --lora-model-list-bf16 <file>  JSON/JSONC list for z-image-turbo-BF16-lora IDs\n"
              << "                                      (Default: ./z_image_bf16_lora_models.jsonc, then .json)\n"
              << "      --lora-model-list-flux <file>  JSON/JSONC list for flux.2-klein-4B-lora IDs\n"
              << "                                      (Default: ./flux_klein_lora_models.jsonc, then .json)\n"
              << "      --lora-model-list-flux-edit <file>  JSON/JSONC list for flux.2-klein-4B-edit-lora IDs\n"
              << "                                      (Default: ./flux_klein_edit_lora_models.jsonc, then .json)\n"
            //   << "      --lora-source <repo/url>   Hugging Face LoRA repo id or file URL\n"
            //   << "      --lora-rank <int>          LoRA rank (default: 0 = use model default)\n"
            //   << "      --lora-file <path>         Path to a single LoRA .safetensors file\n"
              << "      --gguf-path <path>         Local GGUF file to parse and pack for q41 workers\n"
              << "      --gguf-repo <repo>         Hugging Face GGUF repo (default: unsloth/Z-Image-Turbo-GGUF)\n"
              << "      --gguf-file <file>         GGUF filename (default: z-image-turbo-Q4_1.gguf)\n"
              << "      --gguf-revision <rev>      Hugging Face revision for GGUF download (default: main)\n"
              << "      --gguf-hf-token <token>    Hugging Face token; defaults to HF_TOKEN in worker\n"
              << "      --force-gguf-download      Re-download GGUF before packing q41 weights\n"
              << "      --z-image-bf16-repo <repo>       Hugging Face repo for Z-Image BF16/shared weights\n"
              << "      --z-image-bf16-revision <rev>    Hugging Face revision for Z-Image BF16/shared weights\n"
              << "      --z-image-bf16-cache-dir <dir>   Cache directory for BF16 downloads\n"
              << "      --z-image-bf16-hf-token <token>  Hugging Face token; defaults to HF_TOKEN\n"
              << "      --force-z-image-bf16-download    Re-download and re-export BF16 weights\n"
              << "      --no-z-image-bf16-auto-download  Fail instead of downloading missing BF16 weights\n"
              << "      --flux-klein-bf16-repo <repo>       Hugging Face repo for FLUX NPU or raw weights\n"
              << "      --flux-klein-bf16-revision <rev>    Hugging Face revision for FLUX BF16 weights\n"
              << "      --flux-klein-bf16-cache-dir <dir>   Cache directory for FLUX BF16 downloads\n"
              << "      --flux-klein-bf16-hf-token <token>  Hugging Face token; defaults to HF_TOKEN\n"
              << "      --force-flux-klein-bf16-download    Re-download FLUX BF16 weights\n"
              << "      --no-flux-klein-bf16-auto-download  Fail instead of downloading missing FLUX BF16 weights\n"
              << "  -k, --keep-images <true/false> Keep generated images on disk (default: false)\n"
              << "      --seed <int>               Default random seed (default: 42)\n"
              << "      --steps <int>              Default number of sample steps (default: 4)\n"
              << "  -H, --height <int>             Default image height, in pixel space (default: 1024)\n"
              << "  -W, --width <int>              Default image width, in pixel space (default: 1024)\n"
              << "      --prompt <string>          Default text prompt\n"
              << "  -h, --help                     Show this help message\n";
}

// Returns: 0 = show help and exit, 1 = ok, -1 = error.
int parse_cli(int argc, char* argv[], bool& output_dir_specified) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (!arg.empty() && arg[0] != '-') {
            config::model_id = arg;
            continue;
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "-w" || arg == "--weights-path") {
            if (i + 1 >= argc) return -1;
            config::weights_path = argv[++i];
        } else if (arg == "-n" || arg == "--npu-files-path") {
            if (i + 1 >= argc) return -1;
            config::npu_base_dir = argv[++i];
        } else if (arg == "-o" || arg == "--output-dir") {
            if (i + 1 >= argc) return -1;
            config::output_dir = argv[++i];
            output_dir_specified = true;
        } else if (arg == "-m" || arg == "--model-id") {
            if (i + 1 >= argc) return -1;
            config::served_model_id = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 >= argc) return -1;
            config::port = static_cast<unsigned short>(std::stoi(argv[++i]));
            config::public_base_url = "http://127.0.0.1:" + std::to_string(config::port);
        } else if (arg == "-e" || arg == "--exe-path") {
            if (i + 1 >= argc) return -1;
            config::custom_exe = argv[++i];
        } else if (arg == "--exe-base-dir" || arg == "--exe_base_dir") {
            if (i + 1 >= argc) return -1;
            config::exe_base_dir = argv[++i];
        } else if (arg == "-d" || arg == "--workdir") {
            if (i + 1 >= argc) return -1;
            config::workdir = argv[++i];
        } else if (arg == "--lora-dir" || arg == "--lora_dir") {
            if (i + 1 >= argc) return -1;
            config::lora_dir = argv[++i];
        } else if (arg == "--lora-catalog-dir" || arg == "--lora_catalog_dir") {
            if (i + 1 >= argc) return -1;
            config::lora_catalog_dir = argv[++i];
        } else if (arg == "--lora-model-list" || arg == "--lora_model_list") {
            if (i + 1 >= argc) return -1;
            config::lora_model_list = argv[++i];
        } else if (arg == "--lora-model-list-q41" || arg == "--lora_model_list_q41") {
            if (i + 1 >= argc) return -1;
            config::lora_model_list_q41 = argv[++i];
        } else if (arg == "--lora-model-list-bf16" || arg == "--lora_model_list_bf16") {
            if (i + 1 >= argc) return -1;
            config::lora_model_list_bf16 = argv[++i];
        } else if (arg == "--lora-model-list-flux" || arg == "--lora_model_list_flux") {
            if (i + 1 >= argc) return -1;
            config::lora_model_list_flux = argv[++i];
        } else if (arg == "--lora-model-list-flux-edit" || arg == "--lora_model_list_flux_edit") {
            if (i + 1 >= argc) return -1;
            config::lora_model_list_flux_edit = argv[++i];
        } else if (arg == "--lora-source" || arg == "--lora_source") {
            if (i + 1 >= argc) return -1;
            config::lora_source = argv[++i];
        } else if (arg == "--lora-rank" || arg == "--lora_rank") {
            if (i + 1 >= argc) return -1;
            config::lora_rank = std::stoi(argv[++i]);
        } else if (arg == "--lora-file" || arg == "--lora_file") {
            if (i + 1 >= argc) return -1;
            config::lora_file = argv[++i];
        } else if (arg == "--gguf-path" || arg == "--gguf_path") {
            if (i + 1 >= argc) return -1;
            config::gguf_path = argv[++i];
        } else if (arg == "--gguf-repo" || arg == "--gguf_repo") {
            if (i + 1 >= argc) return -1;
            config::gguf_repo = argv[++i];
        } else if (arg == "--gguf-file" || arg == "--gguf_file") {
            if (i + 1 >= argc) return -1;
            config::gguf_file = argv[++i];
        } else if (arg == "--gguf-revision" || arg == "--gguf_revision") {
            if (i + 1 >= argc) return -1;
            config::gguf_revision = argv[++i];
        } else if (arg == "--gguf-hf-token" || arg == "--gguf_hf_token") {
            if (i + 1 >= argc) return -1;
            config::gguf_hf_token = argv[++i];
        } else if (arg == "--force-gguf-download" || arg == "--force_gguf_download") {
            config::force_gguf_download = true;
        } else if (arg == "--z-image-bf16-repo" || arg == "--z_image_bf16_repo") {
            if (i + 1 >= argc) return -1;
            config::z_image_bf16_repo = argv[++i];
        } else if (arg == "--z-image-bf16-revision" || arg == "--z_image_bf16_revision") {
            if (i + 1 >= argc) return -1;
            config::z_image_bf16_revision = argv[++i];
        } else if (arg == "--z-image-bf16-cache-dir" || arg == "--z_image_bf16_cache_dir") {
            if (i + 1 >= argc) return -1;
            config::z_image_bf16_cache_dir = argv[++i];
        } else if (arg == "--z-image-bf16-hf-token" || arg == "--z_image_bf16_hf_token") {
            if (i + 1 >= argc) return -1;
            config::z_image_bf16_hf_token = argv[++i];
        } else if (arg == "--force-z-image-bf16-download" || arg == "--force_z_image_bf16_download") {
            config::force_z_image_bf16_download = true;
        } else if (arg == "--no-z-image-bf16-auto-download" || arg == "--no_z_image_bf16_auto_download") {
            config::auto_download_z_image_bf16 = false;
        } else if (arg == "--flux-klein-bf16-repo" || arg == "--flux_klein_bf16_repo") {
            if (i + 1 >= argc) return -1;
            config::flux_klein_bf16_repo = argv[++i];
        } else if (arg == "--flux-klein-bf16-revision" || arg == "--flux_klein_bf16_revision") {
            if (i + 1 >= argc) return -1;
            config::flux_klein_bf16_revision = argv[++i];
        } else if (arg == "--flux-klein-bf16-cache-dir" || arg == "--flux_klein_bf16_cache_dir") {
            if (i + 1 >= argc) return -1;
            config::flux_klein_bf16_cache_dir = argv[++i];
        } else if (arg == "--flux-klein-bf16-hf-token" || arg == "--flux_klein_bf16_hf_token") {
            if (i + 1 >= argc) return -1;
            config::flux_klein_bf16_hf_token = argv[++i];
        } else if (arg == "--force-flux-klein-bf16-download" || arg == "--force_flux_klein_bf16_download") {
            config::force_flux_klein_bf16_download = true;
        } else if (arg == "--no-flux-klein-bf16-auto-download" || arg == "--no_flux_klein_bf16_auto_download") {
            config::auto_download_flux_klein_bf16 = false;
        } else if (arg == "-k" || arg == "--keep-images") {
            if (i + 1 >= argc) return -1;
            const std::string val = argv[++i];
            config::keep_images = (val == "true" || val == "1");
        } else if (arg == "--seed") {
            if (i + 1 >= argc) return -1;
            config::default_seed = std::stoi(argv[++i]);
        } else if (arg == "--steps") {
            if (i + 1 >= argc) return -1;
            config::default_steps = std::stoi(argv[++i]);
        } else if (arg == "-H" || arg == "--height") {
            if (i + 1 >= argc) return -1;
            config::default_height = std::stoi(argv[++i]);
        } else if (arg == "-W" || arg == "--width") {
            if (i + 1 >= argc) return -1;
            config::default_width = std::stoi(argv[++i]);
        } else if (arg == "--prompt") {
            if (i + 1 >= argc) return -1;
            config::default_prompt = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return -1;
        }
    }

    return 1;
}

}  // namespace server
