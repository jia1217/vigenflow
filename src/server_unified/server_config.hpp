#pragma once

#include <string>

namespace server::config {

extern const std::string kInstallDir;

extern std::string model_id;
extern std::string served_model_id;
extern std::string weights_path;
extern std::string npu_base_dir;
extern std::string exe_base_dir;
extern std::string custom_exe;
extern std::string workdir;

extern std::string lora_dir;
extern std::string lora_catalog_dir;
extern std::string lora_model_list;
extern std::string lora_model_list_q41;
extern std::string lora_model_list_bf16;
extern std::string lora_model_list_flux;
extern std::string lora_model_list_flux_edit;
extern std::string lora_source;
extern std::string lora_file;
extern std::string lora_revision;
extern std::string lora_cache_dir;
extern std::string lora_export_dir;
extern bool lora_force;
extern bool lora_keep_cache;
extern int lora_rank;

extern std::string gguf_path;
extern std::string gguf_repo;
extern std::string gguf_file;
extern std::string gguf_revision;
extern std::string gguf_hf_token;
extern bool force_gguf_download;

extern bool auto_download_z_image_bf16;
extern bool force_z_image_bf16_download;
extern std::string z_image_bf16_repo;
extern std::string z_image_bf16_revision;
extern std::string z_image_bf16_cache_dir;
extern std::string z_image_bf16_hf_token;

extern bool auto_download_flux_klein_bf16;
extern bool force_flux_klein_bf16_download;
extern std::string flux_klein_bf16_repo;
extern std::string flux_klein_bf16_revision;
extern std::string flux_klein_bf16_cache_dir;
extern std::string flux_klein_bf16_hf_token;

extern unsigned short port;
extern std::string public_base_url;
extern std::string output_dir;

extern bool keep_images;

extern int default_seed;
extern int default_steps;
extern int default_height;
extern int default_width;
extern std::string default_prompt;

void initialize_default_paths(const char* argv0);

}  // namespace server::config
