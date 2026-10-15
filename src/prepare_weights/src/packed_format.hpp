#pragma once
// Pre-packed NPU weight folders, for uploading to Hugging Face: every file of all_model_weights/<id> is
// one U8 tensor named by its path relative to that folder, in safetensors shards of about --shard_gb each
// (model-00001-of-0000N.safetensors), plus model.safetensors.index.json. Unpacking restores the folder
// byte for byte, prepare_manifest.json included.
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace prep {

inline constexpr const char* kPackedFormat = "vgf-npu-weights-1";
inline constexpr const char* kPackedIndex = "model.safetensors.index.json";

struct PackResult {
    size_t files = 0;
    uint64_t bytes = 0;
    std::vector<std::string> shards;
};

// Packs every file under folder into out_dir (which is emptied of earlier shards first).
PackResult pack_folder(const std::filesystem::path& folder, const std::filesystem::path& out_dir, const std::string& id,
                       uint64_t shard_bytes, const std::function<void(const std::string&)>& log);

// Writes the files stored in one shard under dir. Returns (files, bytes). Rejects names that would land
// outside dir.
std::pair<size_t, uint64_t> unpack_shard(const std::filesystem::path& shard, const std::filesystem::path& dir);

}  // namespace prep
