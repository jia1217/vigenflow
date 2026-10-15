#include "packed_format.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cli_common.hpp"
#include "safetensors_reader.hpp"

namespace prep {

namespace {

constexpr size_t kCopyChunk = 64u << 20;

void copy_bytes(std::istream& in, std::ostream& out, uint64_t n, const std::string& what) {
    std::vector<char> buf(std::min<uint64_t>(n, kCopyChunk));
    while (n) {
        const auto k = std::min<uint64_t>(n, buf.size());
        in.read(buf.data(), std::streamsize(k));
        if (!in) throw std::runtime_error("short read while copying " + what);
        out.write(buf.data(), std::streamsize(k));
        n -= k;
    }
    if (!out) throw std::runtime_error("write failed while copying " + what);
}

// A stored name must stay inside the model folder: relative, no "..", no drive letters or backslashes.
bool safe_name(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.find_first_of("\\:") != std::string::npos) return false;
    for (const auto& part : fs::path(name))
        if (part == ".." || part == ".") return false;
    return true;
}

}  // namespace

PackResult pack_folder(const fs::path& folder, const fs::path& out_dir, const std::string& id, uint64_t shard_bytes,
                       const std::function<void(const std::string&)>& log) {
    std::vector<std::pair<std::string, uint64_t>> files;  // relative path, size; sorted, so packing is reproducible
    for (const auto& e : fs::recursive_directory_iterator(folder))
        if (e.is_regular_file()) files.emplace_back(fs::relative(e.path(), folder).generic_string(), e.file_size());
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error(folder.string() + " has no files");

    // Files in order, a new shard when the next file would push the current one past shard_bytes.
    std::vector<std::vector<size_t>> shards(1);
    uint64_t in_shard = 0, total = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        if (in_shard && in_shard + files[i].second > shard_bytes) shards.emplace_back(), in_shard = 0;
        shards.back().push_back(i);
        in_shard += files[i].second;
        total += files[i].second;
    }

    fs::create_directories(out_dir);
    for (const auto& e : fs::directory_iterator(out_dir)) {  // earlier shards of this model
        const auto n = e.path().filename().string();
        if ((n.starts_with("model-") && n.ends_with(".safetensors")) || n == kPackedIndex) fs::remove(e.path());
    }

    // Provenance for the index: the folder's prepare_manifest.json, when prepare.exe made it.
    nlohmann::ordered_json meta = {{"format", kPackedFormat}, {"id", id}, {"total_size", total}, {"file_count", files.size()}};
    if (const auto man = read_manifest(folder))
        for (const char* k : {"recipe", "recipe_version", "source", "revision", "commit"})
            if (man->contains(k)) meta[k] = (*man)[k];

    PackResult result{files.size(), total, {}};
    nlohmann::ordered_json weight_map = nlohmann::ordered_json::object();
    char name[64];
    for (size_t s = 0; s < shards.size(); ++s) {
        std::snprintf(name, sizeof(name), "model-%05zu-of-%05zu.safetensors", s + 1, shards.size());
        nlohmann::ordered_json header = {{"__metadata__", {{"format", kPackedFormat}, {"id", id}}}};
        uint64_t offset = 0;
        for (size_t i : shards[s]) {
            header[files[i].first] = {{"dtype", "U8"}, {"shape", {files[i].second}}, {"data_offsets", {offset, offset + files[i].second}}};
            offset += files[i].second;
            weight_map[files[i].first] = name;
        }
        std::string h = header.dump();
        h.append((8 - h.size() % 8) % 8, ' ');  // data starts 8-byte aligned
        const fs::path out = out_dir / name, partial = out.string() + ".partial";
        {
            std::ofstream f(partial, std::ios::binary | std::ios::trunc);
            const uint64_t len = h.size();
            f.write(reinterpret_cast<const char*>(&len), 8);
            f.write(h.data(), std::streamsize(h.size()));
            for (size_t i : shards[s]) {
                std::ifstream in(folder / files[i].first, std::ios::binary);
                copy_bytes(in, f, files[i].second, files[i].first);
            }
            if (!f) throw std::runtime_error("cannot write " + partial.string());
        }
        rename_retrying(partial, out);
        result.shards.push_back(name);
        log(std::string(name) + ": " + std::to_string(shards[s].size()) + " files, " + gib(offset) + " GiB");
    }
    std::ofstream(out_dir / kPackedIndex) << nlohmann::ordered_json{{"metadata", meta}, {"weight_map", weight_map}}.dump(2) << "\n";
    return result;
}

std::pair<size_t, uint64_t> unpack_shard(const fs::path& shard, const fs::path& dir) {
    SafetensorsIndex index;
    index.add_file(shard);
    std::vector<const std::pair<const std::string, TensorInfo>*> tensors;
    for (const auto& t : index.all()) tensors.push_back(&t);
    std::sort(tensors.begin(), tensors.end(), [](auto* a, auto* b) { return a->second.begin < b->second.begin; });

    std::ifstream in(shard, std::ios::binary);
    uint64_t bytes = 0;
    for (const auto* t : tensors) {
        const auto& [name, info] = *t;
        const uint64_t n = info.end - info.begin;
        if (!safe_name(name)) throw std::runtime_error(shard.string() + ": refusing to write \"" + name + "\"");
        if (info.dtype != "U8" || info.shape.size() != 1 || info.shape[0] != n)
            throw std::runtime_error(shard.string() + ": " + name + " is not a packed file (" + info.dtype + ")");
        const fs::path out = dir / name;
        fs::create_directories(out.parent_path());
        std::ofstream f(out, std::ios::binary | std::ios::trunc);
        in.seekg(std::streamoff(info.begin));
        copy_bytes(in, f, n, name);
        bytes += n;
    }
    return {tensors.size(), bytes};
}

}  // namespace prep
