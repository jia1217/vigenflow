#pragma once
// Hugging Face access for prepare.exe: resolve a revision to a commit and its file list, then find each
// file in the standard HF hub cache (reused as is) or download it with curl into our cache, resuming
// partial downloads. The token reaches curl on its stdin, never on the command line or on disk.
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace prep {

struct HfFile {
    std::string path;   // in the repo, e.g. "transformer/diffusion_pytorch_model.safetensors"
    uint64_t size = 0;    // 0 when unknown (offline)
    std::string blob_id;  // git blob id: changes whenever the file's content does ("" offline)
};

struct HfSnapshot {
    std::string repo, revision, commit;
    std::vector<HfFile> files;
    bool offline = false;  // resolved from the local hub cache without contacting Hugging Face
};

class HfClient {
public:
    HfClient(std::string token, std::filesystem::path cache_dir);
    ~HfClient();

    HfSnapshot resolve(const std::string& repo, const std::string& revision);

    // Local path of the file, downloading it if needed. progress(bytes_done, bytes_total) runs every ~2 s.
    std::filesystem::path fetch(const HfSnapshot& snapshot, const HfFile& file,
                                const std::function<void(uint64_t, uint64_t)>& progress);

    // Parallel connections per large file (1 = a single stream).
    void set_connections(int n) { connections_ = n < 1 ? 1 : n; }

private:
    // curl's exit code; *http gets the HTTP status of the last response ("000" when there was none).
    int run_curl(const std::string& args, std::string* http = nullptr) const;
    std::string http_problem(const std::string& http, const std::string& what) const;
    void fetch_ranges(const std::string& url, const std::filesystem::path& partial, uint64_t size, const std::string& what,
                      const std::function<void(uint64_t, uint64_t)>& progress);
    std::filesystem::path hub_repo_dir(const std::string& repo) const;

    std::string token_;
    std::filesystem::path cache_dir_;
    int connections_ = 8;
    bool curl_has_variable_ = false;  // curl >= 8.3: --variable %ENV, to pass the token to range downloads
};

}  // namespace prep
