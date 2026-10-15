#pragma once
// Shared by prepare.exe (convert from the upstream checkpoints) and prepare_base.exe (download the
// pre-packed NPU folders): prepare_models.jsonc, paths, progress lines, manifests, swap-in, cleanup.
#include <algorithm>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "fs_retry.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace prep {
namespace fs = std::filesystem;

inline constexpr const char* kManifest = "prepare_manifest.json";
inline std::string g_log_tag = "[prepare]";  // the server relays lines starting with it

template <typename... Args>
void say(const std::string& id, Args&&... args) {
    static std::mutex m;
    std::ostringstream o;
    o << g_log_tag << " " << id << ": ";
    (o << ... << args);
    std::lock_guard<std::mutex> lock(m);
    std::cout << o.str() << std::endl;
}

inline std::string gib(uint64_t bytes) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << double(bytes) / (1ull << 30);
    return o.str();
}

// Folder of the running executable, so default paths do not depend on the current directory (vgf-serve
// resolves its model folders the same way).
inline fs::path exe_dir(const char* argv0) {
#ifdef _WIN32
    (void)argv0;
    std::vector<wchar_t> path(512);
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (!n) throw std::system_error(GetLastError(), std::system_category(), "GetModuleFileNameW");
        if (n < path.size()) return fs::path(std::wstring(path.data(), n)).parent_path();
        path.resize(path.size() * 2);
    }
#else
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    return (ec ? fs::absolute(argv0) : self).parent_path();
#endif
}

inline bool is_within(const fs::path& path, const fs::path& dir) {
    const auto rel = fs::weakly_canonical(path).lexically_relative(fs::weakly_canonical(dir));
    return !rel.empty() && *rel.begin() != "..";
}

struct ModelEntry {
    std::string id, recipe, source, revision = "main";
    bool enabled = true, keep_sources = false;
    // prepare_base.exe: pre-packed folder at <packed_repo>@<packed_revision>/<packed_path>
    std::string packed_repo, packed_revision, packed_path;
};

struct Config {
    std::vector<ModelEntry> models;
    fs::path weights_root;  // absolute, or empty when the file does not set it
};

inline Config load_config(const fs::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    // ignore_comments: .jsonc (// and /* */ comments; trailing commas are still errors)
    const auto j = nlohmann::json::parse(f, nullptr, true, true);
    Config config;
    if (j.contains("weights_root"))  // relative to the config file, not to the current directory
        config.weights_root = (fs::absolute(path).parent_path() / j.at("weights_root").get<std::string>()).lexically_normal();
    const std::string packed_repo = j.value("packed_repo", std::string()), packed_rev = j.value("packed_revision", std::string("main"));
    for (const auto& m : j.at("models")) {
        ModelEntry e;
        e.id = m.at("id").get<std::string>();
        e.recipe = m.value("recipe", std::string());
        e.source = m.value("source", std::string());
        e.revision = m.value("revision", std::string("main"));
        e.enabled = m.value("enabled", true);
        e.keep_sources = m.value("keep_sources", false);
        e.packed_repo = m.value("packed_repo", packed_repo);
        e.packed_revision = m.value("packed_revision", packed_rev);
        e.packed_path = m.value("packed_path", e.id);  // "" = the repo root
        if (e.id.empty() || e.id.find_first_of("/\\") != std::string::npos || e.id.starts_with("."))
            throw std::runtime_error("invalid id \"" + e.id + "\" in " + path.string());
        config.models.push_back(e);
    }
    return config;
}

// Options both tools take; parse_common_arg returns false for an option it does not know.
struct CommonOptions {
    fs::path config, weights_root, cache_dir;  // weights_root / cache_dir empty: resolved by resolve_paths
    std::set<std::string> only;
    std::string hf_token;
    int connections = 8;  // parallel connections per large download
    bool force = false, keep_sources = false;

    bool selected(const ModelEntry& m) const { return only.empty() ? m.enabled : only.count(m.id) > 0; }
};

inline bool parse_common_arg(CommonOptions& o, const std::string& a, const std::function<std::string()>& value) {
    if (a == "--config") o.config = value();
    else if (a == "--weights_root") o.weights_root = value();
    else if (a == "--cache_dir") o.cache_dir = value();
    else if (a == "--hf_token") o.hf_token = value();
    else if (a == "--connections") o.connections = std::max(1, std::stoi(value()));
    else if (a == "--force") o.force = true;
    else if (a == "--keep_sources") o.keep_sources = true;
    else if (a == "--only") {
        std::stringstream ss(value());
        for (std::string id; std::getline(ss, id, ',');)
            if (!id.empty()) o.only.insert(id);
    } else return false;
    return true;
}

// --weights_root, else the config's weights_root, else ../../all_model_weights relative to the executable.
inline void resolve_paths(CommonOptions& o, const Config& config, const char* argv0) {
    if (o.weights_root.empty())
        o.weights_root = config.weights_root.empty() ? exe_dir(argv0) / ".." / ".." / "all_model_weights" : config.weights_root;
    o.weights_root = fs::absolute(o.weights_root).lexically_normal();
    if (o.cache_dir.empty()) o.cache_dir = o.weights_root / ".hf_sources";
}

inline std::optional<nlohmann::json> read_manifest(const fs::path& dir) {
    std::ifstream f(dir / kManifest);
    if (!f) return std::nullopt;
    try {
        return nlohmann::json::parse(f);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// Swap a finished <id>.partial folder in: the old folder is kept until the new one is in place.
inline void swap_in(const fs::path& stage, const fs::path& out) {
    std::error_code ec;
    const fs::path old = out.string() + ".old";
    fs::remove_all(old, ec);
    if (fs::exists(out)) rename_retrying(out, old);
    rename_retrying(stage, out);
    fs::remove_all(old, ec);
}

// After a successful prepare: delete the files this run downloaded (under cache_dir) and the folders that
// leaves empty. Files read from the shared Hugging Face cache stay: other tools use them.
inline void remove_downloads(const std::string& id, const std::vector<fs::path>& sources, const fs::path& cache_dir,
                             bool keep, const std::string& what, bool quiet = false) {
    uint64_t own = 0, shared = 0;
    std::vector<fs::path> own_files;
    for (const auto& p : sources) {
        std::error_code ec;
        const uint64_t n = fs::file_size(p, ec);
        if (is_within(p, cache_dir)) own += n, own_files.push_back(p);
        else shared += n;
    }
    if (shared && !quiet) say(id, gib(shared), " GiB of the ", what, " came from the shared Hugging Face cache; left in place");
    if (own_files.empty()) return;
    if (keep) {
        if (!quiet) say(id, "kept the downloaded ", what, " (", gib(own), " GiB) in ", cache_dir.string());
        return;
    }
    const fs::path root = fs::weakly_canonical(cache_dir);
    for (const auto& p : own_files) {
        std::error_code ec;
        // Resolve while the path exists; a deleted Windows file can remain
        // delete-pending until the last handle is closed.
        fs::path d = fs::weakly_canonical(p).parent_path();
        fs::remove(p, ec);
        for (; d != root && !d.lexically_relative(root).empty() && *d.lexically_relative(root).begin() != "..";
             d = d.parent_path())
            if (!fs::remove(d, ec)) break;  // stops at the first folder that is not empty
    }
    std::error_code ec;
    fs::remove(root, ec);  // only if nothing else is cached there
    if (!quiet) say(id, "deleted the downloaded ", what, " (", gib(own), " GiB)");
}

// One prepare.exe / prepare_base.exe at a time per weights root: two runs would write the same .partial
// files and download into the same cache. The lock (flock, LockFileEx on Windows) is released when the
// process exits, however it exits, so a crash leaves no stale lock.
class WeightsLock {
public:
    explicit WeightsLock(const fs::path& weights_root) {
        const fs::path file = weights_root / ".prepare.lock";
#ifdef _WIN32
        handle_ = CreateFileW(file.c_str(), GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::system_error(GetLastError(), std::system_category(), "cannot open " + file.string());
        OVERLAPPED offset{};
        if (!LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD, &offset)) {
            const DWORD error = GetLastError();
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
            if (error == ERROR_LOCK_VIOLATION)
                throw std::runtime_error("another prepare.exe or prepare_base.exe is working in " +
                                         weights_root.string() + "; wait for it to finish");
            throw std::system_error(error, std::system_category(), "cannot lock " + file.string());
        }
        const std::string me = std::to_string(GetCurrentProcessId());
        DWORD written = 0;
        SetEndOfFile(handle_);
        WriteFile(handle_, me.data(), static_cast<DWORD>(me.size()), &written, nullptr);
#else
        fd_ = ::open(file.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) throw std::runtime_error("cannot open " + file.string());
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            char pid[32] = {};
            if (::pread(fd_, pid, sizeof(pid) - 1, 0) < 0) pid[0] = '?';
            ::close(fd_);
            throw std::runtime_error("another prepare.exe or prepare_base.exe (pid " + std::string(pid) + ") is working in " +
                                     weights_root.string() + "; wait for it to finish");
        }
        const std::string me = std::to_string(::getpid());
        if (::ftruncate(fd_, 0) != 0 || ::pwrite(fd_, me.data(), me.size(), 0) < 0) {
        }  // the pid is only for the message above
#endif
    }
    ~WeightsLock() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            OVERLAPPED offset{};
            UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &offset);
            CloseHandle(handle_);
        }
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    WeightsLock(const WeightsLock&) = delete;
    WeightsLock& operator=(const WeightsLock&) = delete;

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

inline std::string utc_stamp() {
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    return stamp;
}

}  // namespace prep
