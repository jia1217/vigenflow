#include "hf_download.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#include <sys/wait.h>
#endif

#include <nlohmann/json.hpp>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
constexpr const char* kPopenRead = "rb";
#else
constexpr const char* kPopenRead = "r";
#endif
// Range downloads read file data from curl's stdout, so their token comes from this environment variable
// (curl --variable) instead of stdin; like stdin, and unlike the command line, other users cannot see it.
constexpr const char* kTokenEnv = "VGF_HF_TOKEN";

namespace prep {
namespace fs = std::filesystem;

namespace {

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

fs::path home_dir() {
    std::string h = env("HOME");
    if (h.empty()) h = env("USERPROFILE");
    return h;
}

fs::path hf_home() {
    if (auto v = env("HF_HOME"); !v.empty()) return v;
    return home_dir() / ".cache" / "huggingface";
}

fs::path hub_cache() {
    if (auto v = env("HF_HUB_CACHE"); !v.empty()) return v;
    return hf_home() / "hub";
}

std::string endpoint() {
    auto e = env("HF_ENDPOINT");
    if (e.empty()) e = "https://huggingface.co";
    while (!e.empty() && e.back() == '/') e.pop_back();
    return e;
}

std::string url_encode(const std::string& s, bool keep_slash) {
    std::ostringstream o;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (keep_slash && c == '/')) o << c;
        else o << '%' << std::uppercase << std::hex << (c >> 4) << (c & 15) << std::nouppercase << std::dec;
    }
    return o.str();
}

std::string quote(const std::string& s) {
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
#endif
}

std::string repo_dir_name(const std::string& repo) {  // org/name -> org--name
    return std::regex_replace(repo, std::regex("/"), "--");
}

bool is_commit(const std::string& s) { return std::regex_match(s, std::regex("^[0-9a-f]{40}$")); }

std::string read_text(const fs::path& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

}  // namespace

HfClient::HfClient(std::string token, fs::path cache_dir) : token_(std::move(token)), cache_dir_(std::move(cache_dir)) {
    if (token_.empty()) token_ = env("HF_TOKEN");
    if (token_.empty() && fs::exists(hf_home() / "token")) token_ = read_text(hf_home() / "token");
    if (FILE* p = popen("curl --version", kPopenRead)) {
        int major = 0, minor = 0;
        if (std::fscanf(p, "curl %d.%d", &major, &minor) == 2) curl_has_variable_ = major > 8 || (major == 8 && minor >= 3);
        pclose(p);
    }
    if (!token_.empty()) {
#ifdef _WIN32
        _putenv_s(kTokenEnv, token_.c_str());
#else
        setenv(kTokenEnv, token_.c_str(), 1);
#endif
    }
}

HfClient::~HfClient() = default;

int HfClient::run_curl(const std::string& args, std::string* http) const {
    std::string cmd = "curl -L --fail --silent --show-error --retry 3 --retry-delay 2 " + args;
    fs::path code_file;
    if (http) {
        code_file = fs::temp_directory_path() /
                    ("prepare_http_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        cmd += " -w %{http_code} > " + quote(code_file.string());
    }
    int status;
    if (token_.empty()) {
        status = std::system(cmd.c_str());
    } else {
        // -H @- reads the header from stdin: nothing is left behind if prepare.exe is killed mid-download.
        FILE* p = popen((cmd + " -H @-").c_str(), "w");
        if (!p) return -1;
        std::fputs(("Authorization: Bearer " + token_ + "\n").c_str(), p);
        status = pclose(p);
    }
    if (http) {
        *http = read_text(code_file);
        std::error_code ec;
        fs::remove(code_file, ec);
    }
#ifndef _WIN32
    if (status != -1) status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    return status;
}

// A message for an HTTP status the user can fix (gated repo, bad token, wrong name); "" otherwise.
std::string HfClient::http_problem(const std::string& http, const std::string& what) const {
    const auto slash = what.find('/', what.find('/') + 1);  // "org/name[/file]" -> "org/name"
    const std::string repo = what.substr(0, slash == std::string::npos ? what.find('@') : slash);
    if (http == "401" || http == "403")
        return token_.empty() ? what + " needs a Hugging Face login: accept the terms at https://huggingface.co/" + repo +
                                    ", then run `huggingface-cli login` (or set HF_TOKEN, or pass --hf_token)"
                              : what + ": this Hugging Face token has no access; accept the terms at https://huggingface.co/" +
                                    repo + " with the token's account";
    if (http == "404") return what + " does not exist; check source and revision in prepare_models.jsonc";
    return "";
}

fs::path HfClient::hub_repo_dir(const std::string& repo) const { return hub_cache() / ("models--" + repo_dir_name(repo)); }

HfSnapshot HfClient::resolve(const std::string& repo, const std::string& revision) {
    HfSnapshot snap{repo, revision};
    // The cache folder is only created when something is downloaded.
    const fs::path api_out = fs::temp_directory_path() / ("prepare_api_" + repo_dir_name(repo) + "_" +
                                                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    const std::string url = endpoint() + "/api/models/" + url_encode(repo, true) + "/revision/" + url_encode(revision, false) + "?blobs=true";
    std::string http;
    const int rc = run_curl("-o " + quote(api_out.string()) + " " + quote(url), &http);
    std::error_code ec;
    if (rc == 0) {
        const auto j = nlohmann::json::parse(read_text(api_out));
        snap.commit = j.at("sha").get<std::string>();
        for (const auto& s : j.at("siblings"))
            snap.files.push_back({s.at("rfilename").get<std::string>(), s.value("size", uint64_t(0)), s.value("blobId", std::string())});
        fs::remove(api_out);
        return snap;
    }
    fs::remove(api_out, ec);
    // Hugging Face answered with an error the user can fix; otherwise we are offline.
    std::string why = http_problem(http, repo + "@" + revision);
    if (why.empty()) why = "cannot reach Hugging Face for " + repo + "@" + revision;
    // Without a usable answer, fall back to the hub cache, which records the commit each branch or tag pointed to.
    const fs::path repo_dir = hub_repo_dir(repo);
    if (is_commit(revision)) snap.commit = revision;
    else if (fs::exists(repo_dir / "refs" / revision)) snap.commit = read_text(repo_dir / "refs" / revision);
    const fs::path snap_dir = repo_dir / "snapshots" / snap.commit;
    if (snap.commit.empty() || !fs::is_directory(snap_dir) || fs::is_empty(snap_dir))
        throw std::runtime_error(why + " (and it is not in the local Hugging Face cache)");
    for (const auto& e : fs::recursive_directory_iterator(snap_dir))
        // lexically: snapshot files are symlinks into blobs/, and fs::relative would follow them
        if (e.is_regular_file()) snap.files.push_back({e.path().lexically_relative(snap_dir).generic_string(), 0, ""});
    snap.offline = true;
    return snap;
}

fs::path HfClient::fetch(const HfSnapshot& snap, const HfFile& file, const std::function<void(uint64_t, uint64_t)>& progress) {
    auto complete = [&](const fs::path& p) {
        std::error_code ec;
        return fs::is_regular_file(p, ec) && (file.size == 0 || fs::file_size(p, ec) == file.size);
    };
    // 1. The standard hub cache (huggingface-cli, diffusers, an earlier vgf-serve): used in place.
    const fs::path hub = hub_repo_dir(snap.repo) / "snapshots" / snap.commit / file.path;
    if (complete(hub)) return hub;
    // 2. Our cache, keyed by commit so a new revision never mixes files with an old one.
    const fs::path local = cache_dir_ / repo_dir_name(snap.repo) / snap.commit / file.path;
    if (complete(local)) return local;
    if (snap.offline) throw std::runtime_error("offline and " + file.path + " is not cached");

    fs::create_directories(local.parent_path());
    const fs::path partial = local.string() + ".partial";
    const std::string url = endpoint() + "/" + url_encode(snap.repo, true) + "/resolve/" + snap.commit + "/" + url_encode(file.path, true);
    // Large files: parallel ranges, when the server answers a range request with 206.
    bool ranges = connections_ > 1 && file.size >= (64ull << 20) && (token_.empty() || curl_has_variable_);
    if (ranges) {
        const fs::path probe = fs::temp_directory_path() / ("prepare_probe_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::string http;
        run_curl("-r 0-0 -o " + quote(probe.string()) + " " + quote(url), &http);
        std::error_code ec;
        fs::remove(probe, ec);
        if (const std::string why = http_problem(http, snap.repo + "/" + file.path); !why.empty()) throw std::runtime_error(why);
        ranges = http == "206";
    }
    if (ranges) {
        fetch_ranges(url, partial, file.size, snap.repo + "/" + file.path, progress);
    } else if (!(file.size && fs::exists(partial) && fs::file_size(partial) == file.size)) {
        const std::string args = "-C - -o " + quote(partial.string()) + " " + quote(url);
        std::atomic<bool> done{false};
        int rc = 0;
        std::string http;
        std::thread curl([&] { rc = run_curl(args, &http); done = true; });
        auto last = std::chrono::steady_clock::now();
        while (!done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (progress && std::chrono::steady_clock::now() - last > std::chrono::seconds(2)) {
                last = std::chrono::steady_clock::now();
                std::error_code ec;
                const auto n = fs::exists(partial, ec) ? fs::file_size(partial, ec) : 0;
                progress(n, file.size);
            }
        }
        curl.join();
        if (rc != 0) {
            const std::string why = http_problem(http, snap.repo + "/" + file.path);
            throw std::runtime_error(!why.empty() ? why
                                                  : "download failed (curl exit " + std::to_string(rc) + ", HTTP " + http + "): " +
                                                        snap.repo + "/" + file.path);
        }
    }
    if (file.size && fs::file_size(partial) != file.size)
        throw std::runtime_error(file.path + ": downloaded " + std::to_string(fs::file_size(partial)) + " of " + std::to_string(file.size) + " bytes");
    fs::rename(partial, local);
    if (progress) progress(fs::file_size(local), fs::file_size(local));
    return local;
}

// One connection to the Hugging Face CDN is capped (measured ~80 MB/s for a newly uploaded repo, ~270 MB/s
// over 8), so a large file downloads as connections_ byte ranges at once. Each range streams from curl's
// stdout straight to its place in <file>.partial (sized up front: no joining, no second copy), and
// <file>.partial.ranges records how far each range got, so an interrupted download resumes. Progress is
// committed only from 206 (partial content) responses.
void HfClient::fetch_ranges(const std::string& url, const fs::path& partial, uint64_t size, const std::string& what,
                            const std::function<void(uint64_t, uint64_t)>& progress) {
    const fs::path ranges_file = partial.string() + ".ranges";
    const size_t n = size_t(connections_);
    const uint64_t seg = (size + n - 1) / n;
    std::vector<uint64_t> begin(n), len(n), done(n, 0);
    for (size_t i = 0; i < n; ++i) {
        begin[i] = std::min<uint64_t>(size, i * seg);
        len[i] = std::min<uint64_t>(size, begin[i] + seg) - begin[i];
    }
    std::error_code ec;
    bool resumed = false;
    if (std::ifstream rf{ranges_file}) {
        try {
            const auto j = nlohmann::json::parse(rf);
            if (j.at("size").get<uint64_t>() == size && j.at("done").size() == n && fs::file_size(partial, ec) == size) {
                done = j.at("done").get<std::vector<uint64_t>>();
                resumed = true;
            }
        } catch (const std::exception&) {
        }
    }
    if (!resumed) {  // no record: a .partial left by a single-stream download holds a valid prefix
        const uint64_t prefix = fs::exists(partial, ec) ? fs::file_size(partial, ec) : 0;
        if (prefix == size) return;  // finished earlier, not yet renamed
        for (size_t i = 0; i < n; ++i) done[i] = prefix > begin[i] ? std::min(len[i], prefix - begin[i]) : 0;
    }

    std::mutex m;
    auto save = [&] {  // caller holds m
        const fs::path tmp = ranges_file.string() + ".tmp";
        std::ofstream(tmp) << nlohmann::json{{"size", size}, {"done", done}}.dump();
        fs::rename(tmp, ranges_file);
    };
    {
        std::lock_guard<std::mutex> lock(m);
        save();  // the record exists before the file grows to full size
    }
    if (!fs::exists(partial)) std::ofstream(partial, std::ios::binary);
    if (fs::file_size(partial) != size) fs::resize_file(partial, size);

    const std::string auth = token_.empty() ? "" : std::string(" --variable %") + kTokenEnv + " --expand-header " +
                                                        quote(std::string("Authorization: Bearer {{") + kTokenEnv + "}}");
    std::exception_ptr error;
    std::atomic<size_t> running{0};
    auto worker = [&](size_t i) {
        // Next to the download, under a fixed name: a rerun reuses it, so a killed run leaves nothing behind in /tmp.
        const fs::path err = partial.string() + ".range" + std::to_string(i) + ".log";
        try {
            std::fstream f(partial, std::ios::in | std::ios::out | std::ios::binary);
            std::vector<char> buf(8u << 20);
            for (int attempt = 0;; ++attempt) {
                uint64_t start;
                {
                    std::lock_guard<std::mutex> lock(m);
                    if (error) break;
                    start = done[i];
                }
                if (start >= len[i]) break;
                if (attempt >= 6) throw std::runtime_error("download failed after 6 attempts: " + what + " (" + read_text(err) + ")");
                if (attempt) std::this_thread::sleep_for(std::chrono::seconds(2 * attempt));
                const uint64_t from = begin[i] + start, want = len[i] - start;
                const std::string cmd = "curl -L --fail --silent --show-error" + auth + " -r " + std::to_string(from) + "-" +
                                        std::to_string(begin[i] + len[i] - 1) + " -w %{stderr}HTTP=%{http_code} " + quote(url) +
                                        " 2> " + quote(err.string());
                FILE* p = popen(cmd.c_str(), kPopenRead);
                if (!p) throw std::runtime_error("cannot start curl");
                uint64_t got = 0, unsaved = 0;
                bool overflow = false;
                for (size_t k; (k = std::fread(buf.data(), 1, buf.size(), p)) > 0;) {
                    if (got + k > want) {  // more than asked for: not a range response
                        overflow = true;
                        break;
                    }
                    f.seekp(std::streamoff(from + got));
                    f.write(buf.data(), std::streamsize(k));
                    got += k, unsaved += k;
                    if (unsaved >= (64u << 20)) {  // resume point, only once the response is known to be a range
                        f.flush();
                        std::lock_guard<std::mutex> lock(m);
                        done[i] = start + got;
                        save();
                        unsaved = 0;
                    }
                }
                pclose(p);
                f.flush();
                if (!f) throw std::runtime_error("cannot write " + partial.string());
                const std::string e = read_text(err);
                const auto pos = e.rfind("HTTP=");
                const std::string http = pos == std::string::npos ? "" : e.substr(pos + 5, 3);
                std::lock_guard<std::mutex> lock(m);
                if (http == "206" && !overflow) {
                    done[i] = start + got;  // all of it, or as far as a dropped connection got
                } else {
                    done[i] = start;  // not a 206: nothing from this attempt counts
                    if (const std::string why = http_problem(http, what); !why.empty()) throw std::runtime_error(why);
                }
                save();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(m);
            if (!error) error = std::current_exception();
        }
        std::error_code rm_ec;
        fs::remove(err, rm_ec);
        --running;
    };
    std::vector<std::thread> pool;
    for (size_t i = 0; i < n; ++i) {
        ++running;
        pool.emplace_back(worker, i);
    }
    auto last = std::chrono::steady_clock::now();
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (progress && std::chrono::steady_clock::now() - last > std::chrono::seconds(2)) {
            last = std::chrono::steady_clock::now();
            uint64_t sum = 0;
            {
                std::lock_guard<std::mutex> lock(m);
                for (uint64_t d : done) sum += d;
            }
            progress(sum, size);
        }
    }
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
    for (size_t i = 0; i < n; ++i)
        if (done[i] != len[i]) throw std::runtime_error("download incomplete: " + what);
    fs::remove(ranges_file, ec);
}

}  // namespace prep
