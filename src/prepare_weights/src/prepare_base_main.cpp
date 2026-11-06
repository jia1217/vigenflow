// prepare_base.exe: download the pre-packed NPU weight folders from Hugging Face into
// --weights_root/<id>, without converting anything; or, with --pack, make those packed folders from
// all_model_weights for uploading.
//
// Each model's packed folder is <packed_repo>@<packed_revision>/<packed_path> (prepare_models.jsonc):
// model-0000i-of-0000N.safetensors + model.safetensors.index.json (packed_format.hpp). Shards unpack as
// they finish downloading and each downloaded shard is deleted right after (unless keep_sources); an
// interrupted run resumes with the shards not yet unpacked. The folder is swapped in when complete. A model
// is up to date when the same packed files were unpacked before, or when its folder was converted by
// prepare.exe from the same source commit and recipe (the same bytes).
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <queue>
#include <thread>

#include "cli_common.hpp"
#include "hf_download.hpp"
#include "packed_format.hpp"

using namespace prep;
using nlohmann::json;

namespace {

struct Options : CommonOptions {
    bool pack = false;
    fs::path pack_dir;  // default: <weights_root>/../packed_weights
    double shard_gb = 5;
    bool accept_license = false;  // answers the license prompt of the models that have one
};

void usage() {
    std::cout << "usage: prepare_base.exe [options]\n"
                 "       prepare_base.exe --pack [--pack_dir DIR] [--shard_gb N] [options]\n\n"
                 "Downloads the pre-packed NPU weights of the models in prepare_models.jsonc from Hugging Face\n"
                 "(\"packed_repo\"), one folder per model under --weights_root. With --pack, packs those folders\n"
                 "into safetensors shards for uploading instead.\n\n"
                 "  --config PATH        model list (default: prepare_models.jsonc next to prepare_base.exe)\n"
                 "  --weights_root DIR   model folders (default: the config's \"weights_root\", relative to the config\n"
                 "                       file, else ../../all_model_weights relative to prepare_base.exe)\n"
                 "  --cache_dir DIR      downloaded shards (default: <weights_root>/.hf_sources)\n"
                 "  --only ID[,ID...]    only these ids (enabled or not)\n"
                 "  --hf_token TOKEN     for private repos (default: $HF_TOKEN, then the huggingface-cli login)\n"
                 "  --connections N      parallel connections per shard download (default: 8)\n"
                 "  --force              download even if the folder is up to date or was not made by these tools\n"
                 "  --keep_sources       keep the downloaded shards after unpacking (default: delete them)\n"
                 "  --accept_license     accept model-specific terms (Krea 2 / Qwen Image 2.1 Turbo) instead of\n"
                 "                       typing I agree at the prompt, for unattended runs\n"
                 "  --pack               pack <weights_root>/<id> into <pack_dir>/<packed_path> for uploading\n"
                 "  --pack_dir DIR       default: <weights_root>/../packed_weights\n"
                 "  --shard_gb N         shard size for --pack (default: 5)\n";
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "-h" || a == "--help") usage(), std::exit(0);
        else if (a == "--pack") o.pack = true;
        else if (a == "--pack_dir") o.pack_dir = value();
        else if (a == "--shard_gb") o.shard_gb = std::stod(value());
        else if (a == "--accept_license") o.accept_license = true;
        else if (!parse_common_arg(o, a, value)) throw std::runtime_error("unknown option " + a + " (see --help)");
    }
    if (o.config.empty()) o.config = exe_dir(argv[0]) / "prepare_models.jsonc";
    if (!(o.shard_gb > 0)) throw std::runtime_error("--shard_gb must be positive");
    return o;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool is_qwen_image_turbo(const ModelEntry& m) {
    return m.id == "qwen_image_2_1_turbo" || m.recipe == "qwen_image_2_1_turbo" ||
           m.source == "Qwen/Qwen-Image-2.1-Turbo";
}

void require_model_notices(const ModelEntry& m, const fs::path& folder) {
    if (!is_qwen_image_turbo(m)) return;
    for (const char* name : {"LICENSE", "Notice"}) {
        const fs::path file = folder / name;
        if (!fs::is_regular_file(file) || fs::file_size(file) == 0)
            throw std::runtime_error(folder.string() + ": missing or empty " + name +
                                     "; Qwen weights must retain the upstream LICENSE and attribution Notice");
    }
}

void pack_model(const ModelEntry& m, const Options& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    const fs::path folder = opt.weights_root / m.id;
    if (!fs::is_directory(folder)) throw std::runtime_error(folder.string() + " does not exist");
    require_model_notices(m, folder);
    if (!read_manifest(folder))
        say(m.id, "note: no ", kManifest, " (made by the older tools); `prepare.exe --only ", m.id,
            " --force` adds one, so prepare_base.exe can tell such a folder is up to date");
    const fs::path out = opt.pack_dir / (m.packed_path.empty() ? m.id : m.packed_path);
    const auto r = pack_folder(folder, out, m.id, uint64_t(opt.shard_gb * double(1ull << 30)),
                               [&](const std::string& line) { say(m.id, "pack ", line); });
    say(m.id, "packed ", r.files, " files, ", gib(r.bytes), " GiB into ", r.shards.size(), " shards in ", std::fixed,
        std::setprecision(1), seconds_since(t0), " s -> ", out.string());
}

// Model-specific terms; do not infer a license for other Qwen checkpoints from this one.
std::string own_license(const ModelEntry& m) {
    const std::string packed_url = "https://huggingface.co/" + m.packed_repo + "/blob/" + m.packed_revision + "/" +
                                   (m.packed_path.empty() ? "" : m.packed_path + "/");
    if (is_qwen_image_turbo(m))
        return "Qwen RESEARCH LICENSE AGREEMENT (" + packed_url + "LICENSE). "
               "Research/evaluation only; commercial use requires a separate license from "
               "model-business@notice.qwencloud.com. Retain LICENSE and Notice with redistributed weights.";
    if (m.recipe == "krea2_turbo" || m.source.starts_with("krea/"))
        return "Krea 2 Community License Agreement (" + packed_url +
               "LICENSE.pdf) and Krea Acceptable Use Policy (https://www.krea.ai/krea-2-use-policy)";
    return "";
}

struct NotAccepted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Before a model with its own license downloads: the user types I agree (or passed --accept_license). The
// packed repos need no Hugging Face login, so this prompt is where the terms are accepted.
void confirm_license(const ModelEntry& m, const Options& opt) {
    const std::string terms = own_license(m);
    if (terms.empty()) return;
    say(m.id, "this model has its own license: ", terms);
    if (opt.accept_license) {
        say(m.id, "terms accepted with --accept_license");
        return;
    }
    std::cout << g_log_tag << " " << m.id << ": type I agree to accept these terms and continue: " << std::flush;
    std::string answer;
    bool line_ended = bool(std::getline(std::cin, answer));  // a terminal echoes the Enter; a pipe or no input does not
#ifndef _WIN32
    line_ended = line_ended && ::isatty(STDIN_FILENO);
#endif
    if (!line_ended) std::cout << std::endl;
    const auto first = answer.find_first_not_of(" \t\r"), last = answer.find_last_not_of(" \t\r");
    answer = first == std::string::npos ? "" : answer.substr(first, last - first + 1);
    std::transform(answer.begin(), answer.end(), answer.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (answer != "i agree")
        throw NotAccepted("the license terms were not accepted, so nothing was downloaded (run again and type I agree, "
                          "or pass --accept_license)");
}

void download_model(const ModelEntry& m, HfClient& hf, const Options& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    if (m.packed_repo.empty()) throw std::runtime_error("no \"packed_repo\" in prepare_models.jsonc");
    const fs::path out = opt.weights_root / m.id, stage = opt.weights_root / (m.id + ".partial");
    const std::string prefix = m.packed_path.empty() ? "" : m.packed_path + "/";
    const std::string where = m.packed_repo + "@" + m.packed_revision + "/" + prefix;
    const auto local = fs::exists(out) ? read_manifest(out) : std::nullopt;

    say(m.id, "resolving ", where);
    HfSnapshot snap;
    try {
        snap = hf.resolve(m.packed_repo, m.packed_revision);
    } catch (const std::exception& e) {
        if (!opt.force && local) {
            say(m.id, "cannot check for updates (", e.what(), "); keeping the existing weights");
            return;
        }
        throw;
    }

    std::optional<HfFile> index_file;
    std::vector<HfFile> shards;
    for (const HfFile& f : snap.files) {
        if (!f.path.starts_with(prefix) || f.path.find('/', prefix.size()) != std::string::npos) continue;
        const std::string name = f.path.substr(prefix.size());
        if (name == kPackedIndex) index_file = f;
        else if (name.starts_with("model-") && name.ends_with(".safetensors")) shards.push_back(f);
    }
    if (!index_file || shards.empty())
        throw std::runtime_error("no packed weights at " + where + " (upload the folder made by `prepare_base.exe --pack`)");
    std::sort(shards.begin(), shards.end(), [](const HfFile& a, const HfFile& b) { return a.path < b.path; });
    json blobs = {{index_file->path, index_file->blob_id}};
    for (const HfFile& f : shards) blobs[f.path] = f.blob_id;

    const fs::path index_path = hf.fetch(snap, *index_file, nullptr);
    json index, meta;
    {
        std::ifstream f(index_path);
        index = json::parse(f);
        meta = index.at("metadata");
    }
    if (meta.value("format", "") != kPackedFormat)
        throw std::runtime_error(where + kPackedIndex + " is not a " + kPackedFormat + " index");
    if (is_qwen_image_turbo(m)) {
        for (const char* name : {"LICENSE", "Notice"}) {
            if (!index.at("weight_map").contains(name))
                throw std::runtime_error("Qwen packed index must include " + std::string(name) + " inside the shards");
        }
    }

    if (fs::exists(out) && !opt.force) {
        if (!local) {
            say(m.id, "skipped: ", out.string(), " exists but was not made by prepare.exe or prepare_base.exe (use --force)");
            remove_downloads(m.id, {index_path}, opt.cache_dir, false, "", true);
            return;
        }
        const bool same_pack = local->contains("packed") && (*local)["packed"].value("repo", "") == m.packed_repo &&
                               (*local)["packed"].value("blobs", json()) == blobs;
        const bool same_source = meta.contains("commit") && local->value("recipe", "") == meta.value("recipe", "") &&
                                 local->value("recipe_version", 0) == meta.value("recipe_version", 0) &&
                                 local->value("source", "") == meta.value("source", "") &&
                                 local->value("commit", "") == meta.value("commit", "");
        if (same_pack || same_source) {
            require_model_notices(m, out);
            say(m.id, "up to date");
            remove_downloads(m.id, {index_path}, opt.cache_dir, false, "", true);
            return;
        }
        say(m.id, "updating (the packed weights changed)");
    }
    try {
        confirm_license(m, opt);
    } catch (const NotAccepted&) {
        remove_downloads(m.id, {index_path}, opt.cache_dir, false, "", true);
        throw;
    }

    // Unpacking resumes: <id>.partial records the shards already unpacked into it, valid while the packed files
    // are the same ones (blob ids). Without that, a failed run would download every deleted shard again.
    const fs::path progress_file = stage / ".prepare_base_progress.json";
    std::set<std::string> done;
    if (std::ifstream pf{progress_file}) {
        try {
            const json p = json::parse(pf);
            if (p.at("blobs") == blobs) done = p.at("done").get<std::set<std::string>>();
        } catch (const std::exception&) {
        }
    }
    std::error_code ec;
    if (done.empty()) {
        fs::remove_all(stage, ec);
        fs::create_directories(stage);
    } else {
        say(m.id, "resuming: ", done.size(), " of ", shards.size(), " shards already unpacked");
    }
    auto save_progress = [&] {
        const fs::path tmp = progress_file.string() + ".tmp";
        std::ofstream(tmp) << json{{"blobs", blobs}, {"done", done}}.dump() << "\n";
        rename_retrying(tmp, progress_file);
    };
    const bool keep = opt.keep_sources || m.keep_sources;
    std::vector<const HfFile*> todo;
    for (const HfFile& f : shards)
        if (!done.count(f.path)) todo.push_back(&f);

    // Shards unpack in order as the downloader thread finishes each one. A downloaded shard is deleted as soon
    // as it is unpacked (unless keep_sources), and the downloader waits while two finished shards are on disk
    // (the one unpacking and the next), so the disk never holds a second copy of the model.
    std::queue<std::pair<fs::path, std::exception_ptr>> ready;
    std::mutex q_mutex;
    std::condition_variable q_cv, space_cv;
    size_t pending = 0;  // downloaded, not yet unpacked and deleted
    bool stop = false;   // the unpacking side failed: download nothing more
    std::thread downloader([&] {
        for (const HfFile* f : todo) {
            {
                std::unique_lock<std::mutex> lock(q_mutex);
                space_cv.wait(lock, [&] { return pending <= 1 || stop; });
                if (stop) return;
            }
            std::pair<fs::path, std::exception_ptr> item;
            try {
                item.first = hf.fetch(snap, *f, [&](uint64_t n, uint64_t total) {
                    say(m.id, "download ", f->path, " ", gib(n), total ? "/" + gib(total) : std::string(), " GiB");
                });
            } catch (...) {
                item.second = std::current_exception();
            }
            const bool failed = bool(item.second);
            {
                std::lock_guard<std::mutex> lock(q_mutex);
                ready.push(std::move(item));
                ++pending;
            }
            q_cv.notify_one();
            if (failed) return;
        }
    });

    uint64_t deleted = 0, kept = 0, shared = 0;
    std::exception_ptr error;
    for (size_t i = 0; i < todo.size() && !error; ++i) {
        std::pair<fs::path, std::exception_ptr> item;
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            q_cv.wait(lock, [&] { return !ready.empty(); });
            item = std::move(ready.front());
            ready.pop();
        }
        try {
            if (item.second) std::rethrow_exception(item.second);
            const auto [n, b] = unpack_shard(item.first, stage);
            done.insert(todo[i]->path);
            save_progress();
            const uint64_t size = fs::file_size(item.first);
            // Only the file: its folder is pruned at the end, never while the downloader writes the next shard there.
            if (!is_within(item.first, opt.cache_dir)) shared += size;
            else if (keep) kept += size;
            else fs::remove(item.first), deleted += size;
            say(m.id, "unpack ", done.size(), "/", shards.size(), " (", n, " files, ", gib(b), " GiB)");
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(q_mutex);
            --pending;
            stop = bool(error);
        }
        space_cv.notify_one();
    }
    downloader.join();
    if (error) std::rethrow_exception(error);
    fs::remove(progress_file);

    // The finished folder against the index.
    size_t files = 0;
    uint64_t bytes = 0;
    for (const auto& e : fs::recursive_directory_iterator(stage))
        if (e.is_regular_file()) ++files, bytes += e.file_size();
    if (files != meta.value("file_count", size_t(0)) || bytes != meta.value("total_size", uint64_t(0)))
        throw std::runtime_error("unpacked " + std::to_string(files) + " files / " + std::to_string(bytes) + " bytes; the index lists " +
                                 std::to_string(meta.value("file_count", size_t(0))) + " / " + std::to_string(meta.value("total_size", uint64_t(0))));
    require_model_notices(m, stage);

    // The packed prepare_manifest.json (provenance) plus where these files came from.
    json man = json::object();
    if (const auto packed_man = read_manifest(stage)) man = *packed_man;
    else man["id"] = m.id;
    man["packed"] = {{"repo", m.packed_repo}, {"revision", m.packed_revision}, {"path", m.packed_path},
                     {"commit", snap.commit}, {"blobs", blobs}, {"unpacked_at", utc_stamp()}};
    std::ofstream(stage / kManifest) << man.dump(2) << "\n";
    swap_in(stage, out);
    remove_downloads(m.id, {index_path}, opt.cache_dir, keep, "", true);  // the index; prunes the emptied folders
    if (shared) say(m.id, gib(shared), " GiB of the packed weights came from the shared Hugging Face cache; left in place");
    if (deleted) say(m.id, "deleted each downloaded shard right after unpacking it (", gib(deleted), " GiB)");
    if (kept) say(m.id, "kept the downloaded shards (", gib(kept), " GiB) in ", opt.cache_dir.string());
    say(m.id, "ready in ", std::fixed, std::setprecision(1), seconds_since(t0), " s: ", files, " files, ", gib(bytes),
        " GiB -> ", out.string());
}

}  // namespace

int main(int argc, char** argv) {
    g_log_tag = "[prepare_base]";
    try {
        Options opt = parse_args(argc, argv);
        const Config config = load_config(opt.config);
        resolve_paths(opt, config, argv[0]);
        if (opt.pack_dir.empty()) opt.pack_dir = opt.weights_root.parent_path() / "packed_weights";
        opt.pack_dir = fs::absolute(opt.pack_dir).lexically_normal();
        std::cout << g_log_tag << " weights root: " << opt.weights_root.string() << std::endl;
        if (opt.pack) std::cout << g_log_tag << " pack dir: " << opt.pack_dir.string() << std::endl;
        else fs::create_directories(opt.weights_root);
        const WeightsLock lock(opt.weights_root);  // --pack too: packing must not read a folder being rebuilt
        HfClient hf(opt.hf_token, opt.cache_dir);
        hf.set_connections(opt.connections);
        int failures = 0;
        for (const auto& m : config.models) {
            if (!opt.selected(m)) continue;
            try {
                if (opt.pack) pack_model(m, opt);
                else download_model(m, hf, opt);
            } catch (const NotAccepted& e) {
                say(m.id, "skipped: ", e.what());
                ++failures;
            } catch (const std::exception& e) {
                say(m.id, "FAILED: ", e.what());
                ++failures;
            }
        }
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << g_log_tag << " error: " << e.what() << "\n";
        return 2;
    }
}
