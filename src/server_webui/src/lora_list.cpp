#include "lora_list.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace server {

// Same rules as the hosts' lora_config.hpp: JSON with // comments, "name" and "file" required,
// "file" relative to the list's folder, unique names other than "none".
std::optional<std::vector<LoraInfo>> read_loras(const config::ModelConfig& model) {
    // run.exe starts in its own folder and reads loras.jsonc from there by default.
    const fs::path path = fs::path(model.worker_exe).parent_path() / "loras.jsonc";
    std::ifstream in(path);
    if (!in) return std::nullopt;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<LoraInfo> out;
    try {
        const json j = json::parse(text, nullptr, true, /*ignore_comments=*/true);
        for (const auto& e : j.at("loras")) {
            LoraInfo x;
            x.name = e.at("name").get<std::string>();
            fs::path file = e.at("file").get<std::string>();
            if (file.is_relative()) file = (path.parent_path() / file).lexically_normal();
            x.file = file.filename().string();
            x.scale = e.value("scale", 1.0f);
            x.trigger = e.value("trigger", std::string());
            x.license = e.value("license", std::string());
            x.enabled = e.value("enabled", false);
            std::error_code ec;
            x.status = fs::exists(file, ec) ? "ok" : e.value("source", std::string()).empty() ? "missing" : "download";
            if (x.name.empty() || x.name == "none") {
                throw std::runtime_error("an entry needs a \"name\" other than \"none\"");
            }
            for (const auto& o : out) {
                if (o.name == x.name) throw std::runtime_error("two entries are named \"" + x.name + "\"");
            }
            out.push_back(std::move(x));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(path.string() + ": " + e.what());
    }
    return out;
}

}  // namespace server
