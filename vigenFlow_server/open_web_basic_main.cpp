
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <nlohmann/json.hpp>

#include <regex>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <ctime>
#include <mutex>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace bp = boost::process;
namespace fs = std::filesystem;
using json = nlohmann::json;
using tcp = net::ip::tcp;

// --------------------------------------------------
// Global defaults
// --------------------------------------------------
std::string g_weights_path = "../";
std::string g_npu_files_path = "../npu_files/Z-Image-Turbo";

std::string g_output_dir = "../images";
std::string g_public_base_url = "http://127.0.0.1:11281";
std::string g_model_id = "local-image-1";
bool g_keep_images = true; // Default to keeping images
std::string g_exe_path =
    "../run.exe";
std::string g_workdir =
    ".";

unsigned short g_port = 11281;

// serialize NPU / run.exe access
std::mutex g_infer_mutex;

// --------------------------------------------------
// Params
// --------------------------------------------------
struct GenParams {
    int H = 1024;
    int W = 1024;
    int steps = 4;
    int seed = 42;
    std::string prompt =
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.";
};

// --------------------------------------------------
// Helpers
// --------------------------------------------------
static void ensure_output_dir() {
    std::error_code ec;
    fs::create_directories(g_output_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create output dir: " + g_output_dir + " : " + ec.message());
    }
}

static bool is_safe_filename(const std::string& name) {
    if (name.empty()) return false;
    if (name.find("..") != std::string::npos) return false;
    if (name.find('/') != std::string::npos) return false;
    if (name.find('\\') != std::string::npos) return false;
    return true;
}

static std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<size_t>(size));
    if (size > 0 && !file.read(data.data(), size)) {
        throw std::runtime_error("failed to read file: " + path);
    }

    return data;
}

static std::string base64_encode(const std::vector<char>& data) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data.data());
    const size_t len = data.size();

    while (i + 2 < len) {
        unsigned int n = (static_cast<unsigned int>(bytes[i]) << 16) |
                         (static_cast<unsigned int>(bytes[i + 1]) << 8) |
                         static_cast<unsigned int>(bytes[i + 2]);

        out.push_back(table[(n >> 18) & 63]);
        out.push_back(table[(n >> 12) & 63]);
        out.push_back(table[(n >> 6) & 63]);
        out.push_back(table[n & 63]);
        i += 3;
    }

    if (i < len) {
        unsigned int n = static_cast<unsigned int>(bytes[i]) << 16;
        out.push_back(table[(n >> 18) & 63]);

        if (i + 1 < len) {
            n |= static_cast<unsigned int>(bytes[i + 1]) << 8;
            out.push_back(table[(n >> 12) & 63]);
            out.push_back(table[(n >> 6) & 63]);
            out.push_back('=');
        } else {
            out.push_back(table[(n >> 12) & 63]);
            out.push_back('=');
            out.push_back('=');
        }
    }

    return out;
}

// static std::string get_response_format(const std::string& body) {
//     try {
//         json j = json::parse(body);
//         if (j.contains("response_format") && j["response_format"].is_string()) {
//             return j["response_format"].get<std::string>();
//         }
//     } catch (...) {
//     }
//     return "url";
// }
static std::string get_response_format(const std::string& body) {
    try {
        json j = json::parse(body);
        if (j.contains("response_format") && j["response_format"].is_string()) {
            return j["response_format"].get<std::string>();
        }
    } catch (...) {
    }
    return "b64_json";
}
// --------------------------------------------------
// Parse request
// --------------------------------------------------
GenParams parse_request(const std::string& body)
{
    std::cout << "========== RAW HTTP BODY ==========\n";
    std::cout << body << "\n";
    std::cout << "===================================\n";

    GenParams params;

    try {
        json j = json::parse(body);

        if (j.contains("prompt") && j["prompt"].is_string()) {
            params.prompt = j["prompt"].get<std::string>();
        }

        if (j.contains("steps") && j["steps"].is_number_integer()) {
            params.steps = j["steps"].get<int>();
        }

        if (j.contains("seed") && j["seed"].is_number_integer()) {
            params.seed = j["seed"].get<int>();
        }

        if (j.contains("size") && j["size"].is_string()) {
            std::string size = j["size"].get<std::string>();

            if (size == "3:4") {
                params.H = 1024; params.W = 768;
            }
            else if (size == "4:3") {
                params.H = 768; params.W = 1024;
            }
            else if (size == "9:16") {
                params.H = 1024; params.W = 576;
            }
            else if (size == "16:9") {
                params.H = 576; params.W = 1024;
            }
            else if (size == "1:1") {
                params.H = 1024; params.W = 1024;
            }
            else if (size == "1024x1024") {
                params.H = 1024; params.W = 1024;
            }
            else if (size == "1024x1536") {
                params.H = 1536; params.W = 1024;
            }
            else if (size == "1536x1024") {
                params.H = 1024; params.W = 1536;
            }
        }

        return params;
    }
    catch (...) {
        // fallback parsing below
    }

    if (body.find("3:4") != std::string::npos) {
        params.H = 1024; params.W = 768;
    }
    else if (body.find("4:3") != std::string::npos) {
        params.H = 768; params.W = 1024;
    }
    else if (body.find("9:16") != std::string::npos) {
        params.H = 1024; params.W = 576;
    }
    else if (body.find("16:9") != std::string::npos) {
        params.H = 576; params.W = 1024;
    }
    else if (body.find("1:1") != std::string::npos) {
        params.H = 1024; params.W = 1024;
    }

    std::regex steps_regex(R"("steps"\s*:\s*(\d+))");
    std::smatch match_steps;
    if (std::regex_search(body, match_steps, steps_regex)) {
        params.steps = std::stoi(match_steps[1].str());
    }

    std::regex seed_regex(R"("seed"\s*:\s*(\d+))");
    std::smatch match_seed;
    if (std::regex_search(body, match_seed, seed_regex)) {
        params.seed = std::stoi(match_seed[1].str());
    }

    std::string prompt_key = "\"prompt\":";
    size_t p_key_pos = body.find(prompt_key);
    if (p_key_pos != std::string::npos) {
        size_t v_start = body.find("\"", p_key_pos + prompt_key.length());
        size_t v_end = body.find("\"", v_start + 1);
        if (v_start != std::string::npos && v_end != std::string::npos) {
            params.prompt = body.substr(v_start + 1, v_end - v_start - 1);
        }
    }

    return params;
}

// --------------------------------------------------
// Run worker
// --------------------------------------------------
std::string run_worker(const GenParams& p)
{
    ensure_output_dir();

    static std::atomic<uint64_t> job_counter{0};
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    std::string filename = "output_" + std::to_string(ms) + "_" + std::to_string(++job_counter) + ".png";
    std::string fullpath = g_output_dir + "/" + filename;

    std::cout << "[INFO] Request received. Starting AI model (" << p.W << "x" << p.H << ")...\n";
    std::cout << "[INFO] Using model weights: " << g_weights_path << "\n";
    std::cout << "[INFO] Using npu files: " << g_npu_files_path << "\n";
    std::cout << "[INFO] Using random seed data: " << p.seed << "\n";
    std::cout << "[INFO] Model request: " << p.W << "x" << p.H << " for " << p.steps << " steps.\n";
    std::cout << "[INFO] Prompt: " << p.prompt << "\n";
    std::cout << "[INFO] Output path: " << fullpath << "\n";

    auto start_time = std::chrono::steady_clock::now();

    // bp::child c(
    //     g_exe_path,
    //     g_weights_path,
    //     g_npu_files_path,
    //     std::to_string(p.seed),
    //     std::to_string(p.H),
    //     std::to_string(p.W),
    //     std::to_string(p.steps),
    //     p.prompt,
    //     fullpath,
    //     bp::start_dir = g_workdir
    // );
    bp::child c(
        g_exe_path,
        "--weights_path", g_weights_path,
        "--npu_files_path", g_npu_files_path,
        "--seed", std::to_string(p.seed),
        "--image_H", std::to_string(p.H),
        "--image_W", std::to_string(p.W),
        "--step", std::to_string(p.steps), // Note: singular 'step' matching your main.cpp
        "--prompt", p.prompt,
        "--output_path", fullpath,
        bp::start_dir = g_workdir
    );

    c.wait();

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    if (c.exit_code() != 0) {
        std::cerr << "[ERROR] Worker failed with exit code " << c.exit_code() << "\n";
        throw std::runtime_error("worker failed with exit code " + std::to_string(c.exit_code()));
    }

    if (!fs::exists(fullpath)) {
        throw std::runtime_error("worker finished but output image was not created");
    }

    std::cout << "[INFO] Generation completed in " << elapsed_ms << " ms\n";
    std::cout << "[INFO] Waiting for next request\n";

    return fullpath;
}

// --------------------------------------------------
// Response helpers
// --------------------------------------------------
http::response<http::string_body>
make_json_response(http::status status, unsigned version, bool keep_alive, const json& body)
{
    http::response<http::string_body> res{status, version};
    res.set(http::field::content_type, "application/json");
    res.keep_alive(keep_alive);
    res.body() = body.dump();
    res.prepare_payload();
    return res;
}

http::response<http::vector_body<char>>
make_image_response(unsigned version, bool keep_alive, std::vector<char>&& data, const std::string& mime)
{
    http::response<http::vector_body<char>> res{http::status::ok, version};
    res.set(http::field::content_type, mime);
    res.keep_alive(keep_alive);
    res.body() = std::move(data);
    res.prepare_payload();
    return res;
}

// --------------------------------------------------
// Handle request
// --------------------------------------------------
http::message_generator
handle_request(http::request<http::string_body>&& req)
{
    const bool keep_alive = req.keep_alive();
    const std::string target = std::string(req.target());
    std::cout << "[REQ] " 
    << req.method_string() << " " 
    << std::string(req.target()) << "\n";
    try {
       
        if (req.method() == http::verb::get && target == "/health") {
            return make_json_response(
                http::status::ok,
                req.version(),
                keep_alive,
                json{{"status", "ok"}}
            );
        }

        if (req.method() == http::verb::get && target == "/v1/models") {
            json body = {
                {"object", "list"},
                {"data", json::array({
                    {
                        {"id", g_model_id},
                        {"object", "model"},
                        {"created", 0},
                        {"owned_by", "local"}
                    }
                })}
            };
            return make_json_response(http::status::ok, req.version(), keep_alive, body);
        }

        if (req.method() == http::verb::get && target.rfind("/images/", 0) == 0) {
            std::string filename = target.substr(std::string("/images/").size());

            if (!is_safe_filename(filename)) {
                return make_json_response(
                    http::status::bad_request,
                    req.version(),
                    keep_alive,
                    json{{"error", "invalid filename"}}
                );
            }

            std::string fullpath = g_output_dir + "/" + filename;
            if (!fs::exists(fullpath)) {
                return make_json_response(
                    http::status::not_found,
                    req.version(),
                    keep_alive,
                    json{{"error", "image not found"}}
                );
            }

            auto data = read_binary_file(fullpath);

            std::string ext = fs::path(fullpath).extension().string();
            std::string mime = "image/png";
            if (ext == ".jpg" || ext == ".jpeg") mime = "image/jpeg";
            else if (ext == ".webp") mime = "image/webp";

            return make_image_response(req.version(), keep_alive, std::move(data), mime);
        }
        
        if (req.method() == http::verb::post && target == "/v1/images/generations") {
            GenParams params = parse_request(req.body());
            std::string response_format = get_response_format(req.body());

            std::string image_path;
            {
                std::lock_guard<std::mutex> lock(g_infer_mutex);
                image_path = run_worker(params);
            }

            std::string filename = fs::path(image_path).filename().string();
            std::string public_url = g_public_base_url + "/images/" + filename;

            auto data = read_binary_file(image_path);

            json item = {
                {"b64_json", base64_encode(data)},
                {"revised_prompt", params.prompt}
            };

            // --- NEW CLEANUP LOGIC ---
            // Only delete if the flag is false AND we aren't using URLs
            if (!g_keep_images && response_format != "url") {
                std::error_code ec;
                fs::remove(image_path, ec);
                if (ec) {
                    std::cerr << "[WARNING] Failed to delete image: " << ec.message() << "\n";
                } else {
                    std::cout << "[INFO] Cleaned up temporary image: " << image_path << "\n";
                }
            }
            // -------------------------

            json body = {
                {"created", static_cast<long long>(std::time(nullptr))},
                {"data", json::array({item})}
            };

            return make_json_response(http::status::ok, req.version(), keep_alive, body);
        }
        // if (req.method() == http::verb::post && target == "/v1/images/generations") {
        //     GenParams params = parse_request(req.body());
        //     std::string response_format = get_response_format(req.body());

        //     std::string image_path;
        //     {
        //         std::lock_guard<std::mutex> lock(g_infer_mutex);
        //         image_path = run_worker(params);
        //     }

        //     std::string filename = fs::path(image_path).filename().string();
        //     std::string public_url = g_public_base_url + "/images/" + filename;

        //     auto data = read_binary_file(image_path);

        //     json item = {
        //         {"b64_json", base64_encode(data)},
        //         {"revised_prompt", params.prompt}
        //     };

        //     json body = {
        //         {"created", static_cast<long long>(std::time(nullptr))},
        //         {"data", json::array({item})}
        //     };

        //     return make_json_response(http::status::ok, req.version(), keep_alive, body);
           
        // }
        
        
        if (req.method() == http::verb::post && target == "/v1/chat/completions") {
            json body = {
                {"id", "chatcmpl-local"},
                {"object", "chat.completion"},
                {"created", static_cast<long long>(std::time(nullptr))},
                {"model", g_model_id},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", " "}
                        }},
                        {"finish_reason", "stop"}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", 0},
                    {"completion_tokens", 0},
                    {"total_tokens", 0}
                }}
            };
        
            return make_json_response(
                http::status::ok,
                req.version(),
                keep_alive,
                body
            );
        }
        if (target == "/v1/models" || target == "/v1/images/generations" ||
            target == "/health" || target.rfind("/images/", 0) == 0) {
            return make_json_response(
                http::status::method_not_allowed,
                req.version(),
                keep_alive,
                json{{"error", "method not allowed"}}
            );
        }
        std::cout << "[404] "
        << req.method_string() << " "
        << target << "\n";
        return make_json_response(
            http::status::not_found,
            req.version(),
            keep_alive,
            json{{"error", "not found"}}
        );
    }
    catch (std::exception& e) {
        return make_json_response(
            http::status::internal_server_error,
            req.version(),
            keep_alive,
            json{{"error", e.what()}}
        );
    }
}

// --------------------------------------------------
// Session
// --------------------------------------------------
void do_session(tcp::socket socket)
{
    beast::flat_buffer buffer;
    beast::error_code ec;

    for (;;) {
        http::request<http::string_body> req;
        http::read(socket, buffer, req, ec);

        if (ec == http::error::end_of_stream) {
            break;
        }

        if (ec) {
            std::cerr << "read error: " << ec.message() << "\n";
            return;
        }

        bool keep_alive = req.keep_alive();

        beast::write(socket, handle_request(std::move(req)), ec);

        if (ec) {
            std::cerr << "write error: " << ec.message() << "\n";
            return;
        }

        if (!keep_alive) {
            break;
        }
    }

    socket.shutdown(tcp::socket::shutdown_send, ec);
}

// --------------------------------------------------
// Main
// --------------------------------------------------
void print_usage(const char* prog_name) {
    std::cout
        << "Usage: " << prog_name << " [options]\n\n"
        << "Options:\n"
        << "  --weights_path <path>      Path to model weights\n"
        << "  --npu_files_path <path>    Path to NPU files\n"
        << "  --output_dir <path>        Output image directory\n"
        << "  --public_base_url <url>    Public base URL\n"
        << "  --model_id <id>            Model ID\n"
        << "  --port <port>              Server port\n"
        << "  --exe_path <path>          Worker executable path\n"
        << "  --workdir <path>           Working directory\n"
        << "  -h, --help                 Show this help message\n"
        << "  --keep_images <true/false> Keep generated images on disk (default: true)\n";
}
int main(int argc, char* argv[])
{
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
    
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            else if (arg == "--weights_path") {
                if (i + 1 < argc) g_weights_path = argv[++i];
                else { std::cerr << "Error: --weights_path requires an argument.\n"; return 1; }
            }
            else if (arg == "--npu_files_path") {
                if (i + 1 < argc) g_npu_files_path = argv[++i];
                else { std::cerr << "Error: --npu_files_path requires an argument.\n"; return 1; }
            }
            else if (arg == "--output_dir") {
                if (i + 1 < argc) g_output_dir = argv[++i];
                else { std::cerr << "Error: --output_dir requires an argument.\n"; return 1; }
            }
            else if (arg == "--public_base_url") {
                if (i + 1 < argc) g_public_base_url = argv[++i];
                else { std::cerr << "Error: --public_base_url requires an argument.\n"; return 1; }
            }
            else if (arg == "--model_id") {
                if (i + 1 < argc) g_model_id = argv[++i];
                else { std::cerr << "Error: --model_id requires an argument.\n"; return 1; }
            }
            else if (arg == "--port") {
                if (i + 1 < argc) g_port = static_cast<unsigned short>(std::stoi(argv[++i]));
                else { std::cerr << "Error: --port requires an argument.\n"; return 1; }
            }
            else if (arg == "--exe_path") {
                if (i + 1 < argc) g_exe_path = argv[++i];
                else { std::cerr << "Error: --exe_path requires an argument.\n"; return 1; }
            }
            else if (arg == "--workdir") {
                if (i + 1 < argc) g_workdir = argv[++i];
                else { std::cerr << "Error: --workdir requires an argument.\n"; return 1; }
            }
            else if (arg == "--keep_images") {
                if (i + 1 < argc) {
                    std::string val = argv[++i];
                    g_keep_images = (val == "true" || val == "1");
                }
                else { std::cerr << "Error: --keep_images requires an argument (true/false).\n"; return 1; }
            }
            else {
                std::cerr << "Unknown argument: " << arg << "\n\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        ensure_output_dir();

        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {net::ip::make_address("0.0.0.0"), g_port}};

        std::cout << "Server running on http://0.0.0.0:" << g_port << "\n";
        std::cout << "Weights Path: " << g_weights_path << "\n";
        std::cout << "NPU Files Path: " << g_npu_files_path << "\n";
        std::cout << "Executable Path: " << g_exe_path << "\n";
        std::cout << "Working Directory: " << g_workdir << "\n";
        std::cout << "Image Output Dir: " << g_output_dir << "\n";
        std::cout << "Public Base URL: " << g_public_base_url << "\n";
        std::cout << "Model ID: " << g_model_id << "\n";

        for (;;) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            std::thread{do_session, std::move(socket)}.detach();
        }
    }
    catch (std::exception& e) {
        std::cerr << "fatal error: " << e.what() << "\n";
        return 1;
    }
}