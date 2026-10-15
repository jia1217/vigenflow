#include "cli.hpp"
#include "file_utils.hpp"
#include "server_config.hpp"
#include "tcp_server.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

void set_runtime_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void configure_worker_runtime_env() {
    set_runtime_env("OMP_WAIT_POLICY", "PASSIVE");
    set_runtime_env("KMP_AFFINITY", "granularity=fine,compact,1,0");
}

}  // namespace

int main(int argc, char* argv[]) {
    configure_worker_runtime_env();
    try {
        server::config::initialize_default_paths(argc > 0 ? argv[0] : nullptr);
        const int cli_result = server::parse_cli(argc, argv);
        if (cli_result == 0) return 0;
        if (cli_result < 0) return 1;

        server::config::make_paths_absolute();
        for (auto& model : server::config::models) {
            if (server::config::is_worker(model.worker_exe)) continue;
            const std::string missing = std::string(model.id) + " worker not found: " + model.worker_exe;
#ifdef _WIN32
            // Windows serves the models whose workers are built; requests for the others return 400.
            model.available = false;
            std::cerr << "warning: " << missing << "; " << model.id << " is not served\n";
#else
            throw std::runtime_error(missing + " (build this worker first)");
#endif
        }
        if (!server::config::default_model().available) {
            throw std::runtime_error("default model " + server::config::default_model_id +
                                     " has no worker; build it or choose another with --model");
        }

        // Startup fails early if the output directory cannot be created.
        server::ensure_output_dir();

        // This call blocks forever while the HTTP server accepts requests.
        server::run_server();
    } catch (const std::exception& e) {
        std::cerr << "fatal error: " << e.what() << "\n";
        return 1;
    }
}
