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
        for (const auto& model : server::config::models) {
            if (!fs::is_regular_file(model.worker_exe)) {
                throw std::runtime_error(std::string(model.id) + " worker not found: " +
                                         model.worker_exe + " (build this worker first)");
            }
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
