#include "cli.hpp"
#include "file_utils.hpp"
#include "server_config.hpp"
#include "tcp_server.hpp"
#include "worker.hpp"

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
    server::config::initialize_default_paths(argc > 0 ? argv[0] : nullptr);

    bool output_dir_specified = false;

    try {
        // Parse CLI flags first; this fills values in server::config.
        const int cli_result = server::parse_cli(argc, argv, output_dir_specified);
        if (cli_result == 0) return 0;
        if (cli_result < 0) return 1;

        // When generated images are not kept, default to the system temp directory
        // unless the user explicitly requested an output path.
        const bool output_dir_is_temporary =
            !server::config::keep_images && !output_dir_specified;
        if (output_dir_is_temporary) {
            server::config::output_dir = fs::temp_directory_path().string();
        }
        if (!fs::path(server::config::output_dir).is_absolute()) {
            server::config::output_dir =
                fs::absolute(server::config::output_dir).lexically_normal().string();
        }

        // Startup fails early if the output directory cannot be created.
        server::ensure_output_dir();

        // Prepare any configured model weights that can be bootstrapped locally.
        server::ensure_model_weights_ready(server::config::model_id);

        // This call blocks forever while the HTTP server accepts requests.
        server::run_server(output_dir_is_temporary);
    } catch (const std::exception& e) {
        std::cerr << "fatal error: " << e.what() << "\n";
        return 1;
    }
}
