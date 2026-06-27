#pragma once

namespace server {

// Print supported model examples and all command-line options.
void print_usage(const char* prog_name);

// Parse argv into server::config.
// Returns 0 for --help, 1 for success, and -1 for invalid arguments.
int parse_cli(int argc, char* argv[], bool& output_dir_specified);

}  // namespace server
