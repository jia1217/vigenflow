#pragma once

namespace server {

// Start the blocking TCP accept loop and spawn one detached thread per connection.
void run_server(bool output_dir_is_temporary);

}  // namespace server
