#pragma once

namespace server {

// Start the blocking TCP accept loop and spawn one detached thread per connection.
void run_server();

}  // namespace server
