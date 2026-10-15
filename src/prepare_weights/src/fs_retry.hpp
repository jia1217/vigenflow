#pragma once
// Windows: right after a file or folder is written, antivirus or the search indexer can still have it open, and
// renaming it (or replacing a file it has open) fails with access denied for a moment. Every rename in the
// tools goes through rename_retrying, which waits that out for up to a minute.
#include <chrono>
#include <filesystem>
#include <system_error>
#include <thread>

namespace prep {

inline void rename_retrying(const std::filesystem::path& from, const std::filesystem::path& to) {
    for (int attempt = 1;; ++attempt) {
        std::error_code ec;
        std::filesystem::rename(from, to, ec);
        if (!ec) return;
#ifdef _WIN32
        const bool busy = ec.value() == 5 || ec.value() == 32;  // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION
#else
        const bool busy = false;
#endif
        if (!busy || attempt == 120) throw std::filesystem::filesystem_error("rename", from, to, ec);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace prep
