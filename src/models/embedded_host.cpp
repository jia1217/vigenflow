// Linux release workers carry their prebuilt model runtime in a read-only ELF
// section. Load it through an anonymous memory file; no library is extracted to
// disk and the original src/lib directory is only needed at build time.
#include "embedded_host.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
extern const unsigned char vgf_host_start[];
extern const unsigned char vgf_host_end[];
}

namespace vigenflow {
int run_embedded_host(const char* entry_point, int argc, const char* argv[]) {
    const int fd = memfd_create("vigenflow-host", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd == -1) {
        std::fprintf(stderr, "run.exe: cannot create memory file for embedded host: %s\n",
                     std::strerror(errno));
        return 127;
    }

    const auto size = reinterpret_cast<std::uintptr_t>(vgf_host_end)
                    - reinterpret_cast<std::uintptr_t>(vgf_host_start);
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t written = write(fd, vgf_host_start + offset, size - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            std::fprintf(stderr, "run.exe: cannot copy embedded host into memory: %s\n",
                         written < 0 ? std::strerror(errno) : "short write");
            close(fd);
            return 127;
        }
        offset += static_cast<std::size_t>(written);
    }
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == -1) {
        std::fprintf(stderr, "run.exe: cannot seal embedded host memory: %s\n", std::strerror(errno));
        close(fd);
        return 127;
    }

    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    // GLOBAL preserves the symbol visibility of the former directly linked
    // host. Keep the handle loaded until process exit so runtime callbacks and
    // static destructors remain valid after host_main returns.
    void* host = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!host) {
        std::fprintf(stderr, "run.exe: cannot load embedded host: %s\n", dlerror());
        close(fd);
        return 127;
    }
    close(fd);

    dlerror();
    auto entry = reinterpret_cast<int (*)(int, const char*[])>(dlsym(host, entry_point));
    if (const char* error = dlerror()) {
        std::fprintf(stderr, "run.exe: cannot find embedded entry point %s: %s\n", entry_point, error);
        return 127;
    }
    return entry(argc, argv);
}
}  // namespace vigenflow
