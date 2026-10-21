// Windows links the host DLL; Ubuntu embeds the host runtime in run.exe.
#ifdef _WIN32
extern "C" int z_image_turbo_host_main(int argc, const char *argv[]);
#else
#include "../embedded_host.h"
#endif

int main(int argc, const char *argv[]) {
#ifdef _WIN32
    return z_image_turbo_host_main(argc, argv);
#else
    return vigenflow::run_embedded_host("z_image_turbo_host_main", argc, argv);
#endif
}
