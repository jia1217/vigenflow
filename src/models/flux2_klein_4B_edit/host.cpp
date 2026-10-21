// Windows links the host DLL; Ubuntu embeds the host runtime in run.exe.
#ifdef _WIN32
extern "C" int flux2_klein_4B_edit_host_main(int argc, const char *argv[]);
#else
#include "../embedded_host.h"
#endif

int main(int argc, const char *argv[]) {
#ifdef _WIN32
    return flux2_klein_4B_edit_host_main(argc, argv);
#else
    return vigenflow::run_embedded_host("flux2_klein_4B_edit_host_main", argc, argv);
#endif
}
