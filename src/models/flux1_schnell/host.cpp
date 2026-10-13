// Launcher for the FLUX.1-schnell NPU host. The pipeline is in the prebuilt
// src/lib/libhost_flux1_schnell.so.
extern "C" int flux1_schnell_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return flux1_schnell_host_main(argc, argv);
}
