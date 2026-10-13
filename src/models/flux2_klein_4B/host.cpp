// Launcher for the FLUX.2-klein-4B NPU host. The pipeline is in the prebuilt
// src/lib/libhost_flux2_klein_4B.so.
extern "C" int flux2_klein_4B_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return flux2_klein_4B_host_main(argc, argv);
}
