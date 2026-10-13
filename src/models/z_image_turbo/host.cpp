// Launcher for the Z-Image-Turbo NPU host. The pipeline is in the prebuilt
// src/lib/libhost_z_image_turbo.so.
extern "C" int z_image_turbo_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return z_image_turbo_host_main(argc, argv);
}
