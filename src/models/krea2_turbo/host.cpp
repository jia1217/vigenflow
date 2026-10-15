// Launcher for the Krea 2 Turbo NPU host. The pipeline is in the prebuilt
// src/lib/libhost_krea2_turbo.so.
extern "C" int krea2_turbo_host_main(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
    return krea2_turbo_host_main(argc, argv);
}
