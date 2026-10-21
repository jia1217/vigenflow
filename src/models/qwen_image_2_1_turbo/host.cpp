// Ubuntu only: run.exe embeds the prebuilt host runtime from src/lib.
#include "../embedded_host.h"

int main(int argc, const char *argv[]) {
    return vigenflow::run_embedded_host("qwen_image_2_1_turbo_host_main", argc, argv);
}
