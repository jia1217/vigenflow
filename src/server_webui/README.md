# vgf-serve with the built-in web UI (Beta 4)

The source of the `vgf-serve` in the
[v0.2.0 Beta 4 release](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.4): the
OpenAI-compatible server plus the browser UI it serves at `/` (model picker, prompt, image size,
LoRA styles, editing with a before/after slider, PNG download). The page is in `src/web_ui.hpp`;
the LoRA lists the UI shows come from each worker's `loras.jsonc` (`src/lora_list.cpp`).

`../server_unified` is the earlier server source, used by the Beta 3 packaging scripts.

## Build (Ubuntu)

```bash
sudo apt install build-essential libboost-dev libboost-filesystem-dev libpng-dev nlohmann-json3-dev libstb-dev
bash build_ubuntu.sh            # -> vigenflow_ubuntu/vgf-serve
```

Stripped (`strip`), the result is the same file as `server_running/vgf-serve` in the Beta 4 Ubuntu
package.

## Run

`vgf-serve` finds the workers in `exe_models/<model>/` next to itself, the NPU files in
`../all_npu_files/` and the weights in `../../all_model_weights/`, as in the release package. To try
a build, copy it into an extracted package's `server_running/` folder and start it there:

```bash
./vgf-serve --model flux1-schnell   # then open http://127.0.0.1:11283/
```

`./vgf-serve --help` lists the options (port, default model, seed, steps, keeping images).

## Tests (no NPU or weights needed)

```bash
# Server with fake one-shot workers: API, edits, chat, size/LoRA arguments, cleanup;
# --ui also drives the page in headless Chrome (needs google-chrome, node and node-ws).
python3 tests/test_balanced_only.py --ui

# Edit sizes, cropping and request parsing on the CPU.
g++ -std=c++17 -O2 -Isrc tests/edit_sizes_test.cpp src/input_image.cpp src/request_parser.cpp \
  src/server_config.cpp src/lora_list.cpp src/file_utils.cpp -lpng -o /tmp/edit_sizes_test
/tmp/edit_sizes_test

# The page's editing view with stand-in images, against a running server (no worker is started).
node tests/ui_comparison_test.js http://127.0.0.1:11283/ /tmp/vgf-ui-comparison
```

`tests/ui_click_test.js` (`node tests/ui_click_test.js <url> <chrome profile dir> <screenshot.png>`)
clicks through a custom size and the LoRA boxes, then runs one real generation, so it needs a
server with the NPU and model weights.
