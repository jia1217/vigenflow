# VigenFlow Server

Serves four text-to-image NPU workers and the Flux2 image-edit worker from one API (see [Image editing](#image-editing)). The chat dropdown or image request's `model` selects the worker for that request.

| Model ID | Worker folder / accepted alias | Supported steps | Default seed |
|---|---|---|---|
| `z-image-turbo` | `z_image_turbo` | 4, 8 | 42 |
| `flux1-schnell` | `flux1_schnell` | 4 | 0 |
| `flux2-klein-4B` | `flux2_klein_4B` | 4 | 42 |
| `krea2-turbo` | `krea2_turbo` | 8 | 42 |
| `flux2-klein-4B-edit` (image editing) | `flux2_klein_4B_edit` | 4 | 42 |

All models produce one 1024x1024 image and default to 4 steps, except `krea2-turbo`, which is distilled for 8 steps and defaults to 8. Flux1 Schnell and Flux2 Klein are restricted to 4 steps and Krea 2 Turbo to 8; other step counts are rejected. An omitted `model` uses `z-image-turbo`, or the server's `--model` selection. Unknown model IDs return 400; they never silently select another worker.

The model table and paths are in `server_config.cpp`. By default each worker is `../<worker folder>/run.exe` (a development tree), else `exe_models/<worker folder>/run.exe` next to `vgf-serve` (a package); weights are `../../all_model_weights/<weights folder>` and NPU files `../all_npu_files/<worker folder>`, relative to the `vgf-serve` folder. FLUX.2 Edit shares the `flux2_klein_4B` weights folder; **`/base` is appended to the NPU path for FLUX.2 text-to-image**. Startup prints the actual absolute paths for every served model. Every request gets its own model settings; choosing a model does not change another user's selection.

| Worker option | Comes from |
|---|---|
| `--weights_path` | selected model's resolved weights path |
| `--npu_files_path` | selected model's resolved NPU files path |
| `--output` | a new file in the server's `-o` directory: `.bmp`, or `.png` for `krea2-turbo` |
| `--prompt` | request `prompt` (omitted = worker's built-in demo prompt) |
| `--seed` | request `seed`, else selected model's default |
| `--steps` | request `steps`, else selected model's default |

The request `prompt` is passed to the selected worker's text encoder. It must be non-blank text of at most 16384 UTF-8 bytes, with no NUL characters. The workers' tokenizers can truncate longer prompts. `n` must be 1 and `size` must be `1024x1024` if provided. Other unsupported fields are ignored and logged.

In the release package the server is `server_running/vgf-serve`, so it reads the workers from `server_running/exe_models/`, the NPU files from `all_npu_files/` and the weights from `all_model_weights/` next to the package folder, independently of the working directory:

```text
vigenflow_<version>_ubuntu_amd64/
  server_running/vgf-serve
  server_running/exe_models/<model>/run.exe   model launchers (src/exe_models)
  server_running/lib/libhost_<model>.so        prebuilt host libraries (src/lib)
  all_npu_files/<model>/                       NPU kernels (src/npu_files)
  prepare/prepare_base.exe                     downloads the weights (src/prepare_weights)
all_model_weights/<model>/                     filled by prepare/prepare_base.exe
```

## Build (Ubuntu)

```bash
# Requires Boost, nlohmann-json, libpng and libstb (stb_image) development packages.
sudo apt install build-essential libboost-all-dev nlohmann-json3-dev libpng-dev libstb-dev
bash build_ubuntu.sh            # -> vigenflow_ubuntu/vgf-serve
```

`vgf-serve` only finds the workers, NPU files and weights in the package layout above, so build the package and run it from there:

```bash
# The launchers in ../exe_models are prebuilt; after changing one, rebuild it with make -C ../models/<model>.
VERSION=0.2.0-beta.2 bash package_ubuntu.sh
cd release/vigenflow_0.2.0-beta.2_ubuntu_amd64
./prepare/prepare_base.exe      # downloads the weights to ../all_model_weights (once)
cd server_running && ./vgf-serve -p 11283
```

`package_ubuntu.sh` builds `vgf-serve` and `prepare_base.exe` and writes `release/vigenflow_<VERSION>_ubuntu_amd64/` and its `.zip`. `EXE_MODELS_DIR`, `LIB_DIR` and `NPU_FILES_DIR` override where the workers, host libraries and NPU files come from; `SKIP_BUILD=1` packages the binaries that are already built.

Workers support base-model inference only; LoRA adapters and LoRA command-line options are not supported.

On Ubuntu all five worker executables must exist before starting the server (on Windows, see [Windows](#windows)). Each runs in its own folder. Use `--model flux1-schnell` to change the default model; all five remain available. `-e`, `-w`, `-n`, `--seed` and `--steps` override only that default model, regardless of argument order:

```bash
./vgf-serve -p 1128 --model flux1-schnell --seed 7 --steps 4
```

For example, a request explicitly selecting `z-image-turbo` still uses its own seed and steps defaults. To customize paths for several models, edit the path setup in `server_config.cpp`.

## Image editing

`flux2-klein-4B-edit` is always served next to the text-to-image models; no extra option is needed. `POST /v1/images/edits` uses it when the request names no model, and `/v1/images/generations` rejects it because it needs an input image. The edit worker is `exe_models/flux2_klein_4B_edit/run.exe`; it shares the Flux2 weights (`../../all_model_weights/flux2_klein_4B`) and gets the Flux2 NPU root `../all_npu_files/flux2_klein_4B`, whose `base/` and `edit/` folders it reads itself.

`./vgf-serve edit` is short for `./vgf-serve --model flux2-klein-4B-edit`: it makes the edit model the default, so `-e`, `-w`, `-n`, `--seed` and `--steps` apply to it, and generation requests without a `model` return 400.

An edit needs a prompt and one input image (PNG, JPEG, BMP or GIF, up to 40 megapixels). The NPU pipeline only accepts 1024x1024, so the server center-crops the image to a square and resizes it to 1024x1024 before starting the worker; the output is 1024x1024. When several images are sent, the first is edited.

```bash
# OpenAI-style multipart upload
curl -s http://127.0.0.1:1128/v1/images/edits -F model=flux2-klein-4B-edit \
  -F "prompt=change the dress color from white to red" -F image=@photo.png
# JSON: "image" is base64 or a data:image/...;base64 URL
{ printf '{"prompt":"make it night","image":"'; base64 -w0 photo.png; printf '"}'; } |
  curl -s http://127.0.0.1:1128/v1/images/edits -H 'Accept: image/png' --data-binary @- -o edited.png
```

Responses, `response_format`, `seed`, `steps`, `n`, timings and storage work as for `/v1/images/generations`; `size` may also be `auto`.

## Image storage settings

`-k <true|false>` controls saving. It defaults to **false**:

```bash
./vgf-serve -p 1128           # Do not save images (default)
./vgf-serve -p 1128 -k false  # Do not save images
./vgf-serve -p 1128 -k true   # Save images
```

Workers use temporary image files during generation; with `-k false` these are removed after use and the latest 32 image URLs are served from memory until restart or replacement. With `-k true`, generated PNGs stay in the output directory. Changing the setting does not delete previously saved images.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/v1/models` | Lists all five model IDs |
| POST | `/v1/chat/completions` | Generates an inline PNG for Native-mode image chat; JSON or SSE |
| POST | `/v1/images/generations` | Body `{"model":"flux1-schnell","prompt":"a cat","seed":42,"steps":4,"response_format":"b64_json"}`; all fields optional |
| POST | `/v1/images/edits` | Multipart `image`/`image[]` file or JSON base64 `image`, plus `prompt`; see [Image editing](#image-editing) |
| GET | `/images/<file>` | Serves a PNG from memory, or from disk with `-k true` |

```bash
curl -s -d '{"model":"flux2-klein-4B","prompt":"a cat","seed":7,"steps":4}' http://127.0.0.1:11283/v1/images/generations
```

The default response contains PNG bytes in `data[0].b64_json`. Set `"response_format":"url"` to receive a PNG URL instead. The server converts the worker's temporary BMP to a real PNG and removes the BMP. The `krea2-turbo` worker writes PNG itself (whatever the file name), so the server asks it for a `.png` and returns that file unchanged; `writes_png` in `server_config.cpp` marks such a worker.

For a direct image response, send `Accept: image/png`. The response is PNG bytes with `Content-Type: image/png` and `Content-Disposition: inline`:

```bash
curl -sS http://127.0.0.1:1128/v1/images/generations \
  -H 'Content-Type: application/json' -H 'Accept: image/png' \
  -d '{"prompt":"a cat","seed":42,"steps":4}' -o image.png
```

Timings describe different intervals: `metadata.8_E2E_runtime` is the worker-reported sum of measured inference stages; `metadata.9_Worker_wall_time` is elapsed process time including startup, loading and shutdown. Neither includes queue wait, PNG conversion or HTTP transfer. If the worker does not print its inference time, that field is omitted rather than replaced with process time. JSON metadata also includes `0_Model`; direct PNG responses identify the selected model in `X-Image-Model` and expose the same measurements in `Server-Timing` (`inference` and `worker`, in milliseconds).

Requests run one at a time because each `run.exe` uses the whole NPU. Invalid models, prompts, seeds, steps, sizes, image counts or JSON return 400.

## Open WebUI

1. Start the server with `./vgf-serve -p 1128`.
2. Add/refresh the chat connection `http://<server-address>:1128/v1`, with API Type **Chat Completions**. The model list now contains `z-image-turbo`, `flux1-schnell`, `flux2-klein-4B`, `krea2-turbo` and `flux2-klein-4B-edit`.
3. Set **Function Calling = Native** for the selected model and enable **Image** in the message input. Keep each model's **Image Generation** and **Builtin Tools** capabilities enabled.
4. Select any of the four text-to-image models in the chat dropdown and send a prompt such as `a cat`. For `flux2-klein-4B-edit`, see [Editing images in Open WebUI](#editing-images-in-open-webui).

The prompt is the text of your last message. Markup that Open WebUI adds to it, such as the `<attached_files>` list sent with uploads, is removed: every complete `<tag>...</tag>` or `<tag/>` element is dropped (if nothing else is left, the text inside the tags is kept), while other `<` text such as `I <3 cats` stays.

The server uses the chat request's model directly, then returns a Markdown image pointing to `/images/<file>.png`. This matches [Open WebUI's Markdown image renderer](https://github.com/open-webui/open-webui/blob/main/src/lib/components/chat/Messages/Markdown/MarkdownInlineTokens.svelte). After generation finishes, the complete image link arrives in one short SSE event; the chat no longer streams Base64 text. The browser fetches and displays the PNG from the same API host and port, which must be reachable from the browser. Generation timings are included in the response metadata and server log.

Chat and `response_format: "url"` images follow `keep_images` (default **false**). By default, temporary BMP/PNG files are removed after generation, and the latest **32 image URLs** are served from memory. Repeated image previews work while cached; URLs expire when evicted or the server restarts. Use `-k true` for permanent PNG files and chat images that survive a restart. The image API still supports `b64_json` and direct PNG responses.

The Native-mode `generate_image` tool being offered is the signal that Image is enabled. The server performs the generation itself, so Open WebUI's separate Images default model does not override the dropdown and there is no second generation request. Tool-result turns and background chat calls finish without starting another worker. These endpoints are image backends, not general text models; chat without Image enabled returns an empty acknowledgement.

### Editing images in Open WebUI

Start the server as usual (`./vgf-serve -p 1128`), refresh the connection so `flux2-klein-4B-edit` appears, and set it up like the other models (Native function calling; Image Generation, Builtin Tools and **Vision** capabilities on, so image uploads are allowed). In a chat with that model, enable **Image**, attach a picture and type the change, e.g. `change the dress color from white to red`. The reply shows the edited image.

The server edits the newest image in the conversation, so a follow-up message such as `now make it night` edits the previous result without re-attaching it. You can also create an image with a text-to-image model, switch the dropdown to `flux2-klein-4B-edit` and edit it. Earlier results are read from the server's image store, so with the default `-k false` only the latest 32 remain editable, and not after a restart. Without an attached or earlier image, the chat returns an error instead of starting the worker.

For Legacy/Default function calling, enable **Image Edit** in Admin Settings > Images with the OpenAI engine, base URL `http://<server-address>:1128/v1` and model `flux2-klein-4B-edit`; Open WebUI then sends attached images to `/v1/images/edits`.

Keep image generation enabled in Admin Settings > Images (OpenAI engine, base URL `http://<server-address>:1128/v1`, any listed default model, size `1024x1024`) so the Image control is available. **Legacy/Default function-calling mode still uses that separate Images model setting; use Native for dropdown switching.** Direct `/v1/images/generations` calls always select the JSON `model`, or the server default when omitted.

## Check

```bash
python3 test_server.py ./vigenflow_ubuntu/vgf-serve
```

On Windows run `python test_server.py vigenflow_win\vgf-serve.exe` after `build_windows.bat`; it adds the Windows worker lookup, missing workers and command-line quoting. The check uses the installed Pillow package and temporary fake workers; it verifies image pixels and orientation, raw/base64/URL responses, timings, model routing, model-specific paths/defaults, chat dropdown selection, short SSE image URLs, memory-cache eviction/restart, cleanup in no-save mode, persistent PNGs with `-k true`, serialized NPU access, failure handling, and the edit model (multipart/JSON edits, crop/resize of the input, chat follow-up edits, the `edit` shortcut) without using the NPU. Re-run the orientation check after upgrading Boost.GIL, since the current reader needs a top-down BMP workaround.

## Windows

```bat
vcpkg install boost-asio boost-beast boost-filesystem boost-gil boost-process boost-system libpng nlohmann-json stb --triplet x64-windows
build_windows.bat                              &rem -> vigenflow_win\vgf-serve.exe
python test_server.py vigenflow_win\vgf-serve.exe
```

`build_windows.bat` builds `vigenflow_win\vgf-serve.exe` with MSVC (Boost, libpng and stb from vcpkg; see `VCPKG_ROOT` / `BOOST_FILESYSTEM_LIB` in the script), copies the vcpkg DLLs it loads next to it (found with vcpkg's `applocal.ps1`: `libpng16.dll` and `zlib1.dll`, plus Boost DLLs with newer vcpkg Boost versions) and embeds `utf8.manifest`, so prompts and paths reach the workers as UTF-8. It also builds `build-windows\test_worker_stub.exe`, which the check uses to run its Python fake worker as `run.exe`. In the package, `vgf-serve.exe` also needs the VC++ runtime DLLs (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`) beside it unless the VC++ redistributable is installed.

The Windows package has the same layout as the Ubuntu one: `server_running\vgf-serve.exe`, `server_running\exe_models\<model>\run.exe` with each worker's DLLs, `all_npu_files\`, `prepare\prepare_base.exe`, and the weights in `all_model_weights\` next to the package folder. The Windows workers are built from the model hosts' Windows sources, which are not in this repository (`src/exe_models` holds the Ubuntu launchers). Differences from Ubuntu:

- The worker is `exe_models\<model>\run.exe` (a package), or `..\<model>\build-windows\Release\run.exe` (a development tree of the hosts' CMake builds). Only Windows executables count, so a Linux `run.exe` copied into the folder is ignored; an older package's `host.exe` is used when there is no Windows `run.exe`.
- A model without a worker is not served: startup prints a warning, `/v1/models` leaves it out and requests for it return 400. The server refuses to start only when the default model has no worker.
- A model whose weights are only in the package's own `all_model_weights\` (where 0.2.0-beta.1 put them) is read from there; otherwise the weights are expected next to the package folder, as on Ubuntu.
- The worker command line is quoted for the C runtime, so prompts with quotes, backslashes and newlines arrive unchanged.
