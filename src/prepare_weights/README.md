# prepare_base.exe

`prepare_base.exe` downloads the pre-packed NPU weights of every model from Hugging Face and unpacks them
into the folders the model workers read (`all_model_weights/<id>`). It is the `prepare/` tool of the
release package; no Hugging Face login and no BF16 → BFP16 conversion are needed.

## Build and run (Ubuntu)

```bash
sudo apt install g++-13 nlohmann-json3-dev curl
bash build_ubuntu.sh               # -> prepare_base.exe
./prepare_base.exe                 # download + unpack all enabled models into ../../all_model_weights/<id>
./prepare_base.exe --only flux2_klein_4B --weights_root /some/dir
./prepare_base.exe --help          # all options
```

## Build and run (Windows)

```bat
vcpkg install nlohmann-json:x64-windows
build_windows.bat                                  &rem -> vigenflow_win\prepare_base.exe
vigenflow_win\prepare_base.exe --config prepare_models.jsonc --only flux2_klein_4B
```

`build_windows.bat` uses MSVC (it finds Visual Studio with `vswhere`) and `nlohmann/json.hpp` from vcpkg
(`VCPKG_ROOT`, default `C:\dev\vcpkg`). The exe has the C runtime linked in, so in the package it needs only
`prepare_models.jsonc` beside it (`prepare\`). It starts Windows' own `curl.exe` (Windows 10 1803 and later)
directly, not through `cmd.exe`, so paths with spaces, `&` or `%` work. Its manifest allows paths longer
than 260 characters where `LongPathsEnabled` is set in the registry; without that, the download cache names
each commit by 12 hex digits instead of 40 to keep its paths short.

## Check

```bash
python3 tests/test_prepare_base.py ./prepare_base.exe              # Ubuntu
python tests\test_prepare_base.py vigenflow_win\prepare_base.exe   # Windows
```

The check runs the real tool against a local fake Hugging Face server (no network, models or login):
packing, a resumed multi-shard download with parallel byte ranges, a byte-exact unpack, cleanup, the
config found from the exe's folder, the Krea license prompt and the one-run-per-weights-root lock.

`curl` (8.3 or newer) does the downloads. Paths do not depend on the current directory:
`prepare_models.jsonc` is read from next to `prepare_base.exe`, and its `"weights_root"` is relative to that
file. In the release package this folder is `prepare/`, so `../../all_model_weights` is the folder next to
the package, the one `server_running/vgf-serve` reads. `--config` / `--weights_root` on the command line are
relative to the current directory as usual.

## Models

| Model ID / local folder | Packed repository | Size |
|---|---|---:|
| `flux1_schnell` | [Kelsey1217/FLUX.1-schnell-NPU-BFP16](https://huggingface.co/Kelsey1217/FLUX.1-schnell-NPU-BFP16) | 27 GB |
| `flux2_klein_4B` | [Kelsey1217/FLUX.2-klein-4B-NPU-BFP16](https://huggingface.co/Kelsey1217/FLUX.2-klein-4B-NPU-BFP16) | 8 GB |
| `krea2_turbo` | [Kelsey1217/Krea-2-Turbo-NPU-BFP16](https://huggingface.co/Kelsey1217/Krea-2-Turbo-NPU-BFP16) | 19 GB |
| `z_image_turbo` | [Kelsey1217/Z-Image-Turbo-NPU-BFP16](https://huggingface.co/Kelsey1217/Z-Image-Turbo-NPU-BFP16) | 12 GB |

All four repositories are public. FLUX.2 text-to-image and image editing share the `flux2_klein_4B` weights.

The Krea 2 Turbo weights are under the Krea 2 Community License (not Apache 2.0), so before downloading
them `prepare_base.exe` prints the license and use-policy links and waits for you to type `I agree`:

```text
[prepare_base] krea2_turbo: this model has its own license: Krea 2 Community License Agreement (https://huggingface.co/Kelsey1217/Krea-2-Turbo-NPU-BFP16/blob/main/LICENSE.pdf) and Krea Acceptable Use Policy (https://www.krea.ai/krea-2-use-policy)
[prepare_base] krea2_turbo: type I agree to accept these terms and continue: I agree
```

Any other answer, or no input (not a terminal), skips the model: nothing is downloaded and the exit code is
1. `--accept_license` accepts the terms without the prompt, for unattended runs. A folder that is already up
to date is not asked again. `LICENSE.pdf` and `Notice` are unpacked into `all_model_weights/krea2_turbo/`;
keep them with any copy.

## Choosing models: prepare_models.jsonc

- **Leave a model out:** set `"enabled": false` in its entry; it is then skipped on every run.
- **One run only:** `--only ID[,ID...]` takes just those ids and ignores `enabled`.
- **Other packed weights:** `packed_repo`, `packed_revision` (branch, tag or commit; `"main"` follows the
  latest upload) and `packed_path` (the model's folder in the repo, `""` = the repo root) choose what is
  downloaded. `id` is the output folder under the weights root (the `--weights_path` of that model's
  `run.exe`).
- `recipe`, `source` and `revision` describe the checkpoint the packed weights were converted from. They
  identify a model folder that is already up to date, and a `krea/` source turns on the license prompt.
- Check where the folders go before the first run: if your weights are somewhere else, change
  `"weights_root"` or pass `--weights_root`; otherwise the models are downloaded again into the new place.

## What it does

- Packed format `vgf-npu-weights-1` (`src/packed_format.hpp`): ~5 GB `safetensors` shards in which each
  file of the model folder is one `U8` tensor named by its relative path, plus an index. Unpacking
  restores the folder byte for byte, `prepare_manifest.json` included. The shards open with the standard
  `safetensors` library.
- Large files download over 8 parallel connections (`--connections N`), each writing its byte range
  straight into place. Files already in the Hugging Face cache (`~/.cache/huggingface/hub`) are used in
  place; the rest go to `--cache_dir` (default `<weights_root>/.hf_sources`).
- Shards unpack while the next one downloads, and each one is deleted right after it is unpacked
  (`"keep_sources": true` or `--keep_sources` keeps them). At most two finished shards wait on disk, so a
  model needs its own size plus up to ~10 GB free while installing.
- Writes `<id>.partial/` and swaps it in when complete; the old folder stays until then. An interrupted
  run resumes: `<id>.partial/.prepare_base_progress.json` lists the shards already unpacked, and a partly
  downloaded shard continues where it stopped.
- A folder is up to date when the same packed files (Hugging Face blob ids) were unpacked before, or when
  it was converted from the same source commit and recipe. A folder without `prepare_manifest.json` is
  only replaced with `--force`.
- Only one run per weights root at a time (`flock`, `LockFileEx` on Windows, on
  `<weights_root>/.prepare.lock`, released however the process exits).
- A token (`--hf_token`, `$HF_TOKEN` or the `huggingface-cli login` token) is only needed for private
  repos; it reaches curl through the environment, never the command line or a file.

## Packing (maintainers)

`--pack` does the reverse: it packs `<weights_root>/<id>` into `<pack_dir>/<packed_path>` (default
`<weights_root>/../packed_weights`; `--shard_gb` sets the shard size) for uploading to a packed repository:

```bash
./prepare_base.exe --pack --only flux2_klein_4B
hf upload-large-folder Kelsey1217/FLUX.2-klein-4B-NPU-BFP16 ../../packed_weights/flux2_klein_4B --repo-type=model
```

Clients see the new blob ids and update on their next run.
