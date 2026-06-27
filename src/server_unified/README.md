# Unified VigenFlow Server

This folder keeps one server source tree for both Ubuntu and Windows. The runtime code is shared; only the compile command and worker executable name are platform-specific.

## Ubuntu

```bash
bash build_ubuntu.sh
./vigenflow_ubuntu/vgf-serve -p 1128
```

Ubuntu builds `vigenflow_ubuntu/vgf-serve` with `g++`. The model worker binaries are expected to be named `run.exe` under `exe_models/...`.

### Ubuntu Release Zip

```bash
VERSION=0.1.2 bash package_ubuntu.sh
```

This creates `release/vigenflow_0.1.2_ubuntu_amd64.zip` with one top-level folder:

```text
vigenflow_0.1.2_ubuntu_amd64/
  exe_models/
  model_weights/
  npu_files/
  flux_klein_edit_lora_models.jsonc
  flux_klein_lora_models.jsonc
  vgf-serve
  z_image_bf16_lora_models.jsonc
  z_image_q41_lora_models.jsonc
```

By default, the script copies `exe_models` and `npu_files` from `all_host_inference/src`, and uses `deb_lib/vigenflow/model_weights` if `all_host_inference/src/model_weights` is not present. Override paths when needed:

```bash
EXE_MODELS_DIR=/path/to/exe_models \
NPU_FILES_DIR=/path/to/npu_files \
MODEL_WEIGHTS_DIR=/path/to/model_weights \
VERSION=0.1.2 \
bash package_ubuntu.sh
```

## Windows

```bat
build_windows.bat
exe_server\vgf-serve.exe -p 1128
```

Windows builds `exe_server\vgf-serve.exe` with MSVC. The model worker binaries are expected to be named `host.exe` under `exe_models\...`.

The script also copies the matching Boost filesystem/system DLLs from vcpkg into `exe_server` when they are available.

If vcpkg is not installed at `C:\dev\vcpkg`, set `VCPKG_ROOT` before running:

```bat
set VCPKG_ROOT=D:\path\to\vcpkg
build_windows.bat
```

If your Boost filesystem library has a different filename, set `BOOST_FILESYSTEM_LIB`:

```bat
set BOOST_FILESYSTEM_LIB=boost_filesystem-vc143-mt-x64-1_89.lib
build_windows.bat
```

## Auto Dispatcher

```bash
bash build_vgf_serve.sh
```

The dispatcher calls `build_windows.sh` on Windows/Git Bash and `build_ubuntu.sh` on Ubuntu.

## CI/CD

The repo-level workflow `.github/workflows/ci-cd.yml` builds this folder on Ubuntu and Windows, runs `vgf-serve --help`, uploads artifacts, and creates release packages when a version tag such as `v1.0.0` is pushed.
