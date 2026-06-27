# Build from Source with Docker

This guide explains how to use the repo-level `Dockerfile` to create a Linux
build container and compile VigenFlow from the source tree.

The Docker image is intended as a reproducible Ubuntu build environment. It
installs the compiler, CMake/Ninja, Boost, JSON, DRM, UUID, and optional XRT
development headers needed by the source build.

## Requirements

- Docker or Docker Desktop
- A local checkout of this repository
- AMD XDNA/XRT driver support on the host if you plan to run NPU workloads
- Model weights when creating a runnable release package

## Build the Docker Image

Run these commands from the repository root, where the `Dockerfile` is located:

```bash
cd /path/to/vigenflow

docker build \
  -t vigenflow-build-env:ubuntu24.04 \
  --build-arg BASE_IMAGE=ubuntu:24.04 \
  --build-arg UBUNTU_PPA=ppa:lemonade-team/stable \
  --build-arg INSTALL_XRT_DEV=1 \
  .
```

The GitHub Actions workflow uses the same Ubuntu 24.04 base image and XRT PPA.

### Build Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `BASE_IMAGE` | `ubuntu:24.04` | Base Linux image used for the build environment. |
| `UBUNTU_PPA` | empty | Optional Ubuntu PPA to add before installing packages. Use `ppa:lemonade-team/stable` when `libxrt-dev` is not available from the base image repositories. |
| `BACKPORTS` | empty | Optional Debian backports suite when building from a Debian base image. |
| `INSTALL_XRT_DEV` | `1` | Installs `libxrt-dev` when set to `1`. Set to `0` only if you are building pieces that do not need XRT headers. |

## Start the Build Container

On Linux or macOS:

```bash
docker run --rm -it \
  -v "$PWD":/workspace/vigenflow \
  -w /workspace/vigenflow \
  vigenflow-build-env:ubuntu24.04
```

On Windows PowerShell:

```powershell
docker run --rm -it `
  -v "${PWD}:/workspace/vigenflow" `
  -w /workspace/vigenflow `
  vigenflow-build-env:ubuntu24.04
```

All commands below are run inside the container unless noted otherwise.

## Compile the Unified Server

The unified server source lives in `src/server_unified`.

```bash
cd src/server_unified
bash build_vgf_serve.sh
```

On Ubuntu, the dispatcher calls `build_ubuntu.sh` and creates:

```text
src/server_unified/vigenflow_ubuntu/vgf-serve
```

Check that the binary starts:

```bash
./vigenflow_ubuntu/vgf-serve --help
```

You can also call the Ubuntu build script directly:

```bash
bash build_ubuntu.sh
```

## Compile Model Worker Launchers

Model launcher sources live under `src/models`. The Makefiles compile small
`run.exe` launcher binaries and link them against the prebuilt shared libraries
in `src/lib`.

The source tree must contain the matching shared libraries before these builds
will succeed, for example:

```text
src/lib/libhost_flux.so
src/lib/libhost_flux_edit.so
src/lib/libhost_zimage_bf16_lora.so
src/lib/libhost_zimage_q41.so
src/lib/libhost_zimage_q41_lora_fused.so
```

Build the launchers you need:

```bash
cd /workspace/vigenflow

make -C src/models/flux.2-klein-4B-npu check
make -C src/models/flux.2-klein-4B-edit-npu check
make -C src/models/z-image-turbo/bf16_lora check
make -C src/models/z-image-turbo/q41 check
make -C src/models/z-image-turbo/q41_lora_fused check
```

The generated binaries are written under `src/exe_models`, for example:

```text
src/exe_models/flux.2-klein-4B/run.exe
src/exe_models/flux.2-klein-4B-edit/run.exe
src/exe_models/z-image-turbo/bf16/run.exe
src/exe_models/z-image-turbo/q41/run.exe
src/exe_models/z-image-turbo/lora_add_q41/run.exe
```

## Optional CMake Build for a Model Launcher

Each model launcher folder also includes a `CMakeLists.txt`. This is useful if
you prefer CMake/Ninja over the Makefile.

Example:

```bash
cd /workspace/vigenflow/src/models/z-image-turbo/q41
cmake -S . -B build -G Ninja
cmake --build build
cmake --build build --target check
```

## Create an Ubuntu Release Package

After the server and model launchers build, you can create a release ZIP:

```bash
cd /workspace/vigenflow/src/server_unified
VERSION=0.1.2 bash package_ubuntu.sh
```

The package script expects these runtime assets to exist:

```text
src/exe_models
src/npu_files
src/model_weights
```

If your assets are stored somewhere else, pass explicit paths:

```bash
EXE_MODELS_DIR=/path/to/exe_models \
NPU_FILES_DIR=/path/to/npu_files \
MODEL_WEIGHTS_DIR=/path/to/model_weights \
VERSION=0.1.2 \
bash package_ubuntu.sh
```

The ZIP is created under:

```text
src/server_unified/release/
```

## Run the Built Server

For normal NPU execution, run the package or built binaries on a host with the
AMD XDNA driver installed.

From the repository tree:

```bash
cd /workspace/vigenflow/src/server_unified
./vigenflow_ubuntu/vgf-serve
```

The default OpenWebUI-compatible endpoint is:

```text
http://127.0.0.1:2048/v1
```

## Troubleshooting

### `Unable to locate package libxrt-dev`

Build the image with the XRT PPA:

```bash
docker build \
  -t vigenflow-build-env:ubuntu24.04 \
  --build-arg UBUNTU_PPA=ppa:lemonade-team/stable \
  --build-arg INSTALL_XRT_DEV=1 \
  .
```

### `Missing src/lib/libhost_*.so`

The model launcher Makefiles link against prebuilt host shared libraries in
`src/lib`. Copy or build the required `.so` file before running `make`.

### `package_ubuntu.sh` cannot find `model_weights`

Download or prepare the model weights, then rerun the script with:

```bash
MODEL_WEIGHTS_DIR=/path/to/model_weights bash package_ubuntu.sh
```

### Permission errors after building in Docker

Files created inside the container may be owned by root on the host. On Linux,
you can start the container with your host UID/GID:

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v "$PWD":/workspace/vigenflow \
  -w /workspace/vigenflow \
  vigenflow-build-env:ubuntu24.04
```
