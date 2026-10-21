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

On Linux:

```bash
docker run --rm -it \
  -v "$PWD":/workspace/vigenflow \
  -w /workspace/vigenflow \
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

## Compile the Weight Downloader

`prepare_base.exe` downloads the pre-packed NPU weights from Hugging Face. Its
source lives in `src/prepare_weights`:

```bash
cd src/prepare_weights
bash build_ubuntu.sh
./prepare_base.exe --help
```

It is the `prepare/` folder of the release package; see
`src/prepare_weights/README.md` for its options.

## Compile Ubuntu Model Workers

Model worker sources live under `src/models`. The Makefiles embed the prebuilt
host runtimes from `src/lib` into each `run.exe`. The `.so` files are build inputs;
they are not needed alongside the resulting Ubuntu executables.

At startup, an embedded runtime is loaded through an anonymous Linux memory file
(`memfd_create` and `/proc/self/fd`). It is not extracted to disk. This bundles the
existing runtime without changing its model pipeline; it is not a fully static
build. Ubuntu runtime libraries and XRT must still be installed.

The source tree must contain the matching shared libraries before these builds
will succeed, for example:

```text
src/lib/libhost_flux1_schnell.so
src/lib/libhost_flux2_klein_4B.so
src/lib/libhost_flux2_klein_4B_edit.so
src/lib/libhost_z_image_turbo.so
src/lib/libhost_krea2_turbo.so
src/lib/libhost_qwen_image_2_1_turbo.so
src/lib/libhost_qwen_image_2_1_turbo_edit.so
```

Build the workers you need:

```bash
cd /workspace/vigenflow

make -C src/models/flux1_schnell check
make -C src/models/flux2_klein_4B check
make -C src/models/flux2_klein_4B_edit check
make -C src/models/z_image_turbo check
make -C src/models/krea2_turbo check
make -C src/models/qwen_image_2_1_turbo check
make -C src/models/qwen_image_2_1_turbo_edit check
```

The generated binaries are written under `src/exe_models`, for example:

```text
src/exe_models/flux1_schnell/run.exe
src/exe_models/flux2_klein_4B/run.exe
src/exe_models/flux2_klein_4B_edit/run.exe
src/exe_models/z_image_turbo/run.exe
src/exe_models/krea2_turbo/run.exe
src/exe_models/qwen_image_2_1_turbo/run.exe
src/exe_models/qwen_image_2_1_turbo_edit/run.exe
```

Check all seven executables in isolated folders containing no host libraries:

```bash
python3 src/models/check_ubuntu.py
```

## Refresh the Qwen Host Libraries

The two Qwen libraries in `src/lib` are built from the native host projects with
`make libhost`, like `libhost_krea2_turbo.so`. To rebuild them, install stripped
copies in `src/lib` and rebuild both workers:

```bash
QWEN_SOURCE_ROOT=/path/to/NPU_qwen bash src/models/install_qwen_workers.sh
```

The source root contains `NPU_qwen_turbo/` and `NPU_qwen_image/`; it defaults to
the sibling `NPU_qwen/` tree. `SKIP_BUILD=1` installs existing
`build/lib/libhost_<model>.so` files. The hosts need their native build
dependencies, including a PIC build of tokenizers-cpp (`make libhost` builds it
in `build_pic/`); the edit worker also loads libjpeg-turbo and OpenSSL at runtime. Both workers share the prepared
`qwen_image_2_1_turbo` weights and use `qwen_image/base` or `qwen_image/edit`
under the NPU bundle root. See [server setup](../src/server_unified/README.md#qwen-image-21-turbo).

## Compile Model Worker Launchers on Windows

Windows uses the same launcher sources for all seven models; on Windows they
call the host DLL instead of embedding the host runtime. Their prebuilt host
libraries are in `src/lib` next to the Ubuntu ones:
`host_<model>.dll` with its import library `host_<model>.lib`, and the runtime
DLLs they load (Boost, XRT `xrt_coreutil.dll`, LLVM OpenMP, zlib, libjpeg-turbo
for Qwen editing, and the VC++ runtime).

From a Developer Command Prompt, or any prompt with Visual Studio 2022
installed:

```bat
cd vigenflow\src\models
build_windows.bat                    &rem all seven launchers
build_windows.bat flux2_klein_4B     &rem or only some
..\exe_models_win\flux2_klein_4B\run.exe --help
```

The launchers are written to `src/exe_models_win/<model>/run.exe` (the folder
name keeps them apart from the Ubuntu ones). A launcher loads its
`host_<model>.dll` from `..\..\lib\` relative to its own folder, so
it runs in place in the repository and, with the same layout, in a Windows package:

```text
src/exe_models_win/<model>/run.exe  -> server_running/exe_models/<model>/run.exe
src/lib/*.dll                       -> server_running/lib/
```

If the DLL is missing, `run.exe` exits with code 127 and names the folder it
looked in. The launchers need no DLLs of their own: they use the static C
runtime.

`check_windows.ps1` checks the result without the NPU: each launcher imports
only `KERNEL32` and its host DLL, each host DLL exports its entry point, and
`src/lib` has every DLL they need that is not part of Windows (so a package
runs without the VC++ redistributable). On a CPU with AVX-512, which the host
DLLs are built for, it also runs each launcher with `--help`. The
`windows-build.yml` workflow runs both scripts on GitHub's Windows runners.

```bat
powershell -ExecutionPolicy Bypass -File check_windows.ps1
```

The host DLLs are built outside the repository from each model's Windows host
build (for Qwen, the `NPU_qwen_turbo` and `NPU_qwen_image` builds that their
`README_WINDOWS.md` describes), with the same compiler options as its `run.exe`,
and export `<model>_host_main`.

## Optional CMake Build for a Model Launcher

Each model launcher folder also includes a `CMakeLists.txt`. This is useful if
you prefer CMake/Ninja over the Makefile.

Example:

```bash
cd /workspace/vigenflow/src/models/z_image_turbo
cmake -S . -B build -G Ninja
cmake --build build
cmake --build build --target check
```

## Create an Ubuntu Release Package

After the model workers build, create a release ZIP. The script builds
`vgf-serve` and `prepare_base.exe` itself:

```bash
cd /workspace/vigenflow/src/server_unified
VERSION=0.2.0-beta.3 bash package_ubuntu.sh
```

It packages these runtime assets:

```text
src/exe_models    -> server_running/exe_models
src/npu_files     -> all_npu_files
```

If your assets are stored somewhere else, pass explicit paths:

```bash
EXE_MODELS_DIR=/path/to/exe_models \
NPU_FILES_DIR=/path/to/npu_files \
VERSION=0.2.0-beta.3 \
bash package_ubuntu.sh
```

The folder and its ZIP are created under:

```text
src/server_unified/release/vigenflow_0.2.0-beta.3_ubuntu_amd64/
src/server_unified/release/vigenflow_0.2.0-beta.3_ubuntu_amd64.zip
src/server_unified/release/vigenflow_0.2.0-beta.3_ubuntu_amd64.zip.sha256
```

The model weights are not packaged. `prepare/prepare_base.exe` downloads them
into `all_model_weights/`, next to the package folder.

## Create a Windows Release Package

On Windows, `package_windows.ps1` does the same with Visual Studio 2022 and
vcpkg (see `src/server_unified/README.md` and `src/prepare_weights/README.md`
for their dependencies). It builds `vgf-serve.exe`, `prepare_base.exe` and the
seven launchers, then writes the folder and its ZIP:

```bat
cd vigenflow\src\server_unified
powershell -ExecutionPolicy Bypass -File package_windows.ps1 -Version 0.2.0-beta.2
powershell -ExecutionPolicy Bypass -File package_windows.ps1 -Version 0.2.0-beta.2 -SkipBuild
```

`-SkipBuild` packages the binaries that are already built.

For the beta.2-style Windows layout used by the beta.3 package, supply seven
standalone worker folders instead of the repository's DLL launchers:

```powershell
.\package_windows.ps1 -Version 0.2.0-beta.3 -SkipBuild -StandaloneWorkersRoot C:\workers
```

`C:\workers\<model>\` must contain the native Windows `run.exe` and its runtime
DLLs for each of the seven models. The original five worker folders can be
reused from the beta.2 release. For Qwen, build `NPU_qwen_turbo` and
`NPU_qwen_image` as their `README_WINDOWS.md` describes, then stage them there
from `src/models` with
`.\install_qwen_workers_windows.ps1 -QwenSourceRoot C:\path\to\NPU_qwen -OutDir C:\workers -SkipBuild`.
Its default `-OutDir` is `src/exe_models_win`, where it would replace the Qwen
launchers.
This mode keeps every worker's DLLs beside its executable and omits
`server_running\lib`. It still packages the current server, downloader and NPU
bundles. The supplied native workers are not rebuilt by the packaging script.

Without `-StandaloneWorkersRoot`, the default mapping remains:

```text
src/server_unified/vigenflow_win  -> server_running (vgf-serve.exe and its DLLs)
src/exe_models_win                -> server_running/exe_models
src/lib/*.dll                     -> server_running/lib
src/npu_files                     -> all_npu_files
src/prepare_weights/vigenflow_win -> prepare
```

The output is `src/server_unified/release/vigenflow_<version>_windows_amd64/`
and its `.zip`. The weights again go to `all_model_weights\` next to the
extracted package folder.

To check a package on the NPU (one image per model), see
[npu_runner.md](npu_runner.md): `tests/run_npu_check.ps1` builds, unpacks and
checks a Windows package by hand, and the `npu-test.yml` workflow does the same
on a self-hosted runner.

## Run the Built Server

For normal NPU execution, run the package on a host with the AMD XDNA driver
installed. `vgf-serve` finds the workers, NPU files and weights relative to its
own folder, so run it from the package, not from `src/server_unified`:

```bash
cd src/server_unified/release/vigenflow_0.2.0-beta.3_ubuntu_amd64
./prepare/prepare_base.exe
cd server_running
./vgf-serve
```

The default OpenWebUI-compatible endpoint is:

```text
http://127.0.0.1:11283/v1
```

## Troubleshooting

### `permission denied while trying to connect to the Docker daemon socket`

Your user account cannot access `/var/run/docker.sock`. Use `sudo` for the
current build:

```bash
sudo docker build \
  -t vigenflow-build-env:ubuntu24.04 \
  --build-arg BASE_IMAGE=ubuntu:24.04 \
  --build-arg UBUNTU_PPA=ppa:lemonade-team/stable \
  --build-arg INSTALL_XRT_DEV=1 \
  .
```

Or add your user to the `docker` group, then open a new terminal session:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker ps
```

The `DEPRECATED: The legacy builder is deprecated` message is only a warning and
does not stop the image from building.

### `BuildKit is enabled but the buildx component is missing or broken`

Your Docker installation does not have the Buildx plugin. Either run the build
without `DOCKER_BUILDKIT=1`, or install Buildx and rerun the BuildKit command.

Compatible command without BuildKit:

```bash
docker build \
  -t vigenflow-build-env:ubuntu24.04 \
  --build-arg BASE_IMAGE=ubuntu:24.04 \
  --build-arg UBUNTU_PPA=ppa:lemonade-team/stable \
  --build-arg INSTALL_XRT_DEV=1 \
  .
```

On Docker CE installs, the Buildx package is usually available as:

```bash
sudo apt-get update
sudo apt-get install docker-buildx-plugin
```

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

The Ubuntu Makefiles embed prebuilt host shared libraries from `src/lib` into
the executables. Copy or build the required `.so` file before running `make`.
The packaged workers do not need these files at runtime. The package script
rejects older workers that still depend on an external `libhost_*.so`; rebuild
those workers before packaging.

### `package_ubuntu.sh` reports a missing worker

The Ubuntu package needs seven `src/exe_models/<model>/run.exe` workers,
because `vgf-serve` does not start without them. Build them with
`make -C src/models/<model>`.

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
