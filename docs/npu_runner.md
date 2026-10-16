# NPU Test Runner (self-hosted)

GitHub's runners have no NPU, so `.github/workflows/npu-test.yml` runs on a Windows PC with a Ryzen AI
NPU that is registered as a self-hosted runner with the label `npu`. The job:

1. builds the Windows package with `src/server_unified/package_windows.ps1`;
2. unpacks it next to the PC's `all_model_weights` folder;
3. generates one 1024 × 1024 image with each model through `vgf-serve`
   (`src/server_unified/tests/npu_check.py`; the edit model edits the FLUX.2 image);
4. removes the unpacked package and uploads the images, the server log and `summary.json` as the
   `vigenflow-npu-check` artifact.

It runs on pushes to `main` that change `src/`, or by hand from **Actions → Vigenflow NPU Test →
Run workflow** (optionally for some models only). It never runs for pull requests, only one NPU job
runs at a time, and it is skipped until the repository variable `VGF_NPU_RUNNER` is `true`.

## 1. Repository settings (do this first)

The repository is public, and a self-hosted runner runs whatever a workflow tells it to on your PC.

- **Settings → Actions → General → Approval for running fork pull request workflows from
  contributors:** choose **Require approval for all external contributors**. A pull request can
  change the workflow files, so never approve a run for a pull request that touches `.github/` or
  scripts you have not read.
- Only people with write access can push to `main`, which is what starts this workflow.

## 2. Prepare the PC

- The AMD NPU driver (**NPU Compute Accelerator Device** in Device Manager).
- Visual Studio 2022 with the C++ tools, Git, and Python 3 on `PATH`.
- vcpkg at `C:\dev\vcpkg` (or set `VCPKG_ROOT`) with the server and downloader dependencies:
  ```bat
  vcpkg install boost-asio boost-beast boost-filesystem boost-gil boost-process boost-system libpng nlohmann-json stb --triplet x64-windows
  ```
- The model weights in a folder named `all_model_weights` (as `prepare_base.exe` makes it). The
  runner must be able to read it and to write to the folder that contains it, where the package is
  unpacked for the test.

## 3. Register the runner

On GitHub: **Settings → Actions → Runners → New self-hosted runner → Windows, x64**. Follow the
download commands it shows, in a folder of its own (for example `C:\actions-runner`, not inside a
repository checkout or the weights folder). Then configure it with the extra label `npu`, using the
token from that page:

```powershell
.\config.cmd --url https://github.com/jia1217/vigenflow --token <TOKEN> --labels npu --unattended
```

Tell the jobs where the weights are: create `.env` in the runner folder with

```text
VGF_WEIGHTS_ROOT=C:\path\to\all_model_weights
```

## 4. Run it and switch it on

- Start the runner in a terminal with `.\run.cmd`; stop it with Ctrl+C. The NPU runs one
  application at a time, so run the runner only while you are not using the NPU yourself. (It can
  also run as a Windows service, `config.cmd --runasservice`, but NPU access from a service account
  has not been tried.)
- **Settings → Secrets and variables → Actions → Variables → New repository variable:**
  `VGF_NPU_RUNNER` = `true`. Set it to anything else to switch the job off; while it is on and the
  runner is stopped, new jobs wait in the queue (GitHub cancels them after 24 hours).

## Running the check by hand

Without GitHub, on the same PC:

```bat
cd vigenflow\src\server_unified
powershell -ExecutionPolicy Bypass -File tests\run_npu_check.ps1 -WeightsRoot C:\path\to\all_model_weights
```

`-SkipBuild` uses the binaries that are already built, `-Models flux2-klein-4B,flux2-klein-4B-edit`
checks only some models, and `-Out <folder>` keeps the images somewhere else. For a package that is
already unpacked next to its weights (Windows or Ubuntu), run the check directly:

```bash
python3 tests/npu_check.py /path/to/vigenflow_<version>_<platform>_amd64
```
