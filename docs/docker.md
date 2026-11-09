# Run VigenFlow with Docker or Podman

The image holds the [Beta 4 Ubuntu package](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.4):
`vgf-serve` with its built-in browser UI, the seven model workers, their NPU files, the weight
downloader, and the XRT runtime they need. On the host you need only the NPU driver and a container
engine. The model weights and LoRAs stay outside the image, in a volume that you fill once.

## Quick start

On a Ryzen AI 300-series laptop with Ubuntu 24.04, one command sets everything up:

```bash
curl -fsSL https://raw.githubusercontent.com/jia1217/vigenflow/main/docker/install.sh | bash
```

The installer:

1. Checks for the NPU driver. If it is missing, the installer offers to install it; reboot then, and
   run the same command again.
2. Installs Docker if neither Docker nor Podman is installed.
3. Asks which models to download. Press Enter for FLUX.2-klein-4B (8 GB), the fastest,
   or choose **5** for Qwen Image 2.1 Turbo text-to-image and editing (one shared 19.6 GB download).
4. Starts VigenFlow and prints the address to open, **http://localhost:11283/**.

In the browser UI, pick a model you downloaded, type a prompt and generate. The image size and the
LoRA styles are on the same page, and the editing models take a reference image.

Run the same command again to update VigenFlow, update the weights or add models. To run it without
questions, set the models on `bash`, for example
`curl -fsSL .../install.sh | VIGENFLOW_MODELS=flux2_klein_4B,z_image_turbo bash`; the other settings
are listed at the top of [`docker/install.sh`](../docker/install.sh).

`VIGENFLOW_MODELS=qwen_image_2_1_turbo` selects both Qwen tasks; `VIGENFLOW_MODELS=all` selects all
five weight sets. For unattended downloads, `VIGENFLOW_ACCEPT_QWEN_LICENSE=yes` accepts Qwen's terms
and `VIGENFLOW_ACCEPT_KREA_LICENSE=yes` accepts Krea's terms. Each setting applies only to its own
model. Without these settings, the downloader asks on a terminal and skips a model requiring
acceptance when there is no terminal.


### Updating from Beta 3

Run the installer again. It pulls the new image and restarts the `vigenflow` container. Your
downloaded weights stay in the `vigenflow-weights` volume and are reused. Use the built-in UI at
http://localhost:11283/ from now on: the Docker install no longer starts OpenWebUI. The installer
leaves the OpenWebUI container of the Beta 3 install (`vigenflow-webui`) as it is and prints the
commands that remove it and its chats. The OpenAI-compatible API is still at
`http://127.0.0.1:11283/v1`.

The rest of this page does the same steps by hand.

## Requirements

- An AMD Ryzen AI 300-series processor with an XDNA 2 NPU, as for the release packages. The model
  workers use AVX-512, which these processors have.
- A Linux host with the AMD XDNA driver. Install `amdxdna-dkms` from the
  [Lemonade PPA](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable) (Ubuntu), then reboot:

  ```bash
  sudo add-apt-repository ppa:lemonade-team/stable
  sudo apt update
  sudo apt install amdxdna-dkms
  sudo reboot
  ```

  The host does not need XRT; the image has its own copy. Check that the NPU is there with
  `ls -l /dev/accel/accel0`.
- Docker Engine, or Podman. Docker Desktop does not work: it runs containers in a virtual machine
  that cannot see the NPU.
- About 85 GB of disk space for all five weight sets, plus up to 10 GB while they download. Qwen
  alone needs about 19.6 GB and serves both tasks.

## Get the image

Pull the published image:

```bash
docker pull ghcr.io/jia1217/vigenflow:latest
```

`ghcr.io/jia1217/vigenflow:0.2.0-beta.4` is the same image under its version.

Or build it yourself from a checkout of this repository:

```bash
docker build -f docker/Dockerfile -t ghcr.io/jia1217/vigenflow:latest .
```

The build downloads the Beta 4 Ubuntu release zip, checks its sha256, and lays out the package
as it is in the zip. It then checks the package's file hashes and the runtime libraries of every
program. When the build machine has AVX-512, it also starts every worker with `--help`. It needs no
NPU.

## Download the model weights (one time)

The weights go into a named volume, `vigenflow-weights`:

```bash
docker run --rm -it \
  -v vigenflow-weights:/opt/vigenflow/all_model_weights \
  ghcr.io/jia1217/vigenflow:latest prepare
```

`prepare` runs `prepare_base.exe` and takes the same options. Bare `prepare` downloads all five
weight sets (about 85 GB); `prepare --only flux2_klein_4B,z_image_turbo` downloads only some. The
model names are `flux1_schnell`, `flux2_klein_4B`, `z_image_turbo`, `krea2_turbo` and
`qwen_image_2_1_turbo`. The editing models use the FLUX.2 and Qwen weights.

Keep `-it`: before it downloads Krea 2 Turbo or Qwen, `prepare` shows that model's
license terms and waits for you to type `I agree`. To accept the terms for all
models selected in that invocation without a prompt, add `--accept_license`.

Run the same command again at any time to update the weights; it downloads only the models that
changed.

If you already downloaded the weights with a release package, use that folder instead of the volume:
replace `-v vigenflow-weights:...` with `-v /path/to/all_model_weights:/opt/vigenflow/all_model_weights`
here and in the commands below.

## Start the server

```bash
docker run -d --name vigenflow \
  --device /dev/accel \
  --ulimit memlock=-1:-1 \
  -p 127.0.0.1:11283:11283 \
  -v vigenflow-weights:/opt/vigenflow/all_model_weights \
  --restart unless-stopped \
  ghcr.io/jia1217/vigenflow:latest
```

Then open **http://localhost:11283/**.

- `--device /dev/accel` gives the container the NPU.
- `--ulimit memlock=-1:-1` lets XRT lock the memory of the NPU buffers. Docker's default limit, often
  8 MB, is too low.
- `-p 127.0.0.1:11283:11283` makes the server reachable from this machine only. `vgf-serve` has no
  authentication, so publish it on other addresses only on a network you trust.

Server options go after the image name, as for `vgf-serve`. For example, if you did not download
Z-Image-Turbo, the default model, choose another default:

```bash
docker run ... ghcr.io/jia1217/vigenflow:latest --model flux2-klein-4B
```

Run `docker run --rm ghcr.io/jia1217/vigenflow:latest --help` to see all the options.

Follow the server log with `docker logs -f vigenflow`, and stop the server with `docker stop vigenflow`.
The log shows the timings of each image.

### LoRA styles

Each model's styles are listed in the browser UI. A style's weights download the first time you use
it, into `<model>/loras/` in the weights volume, so they are kept across updates. Most Civitai
downloads need a [Civitai API key](https://civitai.com/user/account): pass it to the container with
`-e CIVITAI_TOKEN=<your key>` (and `-e HF_TOKEN=<token>` for gated Hugging Face LoRAs). The installer
passes `CIVITAI_TOKEN` and `HF_TOKEN` on when they are set:

```bash
curl -fsSL .../install.sh | CIVITAI_TOKEN=<your key> bash
```

The style lists are the package's `loras.jsonc` files, one per worker, in
`/opt/vigenflow/app/server_running/exe_models/<worker>/` in the container. To add or remove styles,
copy a list out, edit it, and mount your copy over it:

```bash
docker cp vigenflow:/opt/vigenflow/app/server_running/exe_models/flux1_schnell/loras.jsonc .
# edit loras.jsonc, then add this to the docker run command and start a new container:
#   -v "$PWD/loras.jsonc:/opt/vigenflow/app/server_running/exe_models/flux1_schnell/loras.jsonc"
```

Edits to a mounted list show up after a browser refresh, without a restart.

### Check the API

```bash
curl http://127.0.0.1:11283/v1/models

curl http://127.0.0.1:11283/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "flux2-klein-4B", "prompt": "a red fox sitting in fresh snow, photo", "response_format": "b64_json"}' \
  | python3 -c 'import sys, json, base64; open("fox.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The second command writes `fox.png` after about as long as the README's model table lists for the
model.

Qwen text-to-image and editing go through the same server. With the Qwen weights downloaded
(`prepare --only qwen_image_2_1_turbo`), generate and then edit a PNG:

```bash
curl -fS http://127.0.0.1:11283/v1/images/generations \
  -H 'Content-Type: application/json' -H 'Accept: image/png' \
  -d '{"model":"qwen-image-2.1-turbo","prompt":"a red fox in snow"}' -o generated.png
curl -fS http://127.0.0.1:11283/v1/images/edits \
  -H 'Accept: image/png' -F model=qwen-image-2.1-turbo-edit \
  -F 'prompt=make it night time' -F image=@generated.png -o edited.png
```

### Podman

The same commands work with `podman` in place of `docker`. When Podman runs without root, it cannot
raise the locked-memory limit above your own, so keep your limit instead, and keep your groups so the
container can open the NPU:

```bash
podman run -d --name vigenflow \
  --device /dev/accel \
  --ulimit host \
  --group-add keep-groups \
  -p 127.0.0.1:11283:11283 \
  -v vigenflow-weights:/opt/vigenflow/all_model_weights \
  ghcr.io/jia1217/vigenflow:latest
```

Your own limit is what `vgf-serve` gets when you run the release package directly. If `ulimit -l`
prints a small number, such as 8192, raise it in `/etc/security/limits.d/` and log in again. On
Fedora and other systems with SELinux, add `:Z` to a host-folder mount
(`-v /path/to/all_model_weights:/opt/vigenflow/all_model_weights:Z`).

## Docker Compose

[`docker/compose.yaml`](../docker/compose.yaml) starts the server with the options above. Run these
from the repository root:

```bash
docker compose -f docker/compose.yaml run --rm vigenflow prepare    # once
docker compose -f docker/compose.yaml up -d
```

Then open http://localhost:11283/. Compose passes `CIVITAI_TOKEN` and `HF_TOKEN` from your shell
to the server. To build the image instead of pulling it, run
`docker compose -f docker/compose.yaml build` first.

For Qwen only, select its weights and server default explicitly:

```bash
docker compose -f docker/compose.yaml run --rm vigenflow prepare --only qwen_image_2_1_turbo
VIGENFLOW_MODEL=qwen-image-2.1-turbo docker compose -f docker/compose.yaml up -d
```

## Troubleshooting

### `No NPU in this container: /dev/accel/accel0 is missing`

Start the container with `--device /dev/accel`. If the host has no `/dev/accel/accel0` either,
the AMD XDNA driver is not loaded: install `amdxdna-dkms` as in [Requirements](#requirements)
and reboot.

### Every image fails with `run.exe failed with exit code 134`

Look at `sudo dmesg` on the host. Lines such as `Message callback ret -22` or `Channel in bad state`
mean that the host's NPU driver does not work with the XRT in the image: the driver built into some
kernels does not work with XRT 2.25. Install `amdxdna-dkms` from the Lemonade PPA on the host, as in
[Requirements](#requirements), and keep it updated, because the image always has the PPA's latest XRT.

Errors that mention allocating a buffer object, or `Cannot allocate memory`, mean that the
locked-memory limit is too low: add `--ulimit memlock=-1:-1` (Docker) or raise your own limit
(rootless Podman).

### `Illegal instruction`, or exit code 132

The model workers need a CPU with AVX-512, such as the Ryzen AI 300 series.

### A LoRA style fails to download

Most Civitai LoRAs need an API key: start the container with `-e CIVITAI_TOKEN=<your key>`, as in
[LoRA styles](#lora-styles).

### `Nothing is mounted at /opt/vigenflow/all_model_weights`

`prepare` refuses to download into the container itself, because the weights would be lost when the
container is removed. Add the `-v` option shown above.

## For maintainers: publishing the image

[`docker/Dockerfile`](../docker/Dockerfile) builds the image from a published Ubuntu release zip,
which it pins with two build arguments, `VIGENFLOW_VERSION` and `PACKAGE_SHA256`. For a new
release, set both. The asset's digest comes from the GitHub API without downloading it:

```bash
gh api repos/jia1217/vigenflow/releases/tags/v<version> --jq '.assets[] | "\(.name) \(.digest)"'
```

[`.github/workflows/docker-server.yml`](../.github/workflows/docker-server.yml) builds the image on
pull requests that change the Docker files, and pushes it to `ghcr.io/<owner>/vigenflow` when a `v*`
tag is pushed or the workflow is run by hand. It is published under the `VIGENFLOW_VERSION` it holds
(for example `0.2.0-beta.4`) and as `latest`. Every build that is not cached downloads the release
zip, which adds one to that asset's download count.

Before publishing, CI checks installer selection and license handling, builds the
image, then starts its server without NPU inference to verify health, the browser UI, all seven
model IDs, entrypoint options and package checksums. The same checks can be run
locally:

```bash
python3 docker/tests/test_install.py
python3 docker/tests/check_image.py ghcr.io/jia1217/vigenflow:latest
```

The image check needs Docker access but no model weights or NPU. Pass
`--engine 'sudo docker'` or `--engine podman` when needed. GitHub's runners do not always have
AVX-512; on those, the image build and the image check skip starting the workers. Actual image
generation and editing still require testing on the supported NPU.

A new GHCR package is private. After the first push, open the package's settings on GitHub and set
its visibility to public, so that `docker pull` works without logging in.
