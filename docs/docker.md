# Run VigenFlow with Docker or Podman

The VigenFlow image holds everything the Ubuntu release package holds: `vgf-serve`, the five model
workers, their NPU files, the weight downloader, and the XRT runtime they need. On the host you need
only the NPU driver and a container engine. The model weights stay outside the image, in a volume
that you fill once.

## Quick start

On a Ryzen AI 300-series laptop with Ubuntu 24.04, one command sets everything up:

```bash
curl -fsSL https://raw.githubusercontent.com/jia1217/vigenflow/main/docker/install.sh | bash
```

The installer:

1. Checks for the NPU driver. If it is missing, the installer offers to install it; reboot then, and
   run the same command again.
2. Installs Docker if neither Docker nor Podman is installed.
3. Asks which models to download. Press Enter for FLUX.2-klein-4B (8 GB), the fastest.
4. Starts VigenFlow and OpenWebUI, and prints the address to open, http://localhost:8080.

In OpenWebUI, create an account, pick a VigenFlow model you downloaded, turn on **Image** under
**Integrations** in the message box, and type what you want to see.

Run the same command again to update VigenFlow, update the weights or add models. To run it without
questions, set the models on `bash`, for example
`curl -fsSL .../install.sh | VIGENFLOW_MODELS=flux2_klein_4B,z_image_turbo bash`; the other settings
are listed at the top of [`docker/install.sh`](../docker/install.sh).

The rest of this page does the same steps by hand.

## Requirements

- An AMD Ryzen AI 300-series processor with an XDNA 2 NPU, as for the release packages.
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
- About 65 GB of disk space for the weights of all four model families, plus up to 10 GB while they
  download. You can download fewer models.

## Get the image

Pull the published image:

```bash
docker pull ghcr.io/jia1217/vigenflow:latest
```

Or build it yourself from a checkout of this repository:

```bash
docker build -f docker/Dockerfile -t ghcr.io/jia1217/vigenflow:latest .
```

The build compiles `vgf-serve`, `prepare_base.exe` and the launchers, and lays them out as
`package_ubuntu.sh` does. It takes a few minutes and needs no NPU.

## Download the model weights (one time)

The weights go into a named volume, `vigenflow-weights`:

```bash
docker run --rm -it \
  -v vigenflow-weights:/opt/vigenflow/all_model_weights \
  ghcr.io/jia1217/vigenflow:latest prepare
```

`prepare` runs `prepare_base.exe` and takes the same options, for example
`prepare --only flux2_klein_4B,z_image_turbo` to download only some models. The model names are
`flux1_schnell`, `flux2_klein_4B`, `z_image_turbo` and `krea2_turbo`.

Keep `-it`: before it downloads Krea 2 Turbo, `prepare` shows the Krea license links and waits for
you to type `I agree`. To accept the license without the prompt, add `--accept_license`.

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
The log shows the timings of each image, and OpenWebUI shows them in a line under the image.

### Check that it works

```bash
curl http://127.0.0.1:11283/v1/models

curl http://127.0.0.1:11283/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "flux2-klein-4B", "prompt": "a red fox sitting in fresh snow, photo", "response_format": "b64_json"}' \
  | python3 -c 'import sys, json, base64; open("fox.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The second command writes `fox.png` after about as long as the README's model table lists for the
model.

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

## Docker Compose, with OpenWebUI

[`docker/compose.yaml`](../docker/compose.yaml) starts the server with the options above. Run these
from the repository root:

```bash
docker compose -f docker/compose.yaml run --rm vigenflow prepare    # once
docker compose -f docker/compose.yaml up -d
```

To build the image instead of pulling it, run `docker compose -f docker/compose.yaml build` first.

With `--profile webui`, Compose also starts OpenWebUI v0.11.4, already connected to VigenFlow, at
http://localhost:8080:

```bash
docker compose -f docker/compose.yaml --profile webui up -d
```

OpenWebUI runs on the host network so that it reaches `vgf-serve` at `127.0.0.1:11283`, the
port that `vgf-serve` publishes only to this computer. The
connection and image settings in `compose.yaml` take effect only the first time OpenWebUI starts;
after that, change them in **Admin Panel → Settings**. Set up each model in **Admin Panel →
Settings → Models** as described in the main
[README](../README.md#-connect-with-openwebui-desktop), step 3.

If you already run OpenWebUI, start only VigenFlow and add the connection
`http://127.0.0.1:11283/v1` as in the README. If your OpenWebUI runs in a container on the default
bridge network, it reaches VigenFlow at `http://host.docker.internal:11283/v1` instead (start it
with `--add-host=host.docker.internal:host-gateway`).

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

### OpenWebUI shows "Image unavailable"

VigenFlow sends the image itself in streamed chat replies, and OpenWebUI shows it from its own
server. If an image does not show:

- Check that **Stream Chat Response** is on in the chat's or the model's advanced parameters. Without
  streaming, VigenFlow's reply links the image at `http://127.0.0.1:11283/images/...`, which only a
  browser on the same computer as VigenFlow can open.
- Images in chats made with an older VigenFlow are such links too. They stay unavailable after
  `vgf-serve` restarts, because it keeps the latest 32 images in memory only.

### `Nothing is mounted at /opt/vigenflow/all_model_weights`

`prepare` refuses to download into the container itself, because the weights would be lost when the
container is removed. Add the `-v` option shown above.

## For maintainers: publishing the image

[`.github/workflows/docker-server.yml`](../.github/workflows/docker-server.yml) builds
`docker/Dockerfile` on every pull request that changes it, and pushes the image to
`ghcr.io/<owner>/vigenflow` when a `v*` tag is pushed or the workflow is run by hand. A tag such as
`v0.2.0-beta.2` is published as `0.2.0-beta.2` and `latest`.

A new GHCR package is private. After the first push, open the package's settings on GitHub and set
its visibility to public, so that `docker pull` works without logging in.

The image takes the libraries in `src/lib` and the NPU files in `src/npu_files` from the checkout,
as `package_ubuntu.sh` does, so tag the commit that holds the release's libraries.
