# VigenFlow Ubuntu package

This package includes seven model workers, the API server, NPU instruction and
XCLBIN files, and the weight downloader. The package version is in `VERSION`.
Qwen Image 2.1 Turbo supports text-to-image and image editing at 1024x1024 with
eight steps. Both tasks share one set of weights.

## Ubuntu setup

Use Ubuntu 24.04 x86_64 with an AMD Ryzen AI 300-series XDNA 2 NPU. The local
reference system is a Ryzen AI 9 HX 370. Install a compatible AMD XDNA driver
and XRT runtime before running inference; these are system dependencies and
are not inside the ZIP. See the repository's
[driver setup](https://github.com/jia1217/vigenflow#-install-the-npu-driver-one-time).

Install the Ubuntu runtime libraries and download tools:

```bash
sudo apt update
sudo apt install libboost-program-options1.83.0 libpng16-16t64 \
  libjpeg-turbo8 libssl3t64 libgomp1 libstdc++6 zlib1g libuuid1 curl ca-certificates
```

The `run.exe` and `prepare_base.exe` files in this package are Linux executables.

## Prepare the Qwen weights

Run commands from the extracted package folder. Download Qwen only:

```bash
./prepare/prepare_base.exe --only qwen_image_2_1_turbo
```

The downloader displays the model's license terms for acceptance. Qwen is an
opt-in download of about 19.59 GB (18.24 GiB). Its weights are placed next to
the package folder, where both Qwen workers find them:

```text
parent/
  vigenflow_<version>_ubuntu_amd64/
    server_running/
    all_npu_files/
    prepare/
  all_model_weights/
    qwen_image_2_1_turbo/
      text_encoder/
      denoising/
      vae_decoder/
      vision/
      vae_encoder/
```

If you already have these five prepared stages, reuse them instead of downloading
again. For an empty destination, link your existing weight directory:

```bash
mkdir -p ../all_model_weights
ln -s /absolute/path/to/qwen-npu ../all_model_weights/qwen_image_2_1_turbo
```

Keep the weights and their notices together when moving them. Symlinks need to
point to an existing directory on the machine where you run the package.

## Run Qwen text-to-image and image editing

Start the server from the package folder:

```bash
cd server_running
./vgf-serve --model qwen-image-2.1-turbo -p 11283 -k true
```

`-k true` keeps final PNGs in `server_running/images/`. Qwen intermediate tensor
and diagnostic dumps are disabled. Omit `-k true` to use the server's default
temporary-file cleanup and save images only through your client.

In another terminal, generate an image:

```bash
curl -fS http://127.0.0.1:11283/v1/images/generations \
  -H 'Content-Type: application/json' -H 'Accept: image/png' \
  -d '{"model":"qwen-image-2.1-turbo","prompt":"a red fox sitting in fresh snow"}' \
  -o generated.png
```

Edit that image:

```bash
curl -fS http://127.0.0.1:11283/v1/images/edits \
  -H 'Accept: image/png' -F model=qwen-image-2.1-turbo-edit \
  -F 'prompt=make it night time' -F image=@generated.png -o edited.png
```

Always specify the Qwen edit model in edit requests when the server's default
model is text-to-image. You can also connect an OpenAI-compatible client to
`http://127.0.0.1:11283/v1` and select either Qwen model.

All seven workers are available through `GET /v1/models`:

| Model ID | Task |
|---|---|
| `qwen-image-2.1-turbo` | Text-to-image |
| `qwen-image-2.1-turbo-edit` | Image editing |
| `flux1-schnell` | Text-to-image |
| `flux2-klein-4B` | Text-to-image |
| `flux2-klein-4B-edit` | Image editing |
| `krea2-turbo` | Text-to-image |
| `z-image-turbo` | Text-to-image |

The other models require their own weights. Running `prepare/prepare_base.exe`
without `--only` downloads the four enabled weight sets; Qwen remains opt-in.

## Verify the package

Before generating images, verify the packaged files from the package folder:

```bash
sha256sum --check --quiet SHA256SUMS
```

The adjacent ZIP has its own `.zip.sha256` file. Verify that checksum from the
directory containing the ZIP before extracting it on another machine.
