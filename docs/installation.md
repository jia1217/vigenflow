# Installation

[← Back to VigenFlow](../README.md) · [Quick Install](quick-install.md)

These steps cover the **Beta 3 multi-model packages**. For the **Beta 4 FLUX.1-schnell Ubuntu package**, follow [the Beta 4 release instructions](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.4).

Download the package for your system, extract it, download the model weights once, and start the server. The weights are not included in the packages; Step 3 downloads them from Hugging Face.

## 📥 Step 1: Download the Release Package

Go to the [Beta 3 release](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.3) page and download the package for your operating system:

- 🐧 Ubuntu: `vigenflow_0.2.0-beta.3_ubuntu_amd64.zip`
- 🪟 Windows: `vigenflow_0.2.0-beta.3_windows_amd64.zip`

To be notified about new releases, click **Watch → Custom → Releases** on the [VigenFlow repository](https://github.com/jia1217/vigenflow) page.

## 📂 Step 2: Extract the Package

Extract the downloaded `.zip` package on a drive with room for the model weights. Step 3 saves them next to the extracted folder, not inside it.

### 🐧 Ubuntu

The Ubuntu package needs a few runtime libraries. XRT comes with the [AMD XDNA driver installation](../README.md#-install-the-npu-driver-one-time).

```bash
sudo apt install libboost-program-options1.83.0 libgomp1 libpng16-16t64 curl unzip
unzip vigenflow_0.2.0-beta.3_ubuntu_amd64.zip
cd vigenflow_0.2.0-beta.3_ubuntu_amd64
```

### 🪟 Windows

Extract the ZIP package manually, then open **PowerShell** inside the extracted `vigenflow_0.2.0-beta.3_windows_amd64` folder.

## 💾 Step 3: Download the Model Weights (One Time)

`prepare_base.exe` downloads the pre-packed NPU weights from Hugging Face and saves them in an `all_model_weights` folder **next to** the extracted package folder, so a later VigenFlow version extracted in the same place uses the same weights. Each model has its own repository: [FLUX.1-schnell](https://huggingface.co/Kelsey1217/FLUX.1-schnell-NPU-BFP16), [FLUX.2-klein-4B](https://huggingface.co/Kelsey1217/FLUX.2-klein-4B-NPU-BFP16), [Z-Image-Turbo](https://huggingface.co/Kelsey1217/Z-Image-Turbo-NPU-BFP16) and [Krea 2 Turbo](https://huggingface.co/Kelsey1217/Krea-2-Turbo-NPU-BFP16). No Hugging Face account is needed.

### 🐧 Ubuntu

```bash
./prepare/prepare_base.exe
```

### 🪟 Windows

```powershell
.\prepare\prepare_base.exe
```

When the download reaches Krea 2 Turbo, `prepare_base.exe` shows the Krea license links and waits for you to type `I agree`. Any other answer skips that model.

All four sets of weights together are about **65 GB** (FLUX.1-schnell 27 GB, Krea 2 Turbo 19 GB, Z-Image-Turbo 12 GB, FLUX.2-klein-4B 8 GB), plus up to 10 GB of free space while downloading. The edit model uses the FLUX.2-klein-4B weights.

To download only some models, name their weight folders with `--only`, for example `prepare_base.exe --only flux2_klein_4B,z_image_turbo`. The names are `flux1_schnell`, `flux2_klein_4B`, `z_image_turbo` and `krea2_turbo`. To skip a model on every run, set `"enabled": false` for it in `prepare/prepare_models.jsonc` instead.

`qwen_image_2_1_turbo` downloads the public [NPU+CPU weight package](https://huggingface.co/Kelsey1217/Qwen-Image-2.1-Turbo-NPU-BFP16)
at a pinned, verified revision. It remains opt-in and adds 19.59 GB when selected with
`--only qwen_image_2_1_turbo`; the tool asks for acceptance of the Qwen research license.
Both Qwen server models share this weight folder. See the
[preparation instructions](../src/prepare_weights/README.md#qwen-image-21-turbo)
and [server setup](../src/server_unified/README.md#qwen-image-21-turbo).

If you used v0.2.0 Beta 1, move its `all_model_weights` folder so that it sits next to the new package folder before you run the command. The three models you already have are then not downloaded again.

You can run `prepare_base.exe` again at any time. It only downloads models whose weights have changed, and it resumes an interrupted download.

## ▶️ Step 4: Launch VigenFlow Server

All models are served at once, so you don't need to give a model name when starting the server.

### 🐧 Ubuntu

```bash
cd server_running
./vgf-serve
```

### 🪟 Windows

```powershell
cd server_running
.\vgf-serve.exe
```

The server listens on port **11283** and makes all supported models available to OpenWebUI. To see all server options, run `./vgf-serve -h` (Ubuntu) or `.\vgf-serve.exe -h` (Windows).

The default model is `z-image-turbo`. If you did not download its weights, start the server with `--model` and the ID of a model you have, for example `--model flux2-klein-4B`.

After starting the server, follow the [OpenWebUI Desktop connection guide](openwebui.md).
