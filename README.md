# VigenFlow

Local image generation and editing with **CPU + AMD NPU inference**, connected to **OpenWebUI Desktop**.

The latest release, **[v0.2.0 Beta 1](https://github.com/jia1217/vigenflow/releases/latest)**, uses **BFP16 weights for all supported models**, adds **FLUX.1-schnell**, and serves **image editing** next to the text-to-image models. Running on the NPU, VigenFlow generates images **several times faster than the same laptop's integrated GPU**.

---

## 🚀 Getting Started

Before getting started, please make sure the **AMD XDNA driver** has been installed on your system.

You can find more details in the [AMD XDNA / MLIR-AIE installation guide](https://github.com/Xilinx/mlir-aie) for the Ubuntu system and Windows system in the [Driver Download](https://www.amd.com/en/support/download/drivers.html).

---

## 🧠 Supported Models

The current release supports the following base models and editing workflow:

| Model | Model ID | Task | Weight format | Platforms | Output size | Denoising steps |
| --- | --- | --- | --- | --- | --- | ---: |
| **FLUX.1-schnell** | `flux1-schnell` | Text-to-image  | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |
| **FLUX.2-klein-4B** | `flux2-klein-4B` | Text-to-image  | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |
| **FLUX.2-klein-4B Edit** | `flux2-klein-4B-edit` | Image-to-image | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |
| **Z-Image-Turbo** | `z-image-turbo` | Text-to-image  | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 or 8 |

FLUX.1-schnell and FLUX.2-klein-4B use 256 text tokens; Z-Image-Turbo uses a caption padded to 512 tokens.

All models use the AMD NPU target described in **Supported Devices and Platforms**. Select a model from the OpenWebUI model list after launching `vgf-serve`.

You can switch image generation models inside OpenWebUI without restarting `vgf-serve`.

---

## 🔜 Next Steps

Planned for a future release:

- [ ] **LoRA support** for the BFP16 model pipelines. If you need LoRA today, use [v0.1.2](https://github.com/jia1217/vigenflow/releases/tag/v0.1.2).

---

## 💻 Usage Commands

Run these inside the `server_running` folder of the extracted package.

### 🐧 Ubuntu

```bash
./vgf-serve
```

### 🪟 Windows

```powershell
.\vgf-serve.exe
```

To check available options:

#### 🐧 Ubuntu

```bash
./vgf-serve -h
```

#### 🪟 Windows

```powershell
.\vgf-serve.exe -h
```

---

## 💻 Supported Devices and Platforms

VigenFlow's model workers use **CPU + AMD NPU inference** and currently target **AMD XDNA 2 (`npu2`)**.

| Component | Supported target |
| --- | --- |
| NPU | AMD XDNA 2, using the matching model xclbins and instruction files |
| Reference processor | [AMD Ryzen AI 9 HX PRO 370](https://www.amd.com/en/newsroom/press-releases/2024-10-10-amd-launches-new-ryzen-ai-pro-300-series-processo.html), from the Ryzen AI PRO 300 Series |
| Operating systems | Ubuntu 24.04 x86_64 and Windows x64 |
| Driver and runtime | AMD XDNA driver and an XRT runtime compatible with the installed NPU driver |

Choose the release package for your operating system and keep its matching model workers and NPU assets together. The device target applies to all supported models; support for another processor requires compatible NPU binaries and runtime support.

---

## ✨ Simple Usage

For most users, the complete workflow is:

1. 📥 Download the release package for your system.
2. 📂 Extract the `.zip` file.
3. 💾 Download the model weights (one time).
4. ▶️ Start the VigenFlow server.
5. 🖥️ Open OpenWebUI Desktop.
6. 🔗 Configure the VigenFlow connection and image settings.
7. 🧠 Select a model from the OpenWebUI model list.
8. 🎨 Start generating images with your own local VigenFlow AI.

---

## 📦 Installation

The recommended way to use **VigenFlow** is to download the latest release package.

The latest release provides ready-to-use `.zip` packages for both Ubuntu and Windows:

- 🐧 `vigenflow_0.2.0-beta.1_ubuntu_amd64.zip`
- 🪟 `vigenflow_0.2.0-beta.1_windows_amd64.zip`

You only need to download the package for your system, extract it, download the model weights once, launch `vgf-serve`, and connect it with **OpenWebUI Desktop**. The model weights are not included in the packages; a one-time step downloads them from Hugging Face.

### 🔔 Get Release Notifications

Downloading a VigenFlow package does not automatically subscribe you to future release notifications. To receive updates:

1. Sign in to GitHub and open the [VigenFlow repository](https://github.com/jia1217/vigenflow).
2. Click **Watch → Custom** in the upper-right corner.
3. Select **Releases** and save your preferences.

When a new release is published, you can receive notifications on GitHub or by email, depending on your [notification settings](https://github.com/settings/notifications). See [GitHub's notification guide](https://docs.github.com/en/subscriptions-and-notifications/get-started/configuring-notifications) for details.

You can also check the [Latest Release](https://github.com/jia1217/vigenflow/releases/latest) page at any time.

---

## ⭐ Option A: Install from the Release Package Recommended

### 📥 Step 1: Download the Release Package

Go to the [Latest Release](https://github.com/jia1217/vigenflow/releases/latest) page and download the package for your operating system.

#### 🐧 Ubuntu

Download:

```text
vigenflow_0.2.0-beta.1_ubuntu_amd64.zip
```

#### 🪟 Windows

Download:

```text
vigenflow_0.2.0-beta.1_windows_amd64.zip
```

---

### 📂 Step 2: Extract the Package

Extract the downloaded `.zip` package to your desired location.

#### 🐧 Ubuntu

The Ubuntu package needs a few runtime libraries. XRT comes with the AMD XDNA driver installation from [Getting Started](#-getting-started).

```bash
sudo apt install libboost-program-options1.83.0 libgomp1 libpng16-16t64 curl unzip
unzip vigenflow_0.2.0-beta.1_ubuntu_amd64.zip
cd vigenflow_0.2.0-beta.1_ubuntu_amd64
```

#### 🪟 Windows

Extract the ZIP package manually, then open **PowerShell** inside the extracted `vigenflow_0.2.0-beta.1_windows_amd64` folder.

---

### 💾 Step 3: Download the Model Weights (One Time)

`prepare_base.exe` downloads the pre-packed NPU weights from [Kelsey1217/vigenflow-npu-weights](https://huggingface.co/Kelsey1217/vigenflow-npu-weights) into the `all_model_weights` folder. No Hugging Face account is needed.

#### 🐧 Ubuntu

```bash
./prepare/prepare_base.exe
```

#### 🪟 Windows

```powershell
.\prepare\prepare_base.exe
```

All three models together are about **46 GB** (FLUX.1-schnell 27 GB, Z-Image-Turbo 12 GB, FLUX.2-klein-4B 8 GB), plus up to 10 GB of free space while downloading. The edit model uses the FLUX.2-klein-4B weights. To skip a model, set `"enabled": false` for it in `prepare/prepare_models.jsonc` before running the command.

You can run `prepare_base.exe` again at any time. It only downloads models whose weights have changed, and it resumes an interrupted download.

---

### ▶️ Step 4: Launch VigenFlow Server

All models are served at once, so you don't need to give a model name when starting the server.

#### 🐧 Ubuntu

```bash
cd server_running
./vgf-serve
```

#### 🪟 Windows

```powershell
cd server_running
.\vgf-serve.exe
```

After the server starts successfully, VigenFlow makes all supported models available to OpenWebUI. The server listens on port **11283**.

---

## 🔗 Connect with OpenWebUI Desktop

After launching `vgf-serve`, open **OpenWebUI Desktop** and configure it for your local VigenFlow server. All settings below are in OpenWebUI's **Admin Panel → Settings**.

1. **Add the connection.** In **Connections**, add an OpenAI API connection with this URL:

   ```text
   http://127.0.0.1:11283/v1
   ```

   Keep **API Type** set to **Chat Completions** (the default). No API key is needed. The VigenFlow models then appear in the model list.
2. **Turn on image generation.** In **Images**, turn on **Image Generation**, keep the engine set to **OpenAI**, set the API base URL to `http://127.0.0.1:11283/v1`, and type any text as the API key, then click **Save**. OpenWebUI will not save this setting without a key; VigenFlow ignores it.
3. **Set up each model.** In **Models**, edit each VigenFlow model:
   - Set **Function Calling** to **Native** in its advanced parameters. Recent OpenWebUI versions already use Native by default.
   - Keep **Image Generation** and **Builtin Tools** checked under Capabilities.
   - For `flux2-klein-4B-edit`, also check **Vision** so you can attach images.
4. **Generate an image.** In a chat, pick a VigenFlow model, open **Integrations** in the message box and turn on **Image**, then type your prompt.

You can switch VigenFlow models directly from the OpenWebUI model list.

<img width="1629" height="995" alt="OpenWebUI connection settings" src="https://github.com/user-attachments/assets/59543b3f-3a49-4675-8aaf-f48afae57c73" />

The screenshot shows the **Connections** page from an earlier release, which used port 2048. Use `http://127.0.0.1:11283/v1` for the current release.

---

## 🖼️ Image Editing

`flux2-klein-4B-edit` is always in the OpenWebUI model list next to the text-to-image models, so you don't need to restart the server to use it.

1. Pick `flux2-klein-4B-edit` in a chat. Make sure **Vision** is on for it (step 3 above).
2. Turn on **Image**, attach a picture, and describe the change, for example `change the dress color from white to red`.
3. A follow-up message such as `now make it night` edits the previous result without attaching it again. You can also create an image with a text-to-image model, then switch to `flux2-klein-4B-edit` to edit it.

The input image is center-cropped to a square and resized to 1024 × 1024.

To make the edit model the server's default model, start the server with:

#### 🐧 Ubuntu

```bash
./vgf-serve --model flux2-klein-4B-edit
```

#### 🪟 Windows

```powershell
.\vgf-serve.exe --model flux2-klein-4B-edit
```

---

## 🎬 Demo

Watch the demo video to learn how to set up and use VigenFlow with OpenWebUI Desktop:

<!-- https://youtu.be/kZaqCE0LnRA?si=5h6zOFKBfm6EKJv9 -->

[[Watch the VigenFlow Demo]](https://youtu.be/kZaqCE0LnRA?si=5h6zOFKBfm6EKJv9)

<!-- You can also watch the local demo below:

https://github.com/user-attachments/assets/ad22f2e3-ffca-468e-ab7f-d5fe10e70998 -->

---

> **Note:** If you are using the latest version of OpenWebUI Desktop, we recommend switching to the stable version shown in the video to avoid compatibility issues.

[[Watch the VigenFlow Demo]](https://youtube.com/shorts/dwDsCnBObzQ?si=Aq4BZV3zcW51KjQa)

---

## ✅ Summary

With the latest release package, VigenFlow is now much easier to run:

- 📦 Ready-to-use `.zip` packages for Ubuntu and Windows.
- 🛠️ No manual build required for normal users.
- ▶️ Start the image generation service with one command.
- 🧠 Switch image generation models directly from OpenWebUI.
- 🎨 FLUX.1-schnell, FLUX.2-klein-4B, and Z-Image-Turbo base models, plus FLUX.2-klein-4B image editing.
- ⚡ Several times faster image generation on the NPU than on the same laptop's integrated GPU.
- 🔢 BFP16 weights for all supported models.
- 🖥️ Works together with OpenWebUI Desktop to create your own local VigenFlow AI.

---

## 📄 License

The VigenFlow source code is licensed under the [Apache License 2.0](LICENSE).

Model weights are **not** covered by this license. Each model and LoRA keeps its original license, including the BFP16 weights in the release packages. See [MODEL_LICENSES.md](MODEL_LICENSES.md) for each model's license and whether it allows commercial use. Third-party libraries are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
