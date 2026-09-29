# VigenFlow

Local image generation and editing with **CPU + AMD NPU inference**, connected to **OpenWebUI Desktop**.

The new VigenFlow version uses **BFP16 weights for all supported models**.

---

<!-- ## 🧠 Supported Models

The current release supports the following base models and editing workflow:

| Model | Task | Weight format | Platforms | Output size | Denoising steps |
| --- | --- | --- | --- | --- | ---: |
| **FLUX.1-schnell** | Text-to-image  | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |
| **FLUX.2-klein-4B** | Text-to-image  | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |
| **FLUX.2-klein-4B Edit** | Image-to-image | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |
| **Z-Image-Turbo** | Text-to-image  | BFP16 | Windows / Ubuntu | 1024 × 1024 | 4 |

FLUX.1-schnell and FLUX.2-klein-4B use 256 text tokens; Z-Image-Turbo uses a caption padded to 512 tokens.

All models use the AMD NPU target described in **Supported Devices and Platforms**. Select a model from the OpenWebUI model list after launching `vgf-serve`.

You can switch image generation models inside OpenWebUI without restarting `vgf-serve`.

---

## 🔜 Next Steps

Planned for a future release:

- [ ] **LoRA support** for the BFP16 model pipelines.

--- -->

## 💻 Usage Commands

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
| Operating systems | Ubuntu x86_64 and Windows x64 |
| Driver and runtime | AMD XDNA driver and an XRT runtime compatible with the installed NPU driver |

Choose the release package for your operating system and keep its matching model workers and NPU assets together. The device target applies to all supported models; support for another processor requires compatible NPU binaries and runtime support.

---

## 🚀 Getting Started

Before getting started, please make sure the **AMD XDNA driver** has been installed on your system.

You can find more details in the [AMD XDNA / MLIR-AIE installation guide](https://github.com/Xilinx/mlir-aie) for the Ubuntu system and Windows system in the [Driver Download](https://www.amd.com/en/support/download/drivers.html).

---

## ✨ Simple Usage

For most users, the complete workflow is:

1. 📥 Download the release package for your system.
2. 📂 Extract the `.zip` file.
3. ▶️ Start the VigenFlow server.
4. 🖥️ Open OpenWebUI Desktop.
5. 🔗 Configure the VigenFlow connection.
6. 🧠 Select a model from the OpenWebUI model list.
7. 🎨 Start generating images with your own local VigenFlow AI.

---

## 📦 Installation

The recommended way to use **VigenFlow** is to download the latest release package.

Starting from **VigenFlow v0.1.2**, we provide ready-to-use `.zip` packages for both Ubuntu and Windows:

- 🐧 `vigenflow_0.1.2_ubuntu_amd64.zip`
- 🪟 `vigenflow_0.1.2_windows_amd64.zip`

You only need to download the package for your system, extract it, launch `vgf-serve`, and connect it with **OpenWebUI Desktop**.

---

## ⭐ Option A: Install from the Release Package Recommended

### 📥 Step 1: Download the Release Package

Go to the **Latest Release** page and download the package for your operating system.

#### 🐧 Ubuntu

Download:

```text
vigenflow_0.1.2_ubuntu_amd64.zip
```

#### 🪟 Windows

Download:

```text
vigenflow_0.1.2_windows_amd64.zip
```

---

### 📂 Step 2: Extract the Package

Extract the downloaded `.zip` package to your desired location.

#### 🐧 Ubuntu

```bash
unzip vigenflow_0.1.2_ubuntu_amd64.zip
cd vigenflow_0.1.2_ubuntu_amd64
```

#### 🪟 Windows

Extract the ZIP package manually, then open **PowerShell** or **Command Prompt** inside the extracted folder.

---

### ▶️ Step 3: Launch VigenFlow Server

Starting from **VigenFlow v0.1.2**, image generation models are launched automatically by default.

You no longer need to provide a model name when starting the server for image generation.

#### 🐧 Ubuntu

```bash
./vgf-serve
```

#### 🪟 Windows

```powershell
.\vgf-serve.exe
```

After the server starts successfully, VigenFlow will make the supported image generation models available to OpenWebUI.

---

## 🔗 Connect with OpenWebUI Desktop

After launching `vgf-serve`, open **OpenWebUI Desktop** and configure the connection to your local VigenFlow server.

In OpenWebUI, go to the **Connections** settings and add your VigenFlow server endpoint.

Example local endpoint:

```text
http://127.0.0.1:2048/v1
```

Once the connection is configured, you can select and switch VigenFlow models directly from the OpenWebUI model list.

<img width="1629" height="995" alt="OpenWebUI connection settings" src="https://github.com/user-attachments/assets/59543b3f-3a49-4675-8aaf-f48afae57c73" />

---

## 🖼️ Important Note for Image Editing Models

To select **FLUX.2-klein-4B image editing** as the default model, specify its model ID when launching `vgf-serve`. Editing requests also need an input image and an editing prompt.

Example:

#### 🐧 Ubuntu

```bash
./vgf-serve flux.2-klein-4B-edit
```

#### 🪟 Windows

```powershell
.\vgf-serve.exe flux.2-klein-4B-edit
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
- ⚡ BFP16 weights for all supported models.
- 🖥️ Works together with OpenWebUI Desktop to create your own local VigenFlow AI.
