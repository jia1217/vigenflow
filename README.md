## 🚀 Getting Started

Set up **VigenFlow** in just a few minutes and connect it with **OpenWebUI Desktop** to run your own local VigenFlow AI.

Before getting started, please make sure the **AMD XDNA driver** has been installed on your system.

You can find more details in the [AMD XDNA / MLIR-AIE installation guide](https://github.com/Xilinx/mlir-aie) for the Ubuntu system and Windows system in the [Driver Download](https://www.amd.com/en/support/download/drivers.html).

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
http://localhost:2048/v1
```

Once the connection is configured, you can select and switch VigenFlow models directly from the OpenWebUI model list.

<img width="1629" height="995" alt="OpenWebUI connection settings" src="https://github.com/user-attachments/assets/59543b3f-3a49-4675-8aaf-f48afae57c73" />

---

## 🧠 Supported Image Generation Models

The following image generation models are available from the OpenWebUI model list after launching `vgf-serve`:

- ⚡ `z-image-turbo-BF16`
- 🎨 `z-image-turbo-BF16-lora`
- ⚡ `z-image-turbo-Q4_1-GGUF`
- 🎨 `z-image-turbo-Q4_1-GGUF-lora`
- 🌊 `flux.2-klein-4B`
- 🎨 `flux.2-klein-4B-lora`

You can switch between different base models and LoRA models directly inside OpenWebUI without restarting `vgf-serve`.

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

## 🖼️ Important Note for Image Editing Models

The simplified startup workflow applies to **image generation models**.

For **image editing models**, you still need to specify the model name when launching `vgf-serve` if you want to use a different image editing model.

Example:

#### 🐧 Ubuntu

```bash
./vgf-serve <image-edit-model-name>
```

#### 🪟 Windows

```powershell
.\vgf-serve.exe <image-edit-model-name>
```

---

## 🎬 Demo

Watch the demo video to learn how to set up and use VigenFlow with OpenWebUI Desktop:

<!-- Replace the link below with your YouTube video link -->

[![Watch the VigenFlow Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

You can also watch the local demo below:

https://github.com/user-attachments/assets/ad22f2e3-ffca-468e-ab7f-d5fe10e70998

---

## ✅ Summary

With the latest release package, VigenFlow is now much easier to run:

- 📦 Ready-to-use `.zip` packages for Ubuntu and Windows.
- 🛠️ No manual build required for normal users.
- ▶️ Start the image generation service with one command.
- 🧠 Switch image generation models directly from OpenWebUI.
- 🎨 LoRA model variants are available from the OpenWebUI model list.
- 🖥️ Works together with OpenWebUI Desktop to create your own local VigenFlow AI.