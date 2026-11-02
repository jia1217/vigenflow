# VigenFlow [![Join Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/87DnMjsccu)

**Create images locally with AMD NPU acceleration and a built-in browser UI.**

**[v0.2.0 Beta 4 — FLUX.1-schnell prerelease](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.4)** brings prompts, LoRA styles, image settings, previews, and PNG downloads together in your browser. This package supports **FLUX.1-schnell on Ubuntu x86-64**, using **CPU + AMD NPU inference**. OpenWebUI is optional.

- 🎨 **LoRA styles:** select styles and add their trigger words, or configure your own LoRAs through `loras.jsonc`.
- 🖼️ **Custom resolutions:** 16–2048 pixels per side, in multiples of 16, for square, portrait, and landscape images.
- ⚡ **Two memory modes:** Balanced reduces RAM usage; Performance keeps weights and reusable buffers loaded for faster subsequent images.
- ⏱️ **Visible timings:** E2E runtime in seconds beside the seed and in the completion message, with stage timings below the image.

**[Download Beta 4](https://github.com/jia1217/vigenflow/releases/download/v0.2.0-beta.4/vgf_0.2.0-beta.4_ubuntu_amd64_flux1.zip)** · **[Get started](#beta4-setup)** · **[FLUX.1 results](#beta4-results)** · **[Speed and memory](#beta4-performance)**

> **Upgrading from Beta 3?** The model weight paths are unchanged. Extract Beta 4 beside your existing `all_model_weights` folder to reuse `all_model_weights/flux1_schnell` without downloading the weights again.

VigenFlow also supports **FLUX.2, Z-Image, Krea 2, and Qwen Image** for generation and editing. The **[Beta 3 packages](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.3)** include all seven models on Windows and Ubuntu; the Beta 4 prerelease above includes **FLUX.1-schnell only**. The existing [Quick Install](#-quick-install) installs the multi-model release with OpenWebUI.

---

## 🎨 Gallery

<a id="beta4-results"></a>

### FLUX.1-schnell · Beta 4 showcase

Explore fantasy art, macro imagery, and fashion illustration with these saved FLUX.1-schnell examples. Click an image for the original PNG, or open its **UI preview** to see it in the Beta 4 browser interface. Dimensions below are **width × height**.

<table>
  <tr>
    <td width="33%" align="center" valign="top"><a href="assets/gallery/flux1_beta4/1_1216x832.png"><img src="assets/gallery/flux1_beta4/1_1216x832.png" width="100%" alt="Fantasy artwork of a winged armored knight holding a kitten"></a><br><sub><b>Winged knight and kitten</b><br>832 × 1216 · FLUX.1-schnell<br><a href="assets/gallery/flux1_beta4/ui_1.png">UI preview</a></sub></td>
    <td width="33%" align="center" valign="top"><a href="assets/gallery/flux1_beta4/2_832x1216.png"><img src="assets/gallery/flux1_beta4/2_832x1216.png" width="100%" alt="Macro image of a jeweled ladybug"></a><br><sub><b>Jeweled ladybug</b><br>1216 × 832 · FLUX.1-schnell<br><a href="assets/gallery/flux1_beta4/ui_2.png">UI preview</a></sub></td>
    <td width="33%" align="center" valign="top"><a href="assets/gallery/flux1_beta4/3_768x1024.png"><img src="assets/gallery/flux1_beta4/3_768x1024.png" width="100%" alt="Fashion illustration featuring a flowing gold gown"></a><br><sub><b>Flowing gold gown</b><br>1024 × 768 · FLUX.1-schnell<br><a href="assets/gallery/flux1_beta4/ui_3.png">UI preview</a></sub></td>
  </tr>
</table>

**The Beta 4 browser UI:** choose your LoRAs, set an image size, and preview or download the result from one page.

[![VigenFlow Beta 4 browser UI displaying the saved winged knight and kitten image](assets/gallery/flux1_beta4/ui_1.png)](assets/gallery/flux1_beta4/ui_1.png)

*These UI previews display saved images. The controls shown are not a record of their original prompts, seeds, LoRA selections, or timings.*

### More models and image editing · Beta 3

The multi-model release includes the examples below, generated on the NPU at **1024 × 1024**. Click an image, or open the **[full gallery](docs/gallery.md)**, for more examples and the exact prompts. These additional models are available in Beta 3, outside the FLUX.1-only Beta 4 package.

<!-- One image per text-to-image model here, up to five per row; the editing table below shows an input image and its edits. Each model's full set and prompts go in docs/gallery.md. -->
<table>
  <tr>
    <td width="20%" align="center" valign="top"><a href="docs/gallery.md#flux1-schnell"><img src="assets/gallery/flux1_1.jpg" width="100%" alt="A reading nook with a sleeping orange cat"></a><br><sub><b>FLUX.1-schnell</b></sub></td>
    <td width="20%" align="center" valign="top"><a href="docs/gallery.md#flux2-klein-4b"><img src="assets/gallery/flux2_3.jpg" width="100%" alt="Paper-cut art of a koi among lotus flowers"></a><br><sub><b>FLUX.2-klein-4B</b></sub></td>
    <td width="20%" align="center" valign="top"><a href="docs/gallery.md#krea-2-turbo"><img src="assets/gallery/krea2_1.jpg" width="100%" alt="A young woman holding a purple bellflower in a sunlit forest"></a><br><sub><b>Krea 2 Turbo</b></sub></td>
    <td width="20%" align="center" valign="top"><a href="docs/gallery.md#qwen-image-21-turbo"><img src="assets/gallery/qwen_image_2_1_turbo_3.webp" width="100%" alt="A Qwen Coffee storefront with Chinese and English signage"></a><br><sub><b>Qwen Image 2.1 Turbo</b></sub></td>
    <td width="20%" align="center" valign="top"><a href="docs/gallery.md#z-image-turbo"><img src="assets/gallery/zimage_1.jpg" width="100%" alt="Portrait of an elderly fisherman"></a><br><sub><b>Z-Image-Turbo</b></sub></td>
  </tr>
</table>

**Image editing** with Qwen Image 2.1 Turbo Edit, starting from the Krea 2 Turbo image above:

<table>
  <tr>
    <td width="25%" align="center" valign="top"><a href="docs/gallery.md#qwen-image-21-turbo-edit"><img src="assets/gallery/krea2_1.jpg" width="100%" alt="Input: a young woman holding a purple bellflower in a sunlit forest"></a><br><sub><b>Input image</b></sub></td>
    <td width="25%" align="center" valign="top"><a href="docs/gallery.md#qwen-image-21-turbo-edit"><img src="assets/gallery/qwen_image_2_1_turbo_edit_1.webp" width="100%" alt="The same woman shown full length in her white dress and shoes on a forest path"></a><br><sub><b>Full-body shot</b></sub></td>
    <td width="25%" align="center" valign="top"><a href="docs/gallery.md#qwen-image-21-turbo-edit"><img src="assets/gallery/qwen_image_2_1_turbo_edit_2.webp" width="100%" alt="The same woman redrawn as a full-color manga illustration"></a><br><sub><b>Manga style</b></sub></td>
    <td width="25%" align="center" valign="top"><a href="docs/gallery.md#qwen-image-21-turbo-edit"><img src="assets/gallery/qwen_image_2_1_turbo_edit_3.webp" width="100%" alt="The same woman with her white dress changed to pink"></a><br><sub><b>Pink dress</b></sub></td>
  </tr>
</table>

---

<a id="beta4-performance"></a>

## ⚡ FLUX.1-schnell speed and memory

Measurements from a development-machine run at **1024 × 1024, four steps, without LoRA**, using the prompt `A red fox in fresh snow` and seed `0`:

| Memory mode | Request time | Peak worker RAM |
| --- | ---: | ---: |
| **Balanced** · default | 28.5 s | 10.8 GiB |
| **Performance** · after the first image | 20.6 s | 25.4 GiB |

**Balanced** releases stage resources and exits the worker after each image. **Performance** retains weights and reusable buffers: the first image loads the pipeline, and subsequent images reuse it. Changing the resolution or selected LoRAs reloads the worker. Generating in Balanced mode or stopping the server releases it; otherwise its memory stays allocated while idle.

These are worker request times, including request overhead. The UI reports pipeline **E2E runtime** separately. Peak worker RAM excludes other applications; timing and memory usage vary with hardware and settings. The showcase images above are separate from this benchmark.

<a id="beta4-setup"></a>

## 🚀 Try Beta 4 on Ubuntu

You need **Ubuntu x86-64**, a compatible **AMD Ryzen AI NPU** with a working [NPU driver and XRT installation](#-install-the-npu-driver-one-time), and **`curl` 8.3 or newer**. A fresh model download needs approximately **27 GB**, plus up to **10 GB of temporary space**. Weights download separately from the application ZIP.

Download **[`vgf_0.2.0-beta.4_ubuntu_amd64_flux1.zip`](https://github.com/jia1217/vigenflow/releases/download/v0.2.0-beta.4/vgf_0.2.0-beta.4_ubuntu_amd64_flux1.zip)**, then run:

```bash
unzip vgf_0.2.0-beta.4_ubuntu_amd64_flux1.zip
cd vgf_0.2.0-beta.4_ubuntu_amd64_flux1

# First use only; skip if the required Beta 3 weights are already in place
./prepare/prepare_base.exe

# Start the server and built-in browser UI
bash start.sh
```

Open **http://127.0.0.1:11285/**. Enter a prompt, choose your image settings, and click **Generate image**. Select **Performance** under **Advanced settings** for repeated generation, or enable it at startup:

```bash
bash start.sh --flux1-mode performance
```

The weights folder is shared with Beta 3:

```text
your-install-folder/
├── all_model_weights/
│   └── flux1_schnell/
└── vgf_0.2.0-beta.4_ubuntu_amd64_flux1/
```

**Add your own LoRAs** by editing `server_running/exe_models/flux1_schnell/loras.jsonc` inside the extracted package. Copy an existing entry and set its `name`, weight `file`, `scale`, and optional `trigger` words. Click **Refresh** in the UI, then select it under **Styles (LoRAs)**. **Add words** also selects the style and inserts its trigger words into the prompt.

This is a testing prerelease. Share your images and report issues with your hardware, prompt, seed, resolution, memory mode, selected LoRAs, and relevant server logs in [Issues](https://github.com/jia1217/vigenflow/issues) or [Discord](https://discord.gg/87DnMjsccu).

---

## 🎬 Demo

- ⚡ **[v0.2 speed demo](https://www.youtube.com/watch?v=2mQH2Q0H6Nc&list=PLcZNuhm1Td94)**: VigenFlow generating images on the NPU.
- 🛠️ **[Setup walkthrough](https://youtu.be/kZaqCE0LnRA?si=5h6zOFKBfm6EKJv9)**: how to set up and use VigenFlow with OpenWebUI Desktop.
<!-- - 📱 **[Short demo](https://youtube.com/shorts/dwDsCnBObzQ?si=Aq4BZV3zcW51KjQa)** -->

> **Note:** VigenFlow is tested with Open WebUI **v0.11.4**. If you use a different version and run into compatibility issues, switch to v0.11.4.

---

## 🧠 Supported Models

The table below describes the **Beta 3 multi-model packages** for Windows and Ubuntu. **Beta 4 currently packages FLUX.1-schnell only for Ubuntu**, with LoRAs, custom resolutions, and the memory modes described above.

### 🖼️ Text-to-Image

| Model | Model ID | Denoising steps | Time per image | Windows | Ubuntu |
| --- | --- | ---: | ---: | :---: | :---: |
| **FLUX.1-schnell** | `flux1-schnell` | 4 | ~20 s | ✅ | ✅ |
| **FLUX.2-klein-4B** | `flux2-klein-4B` | 4 | ~9 s | ✅ | ✅ |
| **Krea 2 Turbo** | `krea2-turbo` | 8 | ~48 s | ✅ | ✅ |
| **Z-Image-Turbo** | `z-image-turbo` | 4 or 8 | ~18 s | ✅ | ✅ |
| **Qwen Image 2.1 Turbo** | `qwen-image-2.1-turbo` | 8 | ~35 s | ✅ | ✅ |

### ✏️ Image-to-Image (Editing)

| Model | Model ID | Denoising steps | Time per image | Windows | Ubuntu |
| --- | --- | ---: | ---: | :---: | :---: |
| **FLUX.2-klein-4B Edit** | `flux2-klein-4B-edit` | 4 | ~18 s | ✅ | ✅ |
| **Qwen Image 2.1 Turbo Edit** | `qwen-image-2.1-turbo-edit` | 8 | ~52 s | ✅ | ✅ |

All models use BFP16 weights. The Beta 3 workers make 1024 × 1024 images; Beta 4's FLUX.1-schnell worker also supports custom resolutions. You can switch between installed workers in the OpenWebUI model list without restarting `vgf-serve`.

The times are approximate, for one 1024 × 1024 image on an AMD Ryzen AI 9 HX 370, with prompts encoded at 512 text tokens.

Krea 2 Turbo's weights are under the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing), not Apache-2.0, and its commercial use is limited; see [MODEL_LICENSES.md](MODEL_LICENSES.md). The Flux and Z-Image models in the table above are Apache-2.0. The opt-in Qwen Image 2.1 Turbo weights use the Qwen RESEARCH LICENSE AGREEMENT (research/evaluation only; separate license required for commercial use).

### Compared with AMD Ryzen AI Software 1.8

AMD's [Ryzen AI Software 1.8](https://ryzenai.docs.amd.com/en/latest/sd_demo.html) runs FLUX.1-schnell and FLUX.2-klein-4B on the NPU as preview demos. On the same Ryzen AI 9 HX 370, VigenFlow is faster with prompts twice as long, and it also runs three models that Ryzen AI 1.8 does not:

| Model | VigenFlow | Ryzen AI Software 1.8 |
| --- | --- | --- |
| FLUX.1-schnell | ~20 s, 512 text tokens | ~23 s, 256 text tokens |
| FLUX.2-klein-4B | ~9 s, 512 text tokens | ~10 s, 256 text tokens |
| FLUX.2-klein-4B Edit | ~18 s | Not supported |
| Krea 2 Turbo | ~48 s | Not supported |
| Z-Image-Turbo | ~18 s | Not supported |

All times are for one 1024 × 1024 image, and both FLUX comparisons use 4 steps. In our tests, Ryzen AI 1.8's FLUX demos accept prompts of up to 256 text tokens. At the same 256 text tokens, VigenFlow's FLUX.2-klein-4B takes about 8 s; VigenFlow ships only the 512-token version so that longer prompts fit.

---

## 🔜 Next Steps

Planned for a future release:

- [ ] **SeedVR2** and **VOSR2** support for image upscaling.
- [ ] Extend **BFP16 LoRA support** to more model packages. FLUX.1-schnell LoRAs are available now in the [Beta 4 Ubuntu prerelease](#beta4-setup).

Which model should come next? Vote in [issue #4](https://github.com/jia1217/vigenflow/issues/4).

---

## 🚀 Getting Started

You need:

- [ ] **An AMD Ryzen AI 300-series processor** with an XDNA 2 NPU (`npu2`). VigenFlow is tested on the [AMD Ryzen AI 9 HX 370](https://www.amd.com/en/newsroom/press-releases/2024-10-10-amd-launches-new-ryzen-ai-pro-300-series-processo.html); other processors need matching NPU binaries and runtime support.
- [ ] **Windows x64 or Ubuntu 24.04 x86_64.**
- [ ] **Free disk space for the models you want:** 8 GB for FLUX.2-klein-4B alone, about 65 GB for the original four weight sets, or about 85 GB including Qwen, plus up to 10 GB while they download. [Step 3](#-step-3-download-the-model-weights-one-time) lists each model's size.
- [ ] **The NPU driver**, installed once as shown below.

### 🔌 Install the NPU Driver (One Time)

#### 🪟 Windows

Install the NPU driver from AMD's [Driver Download](https://www.amd.com/en/support/download/drivers.html) page, then restart. **NPU Compute Accelerator Device** should then appear in Device Manager.

#### 🐧 Ubuntu

Install the AMD XDNA driver and XRT from the [Lemonade PPA](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable), then reboot:

```bash
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
sudo apt install amdxdna-dkms libxrt-npu2 libxrt-utils
sudo reboot
```

After the reboot, `xrt-smi examine` should list the NPU.

Ubuntu has no official XRT packages. The PPA's `amdxdna-dkms` driver replaces the one built into the kernel, which does not work with XRT 2.25 on some kernels (see [Troubleshooting](#-troubleshooting)). The Ubuntu Quick Install needs only `amdxdna-dkms`, and offers to install it for you.

### 🧭 Choose How to Install

- **[Beta 4 · FLUX.1-schnell on Ubuntu](#beta4-setup):** download the package and use the built-in browser UI, with LoRAs and custom sizes.
- **[Quick Install](#-quick-install) · multi-model release:** one command installs VigenFlow and OpenWebUI and downloads the models you choose.
- **[Beta 3 release packages](#-installation):** download and start each part yourself, then connect [OpenWebUI Desktop](https://github.com/open-webui/desktop).

---

## ⚡ Quick Install

For **Beta 4's FLUX.1-schnell browser UI**, use the [Beta 4 setup above](#beta4-setup). The commands below install the multi-model release with OpenWebUI.

The quickest way to try VigenFlow: one command installs it, downloads the models you choose, and starts it together with OpenWebUI. Press Enter at the model question for FLUX.2-klein-4B (8 GB), the fastest. Run the same command again to update VigenFlow or add models.

### 🪟 Windows

Install the [NPU driver](#-install-the-npu-driver-one-time) first. Then, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/jia1217/vigenflow/main/install.ps1 | iex
```

The installer checks the NPU driver, downloads the latest release, asks for an install folder and which models to download, and starts VigenFlow and OpenWebUI in their own windows. OpenWebUI runs with [uv](https://github.com/astral-sh/uv), so you need neither Docker nor Python; its first start downloads about 2 GB and takes several minutes, then it opens in your browser. If Windows Firewall asks about `vgf-serve.exe`, either answer works for use on this PC. After a restart, start both again from the Start menu, under VigenFlow.

### 🐧 Ubuntu (Docker or Podman)

```bash
curl -fsSL https://raw.githubusercontent.com/jia1217/vigenflow/main/docker/install.sh | bash
```

The installer checks for the NPU driver and offers to install it, installs Docker if you have neither Docker nor Podman, asks which models to download, and starts VigenFlow together with OpenWebUI. When it finishes, open the address it prints, usually http://localhost:8080. [docs/docker.md](docs/docker.md) shows the same steps by hand, with Podman and Docker Compose.

To use the release packages instead, follow the steps below.

---

## 🤝 Contributors

Contributions to VigenFlow are welcome! Whether you want to contribute code, report bugs, suggest improvements, improve documentation, or test on your hardware, your help is appreciated.

To get involved, open an [issue](https://github.com/jia1217/vigenflow/issues) or submit a [pull request](https://github.com/jia1217/vigenflow/pulls).

Thank you to everyone who contributes to VigenFlow and helps make the project better!

---

## 📚 Citation and Acknowledgement

If you use VigenFlow in research, publications, benchmarks, or projects built on its code, please cite this repository. If VigenFlow's implementation or design ideas inform your own work, an acknowledgement and link are appreciated.

Use GitHub's **Cite this repository** button or the metadata in [CITATION.cff](CITATION.cff). Please include the version or commit you used so others can identify the implementation behind your results.

If your project or product uses VigenFlow, please acknowledge it in your README, project page, or product:

```markdown
Powered by [VigenFlow](https://github.com/jia1217/vigenflow)
```

---

## 📄 License

The VigenFlow source code is licensed under the [Apache License 2.0](LICENSE).

Model weights are **not** covered by this license. Each model and LoRA keeps its original license, including the BFP16 weights in the release packages. See [MODEL_LICENSES.md](MODEL_LICENSES.md) for each model's license and whether it allows commercial use. Third-party libraries are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 📦 Installation

These steps cover the **Beta 3 multi-model packages**. For the **Beta 4 FLUX.1-schnell Ubuntu package**, follow [Try Beta 4 on Ubuntu](#beta4-setup).

Download the package for your system, extract it, download the model weights once, and start the server. The weights are not included in the packages; Step 3 downloads them from Hugging Face.

### 📥 Step 1: Download the Release Package

Go to the [Beta 3 release](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.3) page and download the package for your operating system:

- 🐧 Ubuntu: `vigenflow_0.2.0-beta.3_ubuntu_amd64.zip`
- 🪟 Windows: `vigenflow_0.2.0-beta.3_windows_amd64.zip`

To be notified about new releases, click **Watch → Custom → Releases** on the [VigenFlow repository](https://github.com/jia1217/vigenflow) page.

### 📂 Step 2: Extract the Package

Extract the downloaded `.zip` package on a drive with room for the model weights. Step 3 saves them next to the extracted folder, not inside it.

#### 🐧 Ubuntu

The Ubuntu package needs a few runtime libraries. XRT comes with the [AMD XDNA driver installation](#-install-the-npu-driver-one-time).

```bash
sudo apt install libboost-program-options1.83.0 libgomp1 libpng16-16t64 curl unzip
unzip vigenflow_0.2.0-beta.3_ubuntu_amd64.zip
cd vigenflow_0.2.0-beta.3_ubuntu_amd64
```

#### 🪟 Windows

Extract the ZIP package manually, then open **PowerShell** inside the extracted `vigenflow_0.2.0-beta.3_windows_amd64` folder.

### 💾 Step 3: Download the Model Weights (One Time)

`prepare_base.exe` downloads the pre-packed NPU weights from Hugging Face and saves them in an `all_model_weights` folder **next to** the extracted package folder, so a later VigenFlow version extracted in the same place uses the same weights. Each model has its own repository: [FLUX.1-schnell](https://huggingface.co/Kelsey1217/FLUX.1-schnell-NPU-BFP16), [FLUX.2-klein-4B](https://huggingface.co/Kelsey1217/FLUX.2-klein-4B-NPU-BFP16), [Z-Image-Turbo](https://huggingface.co/Kelsey1217/Z-Image-Turbo-NPU-BFP16) and [Krea 2 Turbo](https://huggingface.co/Kelsey1217/Krea-2-Turbo-NPU-BFP16). No Hugging Face account is needed.

#### 🐧 Ubuntu

```bash
./prepare/prepare_base.exe
```

#### 🪟 Windows

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
[preparation instructions](src/prepare_weights/README.md#qwen-image-21-turbo)
and [server setup](src/server_unified/README.md#qwen-image-21-turbo).

If you used v0.2.0 Beta 1, move its `all_model_weights` folder so that it sits next to the new package folder before you run the command. The three models you already have are then not downloaded again.

You can run `prepare_base.exe` again at any time. It only downloads models whose weights have changed, and it resumes an interrupted download.

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

The server listens on port **11283** and makes all supported models available to OpenWebUI. To see all server options, run `./vgf-serve -h` (Ubuntu) or `.\vgf-serve.exe -h` (Windows).

The default model is `z-image-turbo`. If you did not download its weights, start the server with `--model` and the ID of a model you have, for example `--model flux2-klein-4B`.

---

## 🖼️ Use the Built-in Browser UI

**Beta 4 package:** run `bash start.sh` and open **http://127.0.0.1:11285/**. Its FLUX.1-schnell UI includes LoRAs and custom resolutions; see [the Beta 4 setup](#beta4-setup).

Enter a prompt, select optional LoRAs, choose an image size and seed, and click **Generate image**. The result includes a PNG download, a full-size preview, and E2E runtime in seconds. **Advanced settings** contains the Balanced and Performance memory modes.

The UI is included in the Beta 4 server executable and uses the same model weights as the API. It needs no separate UI installation or account. The Beta 3 packages and current quick installers use OpenWebUI as described below.

---

## 🔗 Connect with OpenWebUI Desktop (Optional)

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

The screenshot shows the **Images** settings page set up for VigenFlow.

<img src="assets/openwebUI.png" width="500" alt="OpenWebUI Images settings for VigenFlow" />

---

## 🔧 Troubleshooting

### Ubuntu: `run.exe failed with exit code 134` for every model

The AMD XDNA driver built into some Linux kernels rejects the messages that XRT 2.25 sends to the NPU. This was reported on kernel `7.0.0-38`. Every model worker then stops, `vgf-serve` prints `run.exe failed with exit code 134`, and `sudo dmesg` shows lines like these:

```text
Message callback ret -22
Channel in bad state
```

To fix it, install the driver from the Lemonade PPA and reboot. The `amdxdna-dkms` package builds the driver for your kernel, and it is loaded instead of the built-in one:

```bash
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
sudo apt install amdxdna-dkms
sudo reboot
```

After the reboot, `modinfo -n amdxdna` should print a path that contains `updates/dkms`, and `xrt-smi examine` should list the NPU.
