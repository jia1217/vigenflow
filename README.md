## Getting Started

Follow the steps below to set up, build, and run the project on your local machine.

Before proceeding, please make sure that the AMD XDNA driver has been installed on your system. You can find more details in the [installation guide](https://github.com/Xilinx/mlir-aie).

### 📥 Download model weights via CLI
The fastest way to download the full weight structure:

```bash
sudo apt install python3.14-venv
python3 -m venv myenv
source myenv/bin/activate
sudo apt update
```

1. Install the Hub library:
   `pip install -U "huggingface_hub[cli]"`
2. Login:
   `hf auth login`
3. Download the whole repo:
   `hf download Kelsey1217/Z-Image-Turbo --local-dir ./Z-Image-Turbo`


## Installation

You can use either of the following methods to run **VigenFlow**.

### Option A: Install from the Release Package (Recommended)

The easiest way to get started is to download the [released](https://github.com/jia1217/vigenflow/releases/tag/vigen1.0) `.deb` package and install it directly:

```bash
sudo dpkg -i vigenflow_1.0_amd64.deb
```

If any dependencies are missing, run:

```bash
sudo apt-get install -f
```

### Start the Server After Installation

After installing the package, you should see a message indicating that `vgf-serve` has been installed successfully and added to your system path.

You can then verify the installation and view the available server options by running:

```bash
vgf-serve -h
```

This command will display the supported arguments, such as the model weights path, NPU files path, output directory, port, and other server settings.

After that, you can launch the server with your desired configuration. For example:

```bash
vgf-serve \
  -w /path/to/model_weights \
  -k false \
  -m model_name \
```

After installation, you can integrate VigenFlow with **Open WebUI** and use Open WebUI to call your own server.




### Option B: Build from Source

You can also clone the repository and build the executable yourself. Please see the docs for more information.

## Documentation

For more detailed build and usage instructions, please refer to:

- `doc/readme_*.md`
- [https://github.com/jia1217/vigenflow/tree/main/docs](YOUR_LINK_HERE)

## Usage

### 📊 Supported Configurations

#### 1. Supported Aspect Ratios
The model is optimized for these specific dimensions. Using other resolutions may result in unexpected cropping or performance degradation.

| Aspect Ratio | Resolution (W x H) | Recommended Use |
| :--- | :--- | :--- |
| **1:1** | 1024 x 1024 | Square (Standard) |
| **4:3** | 1024 x 768 | Landscape (Classic) |
| **3:4** | 768 x 1024 | Portrait |
| **16:9** | 1024 x 576 | Widescreen |
| **9:16** | 576 x 1024 | Vertical / Social Media |

#### 2. Denoising Steps
You can choose the number of steps based on your requirement for speed vs. quality:

* **4 Steps (Turbo):** Highly recommended for speed. Generates an image in approximately **one minute** on supported NPU hardware.
* **8 Steps (High Quality):** Provides more detail and better texture refinement at the cost of longer generation time.


