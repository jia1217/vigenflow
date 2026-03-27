# User_guidance_test

## Getting Started

Follow these steps to set up, build, and run the project on your local machine.

### 📥 Download model weights via CLI
The fastest way to download the full weight structure:

`sudo apt install python3.14-venv`

`python3 -m venv myenv`

`source myenv/bin/activate`


`sudo apt update`
`sudo apt install libstb-dev`

to install the pacakge so that you can get the img.png file directly


1. Install the Hub library:
   `pip install -U "huggingface_hub[cli]"`
2. Login:
   `hf auth login`
3. Download the whole repo:
   `hf download Kelsey1217/Z-Image-Turbo --local-dir ./Z-Image-Turbo`

**Option A: Clone the Repository (Recommended)**
Open your terminal and run the following commands to clone the repo and navigate into the folder:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**Option B: Download the `.zip` File**
### 1. Extract the Files
Once you have downloaded the project `.zip` file, extract its contents. Open your terminal and navigate into the root directory of the extracted folder:

```bash
cd path/to/extracted/folder
```

### 2. Install External Dependencies
Before compiling the main code, you need to pull in and build the required external libraries (such as tokenizers-cpp). We have provided a script to automate this. Run the following command from the root directory

```bash
mkdir external_lib
bash scripts/setup_deps.sh
```

Note: This script will clone the necessary repositories, update submodules, configure Rust, and build the external libraries.

### 3. Build the Project
Once the dependencies are successfully set up, compile the main C++ project using `make`:

```bash
sudo apt update && sudo apt install g++-13 -y
make
```

### 4. Run the Executable

After a successful build, a run.exe file will be generated. You can execute the program by running:

```bash
./run.exe
```

You can view the available command-line arguments (like setting the image height, width, or prompt) by using the help flag:

```bash
./run.exe --help
```


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

### Option 1: Direct Execution (Single Image)

You can run inference directly by executing the `run.exe` binary. This method requires passing all configuration parameters as sequential command-line arguments.

#### Command Syntax

```bash
./run.exe <weights_path> <npu_files_path> <seed> <H> <W> <steps> "<prompt>" <output_png>
```

#### Argument Description

1. **`weights_path`**: Directory containing the model weights.
2. **`npu_files_path`**: Directory containing NPU-specific compiled model files (e.g., context files).
3. **`seed`**: Integer seed for random number generation.
4. **`H`**: Output image height in pixels.
5. **`W`**: Output image width in pixels.
6. **`steps`**: Number of inference steps.
7. **`prompt`**: The text description enclosed in quotes.
8. **`output_png`**: Full path including filename where the resulting PNG should be saved.

#### Example

```bash
 ./run.exe /home/kelsey/repo_test/npu-image-diffusion /home/kelsey/repo_test/npu-image-diffusion/build/Z-Image-T
urbo 42 1024 1024 4 "Cover of 'JACOB VAN RUISDAEL: A COMPLETE CATALOGGW OF HIS PAINTINGS, DRAWINGS AND ETCHINGS' by SEYMOUR SLIVE, featuring a serene landscape pai
nting with figures resting under trees." /home/kelsey/img_test/img_test.png
```

---

### Option 2: Server Mode (API-Based)

This method compiles and runs an HTTP server that listens for generation requests. When a request is received via POST, the server handles the execution of the underlying NPU worker and returns the generated image data.

#### Step 1: Configure Paths (main.cpp)

The server is designed to be portable. The `run_worker` function has been polished to use **relative paths**, ensuring the NPU executable can be found regardless of where the project is installed on the system.

Update your `main.cpp` with the following implementation:

```cpp
#include <string>
#include <atomic>
#include <iostream>

/**
 * Executes the NPU worker process. 
 * Polished for portability using relative paths.
 */
std::string run_worker(const GenParams& p)
{
    const std::string exe =
        "/run.exe";

    const std::string workdir =
        "/";

   ....
}
```
#### Step 2: Compile the Server

Use `g++` to compile the server code. Ensure you link the necessary threads and Boost libraries.

```bash
g++ -std=c++17 -O2 main.cpp -lpthread -lboost_system -o server
```

#### Step 3: Start the Server

Run the generated server binary. You must specify the base paths for the weights and NPU files using command-line flags. These paths act as the global defaults for the server.

```bash
./server --weights_path /home/kelsey/repo_test/npu-image-diffusion --npu_files_path /home/kelsey/repo_test/npu-image-diffusion/build/Z-Image-Turbo
```

By default, the server will run and listen on `http://0.0.0.0:1234`.

#### Step 4: Trigger Generation

Once the server is running, you can send generation requests to it. A sample bash script is provided to quickly test the server. Open a new terminal window and run:

```bash
bash test_server.sh
```