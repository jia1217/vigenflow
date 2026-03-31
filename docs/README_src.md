### Option : Build from Source

You can also clone the repository and build the executable yourself.

#### 1. Clone the Repository

```bash
git clone https://github.com/jia1217/vigenflow.git
cd vigenflow/src
```

#### 2. Install External Dependencies

Before building the main project, you need to set up the required external libraries, such as `tokenizers-cpp`.

Run the following commands from the project root directory:

```bash
mkdir -p external_lib
bash scripts/setup_deps.sh
```

This script will clone the necessary repositories, initialize submodules, configure Rust, and build the external dependencies automatically.

#### 3. Build the Project

```bash
sudo apt update && sudo apt install g++-13 -y
make
```

#### 4. Run the Executable

After a successful build, the `run.exe` executable will be generated. Start the program with:

```bash
./run.exe
```

To view the available command-line arguments, run:

```bash
./run.exe --help
```

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
 ./run.exe /home/kelsey/repo_test/model_weights /home/kelsey/repo_test/vigenflow/build/Z-Image-T
urbo 42 1024 1024 4 "Cover of 'JACOB VAN RUISDAEL: A COMPLETE CATALOGGW OF HIS PAINTINGS, DRAWINGS AND ETCHINGS' by SEYMOUR SLIVE, featuring a serene landscape pai
nting with figures resting under trees." /home/kelsey/img_test/img_test.png
```

---
