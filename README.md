# User_guidance_test

## Getting Started

Follow these steps to set up, build, and run the project on your local machine.

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
bash scripts/setup_deps.sh
```

Note: This script will clone the necessary repositories, update submodules, configure Rust, and build the external libraries.

### 3. Build the Project
Once the dependencies are successfully set up, compile the main C++ project using `make`:

```bash
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