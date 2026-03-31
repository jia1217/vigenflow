#!/bin/bash

# Exit immediately if any command fails
set -e

echo "=== Checking System Dependencies ==="

# 1. CHECK FOR AND INSTALL RUST IF MISSING
if ! command -v rustup &> /dev/null; then
    echo "rustup not found. Installing Rust automatically..."
    # The -y flag installs the default profile without prompting the user
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
    echo "rustup is already installed."
fi

# 2. LOAD RUST ENVIRONMENT
echo "Loading Rust environment..."
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

echo "Setting rustup default to stable..."
rustup default stable

# 3. NAVIGATE TO EXTERNAL_LIB
echo "=== Setting up external_lib ==="
cd "$(dirname "$0")/../external_lib"

echo "Cloning tokenizers-cpp repository..."
if [ ! -d "tokenizers-cpp" ]; then
    git clone https://github.com/mlc-ai/tokenizers-cpp.git
else
    echo "tokenizers-cpp already exists."
fi

cd tokenizers-cpp

echo "Initializing submodules..."
git submodule update --init --recursive

# 4. WIPE OUT ANY BROKEN BUILDS AND START FRESH
echo "Cleaning old build files..."
rm -rf build
mkdir build
cd build

# 5. COMPILE
echo "=== Compiling tokenizers-cpp ==="
echo "Running CMake..."
cmake ..

echo "Running Make..."
make

echo "=== Dependencies setup completed successfully! ==="