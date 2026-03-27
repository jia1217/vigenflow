#!/bin/bash

# Exit immediately if any command fails
set -e

# Navigate to the external_lib directory relative to this script's location
cd "$(dirname "$0")/../external_lib"

echo "Cloning tokenizers-cpp repository..."
# Only clone if the directory doesn't already exist to prevent errors on multiple runs
if [ ! -d "tokenizers-cpp" ]; then
    git clone https://github.com/mlc-ai/tokenizers-cpp.git
else
    echo "tokenizers-cpp already exists. Skipping clone."
fi

cd tokenizers-cpp

echo "Creating build directory..."
mkdir -p build

echo "Initializing and updating submodules..."
git submodule update --init --recursive

cd build

echo "Running CMake..."
cmake ..

echo "Running Make..."
make

echo "Setting rustup default to stable..."
rustup default stable

echo "Running Make again..."
make

echo "Dependencies setup completed successfully!"