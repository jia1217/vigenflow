#!/usr/bin/env bash
# Builds prepare_base.exe, which downloads the pre-packed NPU weights from Hugging Face.
# Needs g++-13 (C++23) and nlohmann-json3-dev; curl at run time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CXX="${CXX:-g++-13}"
OUT="${OUT:-prepare_base.exe}"

mkdir -p "$(dirname "$OUT")"

"$CXX" -std=c++23 -O2 -Wall -Wextra -Wno-missing-field-initializers -pthread ${CXXFLAGS:-} \
  src/prepare_base_main.cpp src/packed_format.cpp src/hf_download.cpp \
  ${LDFLAGS:-} \
  -o "$OUT"

echo "Built $SCRIPT_DIR/$OUT"
