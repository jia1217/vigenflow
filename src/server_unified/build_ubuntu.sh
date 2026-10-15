#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CXX="${CXX:-g++}"
OUT_DIR="${OUT_DIR:-vigenflow_ubuntu}"
OUT="${OUT:-${OUT_DIR}/vgf-serve}"

mkdir -p "$(dirname "$OUT")"

SOURCES=(
  src/server_main.cpp
  src/server_config.cpp
  src/file_utils.cpp
  src/input_image.cpp
  src/request_parser.cpp
  src/worker.cpp
  src/http_handlers.cpp
  src/tcp_server.cpp
  src/cli.cpp
)

"$CXX" -std=c++17 -O2 ${CXXFLAGS:-} \
  "${SOURCES[@]}" \
  -I./src -I./include \
  -pthread \
  -lboost_system \
  -lboost_filesystem -lpng \
  ${LDFLAGS:-} \
  -o "$OUT"

echo "Built $(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
