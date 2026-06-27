#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CXX="${CXX:-g++}"
OUT_DIR="${OUT_DIR:-vigenflow_ubuntu}"
OUT="${OUT:-${OUT_DIR}/vgf-serve}"

mkdir -p "$OUT_DIR"

SOURCES=(
  server_main.cpp
  server_config.cpp
  file_utils.cpp
  request_parser.cpp
  z_image_bf16_export.cpp
  flux_klein_bf16_export.cpp
  worker.cpp
  http_handlers.cpp
  tcp_server.cpp
  cli.cpp
)

"$CXX" -std=c++17 -O2 ${CXXFLAGS:-} \
  "${SOURCES[@]}" \
  -I. -I./include \
  -pthread \
  -lboost_system \
  -lboost_filesystem \
  ${LDFLAGS:-} \
  -o "$OUT"

echo "Built $SCRIPT_DIR/$OUT"
