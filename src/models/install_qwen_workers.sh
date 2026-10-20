#!/usr/bin/env bash
# Install native Ubuntu Qwen executables; weights and NPU bundles are separate.
# QWEN_SOURCE_ROOT defaults to the sibling NPU_qwen development tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QWEN_SOURCE_ROOT="${QWEN_SOURCE_ROOT:-$SRC_ROOT/../../NPU_qwen}"
EXE_MODELS_DIR="${EXE_MODELS_DIR:-$SRC_ROOT/exe_models}"

for task in turbo image; do
  project="$QWEN_SOURCE_ROOT/NPU_qwen_$task"
  model=qwen_image_2_1_turbo
  if [[ "$task" == image ]]; then model=qwen_image_2_1_turbo_edit; fi
  if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    make -C "$project" host -j"${JOBS:-2}"
  fi
  if [[ "$(od -An -tx1 -N4 "$project/run.exe" | tr -d ' \n')" != 7f454c46 ]]; then
    echo "Expected an Ubuntu ELF worker: $project/run.exe" >&2
    exit 1
  fi
  mkdir -p "$EXE_MODELS_DIR/$model"
  # Strip only the installed copy; retain debug symbols in the development tree.
  install -s -m 0755 "$project/run.exe" "$EXE_MODELS_DIR/$model/run.exe"
  "$EXE_MODELS_DIR/$model/run.exe" --help >/dev/null
  echo "Installed $EXE_MODELS_DIR/$model/run.exe"
done
