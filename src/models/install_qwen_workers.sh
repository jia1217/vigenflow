#!/usr/bin/env bash
# Refresh the two prebuilt Qwen host libraries in src/lib from the native host
# projects, then rebuild their Ubuntu workers, which embed them like the others.
# QWEN_SOURCE_ROOT defaults to the sibling NPU_qwen development tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QWEN_SOURCE_ROOT="${QWEN_SOURCE_ROOT:-$SRC_ROOT/../../NPU_qwen}"
LIB_DIR="${LIB_DIR:-$SRC_ROOT/lib}"

for task in turbo image; do
  project="$QWEN_SOURCE_ROOT/NPU_qwen_$task"
  model=qwen_image_2_1_turbo
  if [[ "$task" == image ]]; then model=qwen_image_2_1_turbo_edit; fi
  if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    make -C "$project" libhost -j"${JOBS:-2}"
  fi
  # Strip only the copy; retain debug symbols in the development tree. The
  # dynamic symbol table, which holds <model>_host_main, is kept.
  install -s -m 0755 "$project/build/lib/libhost_$model.so" "$LIB_DIR/libhost_$model.so"
  make -C "$SCRIPT_DIR/$model" check >/dev/null
  echo "Installed $LIB_DIR/libhost_$model.so and rebuilt $SRC_ROOT/exe_models/$model/run.exe"
done
