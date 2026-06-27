#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VERSION="${VERSION:-0.1.2}"
ARCH="${ARCH:-amd64}"
PACKAGE_NAME="${PACKAGE_NAME:-vigenflow_${VERSION}_ubuntu_${ARCH}}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/release}"
SERVER_OUT_DIR="${SERVER_OUT_DIR:-$SCRIPT_DIR/vigenflow_ubuntu}"

resolve_dir() {
  local label="$1"
  local env_name="$2"
  shift 2

  local override="${!env_name:-}"
  if [[ -n "$override" ]]; then
    if [[ -d "$override" ]]; then
      printf '%s\n' "$override"
      return
    fi
    echo "$env_name points to a missing directory: $override" >&2
    exit 1
  fi

  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  echo "Could not find $label. Set $env_name=/path/to/$label and rerun." >&2
  exit 1
}

if ! command -v zip >/dev/null 2>&1; then
  echo "zip is required to create the release package." >&2
  echo "Install it with: sudo apt install zip" >&2
  exit 1
fi

EXE_MODELS_DIR="$(resolve_dir "exe_models" EXE_MODELS_DIR \
  "$SRC_ROOT/exe_models" \
  "$REPO_ROOT/deb_lib/vigenflow/exe_models" \
  "$REPO_ROOT/deb_lib/vgf-zip/vigenflow/exe_models")"
NPU_FILES_DIR="$(resolve_dir "npu_files" NPU_FILES_DIR \
  "$SRC_ROOT/npu_files" \
  "$REPO_ROOT/deb_lib/vigenflow/npu_files" \
  "$REPO_ROOT/deb_lib/vgf-zip/vigenflow/npu_files")"
MODEL_WEIGHTS_DIR="$(resolve_dir "model_weights" MODEL_WEIGHTS_DIR \
  "$SRC_ROOT/model_weights" \
  "$REPO_ROOT/deb_lib/vigenflow/model_weights" \
  "$REPO_ROOT/deb_lib/vgf-zip/vigenflow/model_weights")"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  OUT_DIR="$SERVER_OUT_DIR" bash "$SCRIPT_DIR/build_ubuntu.sh"
fi

SERVER_BIN="${SERVER_BIN:-$SERVER_OUT_DIR/vgf-serve}"
if [[ ! -f "$SERVER_BIN" ]]; then
  echo "Server binary not found: $SERVER_BIN" >&2
  echo "Run build_ubuntu.sh first or set SERVER_BIN=/path/to/vgf-serve." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
STAGE_DIR="$OUT_ROOT/$PACKAGE_NAME"
ZIP_PATH="$OUT_ROOT/$PACKAGE_NAME.zip"

rm -rf "$STAGE_DIR" "$ZIP_PATH"
mkdir -p "$STAGE_DIR"

install -m 0755 "$SERVER_BIN" "$STAGE_DIR/vgf-serve"
cp -a "$EXE_MODELS_DIR" "$STAGE_DIR/exe_models"
cp -a "$NPU_FILES_DIR" "$STAGE_DIR/npu_files"
cp -a "$MODEL_WEIGHTS_DIR" "$STAGE_DIR/model_weights"

find "$STAGE_DIR/model_weights" -type d -name '.*_hf_cache' -prune -exec rm -rf {} +
find "$STAGE_DIR/model_weights" -type d -name '.cache' -prune -exec rm -rf {} +
find "$STAGE_DIR/model_weights" -type f \( \
  -name '*.safetensors' -o \
  -name '*.incomplete' -o \
  -name '*.metadata' \
\) -delete

for catalog in \
  z_image_q41_lora_models.jsonc \
  z_image_bf16_lora_models.jsonc \
  flux_klein_lora_models.jsonc \
  flux_klein_edit_lora_models.jsonc; do
  cp "$SCRIPT_DIR/$catalog" "$STAGE_DIR/$catalog"
done

find "$STAGE_DIR/exe_models" -type f -name 'run.exe' -exec chmod +x {} +

(
  cd "$OUT_ROOT"
  zip -r "$ZIP_PATH" "$PACKAGE_NAME"
)

echo "Created $ZIP_PATH"
echo "Release folder: $STAGE_DIR"
