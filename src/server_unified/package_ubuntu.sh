#!/usr/bin/env bash
# Builds vgf-serve and prepare_base.exe and creates the Ubuntu release zip:
#
#   vigenflow_<VERSION>_ubuntu_amd64/
#     server_running/vgf-serve
#     server_running/exe_models/<model>/run.exe   launchers from src/exe_models
#     server_running/lib/libhost_<model>.so        from src/lib (the launchers' RUNPATH is $ORIGIN/../../lib)
#     server_running/images/
#     all_npu_files/<model>/                       from src/npu_files
#     prepare/prepare_base.exe, prepare_models.jsonc
#     README.md, VERSION, SHA256SUMS
#     LICENSE, MODEL_LICENSES.md, THIRD_PARTY_NOTICES.md
#
# The model weights are not packaged: prepare/prepare_base.exe downloads them into
# all_model_weights/ next to the extracted package folder, where vgf-serve reads them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
PREPARE_DIR="$SRC_ROOT/prepare_weights"

VERSION="${VERSION:-0.2.0-beta.3}"
ARCH="${ARCH:-amd64}"
PACKAGE_NAME="${PACKAGE_NAME:-vigenflow_${VERSION}_ubuntu_${ARCH}}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/release}"
SERVER_OUT_DIR="${SERVER_OUT_DIR:-$SCRIPT_DIR/vigenflow_ubuntu}"
EXE_MODELS_DIR="${EXE_MODELS_DIR:-$SRC_ROOT/exe_models}"
LIB_DIR="${LIB_DIR:-$SRC_ROOT/lib}"
NPU_FILES_DIR="${NPU_FILES_DIR:-$SRC_ROOT/npu_files}"

MODELS=(flux1_schnell flux2_klein_4B flux2_klein_4B_edit krea2_turbo z_image_turbo qwen_image_2_1_turbo qwen_image_2_1_turbo_edit)

if ! command -v zip >/dev/null 2>&1; then
  echo "zip is required to create the release package." >&2
  echo "Install it with: sudo apt install zip" >&2
  exit 1
fi

for dir in "$EXE_MODELS_DIR" "$LIB_DIR" "$NPU_FILES_DIR"; do
  if [[ ! -d "$dir" ]]; then
    echo "Missing directory: $dir" >&2
    exit 1
  fi
done
# vgf-serve refuses to start unless every worker exists.
for model in "${MODELS[@]}"; do
  if [[ ! -f "$EXE_MODELS_DIR/$model/run.exe" ]]; then
    echo "Missing worker: $EXE_MODELS_DIR/$model/run.exe" >&2
    echo "Build existing launchers with make -C src/models/<model>; install Qwen with bash src/models/install_qwen_workers.sh." >&2
    exit 1
  fi
done

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  OUT_DIR="$SERVER_OUT_DIR" bash "$SCRIPT_DIR/build_ubuntu.sh"
  bash "$PREPARE_DIR/build_ubuntu.sh"
fi

SERVER_BIN="${SERVER_BIN:-$SERVER_OUT_DIR/vgf-serve}"
PREPARE_BIN="${PREPARE_BIN:-$PREPARE_DIR/prepare_base.exe}"
for bin in "$SERVER_BIN" "$PREPARE_BIN"; do
  if [[ ! -f "$bin" ]]; then
    echo "Binary not found: $bin" >&2
    echo "Build it first or unset SKIP_BUILD." >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
STAGE_DIR="$OUT_ROOT/$PACKAGE_NAME"
ZIP_PATH="$OUT_ROOT/$PACKAGE_NAME.zip"

rm -rf "$STAGE_DIR" "$ZIP_PATH"
mkdir -p "$STAGE_DIR/server_running/images" "$STAGE_DIR/server_running/exe_models" \
  "$STAGE_DIR/server_running/lib" "$STAGE_DIR/prepare"

install -m 0755 "$SERVER_BIN" "$STAGE_DIR/server_running/vgf-serve"
for model in "${MODELS[@]}"; do
  install -D -m 0755 "$EXE_MODELS_DIR/$model/run.exe" "$STAGE_DIR/server_running/exe_models/$model/run.exe"
done
cp -a "$LIB_DIR"/libhost_*.so "$STAGE_DIR/server_running/lib/"
cp -a "$NPU_FILES_DIR" "$STAGE_DIR/all_npu_files"

install -m 0755 "$PREPARE_BIN" "$STAGE_DIR/prepare/prepare_base.exe"
cp "$PREPARE_DIR/prepare_models.jsonc" "$STAGE_DIR/prepare/"

for doc in LICENSE MODEL_LICENSES.md THIRD_PARTY_NOTICES.md; do
  cp "$REPO_ROOT/$doc" "$STAGE_DIR/$doc"
done
cp -a "$REPO_ROOT/model_licenses" "$STAGE_DIR/model_licenses"
cp "$SCRIPT_DIR/README_ubuntu_package.md" "$STAGE_DIR/README.md"
printf '%s\n' "$VERSION" > "$STAGE_DIR/VERSION"

(
  cd "$STAGE_DIR"
  find . -type f ! -path './SHA256SUMS' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
)

(
  cd "$OUT_ROOT"
  zip -qr "$ZIP_PATH" "$PACKAGE_NAME"
  sha256sum "$PACKAGE_NAME.zip" > "$PACKAGE_NAME.zip.sha256"
)

echo "Created $ZIP_PATH"
echo "Release folder: $STAGE_DIR"
echo "Archive checksum: $ZIP_PATH.sha256"
