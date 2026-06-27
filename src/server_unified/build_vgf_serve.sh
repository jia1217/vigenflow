#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${OS:-}" == "Windows_NT" ]]; then
  bash "$SCRIPT_DIR/build_windows.sh"
else
  bash "$SCRIPT_DIR/build_ubuntu.sh"
fi
