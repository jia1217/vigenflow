#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v cygpath >/dev/null 2>&1; then
  cmd.exe /c "$(cygpath -w "$SCRIPT_DIR/build_windows.bat")"
else
  cmd.exe /c "$SCRIPT_DIR/build_windows.bat"
fi
