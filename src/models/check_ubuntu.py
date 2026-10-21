#!/usr/bin/env python3
"""Verify that Ubuntu workers run without external model host libraries.

Copies each executable to an isolated temporary folder and runs --help without
model weights or an NPU. System libraries such as XRT must be installed.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

MODELS = (
    "flux1_schnell", "flux2_klein_4B", "flux2_klein_4B_edit", "krea2_turbo",
    "z_image_turbo", "qwen_image_2_1_turbo", "qwen_image_2_1_turbo_edit",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe_models", type=Path, nargs="?",
                        default=Path(__file__).resolve().parent.parent / "exe_models")
    args = parser.parse_args()
    for model in MODELS:
        source = args.exe_models / model / "run.exe"
        dynamic = subprocess.run(["readelf", "-d", str(source)], check=True,
                                 capture_output=True, text=True).stdout
        if "libhost_" in dynamic:
            raise RuntimeError(f"{model} still references an external host library")
        with tempfile.TemporaryDirectory(prefix="vgf-worker-") as folder:
            isolated = Path(folder)
            worker = isolated / "run.exe"
            shutil.copy2(source, worker)
            worker.chmod(0o755)
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            env.pop("LD_PRELOAD", None)
            # No writable temp directory is needed to extract the embedded host.
            env["TMPDIR"] = str(isolated / "no-extraction-directory")
            result = subprocess.run([str(worker), "--help"], cwd=isolated,
                                    env=env, capture_output=True, text=True, timeout=30)
            if result.returncode or "--weights_path" not in result.stdout:
                raise RuntimeError(f"{model}: {result.stdout}\n{result.stderr}")
            if set(isolated.iterdir()) != {worker}:
                raise RuntimeError(f"{model} unexpectedly extracted runtime files")
        print(f"PASS: {model} runs with only run.exe in its folder")


if __name__ == "__main__":
    main()
