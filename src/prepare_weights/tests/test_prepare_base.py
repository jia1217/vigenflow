"""Run prepare_base.exe against a local Hugging Face fixture; no models, network or credentials needed.

python3 tests/test_prepare_base.py ./prepare_base.exe              (Ubuntu)
python tests\\test_prepare_base.py vigenflow_win\\prepare_base.exe  (Windows, after build_windows.bat)
"""
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import urllib.parse

COMMIT = "a" * 40
# The cache folder names the commit by 12 hex digits on Windows (path length), by all 40 elsewhere.
COMMIT_DIR = COMMIT[:12] if sys.platform == "win32" else COMMIT
TOKEN = "hf_prepare_base_fixture_only"


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "prepare_base.exe").resolve()
    repos = {}
    requests = []
    entered, release = threading.Event(), threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if self.headers.get("Authorization") != "Bearer " + TOKEN:
                self.send_error(401)
                return
            requests.append((path, self.headers.get("Range")))
            if path.startswith("/api/models/"):
                repo = path[len("/api/models/"):].split("/revision/")[0]
                if repo == "fixture/blocked":
                    entered.set()
                    release.wait(20)
                    self.send_error(404)
                    return
                if repo not in repos:
                    self.send_error(404)
                    return
                data = json.dumps({"sha": COMMIT, "siblings": [
                    {"rfilename": name, "size": len(blob), "blobId": hashlib.sha256(blob).hexdigest()}
                    for name, blob in repos[repo].items()
                ]}).encode()
            else:
                repo, name = path.lstrip("/").split("/resolve/" + COMMIT + "/", 1)
                data = repos[repo][name]
            size = len(data)
            range_header = self.headers.get("Range")
            if range_header:
                start, end = range_header.removeprefix("bytes=").split("-")
                start, end = int(start), int(end) if end else size - 1
                data = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # Spaces, shell metacharacters and an environment-variable-looking name in every path.
        with tempfile.TemporaryDirectory(prefix="prepare base ") as temporary:
            root = Path(temporary) / "models & %VGF_TEST_MARKER%"
            bindir = root / "bin"
            bindir.mkdir(parents=True)
            shutil.copy2(binary, bindir / "prepare_base.exe")
            env = os.environ.copy()
            env.update(HF_ENDPOINT=f"http://127.0.0.1:{server.server_port}", HF_TOKEN=TOKEN,
                       HF_HOME=str(root / "hf home"), HF_HUB_CACHE=str(root / "hf home/hub"),
                       VGF_TEST_MARKER="must-not-expand", NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
            cwd = root / "unrelated cwd"
            cwd.mkdir()

            def run(*options, expected=0, stdin_text=None, argv0=None):
                result = subprocess.run([argv0 or str(bindir / "prepare_base.exe"), *map(str, options)],
                                        executable=str(bindir / "prepare_base.exe"), cwd=cwd, env=env, text=True,
                                        input=stdin_text, stdin=None if stdin_text is not None else subprocess.DEVNULL,
                                        capture_output=True, timeout=60)
                if result.returncode != expected:
                    raise AssertionError(f"{options}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
                return result.stdout + result.stderr

            def write_config(models):
                (bindir / "prepare_models.jsonc").write_text(
                    "// comments are allowed\n" + json.dumps({"weights_root": "../weights", "models": models}),
                    encoding="utf-8")

            tiny = {"id": "tiny", "recipe": "flux2_klein_4b", "source": "fixture/source",
                    "packed_repo": "fixture/packed", "packed_path": "tiny"}
            write_config([tiny])
            assert "--pack" in run("--help")
            run("--unknown", expected=2)

            # A model folder to pack. Every byte value in the large file catches text-mode pipe corruption.
            weights = root / "weights/tiny"
            (weights / "sub dir").mkdir(parents=True)
            (weights / "sub dir/small.bin").write_bytes(b"\x00\r\n\x1a\xff" * 100)
            (weights / "tokenizer.json").write_bytes(b'{"fixture":true}\r\n')
            (weights / "large.bin").write_bytes(bytes(range(256)) * ((65 << 20) // 256))
            packed = root / "packed"
            run("--pack", "--pack_dir", packed, "--shard_gb", "0.01")
            blobs = {"tiny/" + p.name: p.read_bytes() for p in (packed / "tiny").iterdir() if p.is_file()}
            shards = sorted(name for name in blobs if name.endswith(".safetensors"))
            assert len(shards) > 1 and "tiny/model.safetensors.index.json" in blobs, sorted(blobs)
            repos["fixture/packed"] = blobs
            print("PASS: help, unknown option, config next to the exe, pack into shards", flush=True)

            # Resume an interrupted install: a small shard is already unpacked, and the large shard's
            # parallel ranges are partly downloaded.
            large_name = next(name for name, blob in blobs.items() if len(blob) >= 64 << 20)
            large = blobs[large_name]
            unpacked = next(name for name in shards if name != large_name)
            downloaded = root / "downloaded"
            stage = downloaded / "tiny.partial"
            stage.mkdir(parents=True)
            first = blobs[unpacked]
            header_len = struct.unpack_from("<Q", first)[0]
            for name, info in json.loads(first[8:8 + header_len]).items():
                if name == "__metadata__":
                    continue
                out = stage / name
                out.parent.mkdir(parents=True, exist_ok=True)
                begin, end = info["data_offsets"]
                out.write_bytes(first[8 + header_len + begin:8 + header_len + end])
            (stage / ".prepare_base_progress.json").write_text(json.dumps({
                "blobs": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
                "done": [unpacked],
            }))
            partial = downloaded / ".hf_sources/fixture--packed" / COMMIT_DIR / (large_name + ".partial")
            partial.parent.mkdir(parents=True)
            with partial.open("wb") as f:
                f.write(large[:19])
                f.truncate(len(large))
            Path(str(partial) + ".ranges").write_text(json.dumps({"size": len(large), "done": [19, 0, 0, 0]}))
            log = run("--weights_root", downloaded, "--connections", "4")
            assert "resuming: 1" in log, log
            assert any(path.endswith(large_name) and byte_range and byte_range.startswith("bytes=19-")
                       for path, byte_range in requests), (requests, log)
            for original in weights.rglob("*"):
                if original.is_file():
                    assert original.read_bytes() == (downloaded / "tiny" / original.relative_to(weights)).read_bytes(), original
            assert (downloaded / "tiny/prepare_manifest.json").is_file()
            assert not stage.exists() and not (downloaded / ".hf_sources").exists()
            assert "up to date" in run("--weights_root", downloaded)
            print("PASS: download, shard and range resume, byte-exact unpack, cleanup, up to date", flush=True)

            # Started by name, as from PATH: the config is still the one next to the exe, not in the cwd.
            shutil.rmtree(weights)
            log = run(argv0="prepare_base.exe")
            assert f"weights root: {root / 'weights'}" in log, log
            assert (weights / "large.bin").read_bytes() == (downloaded / "tiny/large.bin").read_bytes()
            print("PASS: config and weights root found from the exe's own folder", flush=True)

            # A Krea model asks for its license first; no answer or another answer downloads nothing.
            krea = dict(tiny, id="krea2_turbo", recipe="krea2_turbo", source="krea/fixture")
            write_config([krea])
            krea_root = root / "krea weights"
            for answer in (None, "no\n"):
                log = run("--weights_root", krea_root, expected=1, stdin_text=answer)
                assert "own license" in log and "not accepted" in log, log
                assert not (krea_root / "krea2_turbo").exists() and not (krea_root / ".hf_sources").exists(), log
            log = run("--weights_root", krea_root, stdin_text="  I AGREE \n")
            assert (krea_root / "krea2_turbo/large.bin").is_file(), log
            assert "up to date" in run("--weights_root", krea_root, "--accept_license")
            shutil.rmtree(krea_root / "krea2_turbo")
            log = run("--weights_root", krea_root, "--accept_license")
            assert "accepted with --accept_license" in log and (krea_root / "krea2_turbo/large.bin").is_file(), log
            print("PASS: license prompt (no input, refusal, I agree, --accept_license)", flush=True)

            # One run per weights root; the lock goes away with the process that held it.
            write_config([dict(tiny, packed_repo="fixture/blocked")])
            lock_root = root / "lock weights"
            first_process = subprocess.Popen([str(bindir / "prepare_base.exe"), "--weights_root", str(lock_root)],
                                             cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                assert entered.wait(10), "the first process did not reach the fixture API"
                message = run("--weights_root", lock_root, expected=2)
                assert "another prepare.exe or prepare_base.exe" in message, message
            finally:
                first_process.kill()
                release.set()
                first_process.communicate(timeout=10)
            write_config([])
            run("--weights_root", lock_root)
            print("PASS: one run per weights root, lock released when the process ends", flush=True)
    finally:
        release.set()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
