#!/usr/bin/env python3
"""Check a built Docker/Podman image without an NPU or model weights.

python3 docker/tests/check_image.py vigenflow:beta4 [--engine 'sudo docker']
Starts a temporary container, checks server discovery and packaged checksums, then removes it.
This does not run image generation or editing. The workers need AVX-512, so their --help check is
skipped on a machine without it.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from urllib.request import urlopen

MODELS = {'flux1-schnell', 'flux2-klein-4B', 'flux2-klein-4B-edit', 'krea2-turbo',
          'z-image-turbo', 'qwen-image-2.1-turbo', 'qwen-image-2.1-turbo-edit'}
APP = '/opt/vigenflow/app'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--engine', default='docker')
    args = parser.parse_args()
    engine = shlex.split(args.engine)

    def run(*command):
        result = subprocess.run([*engine, *command], capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    # Invoke the server directly to check HTTP routing without the entrypoint's NPU prerequisite.
    container = run('run', '-d', '-p', '127.0.0.1::11283', args.image,
                    APP + '/server_running/vgf-serve', '--model', 'qwen-image-2.1-turbo')
    try:
        address = run('port', container, '11283/tcp').splitlines()[0]
        base = 'http://' + address
        for _ in range(100):
            try:
                with urlopen(base + '/health', timeout=1) as response:
                    assert response.status == 200
                break
            except OSError:  # URLError, or a reset while Docker's port proxy waits for the server
                time.sleep(0.1)
        else:
            raise RuntimeError('Server did not become healthy')
        with urlopen(base + '/v1/models', timeout=5) as response:
            served = {m['id'] for m in json.loads(response.read())['data']}
        assert served == MODELS, served
        run('exec', '-w', APP, container, 'sha256sum', '--check', '--quiet', 'SHA256SUMS')
        help_text = run('exec', container, '/usr/local/bin/vigenflow', '--help')
        assert all(model in help_text for model in MODELS)
        run('exec', container, '/usr/local/bin/vigenflow', 'prepare', '--help')
        with urlopen(base + '/', timeout=5) as response:
            assert response.status == 200 and b'<html' in response.read().lower(), 'no browser UI at /'
        run('exec', container, 'test', '!', '-e', APP + '/server_running/lib')
        checked = 'workers without lib/'
        if 'avx512f' in Path('/proc/cpuinfo').read_text().split():
            run('exec', container, 'sh', '-ec',
                'for worker in ' + APP + '/server_running/exe_models/*/run.exe; do '
                '"$worker" --help >/dev/null; done')
            checked += ' and their --help'
        else:
            print('No AVX-512 on this machine: worker --help check skipped')
        print('PASS: image checksums, entrypoint help, health, browser UI, all seven model IDs, ' + checked)
    except Exception:
        print(run('logs', container))
        raise
    finally:
        run('rm', '-f', container)


if __name__ == '__main__':
    main()
