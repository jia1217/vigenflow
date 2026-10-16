#!/usr/bin/env python3
"""Real NPU check of a release package: one image per model through its vgf-serve.

python tests/npu_check.py <package folder> [--port 11290] [--out npu_check] [--models flux2-klein-4B,...]

The package must sit next to an all_model_weights folder with the models' weights, as after
prepare/prepare_base.exe. Each text-to-image model gets one request; flux2-klein-4B-edit edits the
flux2-klein-4B image (or a gray image when flux2-klein-4B is not checked). A model passes when vgf-serve
returns a 1024x1024 PNG and the worker's end-to-end time. The images, the server log and summary.json
go to --out. Exit code 0 only when every model passes. Standard library only.
"""
import argparse
import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib

TEXT_MODELS = ['z-image-turbo', 'flux1-schnell', 'flux2-klein-4B', 'krea2-turbo']
EDIT_MODEL = 'flux2-klein-4B-edit'
PROMPT = 'a red fox sitting in fresh snow, photo'
PNG_SIGNATURE = bytes([0x89]) + b'PNG' + bytes([13, 10, 26, 10])

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('package', type=Path)
parser.add_argument('--port', type=int, default=11290)
parser.add_argument('--out', type=Path, default=Path('npu_check'))
parser.add_argument('--models', default=','.join(TEXT_MODELS + [EDIT_MODEL]))
parser.add_argument('--timeout', type=int, default=900, help='seconds per request')
args = parser.parse_args()
models = [m for m in args.models.split(',') if m]
unknown = sorted(set(models) - set(TEXT_MODELS + [EDIT_MODEL]))
if unknown:
    sys.exit(f'unknown models: {", ".join(unknown)}')

package = args.package.resolve()
server = package / 'server_running' / ('vgf-serve.exe' if sys.platform == 'win32' else 'vgf-serve')
weights = package.parent / 'all_model_weights'
if not server.is_file():
    sys.exit(f'no {server}')
if not weights.is_dir():
    sys.exit(f'no {weights}: the package must be next to the all_model_weights folder')
args.out.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{args.port}'


def png_size(data):
    if data[:8] != PNG_SIGNATURE or data[12:16] != b'IHDR':
        return None
    return struct.unpack('>II', data[16:24])


def gray_png(size=1024):
    row = b'\x00' + bytes([128] * size * 3)
    chunk = lambda kind, body: struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    return (PNG_SIGNATURE + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(row * size)) + chunk(b'IEND', b''))


def post(path, body):
    request = Request(base + path, data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    start = time.time()
    try:
        with urlopen(request, timeout=args.timeout) as response:
            return response.status, json.loads(response.read()), time.time() - start
    except HTTPError as error:
        return error.code, error.read().decode(errors='replace'), time.time() - start


summary = {'package': str(package), 'models': {}}
log = (args.out / 'server.log').open('w', encoding='utf-8', errors='replace')
proc = subprocess.Popen([str(server), '-p', str(args.port), '-o', str(args.out / 'tmp')],
                        cwd=server.parent, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            urlopen(base + '/health', timeout=1).close()
            break
        except URLError:
            if proc.poll() is not None:
                sys.exit(f'vgf-serve exited with {proc.returncode}; see {args.out / "server.log"}')
            time.sleep(0.1)
    else:
        sys.exit('vgf-serve did not start listening')
    with urlopen(base + '/v1/models', timeout=5) as response:
        served = [m['id'] for m in json.loads(response.read())['data']]
    print('served:', ', '.join(served), flush=True)

    images = {}
    for model in models:
        if model == EDIT_MODEL:
            source = images.get('flux2-klein-4B') or gray_png()
            status, body, seconds = post('/v1/images/edits', {
                'model': model, 'prompt': 'make it night time',
                'image': 'data:image/png;base64,' + base64.b64encode(source).decode()})
        else:
            status, body, seconds = post('/v1/images/generations', {'model': model, 'prompt': PROMPT})
        result = {'http': status, 'seconds': round(seconds, 1)}
        if status == 200:
            data = base64.b64decode(body['data'][0]['b64_json'])
            images[model] = data
            (args.out / f'{model}.png').write_bytes(data)
            meta = body.get('metadata', {})
            result.update(size=png_size(data), e2e=meta.get('8_E2E_runtime'), worker_wall=meta.get('9_Worker_wall_time'))
            result['passed'] = result['size'] == (1024, 1024) and bool(result['e2e'])
        else:
            result.update(passed=False, error=str(body)[:500])
        summary['models'][model] = result
        state = 'PASS' if result['passed'] else 'FAIL'
        print(f'{state} {model}: HTTP {status} in {seconds:.1f} s, E2E {result.get("e2e")}', flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()

passed = sum(r['passed'] for r in summary['models'].values())
summary['passed'] = passed == len(models)
(args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'{passed} of {len(models)} models passed')
sys.exit(0 if summary['passed'] else 1)
