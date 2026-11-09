#!/usr/bin/env python3
"""Linux regression test with fake one-shot workers: every request runs balanced run.exe; no NPU required.

Usage: tests/test_balanced_only.py [vgf-serve] [--ui]   (--ui also drives the page in headless Chrome)
"""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from PIL import Image

here = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if a != '--ui']
binary = (Path(args[0]) if args else here.parent / 'vigenflow_ubuntu/vgf-serve').resolve()
with tempfile.TemporaryDirectory(prefix='vgf-balanced-test-') as tmp:
    root = Path(tmp)
    folders = ['z_image_turbo', 'flux1_schnell', 'flux2_klein_4B', 'krea2_turbo',
               'qwen_image_2_1_turbo', 'flux2_klein_4B_edit', 'qwen_image_2_1_turbo_edit']
    script = '''#!/usr/bin/env python3
import json, os, pathlib, sys, time
from PIL import Image
args = sys.argv[1:]
root = pathlib.Path.cwd().parent
with (root / 'launches.jsonl').open('a') as f:
    f.write(json.dumps({'pid': os.getpid(), 'model': pathlib.Path.cwd().name, 'args': args})+'\\n')
if '--worker' in args or '--help' in args or '--mode' in args:
    sys.exit(2)
out = pathlib.Path(args[args.index('--output')+1])
seed = int(args[args.index('--seed')+1])
if '--input_image' in args:
    image = Image.open(args[args.index('--input_image')+1])
    image.load()
    assert image.size == (256, 256)
if seed == -1:
    out.write_bytes(b'partial')
    sys.exit(7)
if seed == -3:
    time.sleep(1)
Image.new('RGB', (2, 2), (seed % 255, 40, 90)).save(out)
print('E2E runtime: 0.02 s')
'''
    for folder in folders:
        target = root / folder / 'run.exe'
        target.parent.mkdir()
        target.write_text(script)
        target.chmod(0o755)
    (root / 'flux1_schnell/loras.jsonc').write_text(json.dumps({'loras': [
        {'name': 'a', 'file': 'a.safetensors'}, {'name': 'b', 'file': 'b.safetensors'}]}))
    server_dir = root / 'server_test'
    server_dir.mkdir()
    exe = server_dir / 'vgf-serve'
    exe.symlink_to(binary)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    rejected = subprocess.run([str(exe), '--model', 'flux1-schnell', '--flux1-mode', 'performance',
                               '-p', str(port)], capture_output=True, text=True, timeout=10)
    assert rejected.returncode != 0 and 'Unknown argument' in rejected.stdout + rejected.stderr, rejected
    help_text = subprocess.run([str(exe), '--help'], capture_output=True, text=True, timeout=10)
    help_text = (help_text.stdout + help_text.stderr).lower()
    assert '--flux1-mode' not in help_text and 'memory' not in help_text and 'performance' not in help_text

    log_path = root / 'server.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen([str(exe), '--model', 'flux1-schnell', '-p', str(port),
                                 '-o', str(root / 'images')], stdout=log, stderr=log)
        def launches():
            path = root / 'launches.jsonl'
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        def post(path, data, content_type='application/json'):
            request = Request(base + path, data, {'Content-Type': content_type})
            try:
                with urlopen(request, timeout=15) as response:
                    return response.status, dict(response.headers), response.read()
            except HTTPError as error:
                return error.code, dict(error.headers), error.read()
        def generate(seed=1, model='flux1-schnell', expected=200, **extra):
            data = {'model': model, 'prompt': 'a fox\nwith "quotes"', 'seed': seed, 'response_format': 'b64_json'}
            data.update(extra)
            if model.endswith('-edit'):
                data.setdefault('size', '256x256')
                image = io.BytesIO()
                Image.new('RGB', (320, 192), (seed % 255, 40, 90)).save(image, format='PNG')
                boundary = 'vgf-balanced-test'
                parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
                         for key, value in data.items()]
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\nContent-Type: image/png\r\n\r\n'.encode() + image.getvalue() + b'\r\n')
                parts.append(f'--{boundary}--\r\n'.encode())
                status, headers, body = post('/v1/images/edits', b''.join(parts), f'multipart/form-data; boundary={boundary}')
            else:
                status, headers, body = post('/v1/images/generations', json.dumps(data).encode())
            assert status == expected, (status, body.decode())
            assert not any(key.lower() == 'x-memory-mode' for key in headers), headers
            result = json.loads(body)
            if status == 200:
                assert 'Memory_mode' not in result['metadata'], result['metadata']
            return result
        def last_launch_is_one_shot(model_folder='flux1_schnell'):
            launch = launches()[-1]
            assert launch['model'] == model_folder, launch
            assert not {'--worker', '--mode', '--help'} & set(launch['args']), launch
            return launch
        try:
            for attempt in range(100):
                try:
                    models = json.load(urlopen(base + '/v1/models', timeout=1))['data']
                    break
                except (URLError, ConnectionError):
                    if proc.poll() is not None:
                        raise AssertionError(log_path.read_text())
                    time.sleep(0.05)
            else:
                raise AssertionError('Server did not start')
            assert len(models) == 7
            assert all('memory_modes' not in m and 'default_mode' not in m for m in models), models
            assert not launches(), 'Startup must not launch or probe any run.exe'

            page = urlopen(base + '/', timeout=5).read().decode()
            assert 'mode-field' not in page and 'Memory mode' not in page and 'params.mode' not in page

            generate(); first = last_launch_is_one_shot()
            generate(seed=2); assert last_launch_is_one_shot()['pid'] != first['pid']
            for mode in ['performance', 'balanced', 'invalid']:
                generate(mode=mode); last_launch_is_one_shot()
            assert 'Ignoring request fields not supported by run.exe' in log_path.read_text()

            generate(size='512x512', loras=['a']); launch = last_launch_is_one_shot()
            assert launch['args'][launch['args'].index('--width')+1] == '512'
            assert launch['args'][launch['args'].index('--lora')+1] == 'a'
            generate(loras=[]); launch = last_launch_is_one_shot()
            assert launch['args'][launch['args'].index('--lora')+1] == 'none'

            generate(seed=-1, expected=500)
            assert not list((root / 'images').iterdir()), 'Failed image must be removed'

            for model in [m['id'] for m in models]:
                count = len(launches())
                generate(model=model, mode='performance')
                generate(seed=2, model=model, mode='performance')
                assert len(launches()) == count + 2, model
                a, b = launches()[-2:]
                assert a['pid'] != b['pid'], model
                for launch in (a, b):
                    assert not {'--worker', '--mode', '--help'} & set(launch['args']), (model, launch)
            edit_inputs = [l['args'][l['args'].index('--input_image')+1] for l in launches() if '--input_image' in l['args']]
            assert len(edit_inputs) == 4 and len(set(edit_inputs)) == 4
            assert all(not Path(path).exists() for path in edit_inputs), 'Edit uploads must be cleaned up'

            status, headers, body = post('/v1/chat/completions', json.dumps({
                'model': 'flux1-schnell', 'mode': 'performance', 'seed': 5,
                'messages': [{'role': 'user', 'content': 'a red fox'}],
                'tools': [{'type': 'function', 'function': {'name': 'generate_image'}}]}).encode())
            assert status == 200, body
            reply = json.loads(body)
            assert '![Generated image](' in reply['choices'][0]['message']['content'], reply
            assert 'Memory_mode' not in reply['metadata'], reply['metadata']
            launch = last_launch_is_one_shot()
            assert launch['args'][launch['args'].index('--seed')+1] == '5'

            # The client leaves before the image is ready; the server must survive writing the reply.
            count = len(launches())
            request = json.dumps({'model': 'flux1-schnell', 'prompt': 'slow', 'seed': -3}).encode()
            with socket.create_connection(('127.0.0.1', port)) as raw:
                raw.sendall(b'POST /v1/images/generations HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
                            + f'Content-Length: {len(request)}\r\n\r\n'.encode() + request)
                time.sleep(0.3)
            time.sleep(1.5)
            assert proc.poll() is None, log_path.read_text()
            generate(seed=6)
            assert len(launches()) == count + 2

            if '--ui' in sys.argv:
                profile = root / 'chrome-profile'
                shot = root / 'ui_balanced_only.png'
                subprocess.run(['node', str(here / 'ui_balanced_test.js'), base + '/', str(profile), str(shot)],
                               check=True, timeout=120)
                last_launch_is_one_shot()
            assert proc.poll() is None, log_path.read_text()
            print('PASS: balanced-only: no mode fields or flag, every request one-shot without --mode '
                  '(API, edits, chat' + (', web UI' if '--ui' in sys.argv else '') + '), '
                  'size/LoRA args, failed-image and upload cleanup, client disconnect')
        finally:
            proc.terminate()
            proc.wait(timeout=10)
