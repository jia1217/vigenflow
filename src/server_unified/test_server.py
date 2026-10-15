#!/usr/bin/env python3
"""Run with python3 test_server.py [./vgf-serve]; uses Pillow and a fake worker, no NPU.
On Windows: python test_server.py [vigenflow_win\\vgf-serve.exe], after build_windows.bat."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import shutil
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

windows = sys.platform == 'win32'
binary = Path(sys.argv[1] if len(sys.argv) > 1 else
              'vigenflow_win/vgf-serve.exe' if windows else './vgf-serve').resolve()
with tempfile.TemporaryDirectory(prefix='vgf-server-check-') as tmp:
    root = Path(tmp) / 'NPU_server'
    root.mkdir()
    folders = ['z_image_turbo', 'flux1_schnell', 'flux2_klein_4B', 'krea2_turbo']
    model_ids = ['z-image-turbo', 'flux1-schnell', 'flux2-klein-4B', 'krea2-turbo']
    # Ubuntu runs the fake worker script as run.exe; Windows runs it as run.py from a stub run.exe,
    # in the build-windows/Release folder where the server looks for Windows development workers.
    worker_dir = Path('build-windows/Release') if windows else Path('.')
    worker = root / folders[0] / worker_dir / 'run.exe'
    script = worker.with_name('run.py') if windows else worker
    worker.parent.mkdir(parents=True)
    script.write_text('''#!/usr/bin/env python3
import json, pathlib, random, shutil, struct, sys, time, zlib
# State files go in the model folder, also when the worker runs from build-windows/Release.
state = pathlib.Path.cwd()
if state.name == 'Release':
    state = state.parent.parent
# Fail immediately if two models use the simulated NPU together.
lock = (state.parent / 'npu.lock').open('w')
if sys.platform == 'win32':
    import msvcrt
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
else:
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
args = sys.argv[1:]
(state / 'last_args.json').write_text(json.dumps(args))
if '--input_image' in args:
    shutil.copyfile(args[args.index('--input_image') + 1], state / 'last_input.bmp')
seed = int(args[args.index('--seed') + 1])
prompt = next((arg[len('--prompt='):] for arg in args if arg.startswith('--prompt=')), None)
(state / 'last_prompt.json').write_text(json.dumps(prompt))
with open(state / 'calls', 'a') as f:
    f.write('generation\\n')
out = pathlib.Path(args[args.index('--output') + 1])
if seed == -1:
    out.write_bytes(b'partial worker output')
    sys.exit(1)
# 2x2 top-down BMP with row padding: red/green, blue/white.
pixels = bytes([0,0,255, 0,255,0, 0,0, 255,0,0, 255,255,255, 0,0])
header = struct.pack('<2sIHHI', b'BM', 54 + len(pixels), 0, 0, 54)
if seed == 8:
    pixels = pixels[8:] + pixels[:8]
header += struct.pack('<IiiHHIIiiII', 40, 2, 2 if seed == 8 else -2, 1, 24, 0, len(pixels), 0, 0, 0, 0)
if seed == 10:
    # A larger image must still produce only a short image URL in the SSE reply.
    pixels = random.Random(0).randbytes(80 * 80 * 3)
    header = struct.pack('<2sIHHI', b'BM', 54 + len(pixels), 0, 0, 54)
    header += struct.pack('<IiiHHIIiiII', 40, 80, 80, 1, 24, 0, len(pixels), 0, 0, 0, 0)
if out.suffix == '.png':
    # The same 2x2 picture as a PNG; the comment shows whether the server passed these bytes on unchanged.
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = bytes([0, 255,0,0, 0,255,0, 0, 0,0,255, 255,255,255])
    header = bytes([137, 80, 78, 71, 13, 10, 26, 10]) + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0))
    pixels = chunk(b'tEXt', b'Comment' + bytes(1) + b'written by the worker') + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
out.write_bytes(header + pixels if seed != -2 else b'broken image')
print('Text Encoder:  100%|done| 1/1 [00:00<00:00, 0.01s/it]')
print('Denoising:  100%|done| 4/4 [00:00<00:00, 0.002s/it]')
print('VAE Decoder:  100%|done| 1/1 [00:00<00:00, 0.005s/it]')
if seed != 9:
    print('E2E runtime: 23ms')
time.sleep(0.08)
''')
    script.chmod(0o755)
    if windows:
        shutil.copy2(Path(__file__).resolve().parent / 'build-windows' / 'test_worker_stub.exe', worker)
        os.environ['VGF_TEST_PYTHON'] = sys.executable  # inherited by the server and its workers
    edit_folder, edit_id = 'flux2_klein_4B_edit', 'flux2-klein-4B-edit'
    for folder in folders[1:] + [edit_folder]:
        shutil.copytree(worker.parent, root / folder / worker_dir)
    server_dir = root / 'server_test'
    server_dir.mkdir()
    test_binary = server_dir / binary.name
    if windows:
        # Symlinks need extra rights on Windows: copy the server and its DLLs instead.
        shutil.copy2(binary, test_binary)
        for dll in binary.parent.glob('*.dll'):
            shutil.copy2(dll, server_dir)
    else:
        test_binary.symlink_to(binary)
    for options in (['-k'], ['-k', 'yes'], ['-k', '1'], ['-k', ''],
                    ['--keep-images', 'true'], ['--no-keep-images'], ['--model', 'unknown'], ['--model', 'flux2-klein-4B', '--steps', '8'],
                    ['edit', '--steps', '8'],
                    ['--steps', '8', '--model', 'flux1-schnell'],
                    ['--model', 'krea2-turbo', '--steps', '4'],
                    *(['--model', 'flux1-schnell', '--steps', str(steps)] for steps in (1, 2, 3))):
        result = subprocess.run([str(test_binary), *options], capture_output=True, timeout=5)
        assert result.returncode != 0, options
    help_text = subprocess.check_output([str(test_binary), '--help'], text=True)
    assert '-k <true|false>' in help_text and 'Save generated images (default: false)' in help_text
    assert '--keep-images' not in help_text and '--no-keep-images' not in help_text
    assert '[edit]' in help_text and edit_id in help_text
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    outdir = root / 'images'
    keep_on_disk = False
    with (root / 'server.log').open('w+') as log:
        proc = subprocess.Popen([str(test_binary), '-p', str(port), '-e', str(worker), '-o', str(outdir)], stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    with urlopen(base + '/health', timeout=1) as response:
                        assert response.status == 200
                    break
                except URLError:
                    assert proc.poll() is None, 'server failed to start'
                    time.sleep(0.05)
            else:
                raise AssertionError('server did not become ready')

            def request(path, body, accept='application/json', status=200, content_type='application/json'):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                req = Request(base + path, data=data, headers={'Content-Type': content_type, 'Accept': accept})
                try:
                    response = urlopen(req, timeout=10)
                except HTTPError as error:
                    response = error
                with response:
                    data = response.read()
                    assert response.status == status, (response.status, data)
                    return response.headers, data

            def restart_server(options):
                proc.terminate()
                proc.wait(timeout=5)
                restarted = subprocess.Popen([str(test_binary), '-p', str(port), '-o', str(outdir), *options], stdout=log, stderr=log)
                for _ in range(100):
                    try:
                        with urlopen(base + '/health', timeout=1):
                            return restarted
                    except URLError:
                        assert restarted.poll() is None, 'server failed to restart'
                        time.sleep(0.05)
                restarted.terminate()
                restarted.wait(timeout=5)
                raise AssertionError('server did not restart')

            def expect_missing(url):
                try:
                    with urlopen(url, timeout=5):
                        raise AssertionError('expired image URL must return 404')
                except HTTPError as error:
                    assert error.code == 404

            def check_png(data):
                assert data.startswith(b'\x89PNG\r\n\x1a\n')
                with Image.open(io.BytesIO(data)) as image:
                    assert image.size == (2, 2)
                    assert list(image.convert('RGB').getdata()) == [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)], list(image.convert('RGB').getdata())

            headers, data = request('/v1/images/generations', {}, 'image/png')
            assert headers.get_content_type() == 'image/png'
            assert headers['Content-Disposition'] == 'inline'
            assert 'inference;dur=23.000000' in headers['Server-Timing']
            check_png(data)
            assert not list(outdir.iterdir()), 'raw response should clean up temporary images'

            assert json.loads((root / folders[0] / 'last_prompt.json').read_text()) is None
            # Backslashes before quotes and at the end test the Windows command-line quoting.
            literal_prompt = "--a cat's \"hat\"; $(echo literal) C:\\dir\\ \\\"q\\\"\n猫 🐈\\"
            _, prompted = request('/v1/images/generations', {'prompt': literal_prompt}, 'image/png')
            check_png(prompted)
            assert json.loads((root / folders[0] / 'last_prompt.json').read_text()) == literal_prompt
            for invalid_prompt in (None, 7, [], {}, '', ' \t\n', 'a\0cat', 'a' * 16385):
                request('/v1/images/generations', {'prompt': invalid_prompt}, status=400)

            _, bottom_up = request('/v1/images/generations', {'seed': 8}, 'image/png')
            check_png(bottom_up)

            _, data = request('/v1/images/generations', {})
            result = json.loads(data)
            check_png(base64.b64decode(result['data'][0]['b64_json']))
            assert result['metadata']['8_E2E_runtime'] == '0.02s'
            assert float(result['metadata']['9_Worker_wall_time'][:-1]) >= 0.08
            assert not list(outdir.iterdir())
            _, data = request('/v1/images/generations', {'seed': 9, 'response_format': 'url'})
            result = json.loads(data)
            assert '8_E2E_runtime' not in result['metadata'], 'must not substitute wall time for inference'
            assert result['data'][0]['url'].endswith('.png')
            with urlopen(result['data'][0]['url'], timeout=5) as response:
                assert response.headers.get_content_type() == 'image/png'
                check_png(response.read())
            assert not list(outdir.iterdir()), 'URL responses must obey keep_images=false'

            with urlopen(base + '/v1/models', timeout=5) as response:
                assert [m['id'] for m in json.load(response)['data']] == model_ids + [edit_id]
            for model_id, folder, seed, valid_steps, default_steps in zip(
                    model_ids, folders, [42, 0, 42, 42], [[4, 8], [4], [4], [8]], [4, 4, 4, 8]):
                for name in (model_id, folder):
                    headers, data = request('/v1/images/generations', {'model': name, 'prompt': name, 'n': 1, 'size': '1024x1024'}, 'image/png')
                    assert headers['X-Image-Model'] == model_id
                    check_png(data)
                    args = json.loads((root / folder / 'last_args.json').read_text())
                    assert args[args.index('--seed') + 1] == str(seed)
                    assert args[args.index('--steps') + 1] == str(default_steps)
                    # krea2-turbo's worker writes the PNG itself: served unchanged, no BMP conversion.
                    writes_png = folder == 'krea2_turbo'
                    assert Path(args[args.index('--output') + 1]).suffix == ('.png' if writes_png else '.bmp')
                    assert (b'written by the worker' in data) == writes_png
                    assert args[args.index('--weights_path') + 1] == str(root.parent / 'all_model_weights' / folder)
                    npu_path = root / 'all_npu_files' / folder
                    if folder == 'flux2_klein_4B':
                        npu_path /= 'base'
                    assert args[args.index('--npu_files_path') + 1] == str(npu_path)
                    assert json.loads((root / folder / 'last_prompt.json').read_text()) == name
                for steps in valid_steps:
                    _, data = request('/v1/images/generations', {'model': model_id, 'steps': steps})
                    assert json.loads(data)['metadata']['3_Denoising_steps'] == steps
                for steps in range(1, 9):
                    if steps not in valid_steps:
                        request('/v1/images/generations', {'model': model_id, 'steps': steps}, status=400)
            for seed in (-1, -2):  # failed worker, then a file that is not a PNG
                request('/v1/images/generations', {'model': 'krea2-turbo', 'seed': seed}, status=500)
            assert not list(outdir.iterdir()), 'failed PNG workers must leave no files'
            request('/v1/images/generations', {'steps': 3}, status=400)
            request('/v1/images/generations', {'model': edit_id}, status=400)  # needs an input image
            for invalid in ({'model': 'missing'}, {'model': None}, {'model': {}}, {'model': '../z_image_turbo'},
                            {'n': 2}, {'size': '512x512'}, {'steps': 4294967300}, {'seed': 2**64-1},
                            {'seed': 1.5}, {'response_format': 3}, {'response_format': 'bmp'}):
                request('/v1/images/generations', invalid, status=400)

            def call_counts():
                return [(root / folder / 'calls').read_text().count('generation') for folder in folders]

            def chat_png(content):
                prefix = '![Generated image]('
                assert content.startswith(prefix) and content.endswith('.png)'), content[:100]
                assert 'base64' not in content and len(content) < 1024
                url = content[len(prefix):-1]
                assert url.startswith(base + '/images/')
                assert (outdir / url.rsplit('/', 1)[-1]).is_file() == keep_on_disk
                with urlopen(url, timeout=5) as response:
                    assert response.headers.get_content_type() == 'image/png'
                    assert response.headers['Content-Disposition'] == 'inline'
                    if not keep_on_disk:
                        assert response.headers['Cache-Control'] == 'no-store'
                    return response.read()

            chat = {'messages': [{'role': 'user', 'content': 'a red fox'}], 'tools': [{'type': 'function', 'function': {'name': 'generate_image', 'parameters': {'type': 'object', 'properties': {'prompt': {'type': 'string'}}}}}]}
            for index, model_id in enumerate(model_ids):
                before = call_counts()
                _, data = request('/v1/chat/completions', {**chat, 'model': model_id})
                reply = json.loads(data)
                assert reply['model'] == model_id and reply['metadata']['0_Model'] == model_id
                choice = reply['choices'][0]
                assert choice['finish_reason'] == 'stop' and 'tool_calls' not in choice['message']
                check_png(chat_png(choice['message']['content']))
                check_png(chat_png(choice['message']['content']))  # Preview/reloads can fetch again.
                before[index] += 1
                assert call_counts() == before, 'chat dropdown must run only the selected worker once'
                assert json.loads((root / folders[index] / 'last_prompt.json').read_text()) == 'a red fox'

            # Open WebUI markup in the user's message is not part of the prompt.
            attached = ('<attached_files>\n<file type="file" id="b4e7" url="b4e7" content_type="image/png" '
                        'name="0.png"/>\n</attached_files>\n\n')
            for text, prompt in ((attached + 'help me change the weather to rainy', 'help me change the weather to rainy'),
                                 ('<user_query>a cat</user_query>', 'a cat'),
                                 ('I <3 cats, 2 < 3 > 1 and <tag> stays', 'I <3 cats, 2 < 3 > 1 and <tag> stays')):
                content = [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGk='}}]
                request('/v1/chat/completions', {**chat, 'model': model_ids[0], 'messages': [{'role': 'user', 'content': content}]})
                assert json.loads((root / folders[0] / 'last_prompt.json').read_text()) == prompt
            before = call_counts()
            request('/v1/chat/completions', {**chat, 'messages': [{'role': 'user', 'content': attached}]}, status=400)
            assert call_counts() == before, 'markup-only messages must not start workers'

            headers, data = request('/v1/chat/completions', {**chat, 'model': 'flux2_klein_4B', 'stream': True, 'seed': 10})
            assert headers.get_content_type() == 'text/event-stream'
            events = [line[6:] for line in data.decode().splitlines() if line.startswith('data: ')]
            assert events[-1] == '[DONE]' and len(events) == 3
            assert len(data) < 5000 and b'base64' not in data, 'chat must send a URL, not PNG bytes'
            chunks = [json.loads(event) for event in events[:-1]]
            assert all(c['model'] == 'flux2-klein-4B' for c in chunks)
            assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'
            content = chunks[0]['choices'][0]['delta']['content']
            assert 'content' not in chunks[1]['choices'][0]['delta'], 'image Markdown must arrive complete'
            with Image.open(io.BytesIO(chat_png(content))) as image:
                assert image.size == (80, 80)
            before = call_counts()
            for body in ({**chat, 'messages': chat['messages'] + [{'role': 'tool', 'content': '{"success":true}'}]},
                         {'messages': chat['messages']}, {**chat, 'tool_choice': 'none'}):
                _, data = request('/v1/chat/completions', body)
                assert json.loads(data)['choices'][0]['finish_reason'] == 'stop'
            assert call_counts() == before, 'background calls must not start extra workers'
            for invalid in ({}, {'messages': []}, {**chat, 'stream': 'yes'}, {**chat, 'tools': {}},
                            {**chat, 'tool_choice': {'function': None}}, {**chat, 'model': 'missing'},
                            {**chat, 'model': 'flux2-klein-4B', 'steps': 8},
                            {**chat, 'messages': [{'role': 'user', 'content': ' '}]}):
                request('/v1/chat/completions', invalid, status=400)
            before = call_counts()
            for steps in (1, 2, 3, 8):
                request('/v1/chat/completions', {**chat, 'model': 'flux1-schnell', 'steps': steps}, status=400)
            assert call_counts() == before, 'invalid steps must not start workers'
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(request, '/v1/images/generations', {'model': model_id, 'prompt': model_id}) for model_id in model_ids]
                for model_id, job in zip(model_ids, jobs):
                    _, data = job.result()
                    assert json.loads(data)['model'] == model_id
            for folder, model_id in zip(folders, model_ids):
                assert json.loads((root / folder / 'last_prompt.json').read_text()) == model_id

            assert not list(outdir.iterdir()), 'no-save mode must leave no BMPs or PNGs'
            # The RAM cache is bounded; old URLs expire instead of filling memory forever.
            for _ in range(32):
                _, data = request('/v1/images/generations', {'response_format': 'url'})
                newest_url = json.loads(data)['data'][0]['url']
            expect_missing(content[len('![Generated image]('):-1])
            with urlopen(newest_url, timeout=5) as response:
                check_png(response.read())
            assert not list(outdir.iterdir())

            # Explicit false overrides true, while preserving per-model overrides.
            proc = restart_server(['-k', 'true', '-k', 'false', '--steps', '4', '--seed', '13',
                                   '--model', 'flux1_schnell', '-w', str(root / 'custom weights'),
                                   '-n', str(root / 'custom npu')])
            expect_missing(newest_url)  # A restart clears RAM image URLs.
            _, data = request('/v1/images/generations', {})
            result = json.loads(data)
            assert result['model'] == 'flux1-schnell'
            assert result['metadata']['4_Seed'] == 13 and result['metadata']['3_Denoising_steps'] == 4
            args = json.loads((root / folders[1] / 'last_args.json').read_text())
            assert args[args.index('--weights_path') + 1] == str(root / 'custom weights')
            assert args[args.index('--npu_files_path') + 1] == str(root / 'custom npu')
            _, data = request('/v1/images/generations', {'model': 'z-image-turbo'})
            assert json.loads(data)['metadata']['4_Seed'] == 42
            request('/v1/images/generations', {'seed': -1}, status=500)
            request('/v1/images/generations', {'seed': -2}, status=500)
            _, data = request('/v1/chat/completions', chat)
            check_png(chat_png(json.loads(data)['choices'][0]['message']['content']))
            assert not list(outdir.iterdir()), 'failed workers and no-save chat must not leave files'

            # -k true saves every response type, with the last value winning.
            keep_on_disk = True
            proc = restart_server(['-k', 'false', '-k', 'true'])
            _, data = request('/v1/chat/completions', chat)
            saved_content = json.loads(data)['choices'][0]['message']['content']
            check_png(chat_png(saved_content))
            _, data = request('/v1/images/generations', {'response_format': 'url'})
            saved_url = json.loads(data)['data'][0]['url']
            with urlopen(saved_url, timeout=5) as response:
                check_png(response.read())
            request('/v1/images/generations', {})
            request('/v1/images/generations', {}, 'image/png')
            assert len(list(outdir.glob('*.png'))) == 4 and not list(outdir.glob('*.bmp'))
            saved_files = set(outdir.iterdir())
            request('/v1/images/generations', {'seed': -1}, status=500)
            request('/v1/images/generations', {'seed': -2}, status=500)
            assert set(outdir.iterdir()) == saved_files, 'failure cleanup must preserve earlier PNGs'
            proc = restart_server(['-k', 'true'])
            check_png(chat_png(saved_content))
            with urlopen(saved_url, timeout=5) as response:
                check_png(response.read())
            # The edit model is always served; /v1/images/edits uses it by default.
            keep_on_disk = False
            saved_files = set(outdir.iterdir())
            proc = restart_server([])

            def encoded(image, fmt='PNG'):
                buffer = io.BytesIO()
                image.save(buffer, fmt)
                return buffer.getvalue()

            def multipart(fields, files):
                boundary = 'vgf-check-7d1f'
                body = b''
                for name, value in fields.items():
                    body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
                for name, data in files:
                    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="image.png"\r\n'
                             'Content-Type: image/png\r\n\r\n').encode() + data + b'\r\n'
                return body + f'--{boundary}--\r\n'.encode(), f'multipart/form-data; boundary={boundary}'

            def edit_input():
                args = json.loads((root / edit_folder / 'last_args.json').read_text())
                assert not Path(args[args.index('--input_image') + 1]).exists(), 'temporary input must be removed'
                image = Image.open(root / edit_folder / 'last_input.bmp').convert('RGB')
                assert image.size == (1024, 1024)
                return args, image

            # 300x200: red/blue side bands are cropped away, leaving the green center square.
            wide = Image.new('RGB', (300, 200), (0, 255, 0))
            wide.paste((255, 0, 0), (0, 0, 50, 200))
            wide.paste((0, 0, 255), (250, 0, 300, 200))
            wide_png = encoded(wide)
            body, form_type = multipart({'model': edit_id, 'prompt': 'add a hat', 'n': '1', 'size': '1024x1024',
                                         'response_format': 'b64_json', 'background': 'auto'},
                                        [('image[]', wide_png), ('image[]', encoded(Image.new('RGB', (8, 8))))])
            _, data = request('/v1/images/edits', body, content_type=form_type)
            result = json.loads(data)
            assert result['model'] == edit_id and result['metadata']['0_Model'] == edit_id
            check_png(base64.b64decode(result['data'][0]['b64_json']))
            args, image = edit_input()
            assert set(image.getdata()) == {(0, 255, 0)}, 'first image, center-cropped and resized'
            assert '--prompt=add a hat' in args and args[args.index('--seed') + 1] == '42'
            assert args[args.index('--weights_path') + 1] == str(root.parent / 'all_model_weights' / 'flux2_klein_4B')
            assert args[args.index('--npu_files_path') + 1] == str(root / 'all_npu_files' / 'flux2_klein_4B')

            # JSON body with a data URL, JPEG input, default model and a raw PNG response.
            tall_jpeg = encoded(Image.new('RGB', (40, 90), (200, 30, 30)), 'JPEG')
            jpeg_url = 'data:image/jpeg;base64,' + base64.b64encode(tall_jpeg).decode()
            headers, data = request('/v1/images/edits', {'prompt': 'make it blue', 'image': jpeg_url, 'seed': 5}, 'image/png')
            assert headers['X-Image-Model'] == edit_id
            check_png(data)
            args, image = edit_input()
            assert args[args.index('--seed') + 1] == '5'
            assert all(abs(a - b) <= 4 for a, b in zip(image.getpixel((512, 512)), (200, 30, 30)))
            _, data = request('/v1/images/edits', {'prompt': 'x', 'image': [base64.b64encode(wide_png).decode()],
                                                   'response_format': 'url'})
            with urlopen(json.loads(data)['data'][0]['url'], timeout=5) as response:
                check_png(response.read())

            calls = lambda: (root / edit_folder / 'calls').read_text().count('generation')
            before = calls()
            png_b64 = base64.b64encode(wide_png).decode()
            for invalid in ({'prompt': 'x'}, {'image': png_b64}, {'prompt': 'x', 'image': 'not base64!'},
                            {'prompt': 'x', 'image': base64.b64encode(b'not an image').decode()},
                            {'prompt': 'x', 'image': 'data:image/png,raw'}, {'prompt': 'x', 'image': 7},
                            {'prompt': 'x', 'image': png_b64, 'model': 'z-image-turbo'},
                            {'prompt': 'x', 'image': png_b64, 'size': '512x512'},
                            {'prompt': 'x', 'image': png_b64, 'steps': 8}):
                request('/v1/images/edits', invalid, status=400)
            request('/v1/images/edits', multipart({'prompt': 'x'}, [])[0], status=400, content_type=form_type)
            request('/v1/images/edits', multipart({'prompt': 'x', 'n': 'one'}, [('image', wide_png)])[0],
                    status=400, content_type=form_type)
            request('/v1/images/edits', b'--nope', status=400, content_type='multipart/form-data; boundary=zz')
            request('/v1/images/generations', {'model': edit_id, 'prompt': 'x'}, status=400)
            assert calls() == before, 'invalid edits must not start the worker'
            _, data = request('/v1/images/generations', {'model': 'z-image-turbo'})
            assert json.loads(data)['model'] == 'z-image-turbo'

            # Open WebUI chat: edit the attached image, then keep editing the latest result.
            tools = [{'type': 'function', 'function': {'name': name}} for name in ('generate_image', 'edit_image')]
            wide_url = 'data:image/png;base64,' + png_b64
            first = {'role': 'user', 'content': [{'type': 'text', 'text': attached + 'add a hat'},
                                                 {'type': 'image_url', 'image_url': {'url': wide_url}}]}
            _, data = request('/v1/chat/completions', {'model': edit_id, 'tools': tools, 'messages': [first]})
            reply = json.loads(data)['choices'][0]['message']['content']
            assert reply.startswith('![Edited image](') and len(reply) < 1024
            with urlopen(reply[len('![Edited image]('):-1], timeout=5) as response:
                check_png(response.read())
            args, image = edit_input()
            assert '--prompt=add a hat' in args and set(image.getdata()) == {(0, 255, 0)}
            follow_up = [first, {'role': 'assistant', 'content': reply}, {'role': 'user', 'content': 'now at night'}]
            headers, data = request('/v1/chat/completions', {'model': edit_id, 'tools': tools[1:], 'messages': follow_up, 'stream': True})
            assert headers.get_content_type() == 'text/event-stream' and b'Edited image' in data
            args, image = edit_input()
            assert '--prompt=now at night' in args
            assert image.getpixel((0, 0)) == (255, 0, 0) and image.getpixel((1023, 1023)) == (255, 255, 255), 'previous result'
            _, data = request('/v1/chat/completions', {'model': edit_id, 'tools': tools, 'messages': [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'crop'}, {'type': 'image_url', 'image_url': jpeg_url}]}]})
            assert json.loads(data)['choices'][0]['message']['content'].startswith('![Edited image](')
            before = calls()
            for messages in ([{'role': 'user', 'content': 'add a hat'}],
                             [{'role': 'assistant', 'content': '![old](http://elsewhere/images/missing.png)'}, {'role': 'user', 'content': 'again'}],
                             [{'role': 'user', 'content': [{'type': 'text', 'text': 'x'}, {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}}]}]):
                request('/v1/chat/completions', {'model': edit_id, 'tools': tools, 'messages': messages}, status=400)
            request('/v1/chat/completions', {'model': edit_id, 'messages': [first]})  # background call: no tools
            assert calls() == before, 'chat without an editable image must not start the worker'
            assert set(outdir.iterdir()) == saved_files, 'no-save edits must leave no files'

            # `edit` is short for --model flux2-klein-4B-edit, so -w applies to the edit model.
            proc = restart_server(['edit', '-w', str(root / 'edit weights')])
            request('/v1/images/generations', {}, status=400)
            _, data = request('/v1/images/edits', {'model': 'flux2_klein_4B_edit', 'prompt': 'x', 'image': png_b64})
            assert json.loads(data)['model'] == edit_id
            args, _ = edit_input()
            assert args[args.index('--weights_path') + 1] == str(root / 'edit weights')
            print('PASS: storage defaults/flags, bounded RAM URLs, failure cleanup, PNG retention with -k true, four-model routing, worker-written PNG passed through, chat SSE, PNG pixels/raw/base64/URL, timings and validation, edit model (multipart/JSON edits, crop+resize, chat follow-up edits, `edit` shortcut)')

            if windows:
                # Weights beside the package folder come first; a model only in the package's own folder
                # (installed by 0.2.0-beta.1) is read from there; one that is nowhere is expected beside it.
                shared, local = root.parent / 'all_model_weights', root / 'all_model_weights'
                (shared / folders[0]).mkdir(parents=True)
                (local / folders[1]).mkdir(parents=True)
                proc = restart_server([])
                for folder, weights in zip(folders[:3], (shared, local, shared)):
                    request('/v1/images/generations', {'model': folder, 'prompt': 'x'}, 'image/png')
                    args = json.loads((root / folder / 'last_args.json').read_text())
                    assert args[args.index('--weights_path') + 1] == str(weights / folder), (folder, args)
                shutil.rmtree(shared)
                shutil.rmtree(local)

                # Windows serves the models whose workers are built; the others are listed as missing.
                shutil.rmtree(root / folders[2] / worker_dir)
                result = subprocess.run([str(test_binary), '-p', str(port), '--model', model_ids[2]],
                                        capture_output=True, timeout=5)
                assert result.returncode != 0 and b'has no worker' in result.stderr, result.stderr
                proc = restart_server([])
                with urlopen(base + '/v1/models', timeout=5) as response:
                    listed = [model['id'] for model in json.loads(response.read())['data']]
                assert model_ids[2] not in listed and model_ids[0] in listed and edit_id in listed, listed
                _, data = request('/v1/images/generations', {'model': model_ids[2], 'prompt': 'x'}, status=400)
                assert b'not served' in data, data
                _, data = request('/v1/images/generations', {'model': model_ids[1], 'prompt': 'x'}, 'image/png')
                check_png(data)

                # A package: exe_models/<model>/run.exe beside vgf-serve.exe, with the shared base/ NPU files.
                package_dir = server_dir / 'exe_models' / folders[2]
                shutil.copytree(root / folders[1] / worker_dir, package_dir)
                proc = restart_server(['--model', folders[2]])
                with urlopen(base + '/v1/models', timeout=5) as response:
                    listed = [model['id'] for model in json.loads(response.read())['data']]
                assert model_ids[2] in listed, listed
                long_prompt = 'A mountain village with flowers and a river. ' * 60
                _, data = request('/v1/images/generations', {'prompt': long_prompt}, 'image/png')
                check_png(data)
                args = json.loads((package_dir / 'last_args.json').read_text())
                assert '--prompt=' + long_prompt in args, 'server must forward the complete Flux2 prompt'
                assert args[args.index('--weights_path') + 1] == str(root.parent / 'all_model_weights' / folders[2])
                assert args[args.index('--npu_files_path') + 1] == str(root / 'all_npu_files' / folders[2] / 'base')
                assert args[args.index('--steps') + 1] == '4' and args[args.index('--seed') + 1] == '42'

                # A native run.exe takes precedence over an older host.exe.
                (package_dir / 'host.exe').write_bytes(b'MZinvalid test executable')
                proc = restart_server([])
                _, data = request('/v1/images/generations', {'model': folders[2], 'prompt': 'x'}, 'image/png')
                check_png(data)

                # Older host.exe packages still work, even beside a Linux run.exe.
                shutil.copy2(package_dir / 'run.exe', package_dir / 'host.exe')
                (package_dir / 'run.exe').write_bytes(b'\x7fELFtest Linux worker')
                proc = restart_server([])
                _, data = request('/v1/images/generations', {'model': model_ids[2], 'prompt': 'x'}, 'image/png')
                check_png(data)
                print('PASS (Windows): in-package weights fallback; models without a built worker are skipped, '
                      'listed and rejected; run.exe packages, shared base/ NPU files, full prompts, host.exe fallback')
        except BaseException:
            log.flush()
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            proc.terminate()
            proc.wait(timeout=5)
