#!/usr/bin/env python3
"""Exercise the real installer with fake Docker/curl; no downloads or host changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

INSTALLER = Path(__file__).resolve().parents[1] / 'install.sh'
QWEN = 'qwen_image_2_1_turbo'
KREA = 'krea2_turbo'
ALL = {'flux2_klein_4B', 'z_image_turbo', 'flux1_schnell', KREA, QWEN}

FAKE_DOCKER = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['VGF_TEST_STATE'])
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps(args) + '\n')
state = root / 'installed.json'
installed = json.loads(state.read_text())
if args == ['--version']:
    print('Docker version test')
elif args == ['context', 'show']:
    print('default')
elif args and args[0] == 'run' and 'prepare' in args:
    options = args[args.index('prepare') + 1:]
    assert options[0] == '--only', options
    model = options[1]
    if model not in installed:
        if model in ('qwen_image_2_1_turbo', 'krea2_turbo') and '--accept_license' not in options:
            sys.exit(1)  # no terminal to accept this model's terms
        installed.append(model)
        state.write_text(json.dumps(installed))
elif args and args[0] == 'run' and 'ls' in args:
    print('\n'.join(installed))
elif args and args[0] == 'run' and '-d' in args:
    print('fake-container')
'''


class InstallerTest(unittest.TestCase):
    def run_installer(self, models=None, *, krea=False, qwen=False, installed=()):
        with tempfile.TemporaryDirectory(prefix='vgf-installer-test-') as tmp:
            root = Path(tmp)
            bindir = root / 'bin'
            bindir.mkdir()
            (root / 'installed.json').write_text(json.dumps(list(installed)))
            docker = bindir / 'docker'
            docker.write_text(f'#!{sys.executable}\n' + FAKE_DOCKER)
            docker.chmod(0o755)
            for name, body in [('curl', 'exit 0'), ('sudo', 'exit 99')]:
                executable = bindir / name
                executable.write_text('#!/bin/sh\n' + body + '\n')
                executable.chmod(0o755)
            # Simulate the device on CI. All other file tests use Bash's real builtin.
            startup = root / 'bash_env'
            startup.write_text("function [ {\n"
                               "  if [[ \"$*\" == '! -e /dev/accel/accel0 ]' ]]; then return 1; fi\n"
                               "  builtin [ \"$@\"\n}\n")
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith('VIGENFLOW_')}
            env.update(PATH=str(bindir) + ':' + os.defpath, BASH_ENV=str(startup),
                       VGF_TEST_STATE=str(root), VIGENFLOW_WEBUI='0',
                       VIGENFLOW_IMAGE='vigenflow:test',
                       VIGENFLOW_ACCEPT_KREA_LICENSE='yes' if krea else 'no',
                       VIGENFLOW_ACCEPT_QWEN_LICENSE='yes' if qwen else 'no')
            if models is not None:
                env['VIGENFLOW_MODELS'] = models
            result = subprocess.run(['bash', str(INSTALLER)], cwd=root, env=env,
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    start_new_session=True, timeout=20)
            log = root / 'calls.jsonl'
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            prepares = [c[c.index('prepare') + 1:] for c in calls if c[0] == 'run' and 'prepare' in c]
            starts = [c[c.index('serve') + 1:] for c in calls if c[0] == 'run' and 'serve' in c]
            return result, calls, prepares, starts

    def test_qwen_only_uses_shared_weights_and_correct_default(self):
        result, _, prepares, starts = self.run_installer(QWEN, qwen=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(prepares, [['--only', QWEN, '--accept_license']])
        self.assertEqual(starts, [['--model', 'qwen-image-2.1-turbo']])

    def test_all_explicitly_includes_qwen(self):
        result, _, prepares, starts = self.run_installer('all', krea=True, qwen=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({args[1] for args in prepares}, ALL)
        self.assertEqual(len(prepares), 5)
        self.assertEqual({args[1] for args in prepares if '--accept_license' in args}, {KREA, QWEN})
        self.assertEqual(starts, [['--model', 'z-image-turbo']])

    def test_krea_acceptance_does_not_accept_qwen(self):
        result, _, prepares, starts = self.run_installer('all', krea=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(['--only', KREA, '--accept_license'], prepares)
        self.assertIn(['--only', QWEN], prepares)
        self.assertIn('Some models were not downloaded', result.stdout)
        self.assertEqual(starts, [['--model', 'z-image-turbo']])

    def test_qwen_acceptance_does_not_accept_krea(self):
        result, _, prepares, starts = self.run_installer(KREA + ',' + QWEN, qwen=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(['--only', KREA], prepares)
        self.assertIn(['--only', QWEN, '--accept_license'], prepares)
        self.assertEqual(starts, [['--model', 'qwen-image-2.1-turbo']])

    def test_qwen_without_acceptance_does_not_start_an_empty_install(self):
        result, _, prepares, starts = self.run_installer(QWEN)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(prepares, [['--only', QWEN]])
        self.assertEqual(starts, [])
        self.assertIn('No model weights were downloaded', result.stderr)

    def test_existing_qwen_can_be_reused_without_another_acceptance(self):
        result, _, _, starts = self.run_installer(QWEN, installed=[QWEN])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(starts, [['--model', 'qwen-image-2.1-turbo']])

    def test_default_selection_still_installs_flux2(self):
        result, _, prepares, starts = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(prepares, [['--only', 'flux2_klein_4B']])
        self.assertEqual(starts, [['--model', 'flux2-klein-4B']])

    def test_invalid_weight_id_is_rejected_before_download(self):
        result, calls, prepares, starts = self.run_installer('qwen_image_2_1_turbo_edit')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown model', result.stderr)
        self.assertFalse(any(c[0] == 'pull' for c in calls))
        self.assertEqual(prepares, [])
        self.assertEqual(starts, [])


if __name__ == '__main__':
    unittest.main()
