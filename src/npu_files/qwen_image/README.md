# Qwen Image 2.1 Turbo NPU files

The bundles use the same `base/` and `edit/` directory layout as
`flux2_klein_4B`, for 1024x1024 text-to-image generation and image editing.

```text
qwen_image/
├── base/
│   ├── text_files/
│   ├── denoising_files/
│   ├── vae_files/
│   └── vae_pointwise_files/
├── edit/
│   ├── vision_files/
│   ├── image_edit_text_files/
│   ├── image_edit_denoising_files/
│   ├── vae_encoder_files/
│   ├── vae_files/
│   └── vae_pointwise_files/
├── README.md
└── SHA256SUMS
```

- `base/`: runtime bundles for text-to-image generation.
- `edit/`: runtime bundles for image editing.
  The denoising bundle also supplies the text encoder's NPU attention program.

Each task directory is a complete bundle root for its existing host. Both
include the byte-identical VAE decoder bundles so either task can be distributed
independently. Encoder instruction aliases are stored as regular files.

Run these commands from the corresponding host executable directory, replacing
`/path/to/vigenflow` with the repository location:

```bash
# Text-to-image host:
./run.exe -n /path/to/vigenflow/src/npu_files/qwen_image/base

# Image-editing host:
./run.exe -n /path/to/vigenflow/src/npu_files/qwen_image/edit
```

For Qwen, `-n` (or `--npu_files_path`) points directly to `base/` or `edit/`,
not their common parent. An absolute path is also accepted. Model weights are
supplied separately with `-w` or `--weights_path`. Each host's default NPU path
remains its local `build/`.

Keep the selected task's complete directory, including `xclbins/`, `insts/`
and JSON manifests. The `.txt` instruction dumps are retained alongside runtime
`.bin` instructions. Runtime artifact paths are relative to each bundle.

The server registers `qwen-image-2.1-turbo` with `qwen_image/base` and
`qwen-image-2.1-turbo-edit` with `qwen_image/edit`. Both use the shared
`qwen_image_2_1_turbo` weights folder. The bundles still require their matching
native workers; see [server setup](../../server_unified/README.md#qwen-image-21-turbo).

Verify the package from this directory:

```bash
sha256sum --check SHA256SUMS
```
