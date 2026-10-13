# Flux2 NPU files

- `base/`: Flux2 text-to-image artifacts, also reused by Flux2-edit when byte-identical.
- `edit/`: Flux2-edit instructions and kernels that are not present in `base/`.

The migration removed 27 duplicate local copies of base artifacts and placed
45 files in `edit/`, including the legacy `all_atten_custom.xclbin` fallback. Its Qwen instructions target the
shared denoising `mm_bfp_new.xclbin`; they differ from the base text encoder's
instructions. The edit VAE k-loop and stride-2 kernels also remain separate.

Run these commands from each model's project directory:

```bash
# NPU_server/flux2_klein_4B
./run.exe -n ../all_npu_files/flux2_klein_4B/base

# NPU_server/flux2_klein_4B_edit
./run.exe --input_image /path/to/reference.png -n ../all_npu_files/flux2_klein_4B
```

These are the default NPU paths. For Flux2-edit, `-n` points to the parent
containing **both** `base/` and `edit/`; keep both when relocating the artifacts.
Both hosts default to `../all_model_weights/flux2_klein_4B` for model weights;
Flux2-edit additionally loads `vae_encoder_packed_weights/` from that root.

The edit host no longer loads NPU artifacts from its local `build/` directory.
`tools/gen_qwen_mm_bfp_new_insts.sh` in the edit project writes to
`edit/text_files_mm_bfp_new/insts/` by default; its first argument can override
the output directory.
