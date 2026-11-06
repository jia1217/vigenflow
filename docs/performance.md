# Performance details

The [README performance table](../README.md#-performance) lists approximate inference times, grouped by task and sorted by time. Model IDs and supported step counts are in [Supported Models](../README.md#-supported-models).

## Timing definitions

- **Inference time:** time spent running the model's inference stages. This excludes worker startup and weight loading.
- **Worker wall time:** time from starting the model worker until it exits, including startup, loading, inference, image saving and shutdown.
- **Request time:** time until the server finishes responding, including worker execution and request/response processing.

The README uses inference time consistently. It does not represent how long the complete server request takes.

## Comparison settings

The README figures use an AMD Ryzen AI 9 HX 370, CPU + NPU inference, BFP16 weights, one 1024 × 1024 image and prompts encoded at 512 text tokens. The step count is shown for each model; the Z-Image timing uses four steps.

Generation and editing are grouped separately because editing also processes an input image. Runtime varies with the prompt, reference image and system settings. Models differ in image quality and capabilities, so runtime alone does not determine which model to choose.

## Compared with AMD Ryzen AI Software 1.8

AMD's [Ryzen AI Software 1.8](https://ryzenai.docs.amd.com/en/latest/sd_demo.html) runs FLUX.1-schnell and FLUX.2-klein-4B on the NPU as preview demos. On the same Ryzen AI 9 HX 370, VigenFlow is faster with prompts twice as long, and it also runs three models that Ryzen AI 1.8 does not:

| Model | VigenFlow | Ryzen AI Software 1.8 |
| --- | --- | --- |
| FLUX.1-schnell | ~20 s, 512 text tokens | ~23 s, 256 text tokens |
| FLUX.2-klein-4B | ~9 s, 512 text tokens | ~10 s, 256 text tokens |
| FLUX.2-klein-4B Edit | ~18 s | Not supported |
| Krea 2 Turbo | ~48 s | Not supported |
| Z-Image-Turbo | ~18 s | Not supported |

All times are for one 1024 × 1024 image, and both FLUX comparisons use 4 steps. In our tests, Ryzen AI 1.8's FLUX demos accept prompts of up to 256 text tokens. At the same 256 text tokens, VigenFlow's FLUX.2-klein-4B takes about 8 s; VigenFlow ships only the 512-token version so that longer prompts fit.
