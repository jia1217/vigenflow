# Model Licenses

The VigenFlow source code is licensed under the [Apache License 2.0](LICENSE). **That license covers the code only, not model weights.** Each model and LoRA adapter stays under the license chosen by its original authors. This includes the BFP16 weights that the VigenFlow release packages contain.

Before you use a model, especially commercially, read its license on the upstream model card. The tables below were checked against Hugging Face on 2026-09-30, the Krea 2 Turbo row on 2026-10-12, the Qwen Image 2.1 Turbo license at upstream revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b` on 2026-10-19, and the FLUX.3 image klein license at upstream revision `09561231f44e41381c461caf53d5452005a60ac9` on 2026-11-20.

## Base models

These weights are available through the preparation catalog; Qwen and SeedVR2 are opt-in. Native NPU+CPU conversion does not change the upstream license.

| Model | Upstream | License | Commercial use |
| --- | --- | --- | --- |
| FLUX.1-schnell | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Apache-2.0 | Allowed |
| FLUX.2-klein-4B (text-to-image and editing) | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 | Allowed |
| FLUX.3 image klein (text-to-image and editing) | [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | [FLUX Kommunity License v.1.0](model_licenses/flux3_image_klein/LICENSE) | Limited: the model itself may be used for non-commercial purposes only. Outputs may be used commercially only by a Qualifying User (you and your affiliates below USD 5 million gross annualized revenue). Other commercial use needs a license from Black Forest Labs. |
| Z-Image-Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Apache-2.0 | Allowed |
| Krea 2 Turbo | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | [Krea 2 Community License](https://www.krea.ai/krea-2-licensing) | Limited: only while total company revenue, including affiliates, is below USD 1 million over the past twelve months. Above that, a separate Krea enterprise license is required. |
| SeedVR2-3B (image upscaling; opt-in weight preparation) | [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) | Apache-2.0 | Allowed |
| Qwen Image 2.1 Turbo (text-to-image and editing; opt-in weight preparation) | [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | [Qwen RESEARCH LICENSE AGREEMENT](model_licenses/qwen_image_2_1_turbo/LICENSE) | Research/evaluation only under this agreement; commercial use requires a separate upstream license. |

The text encoders and VAEs are the ones distributed in the upstream repositories above.

SeedVR2-3B weights are downloaded with `prepare_base.exe --only seedvr2`, which saves the upstream `LICENSE` and a `Notice` with them. Keep those files with any copy you make.

Krea 2 Turbo is also subject to the [Krea Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy). `prepare_base.exe` asks you to accept both before it downloads these weights, and it saves `LICENSE.pdf` and `Notice` with them. Keep those two files with any copy you make. The full license agreement controls; the row above is only a summary.

Qwen Image 2.1 Turbo's [upstream agreement](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE)
defines non-commercial use as research or evaluation. Section 2 requires a separate commercial license
from `model-business@notice.qwencloud.com`. Redistribution remains subject to that grant and requires
the agreement, modification notices and attribution. The converted package includes the unchanged
`LICENSE` and an attribution/conversion [Notice](model_licenses/qwen_image_2_1_turbo/Notice), and
`prepare_base.exe` asks for acceptance before downloading its shards. Preserve these files with copies;
the VigenFlow code's Apache-2.0 license does not replace the weight license. The agreement also contains
naming and attribution conditions; consult its full text when publishing derived materials.

The [converted Qwen weight package](https://huggingface.co/Kelsey1217/Qwen-Image-2.1-Turbo-NPU-BFP16)
is publicly available through the opt-in preparation entry. Both tasks share one five-stage package.
Both Qwen workers are integrated into `vgf-serve` and the web UI; the source-code license does not
grant additional rights to the weights.

FLUX.3 image klein is under the [FLUX Kommunity License](model_licenses/flux3_image_klein/LICENSE). Non-commercial use excludes any use that earns revenue, runs in production, affects end users, or trains models for commercial use. Section 2.a lets a Qualifying User use the outputs commercially and run the model in production only to generate them. Section 2.e then requires content filtering or review of outputs before they are shared, and AI-generated disclosures where the law requires them. For any other commercial use see <https://bfl.ai/licensing>. Section 4 also forbids military, surveillance and biometric uses, among others. Redistribution must include the license, the attribution notice, a statement that the model was modified, and no claim that Black Forest Labs endorses it. The converted weights are such a modified version. The unchanged license, the [Notice](model_licenses/flux3_image_klein/Notice) and the Apache-2.0 [license](model_licenses/flux3_image_klein/LICENSE-Qwen3-VL-4B-Instruct) of the Qwen3-VL-4B-Instruct text encoder are in `model_licenses/flux3_image_klein/`. `prepare_base.exe` asks you to accept the license before it downloads the weights, and it unpacks the license files next to them in `all_model_weights/flux3_image_klein/`. Keep them with any copy you make. Both FLUX.3 workers (text-to-image and editing) use these weights.

Other models from the same families use different licenses. For example, [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) are under FLUX non-commercial licenses. If VigenFlow adds such a model, it will be listed here with its license. Its commercial use column will read "Not allowed".

## LoRA adapters

The LoRA catalogs (`src/server_unified/*_lora_models.jsonc`) list adapters that `vgf-serve` downloads from Hugging Face when you select them. These adapters are not included in the release packages.

| LoRA | Base model | Upstream | License |
| --- | --- | --- | --- |
| anime_style | FLUX.2-klein-4B | [Sawata97/flux2_4b_koni_animestyle](https://huggingface.co/Sawata97/flux2_4b_koni_animestyle) | Apache-2.0 |
| numzoo_style | FLUX.2-klein-4B | [goumsss/numzoo-flux2-klein-lora](https://huggingface.co/goumsss/numzoo-flux2-klein-lora) | Apache-2.0 |
| background_removal | FLUX.2-klein-4B edit | [fal/flux-2-klein-4B-background-remove-lora](https://huggingface.co/fal/flux-2-klein-4B-background-remove-lora) | Apache-2.0 |
| zoom_in | FLUX.2-klein-4B edit | [fal/flux-2-klein-4B-zoom-lora](https://huggingface.co/fal/flux-2-klein-4B-zoom-lora) | Apache-2.0 |
| 2x2 sprite sheet | FLUX.2-klein-4B edit | [fal/flux-2-klein-4b-spritesheet-lora](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora) | Apache-2.0 |
| Danrisi_Lenovo_UltraReal | Z-Image-Turbo | [Danrisi/Lenovo_UltraReal_Z_Image](https://huggingface.co/Danrisi/Lenovo_UltraReal_Z_Image) | Apache-2.0 |
| JunkieMonkey69_ChaseInfinity | Z-Image-Turbo | [JunkieMonkey69/Chaseinfinity_ZimageTurbo](https://huggingface.co/JunkieMonkey69/Chaseinfinity_ZimageTurbo) | Apache-2.0 |
| Shakker-Labs-AWPortrait-Z | Z-Image-Turbo | [Shakker-Labs/AWPortrait-Z](https://huggingface.co/Shakker-Labs/AWPortrait-Z) | Apache-2.0 |
| jarod2212_PhysioShade detail Enhancer | Z-Image-Turbo | [jarod2212/PhysioShade_ZIT](https://huggingface.co/jarod2212/PhysioShade_ZIT) | Apache-2.0 |
| ostris_childrens-drawings | Z-Image-Turbo | [ostris/z_image_turbo_childrens_drawings](https://huggingface.co/ostris/z_image_turbo_childrens_drawings) | Apache-2.0 |
| renderartist_classic-painting-z | Z-Image-Turbo | [renderartist/Classic-Painting-Z-Image-Turbo-LoRA](https://huggingface.co/renderartist/Classic-Painting-Z-Image-Turbo-LoRA) | Apache-2.0 |
| renderartist_Technically-Color | Z-Image-Turbo | [renderartist/Technically-Color-Z-Image-Turbo](https://huggingface.co/renderartist/Technically-Color-Z-Image-Turbo) | Apache-2.0 |
| suayptalha-Realism | Z-Image-Turbo | [suayptalha/Z-Image-Turbo-Realism-LoRA](https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA) | Apache-2.0 |
| tarn59_pixel_art_style | Z-Image-Turbo | [tarn59/pixel_art_style_lora_z_image_turbo](https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo) | Apache-2.0 |
| wolfer45_nsfwv2-zit | Z-Image-Turbo | [wolfer45/nsfwv2-zit](https://huggingface.co/wolfer45/nsfwv2-zit) | None declared |

"None declared" means the model card states no license. Such an adapter grants no reuse rights beyond what its author gives you directly.

## Adding a model or LoRA

1. Read the license on the upstream model card and add a row above.
2. If the license restricts commercial use, redistribution, or modification, say so in the table and in the README's model list.
3. Only put converted weights in a release package if the license allows redistributing modified weights. Include the upstream license file with them.
