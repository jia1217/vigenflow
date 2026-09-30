# Third-Party Notices

VigenFlow's binaries include or link to the open-source components below. Each component remains under its own license. For model weights, see [MODEL_LICENSES.md](MODEL_LICENSES.md).

| Component | License | Use in VigenFlow |
| --- | --- | --- |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | JSON parsing in `vgf-serve` and the model libraries |
| [Boost](https://www.boost.org/) (program_options, filesystem, system, beast, process) | Boost Software License 1.0 | `vgf-serve` and the model libraries; the Windows package ships the filesystem and system DLLs |
| [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) | Apache-2.0 | Text tokenization (git submodule `external_lib/tokenizers-cpp`) |
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | Used through tokenizers-cpp |
| [SentencePiece](https://github.com/google/sentencepiece) | Apache-2.0 | Used through tokenizers-cpp |
| [LodePNG](https://github.com/lvandeve/lodepng) | zlib | PNG encoding |
| [stb_image_write](https://github.com/nothings/stb) | MIT or public domain | Image output |
| [AMD XRT](https://github.com/Xilinx/XRT) | Apache-2.0 | NPU runtime; loaded from the installed driver, not shipped with VigenFlow |

The Apache-2.0 components use the same license text as VigenFlow's [LICENSE](LICENSE).

## nlohmann/json

```text
MIT License

Copyright (c) 2013-2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
