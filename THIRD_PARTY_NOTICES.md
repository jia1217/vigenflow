# Third-Party Notices

VigenFlow's binaries include or link to the open-source components below. Each component remains under its own license. For model weights, see [MODEL_LICENSES.md](MODEL_LICENSES.md).

| Component | License | Use in VigenFlow |
| --- | --- | --- |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | JSON parsing in `vgf-serve` and the workers |
| [Boost](https://www.boost.org/) (program_options, filesystem, system, atomic, beast, asio, process) | Boost Software License 1.0 | `vgf-serve` and the workers; the Windows package ships the program_options, filesystem, system and atomic DLLs |
| [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) | Apache-2.0 | Text tokenization in the workers (git submodule `external_lib/tokenizers-cpp`) |
| [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) and its Rust dependencies | Apache-2.0 / MIT | Used through tokenizers-cpp |
| [Oniguruma](https://github.com/kkos/oniguruma) | BSD-2-Clause | Regular expressions in the tokenizers (text below) |
| [SentencePiece](https://github.com/google/sentencepiece) | Apache-2.0 | Used through tokenizers-cpp |
| [stb](https://github.com/nothings/stb) (stb_image, stb_image_write, stb_image_resize) | MIT or public domain | Image input, resizing and output |
| [LodePNG](https://github.com/lvandeve/lodepng) | zlib | PNG encoding |
| [libpng](http://www.libpng.org/pub/png/libpng.html) | PNG Reference Library License v2 | PNG output in `vgf-serve`; the Windows package ships `libpng16.dll` |
| [zlib](https://zlib.net/) | zlib | Compression; the Windows package ships `zlib1.dll` |
| [libjpeg-turbo](https://libjpeg-turbo.org/) | IJG and Modified BSD | JPEG references in the Qwen Image Edit worker; the Windows package ships `jpeg62.dll` (text below) |
| [LLVM OpenMP runtime](https://openmp.llvm.org/) | Apache-2.0 with LLVM exception | Multithreading in the Windows workers (`libomp140.x86_64.dll`, redistributed from Visual Studio) |
| [AMD XRT](https://github.com/Xilinx/XRT) | Apache-2.0 | NPU runtime; each Windows worker ships the `xrt_coreutil.dll` matching the supported NPU driver, and Ubuntu uses the installed XRT |
| Microsoft Visual C++ runtime (`msvcp140*.dll`, `vcruntime140*.dll`) | Microsoft Visual C++ Redistributable terms | C++ runtime for the Windows executables |

The Apache-2.0 components use the same license text as VigenFlow's [LICENSE](LICENSE).

This software is based in part on the work of the Independent JPEG Group.

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

## libjpeg-turbo (Modified BSD License)

```text
Copyright (C)2009-2024 D. R. Commander.  All Rights Reserved.
Copyright (C)2015 Viktor Szathmáry.  All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the libjpeg-turbo Project nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Oniguruma

```text
Copyright (c) 2002-2021  K.Kosako  <kkosako0@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```
