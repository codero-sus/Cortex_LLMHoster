# Cortex native C++ GGUF backend

`cortex-gguf-host` is Cortex's private, supervised C++ model process. It loads
GGUF directly through the llama.cpp C API and serves the private OpenAI-compatible
routes on loopback. It does **not** use Python for model loading, the
`llama-cpp-python` binding, or the external `llama-server` executable.

The numerical kernels and GGUF support come from the pinned llama.cpp/ggml core.
This is a native C++ integration and HTTP adapter, not a new tensor engine or a
claim that Cortex outperforms llama.cpp. Native release builds eliminate the
Python inference layer and enable CPU instruction-set tuning; actual speed
remains model-, hardware-, and build-dependent. No performance claim is made
without benchmarks on the target host.

## Build

Requirements: CMake 3.24+, Git, a C/C++17 compiler, and internet access for the
first configure. CMake fetches pinned source revisions of llama.cpp, cpp-httplib,
and nlohmann/json. No dependency is vendored into the Cortex repository.

Linux CPU build (the default build uses `GGML_NATIVE=ON` for the build machine):

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build native/build --target cortex-gguf-host --parallel 2
```

Use a separate build directory for each hardware target. For example, a CUDA
build requires the CUDA toolkit and a compatible compiler:

```bash
cmake -S native -B native/build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build native/build-cuda --target cortex-gguf-host --parallel 2
```

For Apple Silicon, configure on macOS with `-DGGML_METAL=ON`; for Vulkan, use
`-DGGML_VULKAN=ON` with the Vulkan SDK installed. Consult the pinned llama.cpp
build documentation for OS-specific prerequisites and supported accelerators.

The executable is written to `native/build/bin/cortex-gguf-host` (Windows adds
`.exe`). If it is built elsewhere, set `CORTEX_CPP_HOST` to its full path or
install it on `PATH`.

## Model formats

The llama.cpp inference C API loads GGUF. `cortex-cpp` accepts GGUF directly;
for supported non-GGUF inputs, Cortex runs a one-time conversion through the
converter scripts from the same pinned llama.cpp source tree, caches the output
GGUF, then starts the same native C++ inference process.

- **GGUF:** load directly. No conversion interpreter or converter dependencies
  are needed.
- **Hugging Face SafeTensors:** select a local directory with `config.json` and
  `model.safetensors` (single or sharded). `model_path` may also point at a
  standard weight file inside that directory.
- **Hugging Face PyTorch weights:** standard `pytorch_model.bin` and indexed
  shards are accepted through the upstream converter. Other model families,
  tensor names, and quantization schemes are limited to what the pinned
  llama.cpp converter supports.
- **Legacy LLaMA checkpoints:** selected `.safetensors` and legacy checkpoint
  layouts can use llama.cpp's legacy converter. This is architecture- and
  metadata-dependent; it is not a universal PyTorch checkpoint loader.
- **`.pt`, `.pth`, `.bin`, and `.ckpt`:** these extensions do not define one
  universal layout. Hugging Face `.bin` directories are handled by the HF
  converter. Legacy LLaMA pickle checkpoints require the explicit unsafe
  conversion opt-in in the model editor/TOML. Only use that opt-in for a
  checkpoint from a trusted source; pickle deserialization can execute code.
  Arbitrary training checkpoints (including many `.ckpt` files) must first be
  exported to a supported Hugging Face layout or GGUF.

For non-GGUF conversion, first build the native target so the pinned converter
scripts exist under `native/build/_deps/llama_cpp-src`. Install that upstream
converter's optional Python dependencies in a dedicated interpreter, then set
`CORTEX_CPP_CONVERTER_PYTHON` if it is not the interpreter Cortex normally
selects. `CORTEX_LLAMA_CPP_SOURCE_DIR` can point at another llama.cpp checkout.
Python is used only by the optional import/conversion step; the converted model
is loaded and used for every inference request by `cortex-gguf-host`. Cortex
does not download source weights or install conversion dependencies for you.

Converted files are cached outside the checkout in the per-user cache
(`~/.cache/cortex-llmhoster/converted` on Linux/macOS or `%LOCALAPPDATA%` on
Windows, honoring `XDG_CACHE_HOME` where set). Set `CORTEX_CPP_CACHE_DIR` to change that location;
conversions can consume disk space comparable to the source weights. The model's `conversion_outtype` can be `f16` (default), `q8_0`
(smaller), or `f32` (larger). Delete old cache directories manually when no
longer needed.

## Configure Cortex

Add a model with:

```toml
[[models]]
id = "qwen-native"
runtime = "cortex-cpp"
model_path = "/models/qwen-instruct-q4_k_m.gguf"
capabilities = ["text_generation", "embeddings"]
gpu_layers = 0
context_size = 2048
batch_size = 256
ubatch_size = 64
default = true
```

Only text generation and embeddings are currently supported by this native
adapter. Projector-based vision, audio, video, and other tasks need a different
runtime. Cortex starts one persistent native process for the selected model;
requests share a model context and are serialized at the inference boundary.

The process listens only on `127.0.0.1`, exposes `/health`, chat/completion
routes, and embeddings when enabled, and is started/stopped by Cortex's runtime
manager. Chat/completion accepts OpenAI-style `frequency_penalty` and
`presence_penalty` values in `[-2, 2]` along with temperature, top-p, top-k,
seed, max-token, and stop controls. Do not expose the private backend port
directly.

## Third-party notices

The native inference core is from [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT license); HTTP handling uses
[cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT); JSON handling uses
[nlohmann/json](https://github.com/nlohmann/json) (MIT). Their source and license
files are fetched into the local build tree by CMake and are not copied into this
repository.
