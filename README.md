# Laya Android SDK

A C/C++ inference library for [Laya](https://huggingface.co/convaiinnovations/laya) decision models, built on ggml.

Laya pairs a ModernBERT encoder with a decision head. Given a state and a set of questions, it returns a calibrated
answer for each question. Three question types are supported:

| Type     | Output                                              |
|----------|-----------------------------------------------------|
| `choice` | the most probable option and its probability        |
| `score`  | the expected level, `sum(i * p[i])`                 |
| `noul`   | `P(true)`                                           |

Each answer also includes a `confidence` value, computed as 1 minus the normalized entropy of the calibrated distribution.

The whole model (tokenizer, encoder, head and calibration) is stored in a single GGUF file and runs as one ggml graph.
llama.cpp is used only to read the tokenizer vocabulary from that file.


## Layout

```
include/            public API (laya.h) and internal headers
src/                liblaya: model loading, graph, tokenizer, Unicode NFC
tools/laya-cli.cpp  command-line runner for JSON requests
third_party/        llama.cpp (fetched automatically if missing)
```

## Building

You need CMake 3.14 or newer and a C++17 compiler. The build is CPU-only.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If `third_party/llama.cpp` is missing, CMake fetches the pinned commit (`LAYA_LLAMA_CPP_TAG`).

| Option              | Default | Description                     |
|---------------------|---------|---------------------------------|
| `LAYA_BUILD_CLI`    | `ON`    | build `laya-cli`                |
| `LAYA_LLAMA_CPP_DIR`| `third_party/llama.cpp` | llama.cpp source tree |
| `BUILD_SHARED_LIBS` | `OFF`   | build liblaya as a shared library |

To build only the library:

```sh
cmake -B build -DLAYA_BUILD_CLI=OFF
cmake --build build -j
```

If you move the project directory, delete `build/` first. The CMake cache stores absolute paths.

### Android

Cross-compile with the Android NDK (r26 or newer). Point `NDK` at your installation, for example
`~/Android/Sdk/ndk/<version>`.

```sh
NDK=~/Android/Sdk/ndk/<version>
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLAYA_BUILD_TESTS=OFF \
  -DGGML_OPENMP=OFF
cmake --build build-android -j --target laya-cli
```

- `BUILD_SHARED_LIBS=OFF` produces a single static `laya-cli`, so no `.so` files need to be copied alongside it.
- `GGML_OPENMP=OFF` avoids shipping `libomp.so`; ggml uses its own thread pool instead.
- Use `ANDROID_ABI=x86_64` for the emulator.
- On ARMv8.2+ devices, `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod` speeds up quantized matmuls.

Copy the binary and the model to the device and run them:

```sh
adb push build-android/laya-cli /data/local/tmp/
adb push models/gguf/laya.gguf /data/local/tmp/
adb shell 'cd /data/local/tmp && chmod +x laya-cli && ./laya-cli -m laya.gguf -f request.json'
```

To use liblaya from an Android app instead, build it with `-DBUILD_SHARED_LIBS=ON -DLAYA_BUILD_CLI=OFF`, or add
this `CMakeLists.txt` to the app's Gradle `externalNativeBuild { cmake { ... } }` block. Then call the C API in
`laya.h` through a JNI wrapper.


## Using the CLI

```sh
laya-cli -m models/gguf/laya.gguf -f request.json
echo '{"state": "...", "questions": {...}}' | laya-cli -m models/gguf/laya.gguf
laya-cli -m models/gguf/laya.gguf --jsonl tests/requests.jsonl   # one request per line
```

A request looks like this:

```json
{
  "state": "<string or JSON>",
  "questions": {
    "<id>": { "type": "choice" | "score" | "noul", "instructions": "...", "criteria": ... }
  }
}
```

The response has the same shape as `rl_agent_api.RLAgent.system_one()`.

| Flag                | Description                                 |
|---------------------|---------------------------------------------|
| `-m, --model`       | GGUF model path                             |
| `-f, --file`        | request file (default: stdin)               |
| `-t, --threads`     | number of threads                           |
| `--jsonl`           | read one request per line                   |
| `--logits`          | include raw logits in the output            |
| `--tokenize TEXT`   | print the token ids for TEXT                |
| `--sequence`        | print the encoder inputs only               |
| `--bench N`         | run N iterations and report timing          |
| `-v, --verbose`     | verbose logging                             |

## C API

```c
#include "laya.h"

laya_model * model = laya_model_load("models/gguf/laya.gguf");

struct laya_context_params params = laya_context_default_params();
params.n_threads = 4;
laya_context * ctx = laya_create_context(model, params);

/* ... ask questions (struct laya_question -> struct laya_answer) ... */

laya_context_free(ctx);
laya_model_free(model);
```
[![Listed on laya.tools](https://laya.tools/badge.svg)](https://laya.tools/p/alifthi-laya-android-sdk)
