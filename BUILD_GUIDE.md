# Native Build Guide — bitnet_flutter_ai

This document walks through building `libbitnet_bridge` for every supported platform.
The bridge wraps Microsoft's BitNet b1.58 kernels (via bitnet.cpp / llama.cpp) and is
the only native dependency of the plugin.

---

## Overview

```
bitnet.cpp (fork of llama.cpp with BitNet I2_S kernels)
    ↓  builds to
libllama (.so / .dylib / .dll)
    ↓  linked by
native/bitnet_bridge.cc  →  libbitnet_bridge (.so / .dylib / .a / .dll)
    ↓  loaded at runtime by
Dart FFI (NativeLibrary.open())
```

You need to build **both** `libllama` and `libbitnet_bridge`.
`libllama` ships no prebuilt binaries — you compile it from source.

---

## Prerequisites (all platforms)

| Tool | Version | Download |
|---|---|---|
| Git | any recent | https://git-scm.com |
| CMake | ≥ 3.22 | https://cmake.org/download |
| Python | ≥ 3.10 | https://python.org (needed for bitnet.cpp setup script) |
| Ninja (recommended) | any | https://ninja-build.org |

---

## Step 1 — Clone bitnet.cpp

Microsoft's bitnet.cpp is a llama.cpp fork with the I2\_S BitNet kernels.
Clone it alongside your Flutter package:

```bash
# From the parent directory of bitnet_flutter_ai/
git clone https://github.com/microsoft/BitNet.git bitnet.cpp
cd bitnet.cpp
```

> **Why bitnet.cpp and not vanilla llama.cpp?**
> The BitNet b1.58 I2\_S GGUF format uses 1.58-bit weights that need custom
> dequantization kernels. These only exist in Microsoft's fork. Vanilla llama.cpp
> will refuse to load the file.

---

## Step 2 — Install bitnet.cpp Python dependencies

```bash
cd bitnet.cpp
pip install -r requirements.txt
```

This installs the model download helper and quantisation scripts.

---

## Step 3 — Apply the portable-kernels patch (ALL platforms)

> **Required on every platform, including Linux desktop.** Skipping this
> produces a library that appears to load fine but emits a runaway stream of
> `!` at inference time.

The pristine bitnet.cpp I2\_S matmul kernels
(`src/ggml-bitnet-mad.cpp`) are implemented **only** for AVX2 and ARM NEON,
with no fallback:

- On a target compiled without AVX2 (notably the **Android x86_64 emulator** —
  the NDK x86_64 baseline has no AVX2), the kernels compile to *empty
  functions*. Every matmul then writes nothing, the logits come out flat, and
  greedy sampling returns token 0 (`"!"`) forever.
- The NEON path additionally used a wrong on-disk weight layout, so arm64
  results were incorrect even though the kernels were non-empty.

Apply the bundled patch to the bitnet.cpp checkout. It adds a portable scalar
fallback (bit-exact with the AVX2 kernel) and a corrected canonical-layout NEON
path, and fixes an arm64 `PARALLEL_SIZE` mis-dispatch:

```bash
# From the bitnet.cpp checkout, with bitnet_flutter_ai/ as a sibling:
git apply ../bitnet_flutter_ai/native/patches/0001-i2s-portable-kernels.patch
```

`tool/build_android.sh` applies this patch automatically (idempotently), so
you only need to run the command above for manual desktop/iOS builds.

> **Optional — faster x86_64 emulator.** On a host CPU that supports AVX2 you
> can build the Android x86_64 library with the AVX2 code paths for a large
> speedup: `BITNET_ANDROID_X86_AVX2=1 ./tool/build_android.sh`. Leave it off
> (the default) if the emulator might run on a non-AVX2 host, or the binary
> will `SIGILL`. The scalar kernels are always correct regardless.

---

## Platform builds

---

## Desktop — Windows

### Tools needed

| Tool | Download |
|---|---|
| Visual Studio 2022 (C++ workload) | https://visualstudio.microsoft.com/vs/community/ |
| CMake 3.22+ | Bundled with VS, or https://cmake.org/download |

### 3a. Build libllama (Windows)

Open **x64 Native Tools Command Prompt for VS 2022**, then:

```bat
cd bitnet.cpp
cmake -B build-windows -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DLLAMA_NATIVE=OFF ^
      -DLLAMA_BUILD_TESTS=OFF ^
      -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-windows --config Release --target llama
```

Output: `bitnet.cpp\build-windows\bin\Release\llama.dll`
Import lib: `bitnet.cpp\build-windows\Release\llama.lib`

### 3b. Build bitnet_bridge.dll (Windows)

```bat
cd bitnet_flutter_ai\native
cmake -B build-windows -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DLLAMA_INSTALL_PREFIX=..\..\bitnet.cpp\build-windows
cmake --build build-windows --config Release
```

Output: `native\build-windows\Release\bitnet_bridge.dll`

### 3c. Copy outputs

```bat
:: Create destination folders
mkdir bitnet_flutter_ai\windows
copy native\build-windows\Release\bitnet_bridge.dll  bitnet_flutter_ai\windows\
copy ..\bitnet.cpp\build-windows\bin\Release\llama.dll  bitnet_flutter_ai\windows\
```

> Both DLLs must ship together — `bitnet_bridge.dll` loads `llama.dll` at runtime.

---

## Desktop — macOS

### Tools needed

| Tool | Install |
|---|---|
| Xcode 15+ | App Store |
| Xcode Command Line Tools | `xcode-select --install` |
| CMake | `brew install cmake` |
| Ninja | `brew install ninja` |

### 3a. Build libllama (macOS)

```bash
cd bitnet.cpp
cmake -B build-macos -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"   # universal binary
ninja -C build-macos llama
```

Output: `bitnet.cpp/build-macos/libllama.dylib`

### 3b. Build libbitnet_bridge.dylib (macOS)

```bash
cd bitnet_flutter_ai/native
cmake -B build-macos -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_INSTALL_PREFIX=../../bitnet.cpp/build-macos \
      -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
ninja -C build-macos
```

Output: `native/build-macos/bitnet_bridge.dylib`

### 3c. Fix the rpath so the dylib finds libllama at runtime

```bash
install_name_tool -add_rpath @loader_path \
    native/build-macos/bitnet_bridge.dylib
```

### 3d. Copy outputs

```bash
mkdir -p bitnet_flutter_ai/macos
cp native/build-macos/bitnet_bridge.dylib  bitnet_flutter_ai/macos/
cp bitnet.cpp/build-macos/libllama.dylib   bitnet_flutter_ai/macos/
```

---

## Desktop — Linux

> **Verified pipeline.** The steps below were exercised end-to-end on Ubuntu
> (x86_64, GCC 14, CMake 3.31, Ninja 1.12) and produced ~12 tok/s on a
> Ryzen 7 5700U for the 2B-4T I2_S model.

### Tools needed

```bash
sudo apt-get install -y build-essential cmake ninja-build git python3 python3-pip
pip install --user numpy
```

> **Compiler choice — important.** The bitnet kernels in
> `src/ggml-bitnet-mad.cpp` rely on `-fpermissive`, which the project's
> `CMakeLists.txt` only enables for **GCC**. Building with Clang fails with
> a const-correctness error. Always pass `-DCMAKE_C_COMPILER=gcc
> -DCMAKE_CXX_COMPILER=g++` on Linux.

### 3a. Generate the BitNet kernel header (codegen)

bitnet.cpp's CMake build expects `include/bitnet-lut-kernels.h` to exist. For
the 2B-4T model on x86_64 it reuses the 3B kernel parameters:

```bash
cd bitnet.cpp
mkdir -p include
python3 utils/codegen_tl2.py --model bitnet_b1_58-3B \
    --BM 160,320,320 --BK 96,96,96 --bm 32,32,32
```

### 3b. Build libllama (Linux)

```bash
cd bitnet.cpp
cmake -B build-linux -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBITNET_X86_TL2=OFF \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_NATIVE=OFF
cmake --build build-linux --target llama -- -j$(nproc)
```

Outputs (note the nested paths — bitnet.cpp does not create a flat
`build-linux/lib/`):

```
bitnet.cpp/build-linux/3rdparty/llama.cpp/src/libllama.so
bitnet.cpp/build-linux/3rdparty/llama.cpp/ggml/src/libggml.so
```

### 3c. Stage the headers + libs into the layout the bridge expects

The plugin's [native/CMakeLists.txt](native/CMakeLists.txt) takes a single
`LLAMA_INSTALL_PREFIX` and looks for `include/llama.h` and `lib/libllama.so`
under it. Because bitnet.cpp does not actually run `cmake --install`, we
stage the artefacts via symlinks:

```bash
cd bitnet.cpp
mkdir -p stage/lib stage/include
ln -sf $PWD/build-linux/3rdparty/llama.cpp/src/libllama.so       stage/lib/libllama.so
ln -sf $PWD/build-linux/3rdparty/llama.cpp/ggml/src/libggml.so   stage/lib/libggml.so
ln -sf $PWD/3rdparty/llama.cpp/include/llama.h                   stage/include/llama.h
for h in $PWD/3rdparty/llama.cpp/ggml/include/*.h; do
    ln -sf "$h" stage/include/$(basename "$h")
done
```

### 3d. Build libbitnet_bridge.so

```bash
cd bitnet_flutter_ai/native
cmake -B build-linux -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_INSTALL_PREFIX=$PWD/../../bitnet.cpp/stage
ninja -C build-linux
```

Output: `native/build-linux/libbitnet_bridge.so`

### 3e. Smoke-test the bridge end-to-end

Download the GGUF (1,187,801,280 bytes, SHA-256
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`):

```bash
mkdir -p ../bitnet.cpp/models/BitNet-b1.58-2B-4T
curl -L -C - -o ../bitnet.cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf
```

C smoke test (`native/smoke_test.c`):

```bash
cd bitnet_flutter_ai/native
gcc smoke_test.c -o build-linux/smoke_test \
    -I. -L./build-linux -l:libbitnet_bridge.so \
    -Wl,-rpath,$PWD/build-linux \
    -Wl,-rpath,$PWD/../../bitnet.cpp/build-linux/3rdparty/llama.cpp/src \
    -Wl,-rpath,$PWD/../../bitnet.cpp/build-linux/3rdparty/llama.cpp/ggml/src
./build-linux/smoke_test \
    ../../bitnet.cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    "Hello, who are you?"
```

ABI v2 smoke test (`native/test/abi_v2_smoke.c`) — token counting, KV prefix
reuse, grammar-constrained JSON with log-probabilities, greedy determinism,
abort from another thread, and the typed too-long error:

```bash
cd bitnet_flutter_ai
cc -O2 -pthread -Inative native/test/abi_v2_smoke.c \
    -Lnative/build-linux -lbitnet_bridge -lm \
    -Wl,-rpath,$PWD/native/build-linux -o /tmp/abi_v2_smoke
/tmp/abi_v2_smoke ../bitnet.cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# PrismML: link -lprism_bridge from native/build-linux-prism and pass "chatml"
```

Full-stack Dart test (session → isolate → FFI → C), skipped unless
`BITNET_MODEL` is set:

```bash
cd bitnet_flutter_ai
BITNET_MODEL=../bitnet.cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
LD_LIBRARY_PATH="$PWD/native/build-linux:$PWD/../bitnet.cpp/stage/lib:$PWD/../bitnet.cpp/build-linux/3rdparty/llama.cpp/ggml/src" \
    flutter test test/real_model_test.dart
```

### 3f. Copy outputs into the package's `linux/` desktop folder

```bash
mkdir -p bitnet_flutter_ai/linux
cp native/build-linux/libbitnet_bridge.so       bitnet_flutter_ai/linux/
cp -L bitnet.cpp/stage/lib/libllama.so           bitnet_flutter_ai/linux/
cp -L bitnet.cpp/stage/lib/libggml.so            bitnet_flutter_ai/linux/
```

---

## Android (arm64-v8a)

### Tools needed

| Tool | Download / Install |
|---|---|
| Android Studio | https://developer.android.com/studio |
| NDK r26b (or later) | Android Studio → SDK Manager → SDK Tools → NDK |
| CMake 3.22 (NDK bundled) | Same SDK Manager panel |

> **NDK path** on Windows: `%LOCALAPPDATA%\Android\Sdk\ndk\<version>`  
> **NDK path** on macOS/Linux: `~/Library/Android/sdk/ndk/<version>` or `~/Android/Sdk/ndk/<version>`

### 3a. Cross-compile libllama for arm64-v8a

```bash
export ANDROID_NDK=/path/to/ndk   # e.g. ~/Android/Sdk/ndk/26.3.11579264

cd bitnet.cpp
cmake -B build-android-arm64 -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-24 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF
ninja -C build-android-arm64 llama
```

Output: `bitnet.cpp/build-android-arm64/libllama.so`

### 3b. Cross-compile libbitnet_bridge.so for arm64-v8a

```bash
cd bitnet_flutter_ai/android
cmake -B build-android-arm64 -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-24 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_INCLUDE_DIR=../../bitnet.cpp/include \
      -DLLAMA_LIB=../../bitnet.cpp/build-android-arm64/libllama.so
ninja -C build-android-arm64
```

Output: `android/build-android-arm64/libbitnet_bridge.so`

### 3c. Copy outputs

```bash
mkdir -p bitnet_flutter_ai/android/src/main/jniLibs/arm64-v8a
cp android/build-android-arm64/libbitnet_bridge.so \
   bitnet_flutter_ai/android/src/main/jniLibs/arm64-v8a/
cp bitnet.cpp/build-android-arm64/libllama.so \
   bitnet_flutter_ai/android/src/main/jniLibs/arm64-v8a/
```

> Both `.so` files go in `jniLibs/arm64-v8a/`. Android packages everything in that
> folder into the APK automatically.

---

## iOS (arm64)

### Tools needed

- macOS machine (Xcode builds iOS — cannot cross-compile from Windows/Linux)
- Xcode 15+
- CMake: `brew install cmake`

### 3a. Build libllama as a static library for iOS arm64

```bash
export SDKROOT=$(xcrun --sdk iphoneos --show-sdk-path)

cd bitnet.cpp
cmake -B build-ios-arm64 -G Xcode \
      -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_SYSROOT=$SDKROOT \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF
xcodebuild -project build-ios-arm64/llama.xcodeproj \
           -scheme llama \
           -configuration Release \
           -sdk iphoneos \
           ARCHS=arm64 \
           BUILD_DIR=$(pwd)/build-ios-arm64/output
```

Output: `bitnet.cpp/build-ios-arm64/output/Release-iphoneos/libllama.a`

### 3b. Build libbitnet_bridge.a for iOS arm64

```bash
cd bitnet_flutter_ai/native
cmake -B build-ios-arm64 -G Xcode \
      -DCMAKE_SYSTEM_NAME=iOS \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
      -DIOS=ON \
      -DLLAMA_INSTALL_PREFIX=../../bitnet.cpp/build-ios-arm64/output/Release-iphoneos
xcodebuild -project build-ios-arm64/bitnet_bridge.xcodeproj \
           -scheme bitnet_bridge_static \
           -configuration Release \
           -sdk iphoneos \
           ARCHS=arm64 \
           BUILD_DIR=$(pwd)/build-ios-arm64/output
```

Output: `native/build-ios-arm64/output/Release-iphoneos/libbitnet_bridge.a`

### 3c. Copy outputs

```bash
mkdir -p bitnet_flutter_ai/ios/Frameworks
cp native/build-ios-arm64/output/Release-iphoneos/libbitnet_bridge.a \
   bitnet_flutter_ai/ios/Frameworks/

# llama must also be linked — add it to the podspec or merge into a fat lib:
cp bitnet.cpp/build-ios-arm64/output/Release-iphoneos/libllama.a \
   bitnet_flutter_ai/ios/Frameworks/
```

Then update [ios/bitnet_flutter_ai.podspec](ios/bitnet_flutter_ai.podspec):
```ruby
s.vendored_libraries = 'Frameworks/libbitnet_bridge.a', 'Frameworks/libllama.a'
```

---

## Optional — PrismML (Ternary Bonsai) backend

The `Ternary-Bonsai-*` catalog models use PrismML's `qwen3`/`Q2_0` GGUF format,
which is **incompatible** with the bitnet.cpp libraries built above (different
ggml ABI). They run on a second, self-contained `libprism_bridge.so` built from
PrismML's llama.cpp fork. Skip this entirely if you don't ship Bonsai models.

```bash
# 1. Clone the fork as a sibling of bitnet_flutter_ai/ (pin a known-good commit
#    in production — the fork tracks upstream llama.cpp).
git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git ../prism-llama.cpp

# 2a. Linux desktop / host smoke tests:
./tool/build_linux_prism.sh
#     → native/build-linux-prism/libprism_bridge.so  (also copied to linux/)

# 2b. Android (arm64-v8a + x86_64):
ANDROID_NDK=$HOME/Android/Sdk/ndk/<ver> ./tool/build_android_prism.sh
#     → android/src/main/jniLibs/<abi>/libprism_bridge.so
```

Both scripts use `native/prism/CMakeLists.txt`, which links the fork **static**
into one shared library and applies `native/prism/prism_exports.map` (a GNU-ld
version script) so the `.so` exports **only** the `bn_*` C ABI — its modern
ggml symbols stay hidden and cannot interpose on the bitnet.cpp `libggml.so`
loaded in the same process. Dart selects the library from
`BitNetModelInfo.engineBackend` and dlopen's it `RTLD_LOCAL`.

Smoke-test the bridge directly (host):

```bash
gcc -I native native/prism_smoke.c -L native/build-linux-prism -lprism_bridge -o /tmp/prism_smoke
LD_LIBRARY_PATH=native/build-linux-prism /tmp/prism_smoke <bonsai>.gguf
# expect: bn_init OK … coherent generated text … PASS
```

There is no web (WASM) build of the prism backend yet.

---

## Step 4 — Download the GGUF model (optional manual test)

```bash
cd bitnet.cpp
python run_inference.py -m microsoft/bitnet-b1.58-2B-4T \
    --hf-repo microsoft/bitnet-b1.58-2B-4T-gguf \
    -f ggml-model-i2_s.gguf \
    -p "Hello, who are you?" -n 64
```

Or download directly:

```bash
# Linux/macOS
curl -L -o ggml-model-i2_s.gguf \
  "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"

# Get the SHA-256 to pin in bitnet_model.dart
sha256sum ggml-model-i2_s.gguf          # Linux
shasum -a 256 ggml-model-i2_s.gguf      # macOS
Get-FileHash ggml-model-i2_s.gguf       # Windows PowerShell
```

Pin the hash in [lib/src/model/bitnet_model.dart](lib/src/model/bitnet_model.dart):
```dart
ggufSha256: 'paste_the_real_sha256_hex_here',
```

---

## Step 5 — Verify the bridge works (smoke test)

Write a quick standalone C test to confirm the library loads before wiring it to Flutter:

```bash
cd bitnet_flutter_ai/native
cat > smoke_test.c << 'EOF'
#include "bitnet_bridge.h"
#include <stdio.h>
int main() {
    char* err = NULL;
    BnContext* ctx = bn_init("/path/to/ggml-model-i2_s.gguf", 512, 4, &err);
    if (!ctx) { fprintf(stderr, "bn_init failed: %s\n", err ? err : "unknown"); return 1; }
    printf("Context size: %d\n", bn_context_size(ctx));
    bn_prompt(ctx, "Hello", 1, NULL);
    int done = 0;
    for (int i = 0; i < 10; i++) {
        char* tok = bn_next_token(ctx, &done, NULL);
        if (done || !tok) break;
        printf("%s", tok);
        bn_free_string(tok);
    }
    printf("\n");
    bn_free(ctx);
    return 0;
}
EOF

# Linux/macOS
gcc smoke_test.c -o smoke_test \
    -I. -L./build-<platform> -lbitnet_bridge -Wl,-rpath,./build-<platform>
./smoke_test
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `llama.h: No such file or directory` | Wrong `LLAMA_INSTALL_PREFIX` | Check the path contains `include/llama.h` |
| `undefined symbol: llama_model_load_from_file` | Linked against wrong llama build | Ensure `libllama` was built from the bitnet.cpp fork, not vanilla llama.cpp |
| `SIGILL` on Android at runtime | ABI mismatch | Ensure both `.so` files were built for `arm64-v8a` with the same NDK version |
| iOS link error `missing libllama.a` | Static lib not in Frameworks/ | Re-run step 3c and update the podspec `vendored_libraries` list |
| `DynamicLibrary.open failed` on Windows | DLL not on PATH or wrong folder | Both `bitnet_bridge.dll` and `llama.dll` must be in `windows/` |
| Model loads but output is garbage | Wrong model file or vanilla llama.cpp | Download `ggml-model-i2_s.gguf` from the official HF repo and use bitnet.cpp |
