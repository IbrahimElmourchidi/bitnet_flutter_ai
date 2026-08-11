# bitnet\_flutter\_ai

> **Beta — `0.2.0-beta.1`.**  
> The public API is stable in shape but may change before `1.0.0`. The Web WASM artefact is not yet published to pub.dev — see [Building Native Libraries](#building-native-libraries) below.

Run Microsoft's [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) — and nine other ternary/1-bit models from TII and PrismML — **fully on-device**, no server, no API key. See [Model catalog](#model-catalog).

| Platform | Status |
|---|---|
| Android (arm64-v8a / x86\_64) | Working — prebuilt `.so` shipped in `jniLibs/` |
| iOS (arm64 device + simulator) | Working — prebuilt `.xcframework` shipped in `ios/Frameworks/`; verified in the simulator |
| Linux (x86\_64) | Working — verified end-to-end |
| Web (desktop Chrome / Firefox) | Working — build with `tool/build_web.sh` (single-threaded, ~1–2 tok/s; desktop only; bitnet.cpp backend only) |
| macOS (arm64 / x86\_64) | Planned — native build required |
| Windows (x86\_64) | Planned — native build required |

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Web Setup](#web-setup)
- [Building Native Libraries](#building-native-libraries)
- [Architecture](#architecture)
- [Beta Limitations](#beta-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

| Requirement | Minimum |
|---|---|
| Flutter | 3.24.0 |
| Dart SDK | 3.5.0 |
| Physical RAM | 3072 MB (enforced at runtime) |
| Disk space | ~750 MB (GGUF model file) |
| Android | API 24 (arm64-v8a) |
| iOS | 14.0 (arm64) |
| macOS | 12.0 |
| Web | Chrome 92+ / Firefox 90+ with [COI headers](#web-setup) |

---

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  bitnet_flutter_ai: ^0.2.0-beta.1
```

Then run:

```bash
flutter pub get
```

---

## Quick Start

```dart
import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';

Future<void> main() async {
  final engine = BitNetEngine();

  // 1. Load — downloads the model on first run (~745 MiB), then loads from cache.
  await engine.load(
    onProgress: (progress) {
      final pct = (progress * 100).toInt();
      print('Downloading model: $pct%');
    },
  );

  // 2. Generate — returns a Stream<String> of token pieces.
  final buffer = StringBuffer();
  await for (final token in engine.generate('Explain ternary quantization in one paragraph')) {
    buffer.write(token);
    print(token); // stream tokens as they arrive
  }

  print('\n--- Full response ---\n$buffer');

  // 3. Dispose — frees native memory and kills the inference isolate.
  await engine.dispose();
}
```

### Higher-level APIs

`BitNetEngine` is the low-level token streamer. For most apps, prefer the
higher-level helpers:

```dart
// Pick a model that fits the device.
final profile = await DeviceProfile.current();
final model = BitNetCatalog.compatibleWith(profile.totalRamBytes).first;
print('Estimated speed: ~${profile.estimateTokensPerSecond(model)} tok/s');

// Load engine + chat session.
final engine = BitNetEngine(model: model);
await engine.load();
final session = BitNetSession(engine: engine, model: model)
  ..setSystemPrompt('You are a helpful, concise assistant.');

// Stream a chat reply (history is tracked across calls).
await for (final piece in session.chat('Hello!')) stdout.write(piece);

// One-shot helpers.
final tldr = await session.summarize(longText, style: 'tldr');
final answer = await session.assist('What is BitNet?');

// On-device RAG over bundled assets.
final rag = BitNetRag(
  session: session,
  source: AssetKnowledgeSource(assetPaths: ['assets/knowledge/faq.md']),
);
await rag.warmUp();
final result = await rag.ask('How big is the download?');
print(result.answer);
for (final c in result.citations) print('  - ${c.chunk.source}');
```

See the `example/` app for a complete tabbed demo of every public API.

### Streaming to a Flutter widget

```dart
class ChatPage extends StatefulWidget { ... }

class _ChatPageState extends State<ChatPage> {
  final _engine = BitNetEngine();
  final _response = ValueNotifier('');
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _loadEngine();
  }

  Future<void> _loadEngine() async {
    setState(() => _loading = true);
    await _engine.load(onProgress: (_) {});
    setState(() => _loading = false);
  }

  Future<void> _send(String prompt) async {
    _response.value = '';
    await for (final token in _engine.generate(prompt)) {
      _response.value += token;
    }
  }

  @override
  void dispose() {
    _engine.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const Center(child: CircularProgressIndicator());
    return Column(
      children: [
        ValueListenableBuilder<String>(
          valueListenable: _response,
          builder: (_, text, __) => SelectableText(text),
        ),
        ElevatedButton(
          onPressed: () => _send('Hello, who are you?'),
          child: const Text('Ask'),
        ),
      ],
    );
  }
}
```

### Cancellation

```dart
// Start generation in the background.
final sub = engine.generate('Write a long essay').listen((token) {
  print(token);
});

// Cancel after 2 seconds.
await Future.delayed(const Duration(seconds: 2));
await engine.cancelGeneration();
await sub.cancel();
```

---

## API Reference

### `BitNetEngine`

```dart
factory BitNetEngine({BitNetModel model = BitNetModel.bitnet2B4T})
```

Creates an engine for `model`. Uses a background `Isolate` on native platforms and
`dart:js_interop` on Web.

| Method / getter | Description |
|---|---|
| `Future<void> load({void Function(double)? onProgress})` | Downloads (if needed), verifies SHA-256, and loads the model. `onProgress` receives `[0.0, 1.0]`. |
| `bool get isLoaded` | `true` after a successful `load()`. |
| `Stream<String> generate(String prompt, {int maxNewTokens = 512, bool addBos = true})` | Streams token pieces until EOS or `maxNewTokens`. Throws `BitNetNotLoadedException` if not loaded. |
| `Future<void> cancelGeneration()` | Signals the worker to stop at the next token boundary. |
| `Future<void> dispose()` | Frees all native resources. Do not use the engine after calling this. |

### Model catalog

`BitNetCatalog.all` lists the curated, SHA-256-pinned models. Pick one with
`BitNetCatalog.compatibleWith(ramBytes)` (best fit first) or by id, and pass it
to `BitNetEngine(model: ...)` / `BitNetSession(model: ...)`.

| Model | Params | Quant | Backend | Min RAM | License |
|---|---|---|---|---|---|
| `microsoft/bitnet-b1.58-2B-4T` | 2.0B | I2_S | bitnet.cpp | 3 GB | MIT |
| `tiiuae/Falcon3-1B-Instruct-1.58bit` | 1.7B | I2_S | bitnet.cpp | 3 GB | Falcon LLM |
| `tiiuae/Falcon3-3B-Instruct-1.58bit` | 3.2B | I2_S | bitnet.cpp | 4 GB | Falcon LLM |
| `tiiuae/Falcon3-7B-Instruct-1.58bit` | 7.5B | I2_S | bitnet.cpp | 5 GB | Falcon LLM |
| `tiiuae/Falcon3-10B-Instruct-1.58bit` | 10.3B | I2_S | bitnet.cpp | 6 GB | Falcon LLM |
| `tiiuae/Falcon-E-1B-Instruct` | 1.7B | I2_S | bitnet.cpp | 2 GB | Falcon LLM |
| `tiiuae/Falcon-E-3B-Instruct` | 3.1B | I2_S | bitnet.cpp | 3 GB | Falcon LLM |
| `prism-ml/Ternary-Bonsai-1.7B` | 1.7B | Q2_0 | prismQ2† | 2 GB | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-4B` | 4.0B | Q2_0 | prismQ2† | 3 GB | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-8B` | 8.2B | Q2_0 | prismQ2† | 5 GB | Apache-2.0 |

Each `BitNetModelInfo` carries `id`, `ggufFileName`, `ggufDownloadUrl`,
`ggufSha256`, `minimumRamBytes`/`recommendedRamBytes`, `contextLength` (capped
at 4096), a `chatTemplate` (`ChatTemplateFamily`), and an `engineBackend`.

† **PrismML "Ternary Bonsai"** uses an incompatible `qwen3`/`Q2_0` format, so it
runs on a **second native library** (`libprism_bridge.so`, built from PrismML's
llama.cpp fork — see [Building Native Libraries](#building-native-libraries)).
Both backends share the `bn_*` C ABI; the engine picks the right library from
`model.engineBackend`. The Bonsai models are **native-only** (no WASM build
yet) and their `.so` is separate from the bitnet.cpp trio — ship it only if you
use them.

### `ModelCache`

Static utility — direct use is optional (the engine calls it internally).

```dart
// Check if model is already cached.
final path = await ModelCache.modelPath(BitNetModel.bitnet2B4T);

// Delete model + any partial download.
await ModelCache.clearModel(BitNetModel.bitnet2B4T);
```

### `DeviceInspector`

```dart
final bool capable = await DeviceInspector.instance.meetsMinimumRam();
final int ramBytes = await DeviceInspector.instance.totalPhysicalRamBytes();
```

### Exceptions

All exceptions extend `BitNetException implements Exception`.

| Exception | Thrown when |
|---|---|
| `BitNetUnsupportedDeviceException` | Physical RAM < 3072 MB |
| `BitNetUnsupportedPlatformException` | Platform not supported (e.g. Safari without WASM threads) |
| `BitNetLibraryLoadException` | Native `.so` / `.dylib` / `.dll` could not be opened |
| `BitNetInitException` | `bn_init` returned `NULL` (model file corrupt or wrong path) |
| `BitNetInferenceException` | `bn_prompt` or `bn_next_token` returned an error |
| `BitNetHashMismatchException` | SHA-256 of downloaded file does not match expected |
| `BitNetDownloadException` | Network error or unexpected HTTP status |
| `BitNetNotLoadedException` | `generate()` called before `load()` |
| `BitNetIsolateException` | Inference isolate exited unexpectedly |

```dart
try {
  await engine.load();
} on BitNetUnsupportedDeviceException catch (e) {
  print('Not enough RAM: ${e.availableRamBytes ~/ (1024 * 1024)} MB available');
} on BitNetDownloadException catch (e) {
  print('Download failed (HTTP ${e.httpStatusCode}): ${e.message}');
} on BitNetException catch (e) {
  print('Engine error: $e');
}
```

---

## Web Setup

WASM inference requires `SharedArrayBuffer`, which is gated behind Cross-Origin
Isolation (COI) headers. The bundled `coi_service_worker.js` handles this automatically.

### 1. Register the service worker

Add this snippet **before** any other `<script>` tags in `web/index.html`:

```html
<script>
  if (typeof SharedArrayBuffer === 'undefined') {
    const reloaded = sessionStorage.getItem('coi-reload');
    if (!reloaded) {
      sessionStorage.setItem('coi-reload', '1');
      navigator.serviceWorker.register('/coi_service_worker.js')
        .then(() => location.reload());
    } else {
      sessionStorage.removeItem('coi-reload');
    }
  }
</script>
```

### 2. Load the glue script

```html
<!-- Spawns bitnet_worker.js (owns the WASM module, off the UI thread) and
     defines the async window.BitNetWasm. Only this one tag is needed. -->
<script src="bitnet_glue.js"></script>
```

`tool/build_web.sh` produces four artefacts and stages them into `example/web/`:
`bitnet_bridge.js` + `bitnet_bridge.wasm` (Emscripten module), `bitnet_worker.js`
(the Web Worker that owns the module), and `bitnet_glue.js` (the main-thread
proxy). Copy all four into your own app's `web/` folder; only `bitnet_glue.js`
is referenced from `index.html` — the worker loads the bridge itself via
`importScripts`. `web/bitnet_worker.js` and `web/bitnet_glue.js` ship with the
package.

> **Web is desktop-browser only.** The model is ~1.1 GB and lives in the WASM
> heap during inference, so a tab needs multi-GB of memory — fine on desktop
> Chrome/Firefox, not on mobile browsers. Inference runs in a Web Worker (off
> the UI thread), single-threaded scalar (**~1–2 tok/s** for the 2B model).
> Multi-threading (`-pthread` + `SharedArrayBuffer`, which the bundled COI
> service worker already enables) and a hand-written `wasm_simd128` i2_s kernel
> are the two speed follow-ups — see the comments in `tool/build_web.sh` and
> `src/ggml-bitnet-mad.cpp`.

---

## Building Native Libraries

> **Note:** Pre-built binaries will be distributed via GitHub Releases in a future beta.
> For now, you must compile the C++ bridge yourself.

### Prerequisites

- CMake 3.22+
- A C++17 compiler (Clang on Apple, MSVC or Clang on Windows, GCC/Clang on Linux)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) source (or the BitNet fork: [bitnet.cpp](https://github.com/microsoft/BitNet))

### Steps (Desktop)

```bash
# 1. Clone llama.cpp alongside this package.
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp

# 2. Build the bridge as a shared library.
mkdir build && cd build
cmake .. \
  -DLLAMA_DIR=../../llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 3. Copy the output to the right location.
# macOS:  cp libbitnet_bridge.dylib <your_app>/macos/
# Linux:  cp libbitnet_bridge.so    <your_app>/linux/
# Windows: copy bitnet_bridge.dll   <your_app>\windows\
```

### Android

Cross-compile with the Android NDK using `CMakeLists.txt` in `native/`.
The output `libbitnet_bridge.so` must be placed in
`android/app/src/main/jniLibs/arm64-v8a/`.

### iOS

Prebuilt `.xcframework`s already ship in `ios/Frameworks/` (device `arm64` +
simulator `arm64`/`x86_64`) and are wired up via `vendored_frameworks` in the
podspec — no build step needed to consume the package. `bitnet_bridge.xcframework`
is a static archive merged with llama/ggml, resolved at runtime through
`DynamicLibrary.process()`; `prism_bridge.xcframework` is a self-contained
dynamic framework, resolved through `DynamicLibrary.open(path)`. Rebuild them
only if you're changing the native bridge itself (Xcode + cmake + ninja +
libtool/lipo required):

```bash
BITNET_CPP=../bitnet.cpp ./tool/build_ios.sh              # bitnet_bridge.xcframework
PRISM_LLAMA_CPP=../prism-llama.cpp ./tool/build_ios_prism.sh  # prism_bridge.xcframework
```

### Web (WASM)

```bash
# Requires Emscripten (emsdk) on PATH.
source ~/emsdk/emsdk_env.sh
./tool/build_web.sh
# Produces (staged into example/web/):
#   bitnet_bridge.js + bitnet_bridge.wasm   (Emscripten module)
#   bitnet_glue.js                          (window.BitNetWasm glue, from web/)
```

`tool/build_web.sh` applies `native/patches/0001-i2s-portable-kernels.patch`
(the WASM path uses the portable scalar i2\_s kernels — no AVX2/NEON), builds
static `libllama.a` + `libggml.a` for wasm32 with `-fexceptions`, then links the
embind bridge (`native/bitnet_bridge_wasm.cc`) with a 32 MB stack and a 4 GB
memory ceiling. Copy the three artefacts into your app's `web/` folder and add
the two `<script>` tags from [Web Setup](#web-setup).

### PrismML (Ternary Bonsai) backend

Only needed if you use a `Ternary-Bonsai-*` model. It builds a **separate**
self-contained library from PrismML's llama.cpp fork (a different ggml ABI
than bitnet.cpp — the two libraries never share symbols). Android and iOS
prebuilts already ship in `jniLibs/` and `ios/Frameworks/` respectively —
only rebuild if you're changing the bridge itself.

```bash
git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git ../prism-llama.cpp
./tool/build_linux_prism.sh                                             # Linux desktop / host tests
ANDROID_NDK=$HOME/Android/Sdk/ndk/<ver> ./tool/build_android_prism.sh   # arm64-v8a + x86_64
./tool/build_ios_prism.sh                                               # prism_bridge.xcframework
```

All three scripts drive `native/prism/CMakeLists.txt`, which links the fork
static and version-scripts the output down to just the `bn_*` C ABI. The
Android script drops `libprism_bridge.so` into `jniLibs/<abi>/`; the iOS
script produces `prism_bridge.xcframework` in `ios/Frameworks/`. No web build
yet.

---

## Architecture

```
BitNetEngine (interface)
 ├── NativeEngine          — runs on native (Android/iOS/Desktop)
 │    ├── Isolate           dart:isolate worker (non-blocking UI)
 │    └── NativeLibrary     dart:ffi → libbitnet_bridge
 │         └── BitNetBindings  (ffigen-generated)
 │              └── bitnet_bridge.cc  (llama.cpp C API wrapper)
 │
 └── WebEngine             — runs on Web
      └── dart:js_interop  → window.BitNetWasm (Emscripten WASM)

ModelCache                 — HTTP download, SHA-256 verify, disk cache
DeviceInspector            — platform RAM detection (fail-closed gate)
```

**Inference flow:**

1. `engine.load()` — device check → model download/verify → spawn isolate → `bn_init`
2. `engine.generate(prompt)` — `bn_prompt` (prefill) → `bn_next_token` loop → stream
3. Each `bn_next_token` call: `llama_sampler_sample` → `llama_token_to_piece` → `llama_decode`

---

## Beta Limitations

- Prebuilt libraries ship for Android (`arm64-v8a`, `x86_64`) and iOS (device
  `arm64` + simulator). macOS and Windows still require building the bridge
  from source; web is built with `tool/build_web.sh`. Nothing is distributed
  via pub.dev yet.
- iOS is verified in the **simulator only** — not yet run on a physical device.
- The PrismML (`Ternary-Bonsai-*`, Q2\_0) backend is **native-only**: Android
  and iOS, no Web build of the fork yet.
- Web is **desktop-browser only** and **single-threaded scalar** (~1–2 tok/s for
  the 2B model). The ~1.1 GB model lives in the WASM heap during inference, so a
  tab needs multi-GB of memory — mobile browsers will OOM. Threading and a
  `wasm_simd128` kernel are the planned speed follow-ups.
- `cancelGeneration()` on Web cancels only at `yield` boundaries — there is no
  signal to interrupt the synchronous WASM decode call mid-token.
- macOS and Windows are untested in this beta.
- No sampling parameter control yet (temperature, top-p). Greedy sampling is used.
- The model is not instruction-tuned by default; wrap your prompt in the appropriate chat template for best results.

---

## Contributing

Issues and PRs are welcome at
[github.com/IbrahimElmourchidi/bitnet_flutter_ai](https://github.com/IbrahimElmourchidi/bitnet_flutter_ai).

Please open an issue before sending a large PR so we can align on scope.

---

## License

MIT — see [LICENSE](LICENSE).

Published by [utanium.org](https://pub.dev/publishers/utanium.org/packages).
