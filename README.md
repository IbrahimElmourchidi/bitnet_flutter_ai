# bitnet\_flutter\_ai

> **Beta — `0.3.0-beta.1`.**  
> The public API is stable in shape but may change before `1.0.0` (0.3.0 has
> breaking changes — see the [CHANGELOG](CHANGELOG.md) for migration steps).
> The Web WASM artefact is not published to pub.dev — see
> [Building Native Libraries](#building-native-libraries) below.

Run Microsoft's [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) — and nine other ternary/1-bit models from TII and PrismML — **fully on-device**, no server, no API key. See [Model catalog](#model-catalog).

| Platform | Status |
|---|---|
| Android (arm64-v8a / x86\_64) | Working — prebuilt `.so` shipped in `jniLibs/`; verified on the emulator |
| iOS (arm64 device + simulator) | Prebuilt `.xcframework` shipped in `ios/Frameworks/` (0.3.0 needs the ABI v2 rebuild — see [iOS](#ios)) |
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

### Toolchain

| Tool | Minimum | Notes |
|---|---|---|
| Flutter | **3.50.0** | |
| Dart SDK | **3.14.0** | `environment: sdk: ^3.14.0` |
| Gradle (Android host app) | **9.3.1** | Enforced by Flutter 3.50 (`gradle-wrapper.properties`) |
| Android Gradle Plugin | **9.3.1** recommended | The plugin module is Java-only, so it also builds under AGP 8 |
| JDK | 17 | Gradle 9.5 also runs on JDK 21 / 25 |
| Xcode (iOS / macOS) | 16 | Swift Package Manager or CocoaPods |

### Platforms

| Platform | Minimum OS | ABIs | Notes |
|---|---|---|---|
| Android | **API 24** (Android 7.0) — `minSdk 24` | `arm64-v8a`, `x86_64` | Build with `compileSdk 36`. Thermal status needs API 29+ (reported as `unknown` below). |
| iOS | **14.0** | `arm64` (device), `arm64`/`x86_64` (simulator) | `os_proc_available_memory` (available RAM) needs iOS 13+. |
| macOS | 12.0 | `arm64`, `x86_64` | Bridge must be built from source. |
| Linux | — | `x86_64` | Bridge built from source (CMake; see BUILD_GUIDE.md). |
| Web | Chrome 92+ / Firefox 90+ (desktop) | wasm32 | Requires [COI headers](#web-setup); bitnet.cpp backend only. |

### Device

| Resource | Minimum |
|---|---|
| Physical RAM | per model — `BitNetModelInfo.minimumRamBytes` (2–6 GB), enforced at `load()` |
| Free disk | the model file: 0.46 GB (Bonsai 1.7B) to 4 GB (Falcon3 10B); 1.19 GB for the default BitNet 2B-4T |

---

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  bitnet_flutter_ai: ^0.3.0-beta.1
```

Then run:

```bash
flutter pub get
```

---

## Quick Start

```dart
import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';

final engine = BitNetEngine(model: BitNetCatalog.defaultModel);
await engine.load(onProgress: (p) => print('Downloading: ${(p * 100).toInt()}%'));

final session = BitNetSession(
  engine: engine,
  systemPrompt: 'You are a helpful, concise assistant.',
);
await for (final text in session.chat('Explain ternary quantization')) {
  stdout.write(text);
}

await engine.dispose(); // frees the model and the inference isolate
```

### Structured extraction (e.g. expense parsing)

`extract` returns JSON of exactly the shape you declare — decoding is
grammar-constrained, so it always parses — plus how sure the model was:

```dart
const expense = JsonShape({
  'amount': JsonField.number(description: 'Amount of money spent'),
  'currency': JsonField.enumOf(['EGP', 'USD', 'EUR'],
      description: 'Assume EGP when not stated'),
  'vendor': JsonField.string(nullable: true, description: 'Shop or payee'),
  'category': JsonField.enumOf(['food', 'groceries', 'transport', 'utilities', 'other']),
  'date': JsonField.date(nullable: true),
});

final r = await session.extract(
  'I spent 250 EGP at the supermarket yesterday',
  shape: expense,
  instructions: "Extract the expense from the user's sentence.",
  examples: const [
    ExtractExample('electricity bill 150', {'amount': 150, 'currency': 'EGP',
        'vendor': null, 'category': 'utilities', 'date': null}),
  ],
  today: DateTime.now(),
);

if (r.isConfident(0.5)) {
  save(r.value);                    // {amount: 250.0, currency: EGP, …}
} else {
  askUserToConfirm(r.value, highlight: r.uncertainFields());
}
```

Everything except the input text forms a stable prompt prefix that stays in
the native KV cache, so the second and later extractions only process the new
sentence (3–5× faster prefill in our measurements).

### More helpers

```dart
// Pick a model that fits the device.
final profile = await DeviceProfile.current();
final model = BitNetCatalog.compatibleWith(profile.totalRamBytes).first;

// One-shot helpers (do not touch chat history).
final tldr = await session.summarize(longText, style: SummaryStyle.tldr);
final answer = await session.assist('What is BitNet?', context: snippets);

// On-device RAG over bundled assets.
final rag = BitNetRag(
  session: session,
  source: AssetKnowledgeSource(assetPaths: ['assets/knowledge/faq.md']),
);
final result = await rag.ask('How big is the download?');

// Defer heavy work when the device is hot / saving power / low on memory.
final health = await DeviceInspector.instance.health();
if (health.shouldThrottle) scheduleLater();
```

Several sessions may share one engine (e.g. an extractor and an advisor chat).
Generations are queued in call order and never interleave.

### Cancellation

```dart
final sub = session.chat('Write a long essay').listen(stdout.write);
await session.stop();   // or engine.cancelGeneration(), or sub.cancel()
```

On native platforms cancellation is immediate even during a long prompt
prefill (a shared native flag the C layer polls); on web it takes effect at the
next token.

See the `example/` app for a tabbed demo of every public API.

---

## API Reference

### `BitNetEngine`

| Member | Description |
|---|---|
| `BitNetEngine({BitNetModelInfo model})` | Native isolate engine, or WASM worker engine on web. |
| `BitNetModelInfo get model` | The model it runs (sessions take their chat template from it). |
| `Future<void> load({onProgress})` | Downloads (if needed), verifies SHA-256 and loads. Concurrent calls share one load. |
| `Stream<TokenEvent> generateTokens(String prompt, {GenerationOptions options})` | Raw completion of a templated prompt: text chunks + per-token log-probabilities. |
| `Stream<String> generate(String prompt, {GenerationOptions options})` | Text-only convenience. |
| `Future<int> countTokens(String text, {bool addBos})` | Tokens the text occupies in the context window. |
| `Future<void> cancelGeneration()` | Stops the running generation; its stream completes with what was produced. |
| `Future<void> dispose()` | Frees the model and the isolate / WASM context. |

### `GenerationOptions`

`GenerationOptions()` — light sampling (temperature 0.7, top-k 40, top-p 0.95,
repetition penalty 1.3) for chat. `GenerationOptions.precise()` — greedy, no
penalty, for structured or reproducible output. Both accept `maxNewTokens`, a
GBNF `grammar` (root rule `root`) and extra `stop` strings.

### `BitNetSession`

| Member | Description |
|---|---|
| `chat(message)` | Streams a reply; history is kept and trimmed by token budget. |
| `extract(text, shape:, instructions:, examples:, today:)` | Grammar-constrained JSON + `ExtractResult.confidence` / `fieldConfidence`. |
| `summarize(text, style:)` / `assist(question, context:)` | Stateless one-shots. |
| `stop()` | Cancels the running generation. |
| `history`, `clearHistory()`, `setSystemPrompt()` | Conversation state. |

### Model catalog

`BitNetCatalog.all` lists the curated, SHA-256-pinned models. Pick one with
`BitNetCatalog.compatibleWith(ramBytes)` (best fit first) or by id, and pass it
to `BitNetEngine(model: ...)`.

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

**Languages:** BitNet 2B-4T and Falcon3 are English-centric. For other
languages (e.g. Arabic input) prefer the Qwen3-based Ternary Bonsai models.

† **PrismML "Ternary Bonsai"** uses an incompatible `qwen3`/`Q2_0` format, so it
runs on a **second native library** (`libprism_bridge.so`, built from PrismML's
llama.cpp fork `prism` branch — see [Building Native Libraries](#building-native-libraries)).
Both backends share the `bn_*` C ABI; the engine picks the right library from
`model.engineBackend`. Bonsai is native-only (no WASM build yet).

### `ModelCache`

Static utility — the engine calls it internally. Models are stored outside
device backups: Android `getNoBackupFilesDir()`, iOS/macOS Application Support
flagged `NSURLIsExcludedFromBackupKey`.

```dart
final cached = await ModelCache.isCached(BitNetCatalog.defaultModel);
await ModelCache.clearModel(BitNetCatalog.defaultModel);
```

### `DeviceInspector`

```dart
final health = await DeviceInspector.instance.health();
// DeviceHealth(total, available RAM, lowMemory, thermal, lowPowerMode)
```

Uses Dart 3.14 direct native interop — JNIgen bindings on Android
(`jnigen.yaml`), Foundation via `package:objective_c` on iOS/macOS — so it
works from any isolate (no platform channel).

### Exceptions

All exceptions extend `BitNetException implements Exception`.

| Exception | Thrown when |
|---|---|
| `BitNetUnsupportedDeviceException` | Physical RAM below the model's minimum |
| `BitNetUnsupportedPlatformException` | Platform / backend not supported (e.g. Bonsai on web) |
| `BitNetLibraryLoadException` | Native library missing, or a stale build predating ABI v2 |
| `BitNetInitException` | Model could not be loaded |
| `BitNetInferenceException` | Prefill / decode / sampler (e.g. invalid grammar) error |
| `BitNetPromptTooLongException` | Prompt does not fit in the context window |
| `BitNetExtractionException` | `extract` output was cut short (e.g. `maxNewTokens`) |
| `BitNetHashMismatchException` | SHA-256 of downloaded file does not match |
| `BitNetDownloadException` | Network error / incomplete download (resumes next call) |
| `BitNetNotLoadedException` | Generation requested before `load()` / after `dispose()` |
| `BitNetIsolateException` | Inference isolate died, or engine disposed while loading |

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
podspec. `bitnet_bridge.xcframework`
is a static archive merged with llama/ggml, resolved at runtime through
`DynamicLibrary.process()`; `prism_bridge.xcframework` is a self-contained
dynamic framework, resolved through `DynamicLibrary.open(path)`. Rebuild them
whenever the C ABI changes — **0.3.0 (ABI v2) needs a rebuild**; until then the
engine refuses the old frameworks with a `BitNetLibraryLoadException`. Requires
macOS with Xcode + cmake + ninja + libtool/lipo:

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
than bitnet.cpp — the two libraries never share symbols). The fork checkout
must be **recent enough to define `GGML_TYPE_Q2_0`** (the `prism` branch as of
late 2026 does; an older checkout builds fine but fails to load any Bonsai
GGUF with "tensor ... has offset ..., expected ..."). Check with
`grep GGML_TYPE_Q2_0 <fork>/ggml/include/ggml.h`.

```bash
git clone --branch prism https://github.com/PrismML-Eng/llama.cpp.git ../prism-llama.cpp
./tool/build_linux_prism.sh                                             # Linux desktop / host tests
ANDROID_NDK=$HOME/Android/Sdk/ndk/<ver> ./tool/build_android_prism.sh   # arm64-v8a + x86_64
./tool/build_ios_prism.sh                                               # prism_bridge.xcframework
```

Every script — bitnet and prism — must be re-run after changing the C ABI
(0.3.0 introduced ABI v2; the Dart side refuses older libraries with a clear
error).

All three scripts drive `native/prism/CMakeLists.txt`, which links the fork
static and version-scripts the output down to just the `bn_*` C ABI. The
Android script drops `libprism_bridge.so` into `jniLibs/<abi>/`; the iOS
script produces `prism_bridge.xcframework` in `ios/Frameworks/`. No web build
yet.

---

## Architecture

```
BitNetSession              — chat / extract / summarize / assist (prompt templates,
 │                           token-budget history, JSON grammars, confidence)
 └── BitNetEngine (interface; generations are queued, never interleaved)
      ├── NativeEngine     — Android / iOS / desktop
      │    ├── Isolate      background worker (UI never blocks)
      │    └── NativeLibrary  dart:ffi → bn_* C ABI v2 (ffigen bindings)
      │         ├── bitnet_bridge.cc ─┐  shared core: native/bn_core.inc
      │         └── prism_bridge.cc  ─┘  (sampling, grammar, KV reuse, abort)
      └── WebEngine        — Web: dart:js_interop → Web Worker → WASM

ModelCache                 — resumable download, SHA-256, backup-excluded storage
DeviceInspector            — RAM / thermal / power via JNIgen (Android) and
                             Objective-C interop (iOS/macOS); no platform channel
```

**Inference flow:**

1. `engine.load()` — RAM gate → download / verify → spawn isolate → `bn_init`
   → `bn_abi_version` check.
2. `generateTokens(prompt, options)` — `bn_set_sampling` (only when options
   change) → `bn_prompt` (re-uses the longest cached KV prefix) →
   `bn_next_token_ex` loop (token bytes + log-probability) → UTF-8 reassembly
   → stop-marker detection → stream.
3. `cancelGeneration()` sets a shared native flag that the C layer polls during
   prefill and before every token.

---

## Beta Limitations

- Prebuilt libraries ship for Android (`arm64-v8a`, `x86_64`) and iOS (device
  `arm64` + simulator). macOS and Windows require building the bridge from
  source; web is built with `tool/build_web.sh`.
- iOS has been verified in the **simulator only**, not yet on a physical device.
- The PrismML (`Ternary-Bonsai-*`, Q2\_0) backend is **native-only**, and the
  bundled Android/Linux PrismML libraries must be rebuilt from a recent
  `prism` branch of the fork to load Bonsai models (see
  [PrismML backend](#prismml-ternary-bonsai-backend)).
- Web is **desktop-browser only** and **single-threaded scalar** (~1–2 tok/s for
  the 2B model). The ~1.1 GB model lives in the WASM heap, so mobile browsers
  will run out of memory. Cancellation on web takes effect at the next token.
- macOS and Windows are untested.
- BitNet 2B-4T and Falcon3 are English-centric; use a Bonsai model for other
  languages.

---

## Contributing

Issues and PRs are welcome at
[github.com/IbrahimElmourchidi/bitnet_flutter_ai](https://github.com/IbrahimElmourchidi/bitnet_flutter_ai).

Please open an issue before sending a large PR so we can align on scope.

---

## License

MIT — see [LICENSE](LICENSE).

Published by [utanium.org](https://pub.dev/publishers/utanium.org/packages).
