# bitnet\_flutter\_ai

> **Beta — `0.1.0-beta.1`.**  
> The public API is stable in shape but may change before `1.0.0`. Native binary distribution and the Web WASM artefact are not yet published to pub.dev — see [Building Native Libraries](#building-native-libraries) below.

Run Microsoft's [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) large language model **fully on-device** — no server, no API key.

| Platform | Status |
|---|---|
| Android (arm64-v8a) | Planned — native build required |
| iOS (arm64) | Planned — native build required |
| macOS (arm64 / x86\_64) | Planned — native build required |
| Linux (x86\_64) | Planned — native build required |
| Windows (x86\_64) | Planned — native build required |
| Web (Chrome / Firefox) | Planned — WASM build required |

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
  bitnet_flutter_ai: ^0.1.0-beta.1
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

### `BitNetModel`

```dart
static const BitNetModel bitnet2B4T
```

The only supported model in `0.1.0-beta.1`. Immutable const with all metadata:

| Field | Value |
|---|---|
| `id` | `microsoft/bitnet-b1.58-2B-4T` |
| `ggufFileName` | `ggml-model-i2_s.gguf` |
| `ggufDownloadUrl` | HuggingFace direct link |
| `minimumRamBytes` | 3072 MB |
| `contextLength` | 4096 tokens |

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

### 2. Load the WASM glue script

```html
<!-- Emscripten-generated glue file — exposes window.BitNetWasm -->
<script src="bitnet_bridge.js"></script>
```

> The `bitnet_bridge.js` + `bitnet_bridge.wasm` artefacts are not yet published.
> See [Building Native Libraries](#building-native-libraries) for WASM build instructions.

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

Use Xcode to build a static library (`libbitnet_bridge.a`) and add it as a
`FRAMEWORK_SEARCH_PATHS` entry, or use the CocoaPods vendored\_libraries approach.

### Web (WASM)

```bash
# Requires Emscripten (emsdk).
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
emmake make
# Produces: bitnet_bridge.js + bitnet_bridge.wasm
```

Copy both files into your Flutter app's `web/` folder and reference
`bitnet_bridge.js` from `index.html` (see [Web Setup](#web-setup)).

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

- Pre-built native libraries are not distributed. You must build from source.
- SHA-256 hash for the GGUF file is not yet pinned (`PLACEHOLDER_PIN_AFTER_FIRST_DOWNLOAD`). Hash verification is skipped with a warning until it is pinned in a future release.
- Web WASM artefact is not published. Web support requires a custom Emscripten build.
- Windows and Linux desktop are untested in this beta.
- `cancelGeneration()` on Web cancels only at `yield` boundaries — there is no signal to interrupt a synchronous WASM loop mid-token.
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
