# bitnet_flutter_ai example

A tabbed demo of every public API of `bitnet_flutter_ai`:

| Tab | Shows |
|---|---|
| Device | Device profile, live `DeviceInspector.health()` (RAM, thermal, low-power), model catalog with download status, load / switch model |
| Chat | `BitNetSession.chat` streaming with history and Stop |
| Extract | `BitNetSession.extract` — turns "I spent 250 EGP at the supermarket" into JSON with per-field confidence |
| Summarise | `BitNetSession.summarize` with `SummaryStyle` |
| Ask (RAG) | `BitNetRag` over a bundled Markdown FAQ |
| Logs | `BitNetLog` output |

## Run

Requires Flutter 3.50+ (Dart 3.14+).

```bash
flutter run                # Android device / emulator, iOS, Linux
flutter run -d chrome      # web — build the WASM module first (see the package README)
```

The first **Load model** downloads the selected GGUF (about 1.2 GB for the
default BitNet 2B-4T) and verifies its SHA-256.

## Tests

```bash
# Device interop + (if the model is cached) real inference:
flutter test integration_test/on_device_test.dart -d <device>
# UI flow: load the model, then run an extraction:
flutter test integration_test/app_flow_test.dart -d <device>
```

`on_device_test.dart` explains how to pre-seed the model with `adb` so the
tests don't download it.
