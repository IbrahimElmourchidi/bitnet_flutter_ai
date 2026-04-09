# Changelog

All notable changes to `bitnet_flutter_ai` will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0.0 versions (`0.x.x-beta.y`) may introduce breaking changes between beta releases.

---

## 0.1.0-beta.1 — 2026-04-09

> **Beta release.** The public API is functional but may change before the stable
> `1.0.0` release. SHA-256 pinning, pre-built native binaries, and the Web WASM
> artefact are not yet distributed via pub.dev — see the README for build instructions.

### Added

- **`BitNetEngine`** — unified `load / generate / cancelGeneration / dispose` interface
  backed by a dedicated `Isolate` on native and `dart:js_interop` on Web.
- **`BitNetModel.bitnet2B4T`** — immutable const descriptor for Microsoft BitNet b1.58
  2B-4T (I2\_S GGUF, 4096-token context, ~745 MiB on-disk).
- **`ModelCache`** — downloads the GGUF from HuggingFace with resumable HTTP (`Range`
  header), SHA-256 post-download verification, and atomic rename to final path.
  Uses `getApplicationSupportDirectory()` via `path_provider`.
- **`DeviceInspector`** — platform-agnostic RAM gate (≥ 3072 MB required).
  - Android: `ActivityManager.MemoryInfo.totalMem` via `MethodChannel`.
  - iOS: `ProcessInfo.processInfo.physicalMemory` via `MethodChannel`.
  - macOS / Linux / Windows: `system_info2` (`SysInfo.getTotalPhysicalMemory()`).
  - Web: `navigator.deviceMemory` (fails closed to 0 if unavailable).
- **C bridge** (`native/bitnet_bridge.h` + `native/bitnet_bridge.cc`) — thin C ABI
  wrapping the llama.cpp new C API: `llama_model_load_from_file`,
  `llama_new_context_with_model`, `llama_decode`, `llama_sampler_chain_init`,
  `llama_token_to_piece`, `llama_token_is_eog`.
- **FFI bindings** (`lib/src/ffi/`) — hand-authored ffigen-compatible bindings and
  `NativeLibrary` wrapper with `using()` / Arena pointer lifecycle management.
- **`web/coi_service_worker.js`** — COI shim that injects
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` headers, enabling `SharedArrayBuffer`
  for WASM threads in Chrome and Firefox.
- **Sealed exception hierarchy** — `BitNetException` base with typed subclasses for
  every failure mode (device, platform, library, init, inference, hash mismatch,
  download, not-loaded, isolate).
- **Unit tests** — 12 tests covering `BitNetModel` metadata, `DeviceInspector` RAM
  thresholds, exception messages, and `ModelCache` download/resume/clear flows.

### Known Limitations (beta)

- Pre-built native binaries (`libbitnet_bridge.so`, `.dylib`, `.dll`) are **not** yet
  bundled. Consumers must compile `native/bitnet_bridge.cc` against llama.cpp
  themselves. See [Building Native Libraries](README.md#building-native-libraries).
- The WASM artefact (`BitNetWasm` JS object) is not yet published. Web support requires
  a custom Emscripten build.
- `ggufSha256` is a placeholder sentinel — post-download hash verification is skipped
  until the hash is pinned after the first CI download.
- Windows and Linux desktop are untested in this release. Android and iOS are the
  primary targets.
- `cancelGeneration()` on Web cancels at the next `yield` boundary only (no interrupt
  signal to the synchronous WASM loop).
