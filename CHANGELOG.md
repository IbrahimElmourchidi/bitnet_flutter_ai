# Changelog

All notable changes to `bitnet_flutter_ai` will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0.0 versions (`0.x.x-beta.y`) may introduce breaking changes between beta releases.

---

## Unreleased

## 0.3.0-beta.1 — 2026-11-13

> **Known limitations of this beta** (planned for 0.3.0 stable):
> - The prebuilt **iOS xcframeworks are still ABI v1**, so 0.3.0 refuses to load
>   them with a `BitNetLibraryLoadException`. Rebuild them on a Mac with
>   `tool/build_ios.sh` + `tool/build_ios_prism.sh`.
> - The prebuilt **Android/Linux `libprism_bridge.so`** is still ABI v1 and comes
>   from a PrismML fork without `GGML_TYPE_Q2_0`, so Ternary Bonsai models don't
>   load there. The default BitNet backend is rebuilt and verified on Android,
>   Linux and Web.

Requires **Dart 3.14 / Flutter 3.50** and the **bn C ABI v2** native libraries
(rebuild with `tool/build_*.sh`; a stale library is refused with a clear
`BitNetLibraryLoadException` instead of crashing).

### Breaking changes (migration)

- `BitNetEngine.generate(prompt, maxNewTokens:, addBos:)` →
  `generate(prompt, options: GenerationOptions(maxNewTokens: …))`. The new
  `generateTokens` streams `TokenEvent`s (text + per-token log-probabilities).
- `BitNetSession(engine:, model:)` → the `model` parameter is gone; the
  session reads `engine.model`, so the chat template always matches the loaded
  weights. `systemPrompt:` can be passed to the constructor.
- `BitNetSession.dispose()` removed — sessions do not own the engine (several
  sessions may share one). Call `engine.dispose()`.
- `defaultMaxNewTokens` → `options` (`GenerationOptions`).
  `summarize(style: 'bullets')` → `summarize(style: SummaryStyle.bullets)`.
  `BitNetRag.ask(maxNewTokens:)` → `ask(options:)`.
- `BitNetBusyException` removed: overlapping generations are now queued in
  call order instead of rejected.
- `TurnStopDetector`, `Utf8StreamDecoder` and `kBitNetTurnStopMarkers` are no
  longer exported (internal helpers).
- Android plugin namespace `com.example.bitnet_flutter_ai` →
  `org.utanium.bitnet_flutter_ai`.

### Added

- **Structured extraction:** `BitNetSession.extract(text, shape: JsonShape(…))`
  with `JsonField.number/integer/string/boolean/enumOf/date`, worked
  `ExtractExample`s and `today:` for relative dates. Output is constrained by a
  GBNF grammar (always parses), decoded greedily, and returned as an
  `ExtractResult` with an overall and per-field `confidence`.
- `GenerationOptions` (temperature, top-k/p, repetition penalty, seed, GBNF
  `grammar`, extra `stop` strings) with `GenerationOptions.precise()` for
  deterministic structured output.
- `BitNetEngine.countTokens`, `BitNetEngine.model`.
- `BitNetSession.chat` trims history by **token budget** (context minus the
  reply reserve), not just turn count, and raises
  `BitNetPromptTooLongException` when a single message cannot fit.
- **KV-cache prefix reuse:** a new prompt only prefills what differs from the
  previous one, so a fixed system prompt / few-shot block is processed once
  (measured 3–5× faster prefill on repeated extractions).
- **Instant cancellation:** `cancelGeneration()` writes a shared native flag
  the C layer polls during prefill and decode — no longer waits for the
  current multi-second prefill to finish.
- `DeviceInspector.health()` → `DeviceHealth` (total/available RAM, low-memory
  flag, `ThermalState`, Low Power / battery saver, `shouldThrottle`).
- **Dart 3.14 direct native interop:** Android device queries use JNIgen
  bindings (`jnigen.yaml`), iOS/macOS use Foundation through
  `package:objective_c`. The MethodChannel is gone, so device queries also work
  from background isolates.
- C ABI v2: `bn_abi_version`, `bn_set_sampling`, `bn_next_token_ex`,
  `bn_token_count`, `bn_set_abort_flag`, `bn_last_prompt_reused`. Both
  backends share one implementation (`native/bn_core.inc`).
- `native/test/abi_v2_smoke.c` and the opt-in `test/real_model_test.dart`
  (`BITNET_MODEL=…`) for end-to-end checks against real weights.

### Fixed

- **Generations could interleave on one native context**: cancelling a stream
  and immediately starting another let both decode loops pull tokens from the
  same context, corrupting the second reply. Generations are now serialised on
  both sides of the isolate boundary.
- Concurrent `load()` calls spawned several isolates; `dispose()` during
  `load()` was undone when the load finished (leaking the model).
- A dying inference isolate left `load()` / streams hanging forever; it now
  fails them with `BitNetIsolateException` and unloads the engine.
- The sampler's repetition-penalty history leaked from one prompt into the
  next (no `llama_sampler_reset`).
- PrismML backend: prompts over 2048 tokens failed (`n_batch` not raised).
- Model download: a connection that closed early was accepted as complete
  (pinned hash → full re-download from zero; unpinned → truncated model
  loaded); a complete `.part` got HTTP 416 and could never finish; two
  concurrent downloads appended to the same `.part`;
  `BitNetHashMismatchException.actualHash` was the literal `'<computed>'`.
- Models are stored outside device backups (Android `noBackupFilesDir`, with
  a one-time migration; iOS `NSURLIsExcludedFromBackupKey`) — a >1 GB file in
  the regular files dir exceeded Android Auto Backup's quota and made the
  host app's whole backup fail.
- Web: token pieces were decoded to strings per token, turning multi-byte
  characters split across tokens (Arabic, emoji) into `�`. Raw bytes are now
  reassembled as on native.
- The bare `Human:` stop marker truncated replies that merely mentioned
  "Human:" mid-sentence.
- Unconditional `print` in `ModelCache` and per-prompt `stderr` logging in the
  bridges (now behind `BitNetLog` / `BN_DEBUG_LOG`).

### Example app

- New **Extract** tab: expense parsing with `JsonShape`, few-shot examples,
  per-field confidence bars and a "save / please confirm" verdict.
- **Device health** card on the Device tab (`DeviceInspector.health()`).
- Migrated to the Flutter 3.50 Android toolchain: Gradle 9.5.0, AGP 9.3.1,
  built-in Kotlin, `compileSdk`/`targetSdk` 36; `package:material_ui`.
- `integration_test/`: on-device checks (device health from the main and a
  background isolate, backup-excluded model path, real-model extraction) and
  an end-to-end UI flow (load → extract).

### Tooling / requirements

- Minimum **Dart 3.14 / Flutter 3.50**. Android host apps need Gradle 9.3.1+
  (Flutter 3.50 requirement); the plugin's Android module is now Java-only and
  builds under AGP 8 and AGP 9. `ffigen` 23, `flutter_lints` 6.
- New dependencies: `jni`, `jni_flutter`, `objective_c` (dev: `jnigen`).
- `.pubignore` keeps local docs out of the published package.

### Removed

- `tool/dart_smoke.dart` (could not run under plain `dart`; superseded by
  `test/real_model_test.dart`).

## 0.2.0-beta.1 — 2026-08-11

### Added

- **iOS support (device + simulator), both backends.** Ships prebuilt
  `bitnet_bridge.xcframework` (static archive merged with llama/ggml) and
  `prism_bridge.xcframework` (self-contained dynamic framework) in
  `ios/Frameworks/`, covering `arm64` device and `arm64`/`x86_64` simulator
  slices. The two frameworks export the same `bn_*` names over incompatible
  ggml copies, so they resolve differently at runtime: bitnet through
  `DynamicLibrary.process()`, prism through `.open(path)`. The plugin moved to
  the SwiftPM layout (`ios/bitnet_flutter_ai/Sources/`) so CocoaPods and SPM
  build the same sources, with `Package.swift` added. `bitnet_force_link.m`
  keeps the vendored static archive's `bn_*` symbols — reached only via FFI
  `dlsym`, so nothing references them at link time — from being dead-stripped,
  without resorting to `-force_load` (which breaks clean Xcode builds because
  the linker path is validated before the Copy-XCFrameworks phase stages it).
  Verified end-to-end in the simulator with both the bitnet (I2\_S) and prism
  (Q2\_0) backends.
- **`ModelCache` model-picker helpers.** Added `isCached`, `hasPartialDownload`,
  and `cachedStatus` so a model-picker UI can show download state without
  triggering a load. A `.verified` stamp is written after a successful SHA-256
  check so later loads skip re-hashing a multi-GB file, hashing now runs on a
  background isolate instead of the UI thread, and `onProgress` is throttled
  to 0.5% steps.
- **Expanded model catalog (10 models).** Added six TII models that ship
  pre-built I2\_S GGUFs and run on the existing bitnet.cpp native stack with no
  rebuild — `Falcon3-{1B,3B,7B,10B}-Instruct-1.58bit` and
  `Falcon-E-{1B,3B}-Instruct` — plus three PrismML `Ternary-Bonsai-{1.7B,4B,8B}`
  models on a new second backend (below). All are SHA-256 pinned in
  `BitNetCatalog.all`; `compatibleWith(ram)` now surfaces a real size ladder.
  `contextLength` is capped at 4096 per entry (the native layer allocates the
  full KV cache up-front, and history is trimmed separately).
- **Per-model chat templates.** New `lib/src/session/chat_templates.dart` maps a
  model's `ChatTemplateFamily` (`bitnetLabeled` / `falconTagged` / `chatml` /
  `chatmlQwen3`) to its prompt formatter and end-of-turn stop markers. The wrong
  template produces degenerate output (the "runaway `!`" failure), so
  `BitNetSession` and both engines now dispatch per model instead of hardcoding
  BitNet's format. Verified: Falcon3-1B coherent via `falconTagged`.
- **Second native backend: PrismML Q2_0 (Ternary Bonsai).** PrismML's models use
  a `qwen3`/`Q2_0` ggml format incompatible with bitnet.cpp, so they route to a
  new self-contained `libprism_bridge.so` (`native/prism_bridge.cc` over the
  PrismML llama.cpp fork, built by `tool/build_{linux,android}_prism.sh`). It
  exposes the same `bn_*` C ABI as the bitnet bridge (zero Dart FFI changes) and
  a version script exports *only* `bn_*` so its modern ggml symbols can't clash
  with the bitnet stack loaded in the same process. `BitNetModelInfo.engineBackend`
  selects the library at load time. Verified: Bonsai-1.7B coherent via
  `chatmlQwen3`. Not available on web yet (no WASM build of the fork).
- **Working Web (WASM) inference.** The web engine is now backed by a real
  WebAssembly build of bitnet.cpp (ggml + llama.cpp) + the embind bridge
  (`native/bitnet_bridge_wasm.cc`), produced by `tool/build_web.sh`. The module
  runs inside a **Web Worker** (`web/bitnet_worker.js`) — the analogue of the
  native inference isolate — so model loading and token generation never block
  the UI thread; `web/bitnet_glue.js` is a main-thread async proxy that exposes
  `window.BitNetWasm`. The worker downloads the GGUF (`provisionModel`,
  Cache-API persisted, streamed into the in-memory FS) and frees it after
  `bn_init`. Verified end-to-end against the real 1.1 GB I2\_S model (coherent
  output, clean EOS stop). Single-threaded scalar (~1–2 tok/s for the 2B model);
  threading and a `wasm_simd128` kernel are documented speed follow-ups. Desktop
  browsers only — the 1.1 GB model needs multi-GB per-tab memory.
- **Quiet WASM logging.** The bridge installs a `llama_log_set` filter on web
  that drops everything below `ERROR`. llama.cpp's INFO/WARN chatter (the full
  GGUF metadata dump, a per-special-token "not marked as EOG" line, …) otherwise
  crosses the WASM→JS boundary as thousands of stack-traced `console.error`
  calls during load, which froze the tab/devtools on its own.
- **`DeviceProfile` compiles on web.** Platform detection moved behind a
  conditional import (`platform_info_impl.dart` / `platform_info_web.dart`) so
  the file no longer imports `dart:ffi` / `dart:io` on web (which failed to
  compile). Web reads `navigator.hardwareConcurrency` for the core count.
- **Mobile-first SDK surface.** Added the public types
  `BitNetCatalog`, `BitNetModelInfo`, `BitNetCapability`,
  `BitNetQuantization`, `DeviceProfile`, `BitNetSession`, `ChatRole`,
  `ChatTurn`, `BitNetRag`, `RagAnswer`, `KnowledgeSource`,
  `KnowledgeChunk`, `RetrievedChunk`, `AssetKnowledgeSource`,
  `InMemoryKnowledgeSource`, `Bm25Retriever`, and `TextChunker`.
- **Catalog & compatibility.** `BitNetCatalog.all`,
  `BitNetCatalog.byId`, `BitNetCatalog.withCapability`, and
  `BitNetCatalog.compatibleWith(ramBytes)` for picking a model that fits
  the device.
- **Device profile & speed estimate.** `DeviceProfile.current()` returns
  OS / ABI / cores / RAM, and
  `DeviceProfile.estimateTokensPerSecond(model)` produces a heuristic
  tok/s anchored at 12.5 tok/s on an 8-core x86_64 reference.
- **Chat session.** `BitNetSession` provides streaming `chat()`,
  `setSystemPrompt()`, `clearHistory()`, plus convenience helpers
  `summarize(text, {style})` (paragraph/bullets/tldr) and
  `assist(question, {context})`.
- **On-device RAG.** `BitNetRag` indexes
  `AssetKnowledgeSource(assetPaths: [...])` (or
  `InMemoryKnowledgeSource`) with `Bm25Retriever` + `TextChunker` and
  exposes `ask()` returning `RagAnswer{answer, citations}`.
- **Android prebuilts.** `android/src/main/jniLibs/{arm64-v8a,x86_64}/`
  ships `libbitnet_bridge.so`, `libllama.so`, `libggml.so`. Reproducible
  via `tool/build_android.sh` (NDK 28).
- **Example app.** `example/lib/main.dart` now demos every public API
  across four tabs: Device, Chat, Summarise, Ask (RAG) — backed by
  `example/assets/knowledge/bitnet_faq.md`.
- **End-to-end Linux verification.** The full pipeline (bitnet.cpp →
  `libllama.so` → `libbitnet_bridge.so` → Dart FFI → token stream) is now
  validated on x86_64 Linux. A Ryzen 7 5700U produces ~12 tok/s for the
  BitNet b1.58 2B-4T I2_S model with 749 ms cold model load.
- **`tool/dart_smoke.dart`** — Dart-only CLI integration test that drives
  `NativeLibrary` directly against a local GGUF, useful for iterating on
  the FFI bridge without going through Flutter.
- **`native/smoke_test.c`** — standalone C harness for verifying the
  bridge before Dart is in the loop.
- **`@visibleForTesting BitNetModel.forTest`** — lets unit tests build
  models with arbitrary (or computed) SHA-256 fixtures now that the
  canonical hash is pinned.

### Changed

- **C bridge updated for bitnet.cpp's pinned llama.cpp.** Renamed
  `llama_model_load_from_file` → `llama_load_model_from_file`,
  `llama_model_free` → `llama_free_model`,
  `llama_model_n_ctx_train` → `llama_n_ctx_train`. Switched
  `llama_batch_get_one` to its 4-argument form and adapted
  `llama_model_meta_val_str` to the buffer-output variant (the bridge now
  caches the result in a per-context scratch buffer).
- **`BitNetModel.bitnet2B4T.ggufSha256` is now pinned** to
  `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`,
  enabling real download integrity checks. `ggufSizeBytes` updated to the
  exact 1,187,801,280-byte upload.
- **`native/CMakeLists.txt`**: the Linux/macOS shared library now keeps
  the canonical `lib` prefix (`libbitnet_bridge.so/.dylib`) so it matches
  what `DynamicLibrary.open` expects. The `PREFIX ""` strip is now
  Windows-only.
- **pubspec no longer declares the web/ JS files as Flutter assets.** A
  package asset resolves under `assets/packages/...`, which can never claim
  the service-worker scope `index.html` needs, and it would otherwise be
  bundled into every Android/iOS build for nothing. See "Web Setup" in the
  README for the copy-into-your-own-`web/`-directory workflow.

### Fixed

- **Degenerate logits on arm64 (NaN/garbage output, iOS device + Android
  arm64-v8a).** The classic tree's scalar `ggml_vec_dot_i2_i8_s` block layout
  was wrong, and its buggy `ggml-cpu-i2s.c` GEMM/GEMV fast path was reachable
  on arm64, which has no AVX2. Patch `0002-i2s-arm-scalar-and-fastpath-gate`
  fixes the scalar layout and gates that fast path to AVX2/AVX512 only; the
  x86_64 ABI already used the verified AVX2 kernels either way. Applied by
  `tool/build_android.sh`.
- **Prefill SIGBUS/SIGSEGV past ~130 tokens (classic-tree I2\_S gemm).** The
  prefill path sized a stack VLA as `rows-per-thread * n_ubatch` floats, but
  ggml worker threads only get the pthread default stack (1 MB on Android,
  512 KB on Darwin) — a prompt long enough to fill the micro-batch overran it
  and crashed inside `ggml_vec_dot_i2_i8_s`. Patch
  `0003-i2s-gemm-no-stack-vla` accumulates straight into `dst` instead, so
  prefill stack use no longer scales with the prompt.
- **Runaway `!` output on Android (and any non-AVX2 target).** The BitNet
  I2\_S matmul kernels in bitnet.cpp shipped only AVX2 and NEON
  implementations with no fallback, so on the Android x86_64 emulator (NDK
  baseline has no AVX2) they compiled to empty functions: every matmul wrote
  nothing, logits came out flat, and greedy sampling emitted an endless
  stream of token 0 (`"!"`). Added a portable scalar fallback (validated
  bit-exact against the AVX2 kernel) plus a canonical-layout NEON path,
  distributed as `native/patches/0001-i2s-portable-kernels.patch` and applied
  by `tool/build_android.sh`. Also fixes an arm64 `PARALLEL_SIZE` mis-dispatch
  that corrupted 4-row prefill tail blocks, and a wrong-layout `quantize_i2_s`
  NEON path that produced corrupt GGUFs when quantizing on ARM.
- **Multi-piece end-of-turn markers not detected.** `<|eot_id|>` /
  `<|end_of_text|>` arrive as literal text spread over several BPE pieces; the
  old per-piece equality check never matched, so the marker leaked into
  replies and generation ran on into hallucinated turns. Added
  `TurnStopDetector` (streaming, rolling-tail match) and wired it into both
  the native and Web engines.
- **Degenerate-logits guard.** `bn_prompt` now detects flat/non-finite logits
  after prefill and fails with a clear error instead of streaming garbage —
  surfaces a broken-kernel build immediately rather than as `"!"` spam.
- **Prefill stack overflow on long prompts.** `bn_init` caps `n_ubatch` at 128
  to bound the stack VLA in the I2\_S gemm path, which could otherwise exceed
  a mobile thread's ~1 MB stack for wide FFN tensors.
- **Stop button / `cancelGeneration()` had no effect.** The native inference
  isolate ran generation as a synchronous loop over the blocking FFI
  `nextToken()`, which starved its own message port — the `_CmdCancel` sent by
  `BitNetSession.stop()` was queued but never delivered until generation had
  already finished. The worker now yields to the event loop between tokens, so
  a cancel is honoured at the next token boundary.
- **RAM gate was model-agnostic.** Both engines gated `load()` on a hardcoded
  3072 MB constant (`DeviceInspector.meetsMinimumRam`) instead of the selected
  model's `minimumRamBytes` — wrong in both directions now that models range
  0.6–4 GB on disk: a small model was needlessly blocked on a <3 GB device, and
  a large one passed the gate then OOM'd inside `bn_init`. The gate now checks
  the specific model, and `BitNetUnsupportedDeviceException` reports the actual
  required RAM and model name.
- **i2\_s gemm tail-block out-of-bounds (crash on WASM, silent corruption
  elsewhere).** Under `ACT_PARALLEL`, `ggml_gemm_i2_i8_s` calls the kernel with
  `nrc = cur_r` (activation rows) meaning "one weight row · nrc columns" —
  `_Nx1` semantics. The dispatcher routed any `nrc` not a multiple of
  `PARALLEL_SIZE` (e.g. a 23-token prefill → tail of 3) to `_1x1`, which treats
  `nrc` as *weight rows* and reads the weight pointer out of bounds. On
  bounds-checked WebAssembly this traps (`memory access out of bounds`);
  on native it reads adjacent heap and corrupts the tail rows' logits. The
  scalar and NEON-dotprod `_Nx1` handle any `nrc`, so the tail now routes to
  `_Nx1`; the AVX2 path (which blocks by `PARALLEL_SIZE`) is left byte-identical.
  Folded into `native/patches/0001-i2s-portable-kernels.patch`.

### Documentation

- [BUILD_GUIDE.md](BUILD_GUIDE.md) rewritten with the verified Linux
  recipe (gcc-only requirement, kernel codegen step, staging symlinks,
  and bridge smoke tests).

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
