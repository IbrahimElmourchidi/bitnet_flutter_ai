import 'dart:async';
import 'dart:js_interop';

import '../device/device_inspector.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart';
import '../session/chat_templates.dart';
import 'bitnet_engine.dart';

// ---------------------------------------------------------------------------
// JS interop — `window.BitNetWasm` is a proxy (bitnet_glue.js) over a Web
// Worker that owns the WASM module, so model loading and generation run off
// the UI thread. Every call is therefore asynchronous (returns a Promise).
// index.html only needs: <script src="bitnet_glue.js"></script>
// ---------------------------------------------------------------------------

@JS('BitNetWasm.bn_init')
external JSPromise<JSAny?> _jsInit(
    JSString modelPath, JSNumber nCtx, JSNumber nThreads);

@JS('BitNetWasm.bn_prompt')
external JSPromise<JSNumber> _jsPrompt(JSAny ctx, JSString prompt, JSNumber addBos);

@JS('BitNetWasm.bn_next_token')
external JSPromise<JSObject> _jsNextToken(JSAny ctx);

@JS('BitNetWasm.bn_free')
external JSPromise<JSAny?> _jsFree(JSAny ctx);

/// Downloads the GGUF (Cache-API persisted) and streams it into the worker
/// module's in-memory filesystem at [fsPath]; the worker also frees the file
/// after `bn_init`. [onProgress] receives a fraction in [0, 1].
@JS('BitNetWasm.provisionModel')
external JSPromise<JSAny?> _jsProvisionModel(
    JSString url, JSString fsPath, JSNumber sizeBytes, JSFunction onProgress);

@JS('BitNetWasm')
external JSAny? get _bitNetWasm;

/// Resolves once the async Emscripten module finishes instantiating and
/// `window.BitNetWasm` is defined. May be `null` if the glue script never ran.
@JS('BitNetWasmReady')
external JSPromise<JSAny?>? get _bitNetWasmReady;

// ---------------------------------------------------------------------------
// WebEngine
// ---------------------------------------------------------------------------

/// Web WASM implementation of [BitNetEngine].
///
/// Requires `bitnet_glue.js` in the consumer's `index.html`, which spawns a Web
/// Worker (`bitnet_worker.js`) that owns the WASM module and defines the async
/// `window.BitNetWasm` proxy. Running in the worker keeps model loading and
/// token generation off the UI thread. The module is single-threaded, so no
/// `SharedArrayBuffer` / cross-origin isolation is required.
class WebEngine implements BitNetEngine {
  final BitNetModelInfo _model;

  WebEngine(this._model);

  JSAny? _ctx;
  bool _loaded = false;

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {
    if (_loaded) return;

    // Web ships only the bitnet.cpp WASM module. The PrismML (Q2_0) backend has
    // no WASM build yet — fail clearly rather than trying to load a model its
    // stack can't handle.
    if (_model.engineBackend != EngineBackend.bitnetCpp) {
      throw BitNetUnsupportedPlatformException(
        '${_model.displayName} uses the ${_model.engineBackend.name} backend, '
        'which is not yet supported on web (native platforms only).',
      );
    }

    // 1. Wait for the async Emscripten module, then check it is available.
    if (_bitNetWasm == null && _bitNetWasmReady != null) {
      await _bitNetWasmReady!.toDart;
    }
    if (_bitNetWasm == null) {
      throw const BitNetUnsupportedPlatformException(
        'BitNetWasm is not defined. Add the WASM glue <script> tags '
        '(bitnet_bridge.js then bitnet_glue.js) to your index.html.',
      );
    }

    // 2. Device check — gate on the selected model's RAM floor (see the native
    //    engine for rationale). navigator.deviceMemory is coarse on web.
    final ram = await DeviceInspector.instance.totalPhysicalRamBytes();
    if (ram < _model.minimumRamBytes) {
      throw BitNetUnsupportedDeviceException(
        ram,
        requiredRamBytes: _model.minimumRamBytes,
        modelName: _model.displayName,
      );
    }

    // 3. Download the GGUF (Cache-API persisted) and stream it into the
    //    module's in-memory filesystem. dart:io / path_provider don't exist on
    //    web, so the JS glue owns the download + FS write; we just await it.
    final fsPath = '/models/${_model.ggufFileName}';
    final progressShim = ((JSNumber fraction) {
      if (onProgress != null) onProgress(fraction.toDartDouble);
    }).toJS;
    try {
      await _jsProvisionModel(
        _model.ggufDownloadUrl.toJS,
        fsPath.toJS,
        _model.ggufSizeBytes.toJS,
        progressShim,
      ).toDart;
    } catch (e) {
      throw BitNetDownloadException('WASM model provisioning failed: $e');
    }

    // 4. Initialise the WASM context from the in-memory GGUF (runs in the
    //    worker; it frees the FS copy afterwards). Single-threaded (nThreads=1):
    //    the shipped module is built without pthreads.
    final ctx = await _jsInit(
      fsPath.toJS,
      _model.contextLength.toJS,
      1.toJS,
    ).toDart;

    if (ctx == null) {
      throw const BitNetInitException(
        message: 'bn_init returned null from WASM module (model load failed).',
      );
    }

    _ctx = ctx;
    _loaded = true;
  }

  @override
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  }) async* {
    if (!_loaded || _ctx == null) throw const BitNetNotLoadedException();

    final rc = (await _jsPrompt(_ctx!, prompt.toJS, (addBos ? 1 : 0).toJS).toDart)
        .toDartInt;
    if (rc != 0) {
      throw const BitNetInferenceException(message: 'bn_prompt failed in WASM.');
    }

    var count = 0;
    // Detects this model's end-of-turn markers even when split across several
    // token pieces. Markers are template-specific (see chat_templates.dart).
    final stopDetector =
        TurnStopDetector(chatTemplateFor(_model.chatTemplate).stopMarkers);
    while (maxNewTokens == 0 || count < maxNewTokens) {
      final result = await _jsNextToken(_ctx!).toDart;
      final dartResult = result.dartify() as Map<Object?, Object?>?;
      if (dartResult == null) {
        throw const BitNetInferenceException(
          message: 'bn_next_token returned unexpected null from WASM.',
        );
      }

      final done = dartResult['done'] as bool? ?? false;
      if (done) break;

      final token = dartResult['token'] as String?;
      if (token != null) {
        final emit = stopDetector.feed(token);
        if (emit.isNotEmpty) yield emit;
        if (stopDetector.stopped) {
          // A complete end-of-turn marker was assembled — stop without
          // yielding the marker text itself.
          break;
        }
        count++;
      }
    }

    // Release any text held back pending marker detection (stream ended via
    // EOS or token cap).
    if (!stopDetector.stopped) {
      final rest = stopDetector.flush();
      if (rest.isNotEmpty) yield rest;
    }
  }

  @override
  Future<void> cancelGeneration() async {
    // Generation runs one token per worker round-trip; cancellation takes
    // effect at the next token boundary when the caller stops pulling the
    // stream (the async* loop simply stops requesting more tokens). No
    // explicit interrupt is sent — the worker finishes the current token.
  }

  @override
  Future<void> dispose() async {
    if (_ctx != null) {
      await _jsFree(_ctx!).toDart;
      _ctx = null;
    }
    _loaded = false;
  }
}

/// Factory function resolved by the conditional import in [bitnet_engine.dart].
BitNetEngine createEngine({required BitNetModelInfo model}) => WebEngine(model);
