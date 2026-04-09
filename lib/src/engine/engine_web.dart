import 'dart:async';
import 'dart:js_interop';

import '../cache/model_cache.dart';
import '../device/device_inspector.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model.dart';
import 'bitnet_engine.dart';

// ---------------------------------------------------------------------------
// JS interop — expects `window.BitNetWasm` to be set by the consumer's
// index.html <script> tag loading the Emscripten-generated glue file.
// ---------------------------------------------------------------------------

@JS('BitNetWasm.bn_init')
external JSAny? _jsInit(JSString modelPath, JSNumber nCtx, JSNumber nThreads);

@JS('BitNetWasm.bn_prompt')
external JSNumber _jsPrompt(JSAny ctx, JSString prompt, JSNumber addBos);

@JS('BitNetWasm.bn_next_token')
external JSObject _jsNextToken(JSAny ctx);

@JS('BitNetWasm.bn_free')
external void _jsFree(JSAny ctx);

@JS('BitNetWasm')
external JSAny? get _bitNetWasm;

// ---------------------------------------------------------------------------
// WebEngine
// ---------------------------------------------------------------------------

/// Web WASM implementation of [BitNetEngine].
///
/// Requires `window.BitNetWasm` to be defined before [load] is called.
/// The COI service worker (`coi_service_worker.js`) must be registered in the
/// consumer's `index.html` to enable `SharedArrayBuffer` for WASM threads.
class WebEngine implements BitNetEngine {
  final BitNetModel _model;

  WebEngine(this._model);

  JSAny? _ctx;
  bool _loaded = false;

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {
    if (_loaded) return;

    // 1. Check WASM runtime is available.
    if (_bitNetWasm == null) {
      throw const BitNetUnsupportedPlatformException(
        'BitNetWasm is not defined. '
        'Add the WASM glue <script> to your index.html before loading the engine.',
      );
    }

    // 2. Device check.
    final ok = await DeviceInspector.instance.meetsMinimumRam();
    if (!ok) {
      final ram = await DeviceInspector.instance.totalPhysicalRamBytes();
      throw BitNetUnsupportedDeviceException(ram);
    }

    // 3. Download / verify model (stored in browser cache via path_provider_web).
    final path = await ModelCache.ensureModel(
      _model,
      onProgress: onProgress ?? (_) {},
    );

    // 4. Initialise WASM context.
    final ctx = _jsInit(
      path.toJS,
      _model.contextLength.toJS,
      0.toJS, // nThreads: 0 → hardware_concurrency inside WASM
    );

    if (ctx == null) {
      throw const BitNetInitException(
        message: 'bn_init returned null from WASM module.',
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

    final rc = _jsPrompt(_ctx!, prompt.toJS, (addBos ? 1 : 0).toJS).toDartInt;
    if (rc != 0) {
      throw const BitNetInferenceException(message: 'bn_prompt failed in WASM.');
    }

    var count = 0;
    while (maxNewTokens == 0 || count < maxNewTokens) {
      final result = _jsNextToken(_ctx!);
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
        yield token;
        count++;
      }
    }
  }

  @override
  Future<void> cancelGeneration() async {
    // WASM runs on the main thread synchronously; cancellation only takes
    // effect between yield points (token boundaries) in the async* generator.
    // No explicit signal needed — the caller can cancel the StreamSubscription.
  }

  @override
  Future<void> dispose() async {
    if (_ctx != null) {
      _jsFree(_ctx!);
      _ctx = null;
    }
    _loaded = false;
  }
}

/// Factory function resolved by the conditional import in [bitnet_engine.dart].
BitNetEngine createEngine({required BitNetModel model}) => WebEngine(model);
