import 'dart:async';
import 'dart:js_interop';

import '../device/device_inspector.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart';
import '../session/chat_templates.dart';
import 'bitnet_engine.dart';
import 'generation_options.dart';

// ---------------------------------------------------------------------------
// JS interop — `window.BitNetWasm` is a proxy (bitnet_glue.js) over a Web
// Worker that owns the WASM module, so model loading and generation run off
// the UI thread. Every call is therefore asynchronous (returns a Promise).
// index.html only needs: <script src="bitnet_glue.js"></script>
// ---------------------------------------------------------------------------

@JS('BitNetWasm.bn_init')
external JSPromise<JSAny?> _jsInit(
    JSString modelPath, JSNumber nCtx, JSNumber nThreads);

/// `{ok, error}` from bn_set_sampling.
extension type _OkResult._(JSObject _) implements JSObject {
  external bool get ok;
  external String? get error;
}

/// `{rc, error}` from bn_prompt.
extension type _PromptResult._(JSObject _) implements JSObject {
  external int get rc;
  external String? get error;
}

/// `{done, bytes, logprob, error}` from bn_next_token.
extension type _NextResult._(JSObject _) implements JSObject {
  external bool get done;
  external JSUint8Array? get bytes;
  external double get logprob;
  external String? get error;
}

@JS('BitNetWasm.bn_set_sampling')
external JSPromise<_OkResult> _jsSetSampling(
    JSAny ctx,
    JSNumber temperature,
    JSNumber topK,
    JSNumber topP,
    JSNumber repeatPenalty,
    JSNumber repeatLastN,
    JSNumber seed,
    JSString grammar);

@JS('BitNetWasm.bn_prompt')
external JSPromise<_PromptResult> _jsPrompt(JSAny ctx, JSString prompt, JSNumber addBos);

@JS('BitNetWasm.bn_next_token')
external JSPromise<_NextResult> _jsNextToken(JSAny ctx);

@JS('BitNetWasm.bn_token_count')
external JSPromise<JSNumber> _jsTokenCount(JSAny ctx, JSString text, JSNumber addBos);

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
/// `SharedArrayBuffer` / cross-origin isolation is required — which also means
/// there is no shared memory to cancel a running prefill through: on web,
/// cancellation takes effect at the next token boundary.
class WebEngine implements BitNetEngine {
  @override
  final BitNetModelInfo model;

  WebEngine(this.model);

  JSAny? _ctx;
  bool _loaded = false;
  Future<void>? _loading;
  GenerationOptions? _sampler;

  /// Generations run one at a time, in call order (see [BitNetEngine]).
  Future<void> _tail = Future<void>.value();

  /// The running generation's cancel switch, checked between tokens.
  _WebRun? _active;

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) {
    if (_loaded) return Future<void>.value();
    return _loading ??= _load(onProgress).whenComplete(() => _loading = null);
  }

  Future<void> _load(void Function(double progress)? onProgress) async {

    // Web ships only the bitnet.cpp WASM module. The PrismML (Q2_0) backend has
    // no WASM build yet — fail clearly rather than trying to load a model its
    // stack can't handle.
    if (model.engineBackend != EngineBackend.bitnetCpp) {
      throw BitNetUnsupportedPlatformException(
        '${model.displayName} uses the ${model.engineBackend.name} backend, '
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
    if (ram < model.minimumRamBytes) {
      throw BitNetUnsupportedDeviceException(
        ram,
        requiredRamBytes: model.minimumRamBytes,
        modelName: model.displayName,
      );
    }

    // 3. Download the GGUF (Cache-API persisted) and stream it into the
    //    module's in-memory filesystem. dart:io / path_provider don't exist on
    //    web, so the JS glue owns the download + FS write; we just await it.
    final fsPath = '/models/${model.ggufFileName}';
    final progressShim = ((JSNumber fraction) {
      if (onProgress != null) onProgress(fraction.toDartDouble);
    }).toJS;
    try {
      await _jsProvisionModel(
        model.ggufDownloadUrl.toJS,
        fsPath.toJS,
        model.ggufSizeBytes.toJS,
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
      model.contextLength.toJS,
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
  Stream<TokenEvent> generateTokens(
    String prompt, {
    GenerationOptions options = const GenerationOptions(),
  }) {
    if (!_loaded || _ctx == null) throw const BitNetNotLoadedException();
    final run = _WebRun();
    final controller = StreamController<TokenEvent>(
      onCancel: () => run.cancelled = true,
    );
    _tail = _tail.then((_) async {
      if (run.cancelled) {
        await controller.close();
        return;
      }
      _active = run;
      try {
        await _generate(prompt, options, run, controller);
      } on BitNetException catch (e) {
        if (!run.cancelled) controller.addError(e);
      } catch (e) {
        if (!run.cancelled) {
          controller.addError(BitNetInferenceException(message: e.toString()));
        }
      } finally {
        if (identical(_active, run)) _active = null;
        await controller.close();
      }
    });
    return controller.stream;
  }

  Future<void> _generate(String prompt, GenerationOptions o, _WebRun run,
      StreamController<TokenEvent> out) async {
    final ctx = _ctx;
    if (!_loaded || ctx == null) throw const BitNetNotLoadedException();

    if (_sampler == null || !_sampler!.samplerEquals(o)) {
      _sampler = null;
      final r = await _jsSetSampling(
        ctx,
        o.temperature.toJS,
        o.topK.toJS,
        o.topP.toJS,
        o.repeatPenalty.toJS,
        o.repeatLastN.toJS,
        (o.seed & 0xFFFFFFFF).toJS,
        (o.grammar ?? '').toJS,
      ).toDart;
      if (!r.ok) {
        throw BitNetInferenceException(
            message: 'bn_set_sampling failed', nativeError: r.error);
      }
      _sampler = o;
    }

    final p = await _jsPrompt(ctx, prompt.toJS, (o.addBos ? 1 : 0).toJS).toDart;
    if (p.rc != 0) {
      final err = p.error;
      if (err != null && err.startsWith('prompt_too_long:')) {
        throw BitNetPromptTooLongException(
            err.substring('prompt_too_long:'.length).trim());
      }
      throw BitNetInferenceException(message: 'bn_prompt failed', nativeError: err);
    }

    final utf8 = Utf8StreamDecoder();
    final stop = TurnStopDetector([
      ...chatTemplateFor(model.chatTemplate).stopMarkers,
      ...o.stop,
    ]);
    var pendingLp = <double>[];
    void emit(String text) {
      if (text.isEmpty || run.cancelled) return;
      out.add(TokenEvent(text, pendingLp));
      pendingLp = <double>[];
    }

    var count = 0;
    while (o.maxNewTokens == 0 || count < o.maxNewTokens) {
      if (run.cancelled || _ctx == null) break;
      final r = await _jsNextToken(ctx).toDart;
      final bytes = r.bytes;
      if (bytes == null) {
        if (r.done) break;
        throw BitNetInferenceException(
            message: 'bn_next_token failed', nativeError: r.error);
      }
      count++;
      pendingLp.add(r.logprob);
      final text = utf8.add(bytes.toDart);
      if (text.isEmpty) continue;
      emit(stop.feed(text));
      if (stop.stopped) break;
    }
    if (!stop.stopped) {
      final tail = utf8.flush();
      if (tail.isNotEmpty) emit(stop.feed(tail));
      if (!stop.stopped) emit(stop.flush());
    }
  }

  @override
  Future<int> countTokens(String text, {bool addBos = false}) async {
    final ctx = _ctx;
    if (!_loaded || ctx == null) throw const BitNetNotLoadedException();
    final n = (await _jsTokenCount(ctx, text.toJS, (addBos ? 1 : 0).toJS).toDart)
        .toDartInt;
    if (n < 0) {
      throw const BitNetInferenceException(message: 'bn_token_count failed');
    }
    return n;
  }

  @override
  Future<void> cancelGeneration() async {
    _active?.cancelled = true;
  }

  @override
  Future<void> dispose() async {
    _loaded = false;
    _active?.cancelled = true;
    // Let the running generation reach its next token boundary before the
    // context it is using is freed.
    await _tail;
    final ctx = _ctx;
    _ctx = null;
    _sampler = null;
    if (ctx != null) await _jsFree(ctx).toDart;
  }
}

class _WebRun {
  bool cancelled = false;
}

/// Factory function resolved by the conditional import in [bitnet_engine.dart].
BitNetEngine createEngine({required BitNetModelInfo model}) => WebEngine(model);
