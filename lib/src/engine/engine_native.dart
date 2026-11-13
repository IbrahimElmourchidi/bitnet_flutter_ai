import 'dart:async';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart' show visibleForTesting;

import '../cache/model_cache.dart';
import '../device/device_inspector.dart';
import '../diagnostics/bitnet_log.dart';
import '../exceptions/bitnet_exception.dart';
import '../ffi/native_library.dart';
import '../model/bitnet_model_info.dart';
import '../session/chat_templates.dart';
import 'bitnet_engine.dart';
import 'generation_options.dart';
import 'inference_backend.dart';

// ---------------------------------------------------------------------------
// Isolate message protocol. Every message is a plain final class: isolates
// spawned with Isolate.spawn share a group, so these (and BitNetException
// instances) cross the port as-is.
// ---------------------------------------------------------------------------

final class _CmdInit {
  final String modelPath;
  final int nCtx;
  final int nThreads;
  final EngineBackend backend;

  /// Address of the shared int32 cancellation flag (see [NativeEngine]).
  final int abortFlagAddress;
  final SendPort replyPort;
  _CmdInit(this.modelPath, this.nCtx, this.nThreads, this.backend,
      this.abortFlagAddress, this.replyPort);
}

final class _CmdGenerate {
  final int requestId;
  final String prompt;
  final GenerationOptions options;

  /// End-of-turn markers of the model's chat template plus `options.stop`.
  final List<String> stopMarkers;
  final SendPort replyPort;
  _CmdGenerate(this.requestId, this.prompt, this.options, this.stopMarkers,
      this.replyPort);
}

final class _CmdCount {
  final String text;
  final bool addBos;
  final SendPort replyPort;
  _CmdCount(this.text, this.addBos, this.replyPort);
}

final class _CmdCancel {
  final int requestId;
  _CmdCancel(this.requestId);
}

/// Acked with [_MsgDisposed] once the native context is actually freed. The
/// main isolate waits for that before killing the worker: killing reclaims the
/// Dart heap but NOT the model weights + KV cache (over a gigabyte).
final class _CmdDispose {
  final SendPort replyPort;
  _CmdDispose(this.replyPort);
}

final class _MsgReady {
  final SendPort workerPort;
  _MsgReady(this.workerPort);
}

final class _MsgOk {
  const _MsgOk();
}

final class _MsgToken {
  final String text;
  final List<double> logprobs;
  _MsgToken(this.text, this.logprobs);
}

final class _MsgDone {
  final int tokens;
  final int promptTokensReused;
  _MsgDone(this.tokens, this.promptTokensReused);
}

final class _MsgError {
  final BitNetException error;
  _MsgError(this.error);
}

final class _MsgCount {
  final int count;
  _MsgCount(this.count);
}

final class _MsgDisposed {
  const _MsgDisposed();
}

// ---------------------------------------------------------------------------
// Worker (runs inside the inference isolate)
// ---------------------------------------------------------------------------

void _workerEntry((SendPort, InferenceBackendFactory) args) {
  final (mainPort, factory) = args;
  _InferenceWorker(mainPort, factory);
}

BitNetException _asBitNet(Object e) => e is BitNetException
    ? e
    : BitNetInferenceException(message: e.toString());

class _InferenceWorker {
  final InferenceBackendFactory _factory;
  final ReceivePort _port = ReceivePort();

  InferenceBackend? _backend;
  GenerationOptions? _sampler; // what the native sampler is configured with
  bool _disposed = false;
  final Set<int> _cancelled = <int>{};

  /// Generations run strictly one after another. The listener below is async
  /// (generation yields between tokens so cancel/dispose messages get
  /// through), so without this chain a second _CmdGenerate would start its
  /// own decode loop while the first is parked — two loops on one context.
  Future<void> _queue = Future<void>.value();

  _InferenceWorker(SendPort mainPort, this._factory) {
    mainPort.send(_MsgReady(_port.sendPort));
    _port.listen(_onMessage);
  }

  void _onMessage(Object? raw) {
    switch (raw) {
      case _CmdInit():
        _init(raw);
      case _CmdGenerate():
        _queue = _queue.then((_) => _generate(raw));
      case _CmdCount():
        _count(raw);
      case _CmdCancel():
        _cancelled.add(raw.requestId);
      case _CmdDispose():
        // Runs synchronously: a generation loop is parked on its inter-token
        // yield and observes `_disposed` before touching the context again,
        // so bn_free can never race an in-flight bn_next_token.
        _disposed = true;
        _backend?.free();
        _backend = null;
        raw.replyPort.send(const _MsgDisposed());
        _port.close();
    }
  }

  void _init(_CmdInit cmd) {
    try {
      final backend = _factory(cmd.backend);
      try {
        backend.init(cmd.modelPath, nCtx: cmd.nCtx, nThreads: cmd.nThreads);
        backend.setAbortFlag(cmd.abortFlagAddress);
      } catch (_) {
        backend.free();
        rethrow;
      }
      _backend = backend;
      cmd.replyPort.send(const _MsgOk());
    } catch (e) {
      cmd.replyPort.send(_MsgError(e is BitNetException
          ? e
          : BitNetInitException(message: e.toString())));
    }
  }

  void _count(_CmdCount cmd) {
    final backend = _backend;
    if (backend == null || _disposed) {
      cmd.replyPort.send(_MsgError(const BitNetNotLoadedException()));
      return;
    }
    try {
      cmd.replyPort.send(_MsgCount(backend.tokenCount(cmd.text, addBos: cmd.addBos)));
    } catch (e) {
      cmd.replyPort.send(_MsgError(_asBitNet(e)));
    }
  }

  Future<void> _generate(_CmdGenerate cmd) async {
    final reply = cmd.replyPort;
    final backend = _backend;
    if (backend == null || _disposed) {
      reply.send(_MsgError(const BitNetNotLoadedException()));
      return;
    }

    var count = 0;
    try {
      final o = cmd.options;
      if (_sampler == null || !_sampler!.samplerEquals(o)) {
        _sampler = null; // unknown state if setSampling throws
        backend.setSampling(o);
        _sampler = o;
      }

      if (backend.feedPrompt(cmd.prompt, addBos: o.addBos) ==
          PromptStatus.aborted) {
        reply.send(_MsgDone(0, 0));
        return;
      }
      final reused = backend.lastPromptReused;

      // Reassembles UTF-8 characters split across token pieces, then strips
      // end-of-turn markers that may themselves span several pieces.
      final utf8 = Utf8StreamDecoder();
      final stop = TurnStopDetector(cmd.stopMarkers);
      var pendingLp = <double>[];

      void emit(String text) {
        if (text.isEmpty) return;
        reply.send(_MsgToken(text, pendingLp));
        pendingLp = <double>[];
      }

      while (o.maxNewTokens == 0 || count < o.maxNewTokens) {
        // Let queued _CmdCancel / _CmdDispose messages be delivered: the FFI
        // call below blocks this isolate's event loop for its duration.
        await Future<void>.delayed(Duration.zero);
        if (_disposed || _cancelled.contains(cmd.requestId)) break;

        final tok = backend.nextToken();
        if (tok == null) break; // end of turn, context full, or abort flag
        count++;
        pendingLp.add(tok.logprob);
        final text = utf8.add(tok.bytes);
        if (text.isEmpty) continue;
        emit(stop.feed(text));
        if (stop.stopped) break;
      }

      // Release text held back while it might still have become a marker.
      if (!stop.stopped) {
        final tail = utf8.flush();
        if (tail.isNotEmpty) emit(stop.feed(tail));
        if (!stop.stopped) emit(stop.flush());
      }
      reply.send(_MsgDone(count, reused));
    } catch (e) {
      reply.send(_MsgError(_asBitNet(e)));
    } finally {
      _cancelled.remove(cmd.requestId);
    }
  }
}

// ---------------------------------------------------------------------------
// NativeEngine (main isolate)
// ---------------------------------------------------------------------------

/// Resolves the on-disk model path, downloading it if needed.
typedef ModelPathResolver = Future<String> Function(
    BitNetModelInfo model, void Function(double progress) onProgress);

class _Request {
  final int id;
  final String prompt;
  final GenerationOptions options;
  final StreamController<TokenEvent> controller;
  bool started = false;
  bool finished = false;
  bool consumerGone = false;
  Completer<void>? running;
  ReceivePort? reply;

  _Request(this.id, this.prompt, this.options, this.controller);

  void finish([BitNetException? error]) {
    if (finished) return;
    finished = true;
    reply?.close();
    if (error != null && !consumerGone) controller.addError(error);
    controller.close();
    running?.complete();
  }
}

/// Native, isolate-based [BitNetEngine] (Android / iOS / desktop).
///
/// All native calls run on a dedicated isolate, so a multi-second prefill
/// never blocks the UI. Cancellation is immediate: the main isolate writes a
/// shared native `int32` flag that the C layer polls during prefill and
/// between tokens (no round-trip through the worker's event loop needed).
class NativeEngine implements BitNetEngine {
  @override
  final BitNetModelInfo model;

  final InferenceBackendFactory _backendFactory;
  final ModelPathResolver _resolveModelPath;
  final bool _checkRam;

  /// Production engine: real native library, RAM gate, [ModelCache] download.
  NativeEngine(this.model)
      : _backendFactory = NativeLibrary.factory,
        _resolveModelPath = _defaultResolve,
        _checkRam = true;

  /// Engine over a custom [InferenceBackend] (for tests). [backendFactory]
  /// must be a top-level or static function (it is sent to the isolate).
  @visibleForTesting
  NativeEngine.withBackend(
    this.model, {
    required InferenceBackendFactory backendFactory,
    ModelPathResolver? resolveModelPath,
  })  : _backendFactory = backendFactory, // ignore: prefer_initializing_formals
        _resolveModelPath = resolveModelPath ?? ((_, _) async => '/fake'),
        _checkRam = false;

  static Future<String> _defaultResolve(
          BitNetModelInfo model, void Function(double) onProgress) =>
      ModelCache.ensureModel(model, onProgress: onProgress);

  Isolate? _isolate;
  SendPort? _workerPort;
  ReceivePort? _exitPort;
  ReceivePort? _errorPort;
  Pointer<Int32>? _abortFlag;
  bool _loaded = false;

  /// Incremented by [dispose]; a load that sees it change aborts itself.
  int _epoch = 0;
  Future<void>? _loading;
  Completer<Object?>? _pendingInit;

  int _nextRequestId = 0;
  _Request? _active;
  final List<_Request> _waiting = [];
  Future<void> _tail = Future<void>.value();

  @override
  bool get isLoaded => _loaded;

  // ---------------------------------------------------------------- load

  @override
  Future<void> load({void Function(double progress)? onProgress}) {
    if (_loaded) return Future<void>.value();
    return _loading ??= _load(onProgress).whenComplete(() => _loading = null);
  }

  Future<void> _load(void Function(double)? onProgress) async {
    final epoch = _epoch;
    void checkNotDisposed() {
      if (_epoch != epoch) {
        throw const BitNetIsolateException('Engine disposed while loading.');
      }
    }

    // 1. Gate on the SELECTED model's RAM floor before a >1 GB download.
    if (_checkRam) {
      final ram = await DeviceInspector.instance.totalPhysicalRamBytes();
      if (ram < model.minimumRamBytes) {
        throw BitNetUnsupportedDeviceException(ram,
            requiredRamBytes: model.minimumRamBytes,
            modelName: model.displayName);
      }
      checkNotDisposed();
    }

    // 2. Download / verify.
    final path = await _resolveModelPath(model, onProgress ?? (_) {});
    checkNotDisposed();

    // 3. Spawn the worker, watching for it dying.
    final flag = calloc<Int32>();
    final handshake = ReceivePort();
    final exitPort = ReceivePort();
    final errorPort = ReceivePort();
    final Isolate isolate;
    try {
      isolate = await Isolate.spawn(
        _workerEntry,
        (handshake.sendPort, _backendFactory),
        onExit: exitPort.sendPort,
        onError: errorPort.sendPort,
        debugName: 'bitnet-inference',
      );
    } catch (e) {
      handshake.close();
      exitPort.close();
      errorPort.close();
      calloc.free(flag);
      throw BitNetIsolateException('Could not start inference isolate: $e');
    }
    _isolate = isolate;
    _abortFlag = flag;
    _exitPort = exitPort;
    _errorPort = errorPort;
    final init = _pendingInit = Completer<Object?>();
    exitPort.listen((_) => _onWorkerDied('inference isolate exited'));
    errorPort.listen((e) => _onWorkerDied('inference isolate error: $e'));

    // 4. Handshake + init. Either can be cut short by the worker dying or by
    //    dispose(), both of which complete `init` with an error.
    final initReply = ReceivePort();
    handshake.first.then((m) {
      if (init.isCompleted) return;
      final port = (m as _MsgReady).workerPort;
      _workerPort = port;
      initReply.first.then((r) {
        if (!init.isCompleted) init.complete(r);
      }, onError: (Object _) {});
      port.send(_CmdInit(path, model.contextLength, 0, model.engineBackend,
          flag.address, initReply.sendPort));
    }, onError: (Object _) {});

    final Object? result;
    try {
      result = await init.future;
    } finally {
      handshake.close();
      initReply.close();
      _pendingInit = null;
    }

    if (_epoch != epoch || result is _MsgError) {
      await _teardownWorker();
      checkNotDisposed();
      throw (result as _MsgError).error;
    }
    _loaded = true;
    BitNetLog.log('engine', 'loaded ${model.id}');
  }

  /// The worker exited or threw without being asked to.
  void _onWorkerDied(String reason) {
    if (_isolate == null) return; // deliberate teardown
    BitNetLog.log('engine', 'worker died: $reason');
    final error = BitNetIsolateException(reason);
    _failPendingInit(error);
    _loaded = false;
    _workerPort = null;
    _active?.finish(error);
    _closeWatchPorts();
    _isolate = null;
    // The isolate is gone, so native code no longer touches the flag.
    final flag = _abortFlag;
    _abortFlag = null;
    if (flag != null) calloc.free(flag);
  }

  void _failPendingInit(BitNetException error) {
    final init = _pendingInit;
    if (init != null && !init.isCompleted) init.completeError(error);
  }

  void _closeWatchPorts() {
    _exitPort?.close();
    _errorPort?.close();
    _exitPort = null;
    _errorPort = null;
  }

  // ------------------------------------------------------------ generate

  @override
  Stream<TokenEvent> generateTokens(
    String prompt, {
    GenerationOptions options = const GenerationOptions(),
  }) {
    if (!_loaded) throw const BitNetNotLoadedException();
    final req = _Request(
        _nextRequestId++, prompt, options, StreamController<TokenEvent>());
    // The consumer walking away (widget disposed, `await for` broken out of)
    // must stop the model — otherwise it decodes to maxNewTokens unobserved.
    req.controller.onCancel = () {
      req.consumerGone = true;
      if (req.started && !req.finished) _stop(req);
      _waiting.remove(req);
    };
    _waiting.add(req);
    _tail = _tail.then((_) => _run(req));
    return req.controller.stream;
  }

  Future<void> _run(_Request req) async {
    _waiting.remove(req);
    if (req.consumerGone) return;
    final port = _workerPort;
    final flag = _abortFlag;
    if (!_loaded || port == null || flag == null) {
      req.finish(const BitNetNotLoadedException());
      return;
    }

    req.started = true;
    req.running = Completer<void>();
    _active = req;
    flag.value = 0; // the previous generation has fully finished (queued)

    final reply = req.reply = ReceivePort();
    reply.listen((m) {
      switch (m) {
        case _MsgToken():
          if (!req.consumerGone) req.controller.add(TokenEvent(m.text, m.logprobs));
        case _MsgDone():
          BitNetLog.log('engine',
              'req=${req.id} done tokens=${m.tokens} reused=${m.promptTokensReused}');
          req.finish();
        case _MsgError():
          req.finish(m.error);
      }
    });

    final stops = [
      ...chatTemplateFor(model.chatTemplate).stopMarkers,
      ...req.options.stop,
    ];
    port.send(_CmdGenerate(
        req.id, req.prompt, req.options, stops, reply.sendPort));
    await req.running!.future;
    if (identical(_active, req)) _active = null;
  }

  void _stop(_Request req) {
    _abortFlag?.value = 1;
    _workerPort?.send(_CmdCancel(req.id));
  }

  @override
  Future<void> cancelGeneration() async {
    final active = _active;
    if (active != null && !active.finished) _stop(active);
  }

  @override
  Future<int> countTokens(String text, {bool addBos = false}) async {
    final port = _workerPort;
    if (!_loaded || port == null) throw const BitNetNotLoadedException();
    final reply = ReceivePort();
    try {
      port.send(_CmdCount(text, addBos, reply.sendPort));
      final m = await reply.first;
      if (m is _MsgError) throw m.error;
      return (m as _MsgCount).count;
    } finally {
      reply.close();
    }
  }

  // ------------------------------------------------------------- dispose

  @override
  Future<void> dispose() async {
    _epoch++;
    _loaded = false;
    // A load in flight is waiting on the worker's init reply; release it (it
    // then sees the epoch change and reports the dispose).
    _failPendingInit(
        const BitNetIsolateException('Engine disposed while loading.'));
    final active = _active;
    if (active != null && !active.finished) _stop(active);
    for (final r in List.of(_waiting)) {
      r.finish(const BitNetNotLoadedException());
    }
    _waiting.clear();
    await _teardownWorker();
  }

  Future<void> _teardownWorker() async {
    final port = _workerPort;
    final isolate = _isolate;
    final flag = _abortFlag;
    _workerPort = null;
    _isolate = null; // from here on, an exit is expected (not a "death")
    _abortFlag = null;

    var freed = port == null; // never got a context
    if (port != null) {
      final ack = ReceivePort();
      port.send(_CmdDispose(ack.sendPort));
      freed = await ack.first
          .then((_) => true)
          .timeout(const Duration(seconds: 5), onTimeout: () => false);
      ack.close();
      if (!freed) {
        BitNetLog.log(
            'engine', 'dispose ack timed out — killing isolate (native memory may leak)');
      }
    }
    _active?.finish();
    _active = null;
    isolate?.kill(priority: Isolate.immediate);
    _closeWatchPorts();
    // A wedged worker may still be inside native code that reads the flag:
    // leak 4 bytes rather than risk a use-after-free.
    if (flag != null && freed) calloc.free(flag);
  }
}

/// Factory resolved by the conditional import in `bitnet_engine.dart`.
BitNetEngine createEngine({required BitNetModelInfo model}) => NativeEngine(model);
