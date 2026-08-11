import 'dart:async';
import 'dart:isolate';

import '../cache/model_cache.dart';
import '../device/device_inspector.dart';
import '../diagnostics/bitnet_log.dart';
import '../exceptions/bitnet_exception.dart';
import '../ffi/native_library.dart';
import '../model/bitnet_model_info.dart';
import '../session/chat_templates.dart';
import 'bitnet_engine.dart';

// ---------------------------------------------------------------------------
// Isolate message protocol
// ---------------------------------------------------------------------------

sealed class _Cmd {}

final class _CmdInit extends _Cmd {
  final String modelPath;
  final int nCtx;
  final int nThreads;
  final EngineBackend backend;
  final SendPort replyPort;
  _CmdInit(
    this.modelPath, {
    required this.nCtx,
    required this.nThreads,
    required this.backend,
    required this.replyPort,
  });
}

final class _CmdGenerate extends _Cmd {
  final String prompt;
  final int maxNewTokens;
  final bool addBos;
  final int requestId;
  final SendPort replyPort;

  /// End-of-turn markers for this model's chat template. Passed in because the
  /// isolate worker has no access to the model's [ChatTemplateFamily].
  final List<String> stopMarkers;
  _CmdGenerate({
    required this.prompt,
    required this.maxNewTokens,
    required this.addBos,
    required this.requestId,
    required this.replyPort,
    required this.stopMarkers,
  });
}

final class _CmdCancel extends _Cmd {
  final int requestId;
  _CmdCancel(this.requestId);
}

final class _CmdDispose extends _Cmd {
  /// Acked with [_MsgDisposed] once the native context has actually been freed.
  /// The caller MUST wait for that ack before killing the isolate: killing it
  /// releases the Dart heap but NOT the native allocations behind the context
  /// (model weights + KV cache — over a gigabyte), so a kill that races ahead
  /// of bn_free leaks all of it for the lifetime of the process.
  final SendPort replyPort;
  _CmdDispose(this.replyPort);
}

// ---------------------------------------------------------------------------

sealed class _Msg {}

final class _MsgReady extends _Msg {
  final SendPort workerPort;
  _MsgReady(this.workerPort);
}

final class _MsgToken extends _Msg {
  final String text;
  _MsgToken(this.text);
}

final class _MsgDone extends _Msg {
  final int tokensGenerated;
  _MsgDone(this.tokensGenerated);
}

final class _MsgInitOk extends _Msg {
  _MsgInitOk();
}

/// Ack for [_CmdDispose]: the native context has been freed and the worker's
/// port is closed.
final class _MsgDisposed extends _Msg {
  _MsgDisposed();
}

final class _MsgError extends _Msg {
  final String code;
  final String message;

  /// The descriptive error string from the native layer (out_err), when the
  /// underlying exception carried one. Kept separate from [code] (a coarse
  /// Dart-side category like 'prompt_failed') so the real cause survives the
  /// isolate boundary and reaches the UI.
  final String? nativeError;
  _MsgError(this.code, this.message, {this.nativeError});
}

/// Extracts the native out_err string from whichever BitNetException subtype
/// carries one, so it can be forwarded across the isolate boundary.
String? _nativeErrorOf(Object e) => switch (e) {
      BitNetInitException(:final nativeError) => nativeError,
      BitNetInferenceException(:final nativeError) => nativeError,
      _ => null,
    };

// ---------------------------------------------------------------------------
// Isolate entry point
// ---------------------------------------------------------------------------

void _workerEntry(SendPort mainPort) {
  final receivePort = ReceivePort();
  mainPort.send(_MsgReady(receivePort.sendPort));

  NativeLibrary? lib;
  var cancelRequestId = -1;
  // Set by _CmdDispose. A generation loop parked on its inter-token yield
  // checks this before touching the context again, so bn_free can never race
  // ahead of an in-flight bn_next_token (which would be a use-after-free).
  var disposed = false;

  // The handler is async so the generation loop can yield to the isolate
  // event loop between tokens (see _CmdGenerate). Without that yield the
  // synchronous loop starves the port and a _CmdCancel sent mid-generation is
  // never delivered until generation has already finished on its own.
  receivePort.listen((dynamic raw) async {
    final cmd = raw as _Cmd;

    switch (cmd) {
      case _CmdInit():
        try {
          lib = NativeLibrary.open(backend: cmd.backend);
          lib!.initContext(
            cmd.modelPath,
            nCtx: cmd.nCtx,
            nThreads: cmd.nThreads,
          );
          cmd.replyPort.send(_MsgInitOk());
        } on BitNetException catch (e) {
          lib = null;
          cmd.replyPort.send(
              _MsgError('init_failed', e.message, nativeError: _nativeErrorOf(e)));
        } catch (e) {
          lib = null;
          cmd.replyPort.send(_MsgError('init_failed', e.toString()));
        }

      case _CmdGenerate():
        final replyPort = cmd.replyPort;
        cancelRequestId = -1;

        if (lib == null || disposed) {
          replyPort.send(_MsgError('not_loaded', 'Engine not initialised.'));
          replyPort.send(_MsgDone(0));
          return;
        }

        try {
          lib!.feedPrompt(cmd.prompt, addBos: cmd.addBos);
        } on BitNetException catch (e) {
          replyPort.send(_MsgError('prompt_failed', e.message,
              nativeError: _nativeErrorOf(e)));
          replyPort.send(_MsgDone(0));
          return;
        }

        var count = 0;
        // Reassembles UTF-8 characters that the tokenizer split across token
        // pieces (raw byte fragments), so we never hand the stop detector or
        // the UI a half-decoded code point.
        final utf8Decoder = Utf8StreamDecoder();
        // Detects this model's end-of-turn markers even when they are split
        // across several token pieces. Markers are template-specific and
        // supplied by the caller (the worker has no ChatTemplateFamily).
        final stopDetector = TurnStopDetector(cmd.stopMarkers);
        while (count < cmd.maxNewTokens || cmd.maxNewTokens == 0) {
          // Yield to the isolate event loop so a _CmdCancel queued on this
          // port can actually be delivered (and set cancelRequestId) before we
          // check it. nextToken() below is a blocking FFI call that would
          // otherwise starve the port for the whole generation.
          await Future<void>.delayed(Duration.zero);

          // Check for cancellation / teardown between tokens. `disposed` means
          // the context has already been freed — touching it now would be a
          // use-after-free, so bail before nextToken().
          if (disposed) break;
          if (cancelRequestId == cmd.requestId) break;

          try {
            final pieceBytes = lib!.nextToken();
            if (pieceBytes == null) {
              // EOS
              break;
            }
            // Decode bytes → text, buffering any partial UTF-8 sequence so a
            // character split across pieces is never surfaced half-decoded.
            final piece = utf8Decoder.add(pieceBytes);
            if (piece.isNotEmpty) {
              final emit = stopDetector.feed(piece);
              if (emit.isNotEmpty) {
                replyPort.send(_MsgToken(emit));
              }
              if (stopDetector.stopped) {
                // A complete end-of-turn marker was assembled — stop without
                // emitting the marker text itself.
                break;
              }
            }
            count++;
          } on BitNetException catch (e) {
            replyPort.send(_MsgError('inference_failed', e.message,
                nativeError: _nativeErrorOf(e)));
            replyPort.send(_MsgDone(count));
            return;
          }
        }

        // Release any text held back while waiting to see if it was part of a
        // marker (stream ended via EOS, token cap, or cancellation). Drain the
        // UTF-8 decoder first (its leftover bytes feed the stop detector), then
        // the stop detector.
        if (!stopDetector.stopped) {
          final tail = utf8Decoder.flush();
          if (tail.isNotEmpty) {
            final emit = stopDetector.feed(tail);
            if (emit.isNotEmpty) {
              replyPort.send(_MsgToken(emit));
            }
          }
          if (!stopDetector.stopped) {
            final rest = stopDetector.flush();
            if (rest.isNotEmpty) {
              replyPort.send(_MsgToken(rest));
            }
          }
        }

        replyPort.send(_MsgDone(count));

      case _CmdCancel():
        cancelRequestId = cmd.requestId;

      case _CmdDispose():
        // Runs to completion synchronously: any generation loop is parked on
        // its `await Future.delayed(Duration.zero)` and cannot resume until we
        // return, so it observes `disposed == true` before its next FFI call.
        disposed = true;
        lib?.freeContext();
        lib = null;
        cmd.replyPort.send(_MsgDisposed());
        receivePort.close();
    }
  });
}

// ---------------------------------------------------------------------------
// NativeEngine
// ---------------------------------------------------------------------------

/// Native Isolate-based [BitNetEngine] implementation (Android/iOS/Desktop).
class NativeEngine implements BitNetEngine {
  final BitNetModelInfo _model;

  NativeEngine(this._model);

  SendPort? _workerPort;
  Isolate? _isolate;
  bool _loaded = false;
  int _nextRequestId = 0;

  /// Id of the generation currently in flight, or null when idle. One engine
  /// owns one native context, so overlapping generations are rejected rather
  /// than silently interleaved on it (see [BitNetBusyException]).
  int? _activeRequestId;

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {
    if (_loaded) return;

    // 1. Device check — gate on the SELECTED model's RAM floor, not a fixed
    //    constant, so smaller models load on smaller devices and larger models
    //    are refused before they OOM inside bn_init.
    final ram = await DeviceInspector.instance.totalPhysicalRamBytes();
    if (ram < _model.minimumRamBytes) {
      throw BitNetUnsupportedDeviceException(
        ram,
        requiredRamBytes: _model.minimumRamBytes,
        modelName: _model.displayName,
      );
    }

    // 2. Download / verify model.
    final path = await ModelCache.ensureModel(
      _model,
      onProgress: onProgress ?? (_) {},
    );

    // 3. Spawn inference isolate.
    final handshake = ReceivePort();
    _isolate = await Isolate.spawn(_workerEntry, handshake.sendPort);

    final ready = await handshake.first as _MsgReady;
    handshake.close();
    _workerPort = ready.workerPort;

    // 4. Initialise native context inside the isolate, and wait for it.
    final initReply = ReceivePort();
    _workerPort!.send(_CmdInit(
      path,
      nCtx: _model.contextLength,
      nThreads: 0,
      backend: _model.engineBackend,
      replyPort: initReply.sendPort,
    ));

    final initResult = await initReply.first;
    initReply.close();

    if (initResult is _MsgError) {
      // Tear down the half-spawned isolate so subsequent load() calls can retry.
      _isolate?.kill(priority: Isolate.immediate);
      _isolate = null;
      _workerPort = null;
      throw BitNetInitException(
        message: initResult.message,
        nativeError: initResult.nativeError ?? initResult.code,
      );
    }

    _loaded = true;
  }

  @override
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  }) {
    if (!_loaded || _workerPort == null) {
      throw const BitNetNotLoadedException();
    }
    if (_activeRequestId != null) {
      throw const BitNetBusyException();
    }

    final requestId = _nextRequestId++;
    _activeRequestId = requestId;
    final replyPort = ReceivePort();
    final controller = StreamController<String>();

    // Releases the request slot and the reply port exactly once, whichever
    // terminal path gets there first (done / error / consumer cancellation).
    var settled = false;
    void settle() {
      if (settled) return;
      settled = true;
      if (_activeRequestId == requestId) _activeRequestId = null;
      replyPort.close();
    }

    _workerPort!.send(_CmdGenerate(
      prompt: prompt,
      maxNewTokens: maxNewTokens,
      addBos: addBos,
      requestId: requestId,
      replyPort: replyPort.sendPort,
      stopMarkers: chatTemplateFor(_model.chatTemplate).stopMarkers,
    ));
    BitNetLog.log('engine',
        'generate() req=$requestId prompt=${prompt.length}ch '
        'maxNewTokens=$maxNewTokens addBos=$addBos');

    var tokenCount = 0;
    replyPort.listen((dynamic raw) {
      final msg = raw as _Msg;
      switch (msg) {
        case _MsgToken():
          tokenCount++;
          if (tokenCount <= 3 || tokenCount % 32 == 0) {
            BitNetLog.log('engine',
                'token #$tokenCount req=$requestId piece=${_q(msg.text)}');
          }
          controller.add(msg.text);
        case _MsgDone():
          BitNetLog.log('engine',
              'done req=$requestId tokens=${msg.tokensGenerated} '
              '(seen=$tokenCount)');
          settle();
          controller.close();
        case _MsgError():
          BitNetLog.log('engine',
              'error req=$requestId code=${msg.code} message=${msg.message} '
              'native=${msg.nativeError}');
          controller.addError(BitNetInferenceException(
            message: msg.message,
            nativeError: msg.nativeError ?? msg.code,
          ));
          settle();
          controller.close();
        case _MsgReady():
        case _MsgInitOk():
        case _MsgDisposed():
          break; // unexpected here, ignore
      }
    });

    // The consumer walking away (widget disposed, tab switched, `await for`
    // broken out of) must actually stop the worker. Without this the isolate
    // decodes all the way to maxNewTokens at 100% CPU with nobody listening,
    // and the reply port is never closed.
    controller.onCancel = () {
      if (!settled) {
        BitNetLog.log('engine', 'stream cancelled req=$requestId — stopping');
        _workerPort?.send(_CmdCancel(requestId));
      }
      settle();
    };

    return controller.stream;
  }

  static String _q(String s) {
    final escaped = s
        .replaceAll('\\', '\\\\')
        .replaceAll('\n', '\\n')
        .replaceAll('\r', '\\r')
        .replaceAll('\t', '\\t');
    return '"$escaped"';
  }

  @override
  Future<void> cancelGeneration() async {
    // Target the generation that is actually running. Cancelling by
    // `_nextRequestId - 1` would also fire when idle, arming the worker
    // against an id it may never see again.
    final active = _activeRequestId;
    if (_workerPort == null || active == null) return;
    _workerPort!.send(_CmdCancel(active));
  }

  @override
  Future<void> dispose() async {
    final port = _workerPort;
    _loaded = false;
    _workerPort = null;

    if (port != null) {
      // Stop any in-flight generation first so the worker is not mid-decode
      // when the context goes away.
      final active = _activeRequestId;
      if (active != null) port.send(_CmdCancel(active));

      // Wait for bn_free to actually run before killing the isolate. Killing
      // an isolate reclaims its Dart heap but NOT the native model weights and
      // KV cache behind the context, so a kill that outruns the dispose
      // message leaks well over a gigabyte per model switch. Bounded, with the
      // kill below as the backstop for a wedged worker.
      final ack = ReceivePort();
      port.send(_CmdDispose(ack.sendPort));
      await ack.first.timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          BitNetLog.log('engine',
              'dispose ack timed out — killing isolate (native memory may leak)');
          return _MsgDisposed();
        },
      );
      ack.close();
    }

    _activeRequestId = null;
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
  }
}

/// Factory function resolved by the conditional import in [BitNetEngine].
BitNetEngine createEngine({required BitNetModelInfo model}) => NativeEngine(model);
