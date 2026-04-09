import 'dart:async';
import 'dart:isolate';

import '../cache/model_cache.dart';
import '../device/device_inspector.dart';
import '../exceptions/bitnet_exception.dart';
import '../ffi/native_library.dart';
import '../model/bitnet_model.dart';
import 'bitnet_engine.dart';

// ---------------------------------------------------------------------------
// Isolate message protocol
// ---------------------------------------------------------------------------

sealed class _Cmd {}

final class _CmdInit extends _Cmd {
  final String modelPath;
  final int nCtx;
  final int nThreads;
  _CmdInit(this.modelPath, {required this.nCtx, required this.nThreads});
}

final class _CmdGenerate extends _Cmd {
  final String prompt;
  final int maxNewTokens;
  final bool addBos;
  final int requestId;
  final SendPort replyPort;
  _CmdGenerate({
    required this.prompt,
    required this.maxNewTokens,
    required this.addBos,
    required this.requestId,
    required this.replyPort,
  });
}

final class _CmdCancel extends _Cmd {
  final int requestId;
  _CmdCancel(this.requestId);
}

final class _CmdDispose extends _Cmd {}

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

final class _MsgError extends _Msg {
  final String code;
  final String message;
  _MsgError(this.code, this.message);
}

// ---------------------------------------------------------------------------
// Isolate entry point
// ---------------------------------------------------------------------------

void _workerEntry(SendPort mainPort) {
  final receivePort = ReceivePort();
  mainPort.send(_MsgReady(receivePort.sendPort));

  NativeLibrary? lib;
  var cancelRequestId = -1;

  receivePort.listen((dynamic raw) {
    final cmd = raw as _Cmd;

    switch (cmd) {
      case _CmdInit():
        try {
          lib = NativeLibrary.open();
          lib!.initContext(
            cmd.modelPath,
            nCtx: cmd.nCtx,
            nThreads: cmd.nThreads,
          );
        } on BitNetException catch (e) {
          mainPort.send(_MsgError('init_failed', e.message));
        }

      case _CmdGenerate():
        final replyPort = cmd.replyPort;
        cancelRequestId = -1;

        if (lib == null) {
          replyPort.send(_MsgError('not_loaded', 'Engine not initialised.'));
          replyPort.send(_MsgDone(0));
          return;
        }

        try {
          lib!.feedPrompt(cmd.prompt, addBos: cmd.addBos);
        } on BitNetException catch (e) {
          replyPort.send(_MsgError('prompt_failed', e.message));
          replyPort.send(_MsgDone(0));
          return;
        }

        var count = 0;
        while (count < cmd.maxNewTokens || cmd.maxNewTokens == 0) {
          // Check for cancellation between tokens.
          if (cancelRequestId == cmd.requestId) break;

          try {
            final piece = lib!.nextToken();
            if (piece == null) {
              // EOS
              break;
            }
            replyPort.send(_MsgToken(piece));
            count++;
          } on BitNetException catch (e) {
            replyPort.send(_MsgError('inference_failed', e.message));
            replyPort.send(_MsgDone(count));
            return;
          }
        }

        replyPort.send(_MsgDone(count));

      case _CmdCancel():
        cancelRequestId = cmd.requestId;

      case _CmdDispose():
        lib?.freeContext();
        receivePort.close();
    }
  });
}

// ---------------------------------------------------------------------------
// NativeEngine
// ---------------------------------------------------------------------------

/// Native Isolate-based [BitNetEngine] implementation (Android/iOS/Desktop).
class NativeEngine implements BitNetEngine {
  final BitNetModel _model;

  NativeEngine(this._model);

  SendPort? _workerPort;
  Isolate? _isolate;
  bool _loaded = false;
  int _nextRequestId = 0;

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {
    if (_loaded) return;

    // 1. Device check.
    final ok = await DeviceInspector.instance.meetsMinimumRam();
    if (!ok) {
      final ram = await DeviceInspector.instance.totalPhysicalRamBytes();
      throw BitNetUnsupportedDeviceException(ram);
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

    // 4. Initialise native context inside the isolate.
    _workerPort!.send(_CmdInit(
      path,
      nCtx: _model.contextLength,
      nThreads: 0,
    ));

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

    final requestId = _nextRequestId++;
    final replyPort = ReceivePort();
    final controller = StreamController<String>();

    _workerPort!.send(_CmdGenerate(
      prompt: prompt,
      maxNewTokens: maxNewTokens,
      addBos: addBos,
      requestId: requestId,
      replyPort: replyPort.sendPort,
    ));

    replyPort.listen((dynamic raw) {
      final msg = raw as _Msg;
      switch (msg) {
        case _MsgToken():
          controller.add(msg.text);
        case _MsgDone():
          controller.close();
          replyPort.close();
        case _MsgError():
          controller.addError(BitNetInferenceException(
            message: msg.message,
            nativeError: msg.code,
          ));
          controller.close();
          replyPort.close();
        case _MsgReady():
          break; // unexpected here, ignore
      }
    });

    return controller.stream;
  }

  @override
  Future<void> cancelGeneration() async {
    if (_workerPort == null || _nextRequestId == 0) return;
    _workerPort!.send(_CmdCancel(_nextRequestId - 1));
  }

  @override
  Future<void> dispose() async {
    _workerPort?.send(_CmdDispose());
    _isolate?.kill(priority: Isolate.immediate);
    _workerPort = null;
    _isolate = null;
    _loaded = false;
  }
}

/// Factory function resolved by the conditional import in [BitNetEngine].
BitNetEngine createEngine({required BitNetModel model}) => NativeEngine(model);
