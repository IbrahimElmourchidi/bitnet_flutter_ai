import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../diagnostics/bitnet_log.dart';
import '../engine/generation_options.dart';
import '../engine/inference_backend.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart' show EngineBackend;
import 'bindings.dart';

/// C ABI version this Dart code drives (mirrors `BN_ABI_VERSION`).
const int kBnAbiVersion = 2;

/// `bn_prompt` return code for an aborted prefill (mirrors `BN_ABORTED`).
const int _kBnAborted = -2;

/// Prefix of the native error string for an over-long prompt.
const String _kPromptTooLong = 'prompt_too_long:';

/// [InferenceBackend] over the bn_* C ABI via dart:ffi.
///
/// One instance per inference isolate; NOT safe to share across isolates.
class NativeLibrary implements InferenceBackend {
  final BitNetBindings _b;
  Pointer<BnContext> _ctx = nullptr;

  // Out-parameters reused by every bn_next_token_ex call (the hot loop), so a
  // generation does not malloc/free three scratch cells per token.
  final Pointer<Int32> _done = malloc<Int32>();
  final Pointer<Float> _logprob = malloc<Float>();
  final Pointer<Pointer<Char>> _err = malloc<Pointer<Char>>();
  bool _scratchFreed = false;

  NativeLibrary._(this._b);

  /// Opens the shared library for [backend] and checks its ABI version.
  ///
  /// Throws [BitNetLibraryLoadException] if the library cannot be loaded or is
  /// a stale build that predates ABI v[kBnAbiVersion].
  factory NativeLibrary.open({EngineBackend backend = EngineBackend.bitnetCpp}) {
    final BitNetBindings b;
    try {
      b = BitNetBindings(_openLib(backend));
    } on ArgumentError catch (e) {
      throw BitNetLibraryLoadException(
          'Failed to load ${backend.name} native library: $e');
    }
    final int version;
    try {
      version = b.bn_abi_version();
    } on ArgumentError {
      throw BitNetLibraryLoadException(
          'The ${backend.name} native library predates bn ABI v$kBnAbiVersion '
          '(no bn_abi_version symbol). Rebuild it with the tool/build_*.sh '
          'scripts of this package version.');
    }
    if (version != kBnAbiVersion) {
      throw BitNetLibraryLoadException(
          'The ${backend.name} native library implements bn ABI v$version; '
          'this package needs v$kBnAbiVersion. Rebuild the native libraries.');
    }
    return NativeLibrary._(b);
  }

  /// Backend factory used by the inference isolate in production.
  static InferenceBackend factory(EngineBackend backend) =>
      NativeLibrary.open(backend: backend);

  static DynamicLibrary _openLib(EngineBackend backend) {
    // PrismML backend ships as ONE self-contained library (fork linked static),
    // so no transitive-dependency pre-load is needed.
    if (backend == EngineBackend.prismQ2) {
      if (Platform.isAndroid || Platform.isLinux) {
        return DynamicLibrary.open('libprism_bridge.so');
      }
      if (Platform.isMacOS) return DynamicLibrary.open('libprism_bridge.dylib');
      if (Platform.isWindows) return DynamicLibrary.open('prism_bridge.dll');
      if (Platform.isIOS) {
        // Prism ships as its OWN embedded dynamic framework. Opened by path
        // rather than DynamicLibrary.process(): the bitnet backend is also
        // linked into the process and exports the SAME bn_* names, so
        // .process() would resolve bitnet's bn_init, which cannot load Q2_0.
        return DynamicLibrary.open('prism_bridge.framework/prism_bridge');
      }
      throw BitNetLibraryLoadException(
          'PrismML backend unsupported on ${Platform.operatingSystem}');
    }

    // bitnet.cpp backend: libbitnet_bridge.so → libllama.so → libggml.so.
    if (Platform.isAndroid) {
      // Pre-load transitive dependencies so the dynamic linker can resolve the
      // chain. libomp.so is shipped from the NDK's OpenMP runtime.
      for (final dep in const ['libomp.so', 'libggml.so', 'libllama.so']) {
        try {
          DynamicLibrary.open(dep);
        } catch (e) {
          BitNetLog.log('ffi', 'pre-load $dep failed: $e');
        }
      }
      return DynamicLibrary.open('libbitnet_bridge.so');
    }
    // Statically linked into the app binary on iOS.
    if (Platform.isIOS) return DynamicLibrary.process();
    if (Platform.isMacOS) return DynamicLibrary.open('libbitnet_bridge.dylib');
    if (Platform.isLinux) return DynamicLibrary.open('libbitnet_bridge.so');
    if (Platform.isWindows) return DynamicLibrary.open('bitnet_bridge.dll');
    throw BitNetLibraryLoadException(
        'Unsupported platform: ${Platform.operatingSystem}');
  }

  // ---------------------------------------------------------------------------
  // InferenceBackend
  // ---------------------------------------------------------------------------

  @override
  void init(String modelPath, {required int nCtx, required int nThreads}) {
    BitNetLog.log('ffi', 'bn_init nCtx=$nCtx nThreads=$nThreads');
    using((arena) {
      final pathPtr = modelPath.toNativeUtf8(allocator: arena).cast<Char>();
      final errPtr = arena<Pointer<Char>>()..value = nullptr;
      final result = _b.bn_init(pathPtr, nCtx, nThreads, errPtr);
      if (result == nullptr) {
        throw BitNetInitException(
          message: 'bn_init failed for model: $modelPath',
          nativeError: _readAndFreeErr(errPtr.value),
        );
      }
      _ctx = result;
    });
  }

  @override
  void setAbortFlag(int address) {
    _requireContext();
    _b.bn_set_abort_flag(
        _ctx, address == 0 ? nullptr : Pointer<Int32>.fromAddress(address));
  }

  @override
  void setSampling(GenerationOptions o) {
    _requireContext();
    using((arena) {
      final params = BnSamplingParams.$allocate(
        arena,
        temperature: o.temperature,
        top_k: o.topK,
        top_p: o.topP,
        repeat_penalty: o.repeatPenalty,
        repeat_last_n: o.repeatLastN,
        seed: o.seed & 0xFFFFFFFF,
      );
      final grammar = o.grammar;
      final grammarPtr = (grammar == null || grammar.isEmpty)
          ? nullptr
          : grammar.toNativeUtf8(allocator: arena).cast<Char>();
      final errPtr = arena<Pointer<Char>>()..value = nullptr;
      if (_b.bn_set_sampling(_ctx, params, grammarPtr, errPtr) != 0) {
        throw BitNetInferenceException(
          message: 'bn_set_sampling failed',
          nativeError: _readAndFreeErr(errPtr.value),
        );
      }
    });
  }

  @override
  PromptStatus feedPrompt(String prompt, {required bool addBos}) {
    _requireContext();
    return using((arena) {
      final promptPtr = prompt.toNativeUtf8(allocator: arena).cast<Char>();
      final errPtr = arena<Pointer<Char>>()..value = nullptr;
      final rc = _b.bn_prompt(_ctx, promptPtr, addBos ? 1 : 0, errPtr);
      if (rc == 0) return PromptStatus.ok;
      if (rc == _kBnAborted) return PromptStatus.aborted;
      final nativeErr = _readAndFreeErr(errPtr.value);
      if (nativeErr != null && nativeErr.startsWith(_kPromptTooLong)) {
        throw BitNetPromptTooLongException(
            nativeErr.substring(_kPromptTooLong.length).trim());
      }
      throw BitNetInferenceException(
          message: 'bn_prompt failed', nativeError: nativeErr);
    });
  }

  /// Returns the raw bytes of the next token piece — NOT decoded: a piece is
  /// often only a fragment of a multi-byte UTF-8 character, so the caller
  /// reassembles across pieces (see `Utf8StreamDecoder`).
  @override
  NextToken? nextToken() {
    _requireContext();
    _done.value = 0;
    _err.value = nullptr;
    final tokenPtr = _b.bn_next_token_ex(_ctx, _done, _logprob, _err);
    if (tokenPtr == nullptr) {
      if (_done.value == 1) return null;
      throw BitNetInferenceException(
        message: 'bn_next_token failed',
        nativeError: _readAndFreeErr(_err.value),
      );
    }
    // Copy (asTypedList is a view of native memory) before freeing.
    final utf8Ptr = tokenPtr.cast<Utf8>();
    final bytes =
        Uint8List.fromList(utf8Ptr.cast<Uint8>().asTypedList(utf8Ptr.length));
    _b.bn_free_string(tokenPtr);
    return NextToken(bytes, _logprob.value);
  }

  @override
  int tokenCount(String text, {required bool addBos}) {
    _requireContext();
    return using((arena) {
      final n = _b.bn_token_count(
          _ctx, text.toNativeUtf8(allocator: arena).cast<Char>(), addBos ? 1 : 0);
      if (n < 0) {
        throw const BitNetInferenceException(message: 'bn_token_count failed');
      }
      return n;
    });
  }

  @override
  int get lastPromptReused => _ctx == nullptr ? 0 : _b.bn_last_prompt_reused(_ctx);

  /// Context window size of the loaded model, or -1 if not loaded.
  int contextSize() => _ctx == nullptr ? -1 : _b.bn_context_size(_ctx);

  @override
  void free() {
    if (_ctx != nullptr) {
      _b.bn_free(_ctx);
      _ctx = nullptr;
    }
    if (!_scratchFreed) {
      _scratchFreed = true;
      malloc
        ..free(_done)
        ..free(_logprob)
        ..free(_err);
    }
  }

  // ---------------------------------------------------------------------------

  void _requireContext() {
    if (_ctx == nullptr) throw const BitNetNotLoadedException();
  }

  /// Reads a native C string into Dart, then frees it. Null-safe.
  String? _readAndFreeErr(Pointer<Char> ptr) {
    if (ptr == nullptr) return null;
    final msg = ptr.cast<Utf8>().toDartString();
    _b.bn_free_string(ptr);
    return msg;
  }
}
