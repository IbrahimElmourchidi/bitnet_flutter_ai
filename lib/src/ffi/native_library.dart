import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../diagnostics/bitnet_log.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart' show EngineBackend;
import 'bindings.dart';

/// Loads the platform-specific native library and wraps raw FFI calls with
/// Dart-friendly error handling and pointer lifecycle management.
///
/// One [NativeLibrary] instance is created per inference isolate.
/// It is NOT safe to share across isolates or threads.
class NativeLibrary {
  final BitNetBindings _b;
  Pointer<BnContext> _ctx = nullptr;

  NativeLibrary._(this._b);

  /// Opens the shared library for [backend] on the current platform and
  /// returns a [NativeLibrary]. Both backends expose the identical `bn_*` C
  /// ABI, so the FFI bindings are shared — only the library name differs.
  ///
  /// Throws [BitNetLibraryLoadException] if the library cannot be found or loaded.
  factory NativeLibrary.open({EngineBackend backend = EngineBackend.bitnetCpp}) {
    try {
      final lib = _openLib(backend);
      return NativeLibrary._(BitNetBindings(lib));
    } on ArgumentError catch (e) {
      throw BitNetLibraryLoadException(
        'Failed to load ${backend.name} native library: $e',
      );
    }
  }

  static DynamicLibrary _openLib(EngineBackend backend) {
    // PrismML backend ships as ONE self-contained library (fork linked static),
    // so no transitive-dependency pre-load is needed.
    if (backend == EngineBackend.prismQ2) {
      if (Platform.isAndroid || Platform.isLinux) {
        BitNetLog.log('ffi', 'opening libprism_bridge.so');
        return DynamicLibrary.open('libprism_bridge.so');
      }
      if (Platform.isMacOS) return DynamicLibrary.open('libprism_bridge.dylib');
      if (Platform.isWindows) return DynamicLibrary.open('prism_bridge.dll');
      if (Platform.isIOS) return DynamicLibrary.process();
      throw BitNetLibraryLoadException(
        'PrismML backend unsupported on ${Platform.operatingSystem}',
      );
    }

    // bitnet.cpp backend: libbitnet_bridge.so → libllama.so → libggml.so.
    if (Platform.isAndroid) {
      // Pre-load transitive dependencies so the dynamic linker can resolve
      // the libbitnet_bridge.so → libllama.so → libggml.so chain.
      // libomp.so is shipped from the NDK's OpenMP runtime.
      for (final dep in const ['libomp.so', 'libggml.so', 'libllama.so']) {
        try {
          DynamicLibrary.open(dep);
          BitNetLog.log('ffi', 'pre-loaded $dep');
        } catch (e) {
          BitNetLog.log('ffi', 'pre-load $dep failed: $e');
        }
      }
      BitNetLog.log('ffi', 'opening libbitnet_bridge.so');
      return DynamicLibrary.open('libbitnet_bridge.so');
    }
    if (Platform.isIOS) {
      // Statically linked into the app binary on iOS.
      return DynamicLibrary.process();
    }
    if (Platform.isMacOS) {
      return DynamicLibrary.open('libbitnet_bridge.dylib');
    }
    if (Platform.isLinux) {
      return DynamicLibrary.open('libbitnet_bridge.so');
    }
    if (Platform.isWindows) {
      return DynamicLibrary.open('bitnet_bridge.dll');
    }
    throw BitNetLibraryLoadException(
      'Unsupported platform: ${Platform.operatingSystem}',
    );
  }

  // ---------------------------------------------------------------------------
  // Context lifecycle
  // ---------------------------------------------------------------------------

  /// Initialises the native inference context from a GGUF model file.
  ///
  /// [modelPath]  Absolute path to the `.gguf` file on disk.
  /// [nCtx]       KV-cache context window in tokens (0 → model default).
  /// [nThreads]   CPU thread count (0 → hardware_concurrency).
  ///
  /// Throws [BitNetInitException] if the native call fails.
  void initContext(
    String modelPath, {
    int nCtx = 2048,
    int nThreads = 0,
  }) {
    BitNetLog.log('ffi',
        'bn_init path=$modelPath nCtx=$nCtx nThreads=$nThreads');
    using((arena) {
      final pathPtr = modelPath.toNativeUtf8(allocator: arena).cast<Char>();
      final errPtr = arena<Pointer<Char>>();

      final result = _b.bn_init(
        pathPtr,
        nCtx,
        nThreads,
        errPtr,
      );

      if (result == nullptr) {
        final nativeErr = _readAndFreeErr(errPtr.value);
        throw BitNetInitException(
          message: 'bn_init failed for model: $modelPath',
          nativeError: nativeErr,
        );
      }

      _ctx = result;
    });
    final ctxSize = contextSize();
    BitNetLog.log('ffi', 'bn_init OK ctxSize=$ctxSize');
  }

  /// Tokenizes [prompt] and runs the prefill pass, populating the KV cache.
  ///
  /// Must be called before the first [nextToken] call for each new prompt.
  ///
  /// Throws [BitNetInferenceException] on failure.
  void feedPrompt(String prompt, {bool addBos = true}) {
    _requireContext();
    BitNetLog.log('ffi',
        'bn_prompt len=${prompt.length} addBos=$addBos');
    using((arena) {
      final promptPtr = prompt.toNativeUtf8(allocator: arena).cast<Char>();
      final errPtr = arena<Pointer<Char>>();

      final rc = _b.bn_prompt(
        _ctx,
        promptPtr,
        addBos ? 1 : 0,
        errPtr,
      );

      if (rc != 0) {
        final nativeErr = _readAndFreeErr(errPtr.value);
        throw BitNetInferenceException(
          message: 'bn_prompt failed',
          nativeError: nativeErr,
        );
      }
    });
  }

  /// Runs one auto-regressive step and returns the next token piece.
  ///
  /// Returns `null` when generation is complete (EOS or context full).
  ///
  /// Throws [BitNetInferenceException] on native error.
  String? nextToken() {
    _requireContext();
    return using((arena) {
      final donePtr = arena<Int32>()..value = 0;
      final errPtr = arena<Pointer<Char>>();

      final tokenPtr = _b.bn_next_token(_ctx, donePtr, errPtr);

      if (tokenPtr == nullptr) {
        if (donePtr.value == 1) {
          // EOS — normal completion.
          return null;
        }
        // Error path.
        final nativeErr = _readAndFreeErr(errPtr.value);
        throw BitNetInferenceException(
          message: 'bn_next_token failed',
          nativeError: nativeErr,
        );
      }

      // Read the piece, then immediately free the native string.
      final piece = tokenPtr.cast<Utf8>().toDartString();
      _b.bn_free_string(tokenPtr);
      return piece;
    });
  }

  /// Returns the context window size of the loaded model, or -1 if not loaded.
  int contextSize() {
    if (_ctx == nullptr) return -1;
    return _b.bn_context_size(_ctx);
  }

  /// Returns a model metadata string by GGUF key, or null if not found.
  String? meta(String key) {
    if (_ctx == nullptr) return null;
    return using((arena) {
      final keyPtr = key.toNativeUtf8(allocator: arena).cast<Char>();
      final valPtr = _b.bn_meta(_ctx, keyPtr);
      if (valPtr == nullptr) return null;
      return valPtr.cast<Utf8>().toDartString();
    });
  }

  /// Frees the native context. Safe to call multiple times.
  void freeContext() {
    if (_ctx == nullptr) return;
    _b.bn_free(_ctx);
    _ctx = nullptr;
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  void _requireContext() {
    if (_ctx == nullptr) {
      throw const BitNetNotLoadedException();
    }
  }

  /// Reads a native C string into Dart, then frees it. Returns null if ptr is null.
  String? _readAndFreeErr(Pointer<Char> ptr) {
    if (ptr == nullptr) return null;
    final msg = ptr.cast<Utf8>().toDartString();
    _b.bn_free_string(ptr);
    return msg;
  }
}
