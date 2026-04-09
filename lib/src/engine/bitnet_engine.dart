import 'engine_native.dart'
    if (dart.library.html) 'engine_web.dart';

import '../model/bitnet_model.dart';

/// The public interface for driving BitNet inference.
///
/// Usage:
/// ```dart
/// final engine = BitNetEngine();
/// await engine.load(onProgress: (p) => print('${(p*100).toInt()}%'));
/// await for (final token in engine.generate('Hello')) {
///   stdout.write(token);
/// }
/// await engine.dispose();
/// ```
abstract interface class BitNetEngine {
  /// Downloads (if needed), verifies, and loads [model] into memory.
  ///
  /// [onProgress] receives values in [0.0, 1.0] during the download phase.
  /// Throws [BitNetUnsupportedDeviceException] if RAM < 3072 MB.
  /// Throws [BitNetInitException] if the native context cannot be created.
  Future<void> load({void Function(double progress)? onProgress});

  /// Whether the engine has been successfully loaded and is ready to generate.
  bool get isLoaded;

  /// Streams token pieces for [prompt] until EOS or [maxNewTokens] is reached.
  ///
  /// Throws [BitNetNotLoadedException] if called before [load].
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  });

  /// Signals the current generation to stop at the next token boundary.
  Future<void> cancelGeneration();

  /// Frees native resources and kills the inference isolate / WASM context.
  Future<void> dispose();

  /// Creates an engine for [model] using the correct implementation for the
  /// current platform (native isolate or WASM).
  factory BitNetEngine({BitNetModel model = BitNetModel.bitnet2B4T}) =>
      createEngine(model: model);
}
