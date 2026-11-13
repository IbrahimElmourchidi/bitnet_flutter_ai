/// Base exception for all bitnet_flutter_ai errors.
sealed class BitNetException implements Exception {
  final String message;
  const BitNetException(this.message);

  @override
  String toString() => 'BitNetException($runtimeType): $message';
}

/// Device physical RAM is below the minimum required by the selected model.
final class BitNetUnsupportedDeviceException extends BitNetException {
  final int availableRamBytes;

  /// Minimum RAM the model requires, in bytes. `0` when unknown (legacy call).
  final int requiredRamBytes;

  const BitNetUnsupportedDeviceException(
    this.availableRamBytes, {
    this.requiredRamBytes = 0,
    String? modelName,
  }) : super(
          'Device RAM (${availableRamBytes ~/ (1024 * 1024)} MB) is below the '
          '${requiredRamBytes ~/ (1024 * 1024)} MB minimum required'
          '${modelName != null ? ' by $modelName' : ''}.',
        );
}

/// The current platform is not supported (e.g. Safari without WASM threads).
final class BitNetUnsupportedPlatformException extends BitNetException {
  const BitNetUnsupportedPlatformException(super.message);
}

/// The native shared library could not be loaded (DynamicLibrary.open failed).
final class BitNetLibraryLoadException extends BitNetException {
  const BitNetLibraryLoadException(super.message);
}

/// The native context initialisation failed (bn_init returned NULL).
final class BitNetInitException extends BitNetException {
  /// Error string from the native layer, if available.
  final String? nativeError;

  const BitNetInitException({required String message, this.nativeError})
      : super(message);

  @override
  String toString() =>
      'BitNetInitException: $message'
      '${nativeError != null ? ' (native: $nativeError)' : ''}';
}

/// A native inference call (bn_prompt, bn_next_token, bn_set_sampling)
/// returned an error.
final class BitNetInferenceException extends BitNetException {
  /// Error string from the native layer, if available.
  final String? nativeError;

  const BitNetInferenceException({required String message, this.nativeError})
      : super(message);

  @override
  String toString() =>
      'BitNetInferenceException: $message'
      '${nativeError != null ? ' (native: $nativeError)' : ''}';
}

/// The downloaded GGUF file SHA-256 does not match the expected hash.
/// The corrupt file is deleted before this exception is thrown.
final class BitNetHashMismatchException extends BitNetException {
  final String expectedHash;
  final String actualHash;

  BitNetHashMismatchException(this.expectedHash, this.actualHash)
      : super(
          'Model file SHA-256 mismatch. '
          'Expected: $expectedHash. Got: $actualHash. '
          'The file has been deleted and will be re-downloaded on the next call.',
        );
}

/// A network error occurred during model download.
final class BitNetDownloadException extends BitNetException {
  final int? httpStatusCode;

  const BitNetDownloadException(super.message, {this.httpStatusCode});

  @override
  String toString() =>
      'BitNetDownloadException: $message'
      '${httpStatusCode != null ? ' (HTTP $httpStatusCode)' : ''}';
}

/// A generation was requested before `BitNetEngine.load()` completed (or after
/// `dispose()`).
final class BitNetNotLoadedException extends BitNetException {
  const BitNetNotLoadedException()
      : super('BitNetEngine.load() must complete before generating.');
}

/// The inference isolate exited unexpectedly (e.g. a native crash surfaced as
/// an isolate error). The engine is unloaded; call `load()` again.
final class BitNetIsolateException extends BitNetException {
  const BitNetIsolateException(super.message);
}

/// The prompt does not fit in the model's context window (with room left to
/// generate). `BitNetSession.chat` trims old turns automatically and only
/// raises this when the newest message alone is too long.
final class BitNetPromptTooLongException extends BitNetException {
  const BitNetPromptTooLongException(super.message);

  @override
  String toString() => 'BitNetPromptTooLongException: $message';
}

/// `BitNetSession.extract` could not turn the model output into the requested
/// JSON shape (only possible when generation was cut short, e.g. by
/// `maxNewTokens` or cancellation — the grammar otherwise guarantees it).
final class BitNetExtractionException extends BitNetException {
  /// The raw model output.
  final String raw;

  const BitNetExtractionException(super.message, {required this.raw});

  @override
  String toString() => 'BitNetExtractionException: $message (raw: $raw)';
}
