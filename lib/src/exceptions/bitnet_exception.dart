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

/// A native inference call (bn_prompt or bn_next_token) returned an error.
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

/// [BitNetEngine.generate] was called before [BitNetEngine.load].
final class BitNetNotLoadedException extends BitNetException {
  const BitNetNotLoadedException()
      : super('BitNetEngine.load() must be called and awaited before generate().');
}

/// The inference isolate exited unexpectedly.
final class BitNetIsolateException extends BitNetException {
  const BitNetIsolateException(super.message);
}

/// [BitNetEngine.generate] was called while another generation is still in
/// flight on the same engine.
///
/// A single engine owns ONE native context; two overlapping generations would
/// interleave `bn_prompt` / `bn_next_token` calls on it and split one decode
/// stream between two consumers, corrupting both. Await or cancel the active
/// stream (`BitNetSession.stop()`) before starting the next one.
final class BitNetBusyException extends BitNetException {
  const BitNetBusyException()
      : super('A generation is already in progress on this engine. Await or '
            'cancel it before calling generate() again.');
}
