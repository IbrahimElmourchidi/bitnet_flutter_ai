import 'package:flutter/foundation.dart';

/// Immutable descriptor for a BitNet GGUF model artifact.
///
/// The only supported model for v1 is [BitNetModel.bitnet2B4T].
@immutable
final class BitNetModel {
  /// Microsoft BitNet b1.58 2B-4T, I2_S GGUF quantization.
  ///
  /// Source: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
  ///
  /// **Important:** [ggufSha256] is a placeholder. Pin it to the actual
  /// SHA-256 of the downloaded file before publishing to pub.dev.
  /// While the placeholder sentinel is present, hash verification is skipped
  /// and a warning is printed instead.
  static const BitNetModel bitnet2B4T = BitNetModel._(
    id: 'microsoft/bitnet-b1.58-2B-4T',
    displayName: 'BitNet b1.58 2B-4T',
    ggufFileName: 'ggml-model-i2_s.gguf',
    // Approximate size — exact byte count confirmed on first download.
    ggufSizeBytes: 780000000,
    // TODO(maintainer): replace with the real SHA-256 after first CI download.
    // Run: sha256sum ggml-model-i2_s.gguf
    ggufSha256: 'PLACEHOLDER_PIN_AFTER_FIRST_DOWNLOAD',
    ggufDownloadUrl:
        'https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 3072 * 1024 * 1024,
    contextLength: 4096,
  );

  /// Unique model identifier, matching the HuggingFace repo slug.
  final String id;

  /// Human-readable display name.
  final String displayName;

  /// Filename used when storing the model on-device.
  final String ggufFileName;

  /// Approximate compressed file size in bytes. Used for progress reporting.
  final int ggufSizeBytes;

  /// Expected SHA-256 hex digest of the GGUF file.
  ///
  /// If this starts with `'PLACEHOLDER'`, verification is skipped.
  final String ggufSha256;

  /// Direct download URL for the GGUF file.
  final String ggufDownloadUrl;

  /// Minimum physical RAM required to load this model, in bytes.
  final int minimumRamBytes;

  /// Maximum context window supported by this model, in tokens.
  final int contextLength;

  /// Returns true when [ggufSha256] is a real hash and verification should run.
  bool get hasPinnedHash => !ggufSha256.startsWith('PLACEHOLDER');

  const BitNetModel._({
    required this.id,
    required this.displayName,
    required this.ggufFileName,
    required this.ggufSizeBytes,
    required this.ggufSha256,
    required this.ggufDownloadUrl,
    required this.minimumRamBytes,
    required this.contextLength,
  });

  @override
  String toString() => 'BitNetModel($displayName)';
}
