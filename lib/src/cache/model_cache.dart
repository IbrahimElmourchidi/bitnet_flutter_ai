import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart';

/// Downloads, verifies, and caches the GGUF model file on disk.
///
/// Storage location: `<applicationSupportDirectory>/bitnet_flutter_ai/<filename>`
/// Partial downloads use a `.part` suffix and are resumed on retry.
abstract final class ModelCache {
  ModelCache._();

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Ensures the model file exists on disk and its SHA-256 matches.
  ///
  /// - If the file is already present and valid, returns the path immediately.
  /// - If the file is present but the hash is wrong, deletes it and re-downloads.
  /// - If a `.part` file exists, resumes the download from where it left off.
  ///
  /// [onProgress] receives values in [0.0, 1.0] as bytes arrive.
  /// Pass a custom [client] for testing; production code uses the default client.
  ///
  /// Returns the absolute path to the verified model file.
  static Future<String> ensureModel(
    BitNetModelInfo model, {
    required void Function(double progress) onProgress,
    http.Client? client,
  }) async {
    final dir = await _cacheDir();
    await dir.create(recursive: true);

    final modelFile = File(p.join(dir.path, model.ggufFileName));
    final tempFile = File(p.join(dir.path, '${model.ggufFileName}.part'));

    // Fast path: file exists — verify hash, return if good.
    if (modelFile.existsSync()) {
      if (!model.hasPinnedHash) {
        // ignore: avoid_print
        print('[bitnet_flutter_ai] WARNING: SHA-256 not pinned for '
            '${model.ggufFileName}. Skipping verification.');
        return modelFile.path;
      }
      final valid = await _verifySha256(modelFile, model.ggufSha256);
      if (valid) return modelFile.path;
      // Hash mismatch — delete corrupt file and re-download.
      await modelFile.delete();
    }

    // Download (with optional resume).
    await _download(
      model: model,
      tempFile: tempFile,
      modelFile: modelFile,
      onProgress: onProgress,
      client: client ?? http.Client(),
    );

    // Post-download verification.
    if (model.hasPinnedHash) {
      final valid = await _verifySha256(modelFile, model.ggufSha256);
      if (!valid) {
        // _verifySha256 already deleted the file.
        throw BitNetHashMismatchException(model.ggufSha256, '<computed>');
      }
    }

    return modelFile.path;
  }

  /// Returns the absolute path where [model] would be stored, whether or not
  /// the file currently exists.
  static Future<String> modelPath(BitNetModelInfo model) async {
    final dir = await _cacheDir();
    return p.join(dir.path, model.ggufFileName);
  }

  /// Deletes the model file and any in-progress `.part` file.
  static Future<void> clearModel(BitNetModelInfo model) async {
    final dir = await _cacheDir();
    final modelFile = File(p.join(dir.path, model.ggufFileName));
    final tempFile = File(p.join(dir.path, '${model.ggufFileName}.part'));
    if (modelFile.existsSync()) await modelFile.delete();
    if (tempFile.existsSync()) await tempFile.delete();
  }

  // ---------------------------------------------------------------------------
  // Download
  // ---------------------------------------------------------------------------

  static Future<void> _download({
    required BitNetModelInfo model,
    required File tempFile,
    required File modelFile,
    required void Function(double) onProgress,
    required http.Client client,
  }) async {
    final existingBytes = tempFile.existsSync() ? tempFile.lengthSync() : 0;
    final totalBytes = model.ggufSizeBytes;

    final request = http.Request('GET', Uri.parse(model.ggufDownloadUrl));
    if (existingBytes > 0) {
      request.headers['Range'] = 'bytes=$existingBytes-';
    }

    final http.StreamedResponse response;
    try {
      response = await client.send(request);
    } catch (e) {
      throw BitNetDownloadException('Network error during model download: $e');
    }

    // 206 = partial content (resume), 200 = full download
    if (response.statusCode != 200 && response.statusCode != 206) {
      throw BitNetDownloadException(
        'Unexpected HTTP status: ${response.statusCode}',
        httpStatusCode: response.statusCode,
      );
    }

    // If server ignored Range header and returned 200, restart from zero.
    final isResume = response.statusCode == 206;
    final sink = tempFile.openWrite(
      mode: isResume ? FileMode.append : FileMode.write,
    );
    final startBytes = isResume ? existingBytes : 0;
    var receivedBytes = 0;

    try {
      await for (final chunk in response.stream) {
        sink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes > 0) {
          onProgress(
            ((startBytes + receivedBytes) / totalBytes).clamp(0.0, 1.0),
          );
        }
      }
    } catch (e) {
      await sink.close();
      throw BitNetDownloadException('Stream error during model download: $e');
    }

    await sink.close();

    // Atomic rename: only happens when download completed successfully.
    await tempFile.rename(modelFile.path);
  }

  // ---------------------------------------------------------------------------
  // SHA-256 verification
  // ---------------------------------------------------------------------------

  /// Returns true if the file matches [expectedHex].
  /// Deletes the file and returns false on mismatch.
  static Future<bool> _verifySha256(File file, String expectedHex) async {
    final input = file.openRead();
    final digest = await sha256.bind(input).first;
    final actual = digest.toString();
    if (actual == expectedHex) return true;
    await file.delete();
    return false;
  }

  // ---------------------------------------------------------------------------
  // Path helpers
  // ---------------------------------------------------------------------------

  static Future<Directory> _cacheDir() async {
    final support = await getApplicationSupportDirectory();
    return Directory(p.join(support.path, 'bitnet_flutter_ai'));
  }
}
