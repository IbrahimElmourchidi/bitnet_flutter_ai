import 'dart:io';
import 'dart:isolate';

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
/// A successful SHA-256 check writes a `.verified` stamp beside the model so
/// later loads skip re-hashing a multi-gigabyte file (see [_isStampValid]).
abstract final class ModelCache {
  ModelCache._();

  /// Minimum change in download progress before [ensureModel] notifies
  /// `onProgress` again. The HTTP stream delivers chunks of a few tens of KB,
  /// which is ~20k callbacks for a 1.1 GB model — enough to drive a rebuild per
  /// chunk in a naive UI. 0.5% caps it at 200 callbacks per download.
  static const double _progressEpsilon = 0.005;

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
    final stampFile = File(p.join(dir.path, '${model.ggufFileName}.verified'));

    // Fast path: file exists — verify hash, return if good.
    if (modelFile.existsSync()) {
      if (!model.hasPinnedHash) {
        // ignore: avoid_print
        print('[bitnet_flutter_ai] WARNING: SHA-256 not pinned for '
            '${model.ggufFileName}. Skipping verification.');
        return modelFile.path;
      }
      // Skip re-hashing when this exact file already passed. Hashing a 1.1 GB
      // GGUF costs seconds of CPU on every single load otherwise, and it used
      // to run on the calling isolate — i.e. jank on the UI thread at startup.
      if (await _isStampValid(stampFile, modelFile, model.ggufSha256)) {
        return modelFile.path;
      }
      final valid = await _verifySha256(modelFile, model.ggufSha256);
      if (valid) {
        await _writeStamp(stampFile, modelFile, model.ggufSha256);
        return modelFile.path;
      }
      // Hash mismatch — _verifySha256 already deleted the corrupt file.
      await _deleteIfExists(stampFile);
    }

    // A stale stamp must never outlive the file it describes.
    await _deleteIfExists(stampFile);

    // Download (with optional resume). Only close the client when we own it —
    // a caller-supplied one stays the caller's to manage.
    final httpClient = client ?? http.Client();
    try {
      await _download(
        model: model,
        tempFile: tempFile,
        modelFile: modelFile,
        onProgress: onProgress,
        client: httpClient,
      );
    } finally {
      if (client == null) httpClient.close();
    }

    // Post-download verification.
    if (model.hasPinnedHash) {
      final valid = await _verifySha256(modelFile, model.ggufSha256);
      if (!valid) {
        // _verifySha256 already deleted the file.
        throw BitNetHashMismatchException(model.ggufSha256, '<computed>');
      }
      await _writeStamp(stampFile, modelFile, model.ggufSha256);
    }

    return modelFile.path;
  }

  /// Returns the absolute path where [model] would be stored, whether or not
  /// the file currently exists.
  static Future<String> modelPath(BitNetModelInfo model) async {
    final dir = await _cacheDir();
    return p.join(dir.path, model.ggufFileName);
  }

  /// Whether [model]'s GGUF is already fully downloaded on disk.
  ///
  /// This checks only for the presence of the completed file — a `.part`
  /// in-progress download does not count as cached. It does NOT re-verify the
  /// SHA-256 (that would read the whole file); use [ensureModel] when you need
  /// the integrity guarantee. Intended for cheap UI hints such as a
  /// "Cached / Needs download" badge in a model picker.
  static Future<bool> isCached(BitNetModelInfo model) async {
    final dir = await _cacheDir();
    return File(p.join(dir.path, model.ggufFileName)).exists();
  }

  /// Whether a resumable partial download (`.part`) exists for [model] without
  /// the completed file being present. Useful to surface a "Resume download"
  /// affordance distinct from a fresh download.
  static Future<bool> hasPartialDownload(BitNetModelInfo model) async {
    final dir = await _cacheDir();
    final done = File(p.join(dir.path, model.ggufFileName));
    if (await done.exists()) return false;
    return File(p.join(dir.path, '${model.ggufFileName}.part')).exists();
  }

  /// Returns the on-disk cache status of every model in [models] in one pass,
  /// keyed by [BitNetModelInfo.id]. Convenience for populating a model list UI
  /// without awaiting each [isCached] call separately.
  static Future<Map<String, bool>> cachedStatus(
    Iterable<BitNetModelInfo> models,
  ) async {
    final dir = await _cacheDir();
    final entries = await Future.wait(models.map((m) async {
      final cached = await File(p.join(dir.path, m.ggufFileName)).exists();
      return MapEntry(m.id, cached);
    }));
    return Map<String, bool>.fromEntries(entries);
  }

  /// Deletes the model file, any in-progress `.part` file, and the
  /// verification stamp.
  static Future<void> clearModel(BitNetModelInfo model) async {
    final dir = await _cacheDir();
    await _deleteIfExists(File(p.join(dir.path, model.ggufFileName)));
    await _deleteIfExists(File(p.join(dir.path, '${model.ggufFileName}.part')));
    await _deleteIfExists(
        File(p.join(dir.path, '${model.ggufFileName}.verified')));
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
    // Throttled so a consumer can drive a progress bar straight off this
    // callback without rebuilding once per network chunk (see _progressEpsilon).
    var lastReported = -1.0;

    try {
      await for (final chunk in response.stream) {
        sink.add(chunk);
        receivedBytes += chunk.length;
        if (totalBytes > 0) {
          final fraction =
              ((startBytes + receivedBytes) / totalBytes).clamp(0.0, 1.0);
          if (fraction - lastReported >= _progressEpsilon || fraction >= 1.0) {
            lastReported = fraction;
            onProgress(fraction);
          }
        }
      }
    } catch (e) {
      await sink.close();
      throw BitNetDownloadException('Stream error during model download: $e');
    }

    await sink.close();
    // Always land on a final value, even when the byte total was unknown or the
    // last chunk fell inside the throttle window.
    if (lastReported < 1.0) onProgress(1.0);

    // Atomic rename: only happens when download completed successfully.
    await tempFile.rename(modelFile.path);
  }

  // ---------------------------------------------------------------------------
  // SHA-256 verification
  // ---------------------------------------------------------------------------

  /// Returns true if the file matches [expectedHex].
  /// Deletes the file and returns false on mismatch.
  ///
  /// The hash runs on a short-lived background isolate: SHA-256 over a
  /// multi-gigabyte file is pure Dart CPU work, and on the calling isolate it
  /// blocks whatever runs there — for the default engine, the UI thread.
  static Future<bool> _verifySha256(File file, String expectedHex) async {
    // Bind the path to a local so the isolate closure captures only a String —
    // a File instance is not guaranteed to be sendable.
    final path = file.path;
    final actual = await Isolate.run(() => _hashFile(path));
    if (actual == expectedHex) return true;
    await _deleteIfExists(file);
    return false;
  }

  /// Streams the file at [path] through SHA-256 and returns the hex digest.
  static Future<String> _hashFile(String path) async {
    final digest = await sha256.bind(File(path).openRead()).first;
    return digest.toString();
  }

  // ---------------------------------------------------------------------------
  // Verification stamp
  // ---------------------------------------------------------------------------
  //
  // Records which exact bytes passed verification, so a later load can trust
  // the file without re-reading it. Keyed on the expected hash plus the file's
  // observed length and modification time: any of the three changing means the
  // file on disk is no longer the one that was verified, and we re-hash.

  static Future<bool> _isStampValid(
    File stampFile,
    File modelFile,
    String expectedHex,
  ) async {
    try {
      if (!stampFile.existsSync()) return false;
      final stat = modelFile.statSync();
      return await stampFile.readAsString() ==
          _stampContent(expectedHex, stat.size, stat.modified);
    } catch (_) {
      // An unreadable or malformed stamp is not a failure — just fall back to
      // hashing, which is always authoritative.
      return false;
    }
  }

  static Future<void> _writeStamp(
    File stampFile,
    File modelFile,
    String expectedHex,
  ) async {
    try {
      final stat = modelFile.statSync();
      await stampFile
          .writeAsString(_stampContent(expectedHex, stat.size, stat.modified));
    } catch (_) {
      // Best-effort only: failing to write the stamp costs a re-hash next
      // load, which is never a correctness problem.
    }
  }

  static String _stampContent(String hex, int size, DateTime modified) =>
      '$hex\n$size\n${modified.millisecondsSinceEpoch}\n';

  static Future<void> _deleteIfExists(File file) async {
    try {
      if (file.existsSync()) await file.delete();
    } catch (_) {
      // Racing with another delete is harmless.
    }
  }

  // ---------------------------------------------------------------------------
  // Path helpers
  // ---------------------------------------------------------------------------

  static Future<Directory> _cacheDir() async {
    final support = await getApplicationSupportDirectory();
    return Directory(p.join(support.path, 'bitnet_flutter_ai'));
  }
}
