import 'dart:io';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

/// Builds a [BitNetModelInfo] whose pinned SHA-256 matches [bytes].
/// Allows the cache tests to use small synthetic fixtures while still
/// exercising the real hash verification path.
BitNetModelInfo _modelForBytes(List<int> bytes, {String? sha}) {
  final digest = sha ?? sha256.convert(bytes).toString();
  return BitNetModelInfo.forTest(
    id: 'test/fake-model',
    displayName: 'Fake Test Model',
    ggufFileName: 'ggml-model-i2_s.gguf',
    ggufSizeBytes: bytes.length,
    ggufSha256: digest,
    ggufDownloadUrl: 'https://example.test/fake.gguf',
    minimumRamBytes: 0,
    contextLength: 128,
  );
}

void main() {
  late Directory tempDir;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('bitnet_cache_test_');
    PathProviderPlatform.instance = _FakePathProvider(tempDir.path);
  });

  tearDown(() async {
    await tempDir.delete(recursive: true);
  });

  test('ensureModel skips hash check when SHA is placeholder', () async {
    const placeholderModel = BitNetModelInfo.forTest(
      id: 'test/fake-model',
      displayName: 'Fake Test Model',
      ggufFileName: 'ggml-model-i2_s.gguf',
      ggufSizeBytes: 3,
      ggufSha256: 'PLACEHOLDER_PIN_AFTER_FIRST_DOWNLOAD',
      ggufDownloadUrl: 'https://example.test/fake.gguf',
      minimumRamBytes: 0,
      contextLength: 128,
    );

    // Arrange: pre-create a fake model file.
    final modelFile = File(
      p.join(tempDir.path, 'bitnet_flutter_ai', placeholderModel.ggufFileName),
    );
    await modelFile.parent.create(recursive: true);
    await modelFile.writeAsBytes([1, 2, 3]);

    // Act: should return immediately without downloading.
    final client = MockClient((_) async => throw Exception('should not call'));
    final path = await ModelCache.ensureModel(
      placeholderModel,
      onProgress: (_) {},
      client: client,
    );

    expect(path, modelFile.path);
  });

  test('ensureModel downloads file when absent', () async {
    final fakeBytes = List<int>.generate(100, (i) => i);
    final model = _modelForBytes(fakeBytes);

    final client = MockClient((request) async {
      return http.Response.bytes(
        fakeBytes,
        200,
        headers: {'content-length': '${fakeBytes.length}'},
      );
    });

    double? lastProgress;
    final path = await ModelCache.ensureModel(
      model,
      onProgress: (p) => lastProgress = p,
      client: client,
    );

    expect(File(path).existsSync(), isTrue);
    expect(File(path).lengthSync(), fakeBytes.length);
    expect(lastProgress, isNotNull);
  });

  test('ensureModel sends Range header when .part file exists', () async {
    final fullBytes = [0, 1, 2, 3, 4, 5];
    final model = _modelForBytes(fullBytes);

    // Pre-create a partial file with the first 3 bytes.
    final partFile = File(p.join(
      tempDir.path,
      'bitnet_flutter_ai',
      '${model.ggufFileName}.part',
    ));
    await partFile.parent.create(recursive: true);
    await partFile.writeAsBytes(fullBytes.sublist(0, 3));

    String? rangeHeader;
    final client = MockClient((request) async {
      rangeHeader = request.headers['range'];
      return http.Response.bytes(
        fullBytes.sublist(3), // remaining bytes
        206,
        headers: {'content-length': '3'},
      );
    });

    await ModelCache.ensureModel(
      model,
      onProgress: (_) {},
      client: client,
    );

    expect(rangeHeader, 'bytes=3-');
  });

  test('clearModel deletes model and part files', () async {
    final dir = Directory(p.join(tempDir.path, 'bitnet_flutter_ai'));
    await dir.create(recursive: true);

    final modelFile = File(p.join(dir.path, BitNetCatalog.bitnet2B4T.ggufFileName));
    final partFile = File(p.join(dir.path, '${BitNetCatalog.bitnet2B4T.ggufFileName}.part'));
    await modelFile.writeAsBytes([1]);
    await partFile.writeAsBytes([1]);

    await ModelCache.clearModel(BitNetCatalog.bitnet2B4T);

    expect(modelFile.existsSync(), isFalse);
    expect(partFile.existsSync(), isFalse);
  });

  group('download edge cases', () {
    File part(BitNetModelInfo m) =>
        File(p.join(tempDir.path, 'bitnet_flutter_ai', '${m.ggufFileName}.part'));
    File done(BitNetModelInfo m) =>
        File(p.join(tempDir.path, 'bitnet_flutter_ai', m.ggufFileName));

    test('a connection that ends early keeps the .part for resuming', () async {
      final bytes = List<int>.generate(100, (i) => i);
      final model = _modelForBytes(bytes);
      // Server promises 100 bytes, closes cleanly after 40.
      final client = MockClient.streaming((req, _) async => http.StreamedResponse(
            Stream.value(bytes.sublist(0, 40)),
            200,
            contentLength: 100,
          ));
      await expectLater(
        ModelCache.ensureModel(model, onProgress: (_) {}, client: client),
        throwsA(isA<BitNetDownloadException>()),
      );
      expect(done(model).existsSync(), isFalse);
      expect(part(model).lengthSync(), 40, reason: 'resume from byte 40 next time');

      // The retry resumes from byte 40 and completes.
      String? range;
      final resume = MockClient((req) async {
        range = req.headers['range'];
        return http.Response.bytes(bytes.sublist(40), 206,
            headers: {'content-range': 'bytes 40-99/100'});
      });
      final path =
          await ModelCache.ensureModel(model, onProgress: (_) {}, client: resume);
      expect(range, 'bytes=40-');
      expect(File(path).readAsBytesSync(), bytes);
    });

    test('HTTP 416 on a complete .part promotes and verifies it', () async {
      final bytes = List<int>.generate(64, (i) => i * 3 % 256);
      final model = _modelForBytes(bytes);
      await part(model).parent.create(recursive: true);
      await part(model).writeAsBytes(bytes);
      final client = MockClient((_) async => http.Response('', 416));
      final path =
          await ModelCache.ensureModel(model, onProgress: (_) {}, client: client);
      expect(File(path).readAsBytesSync(), bytes);
      expect(part(model).existsSync(), isFalse);
    });

    test('a 206 for the wrong offset discards the .part', () async {
      final bytes = List<int>.generate(10, (i) => i);
      final model = _modelForBytes(bytes);
      await part(model).parent.create(recursive: true);
      await part(model).writeAsBytes(bytes.sublist(0, 4));
      final client = MockClient((_) async => http.Response.bytes(
          bytes.sublist(2), 206,
          headers: {'content-range': 'bytes 2-9/10'}));
      await expectLater(
        ModelCache.ensureModel(model, onProgress: (_) {}, client: client),
        throwsA(isA<BitNetDownloadException>()),
      );
      expect(part(model).existsSync(), isFalse);
    });

    test('hash mismatch reports the actual digest', () async {
      final bytes = [9, 9, 9];
      final model = _modelForBytes(bytes, sha: 'a' * 64);
      final client = MockClient((_) async => http.Response.bytes(bytes, 200));
      await expectLater(
        ModelCache.ensureModel(model, onProgress: (_) {}, client: client),
        throwsA(isA<BitNetHashMismatchException>().having(
            (e) => e.actualHash, 'actualHash', sha256.convert(bytes).toString())),
      );
    });

    test('concurrent ensureModel calls share one download', () async {
      final bytes = List<int>.generate(50, (i) => i);
      final model = _modelForBytes(bytes);
      var requests = 0;
      final client = MockClient((_) async {
        requests++;
        await Future<void>.delayed(const Duration(milliseconds: 20));
        return http.Response.bytes(bytes, 200);
      });
      final paths = await Future.wait([
        ModelCache.ensureModel(model, onProgress: (_) {}, client: client),
        ModelCache.ensureModel(model, onProgress: (_) {}, client: client),
      ]);
      expect(requests, 1);
      expect(paths[0], paths[1]);
    });
  });
}

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

class _FakePathProvider extends Fake
    with MockPlatformInterfaceMixin
    implements PathProviderPlatform {
  final String _basePath;
  _FakePathProvider(this._basePath);

  @override
  Future<String?> getApplicationSupportPath() async => _basePath;

  @override
  Future<String?> getTemporaryPath() async => _basePath;

  @override
  Future<String?> getApplicationDocumentsPath() async => _basePath;

  @override
  Future<String?> getDownloadsPath() async => _basePath;

  @override
  Future<List<String>?> getExternalStoragePaths({StorageDirectory? type}) async => null;

  @override
  Future<String?> getExternalStoragePath() async => null;

  @override
  Future<String?> getLibraryPath() async => _basePath;
}
