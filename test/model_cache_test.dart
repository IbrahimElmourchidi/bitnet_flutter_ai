import 'dart:io';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

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
    // Arrange: pre-create a fake model file.
    final modelFile = File(
      p.join(tempDir.path, 'bitnet_flutter_ai', BitNetModel.bitnet2B4T.ggufFileName),
    );
    await modelFile.parent.create(recursive: true);
    await modelFile.writeAsBytes([1, 2, 3]);

    // Act: should return immediately without downloading.
    final client = MockClient((_) async => throw Exception('should not call'));
    final path = await ModelCache.ensureModel(
      BitNetModel.bitnet2B4T,
      onProgress: (_) {},
      client: client,
    );

    expect(path, modelFile.path);
  });

  test('ensureModel downloads file when absent', () async {
    final fakeBytes = List<int>.generate(100, (i) => i);
    final client = MockClient((request) async {
      return http.Response.bytes(
        fakeBytes,
        200,
        headers: {'content-length': '${fakeBytes.length}'},
      );
    });

    double? lastProgress;
    final path = await ModelCache.ensureModel(
      BitNetModel.bitnet2B4T,
      onProgress: (p) => lastProgress = p,
      client: client,
    );

    expect(File(path).existsSync(), isTrue);
    expect(File(path).lengthSync(), fakeBytes.length);
    expect(lastProgress, isNotNull);
  });

  test('ensureModel sends Range header when .part file exists', () async {
    // Pre-create a partial file.
    final partFile = File(p.join(
      tempDir.path,
      'bitnet_flutter_ai',
      '${BitNetModel.bitnet2B4T.ggufFileName}.part',
    ));
    await partFile.parent.create(recursive: true);
    await partFile.writeAsBytes([0, 1, 2]); // 3 bytes already downloaded

    String? rangeHeader;
    final client = MockClient((request) async {
      rangeHeader = request.headers['range'];
      return http.Response.bytes(
        [3, 4, 5], // remaining bytes
        206,
        headers: {'content-length': '3'},
      );
    });

    await ModelCache.ensureModel(
      BitNetModel.bitnet2B4T,
      onProgress: (_) {},
      client: client,
    );

    expect(rangeHeader, 'bytes=3-');
  });

  test('clearModel deletes model and part files', () async {
    final dir = Directory(p.join(tempDir.path, 'bitnet_flutter_ai'));
    await dir.create(recursive: true);

    final modelFile = File(p.join(dir.path, BitNetModel.bitnet2B4T.ggufFileName));
    final partFile = File(p.join(dir.path, '${BitNetModel.bitnet2B4T.ggufFileName}.part'));
    await modelFile.writeAsBytes([1]);
    await partFile.writeAsBytes([1]);

    await ModelCache.clearModel(BitNetModel.bitnet2B4T);

    expect(modelFile.existsSync(), isFalse);
    expect(partFile.existsSync(), isFalse);
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
