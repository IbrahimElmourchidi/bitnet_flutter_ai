import 'dart:io';
import 'dart:typed_data';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

BitNetModelInfo _modelForBytes(List<int> bytes) => BitNetModelInfo.forTest(
      id: 'test/fake-model',
      displayName: 'Fake Test Model',
      ggufFileName: 'ggml-model-i2_s.gguf',
      ggufSizeBytes: bytes.length,
      ggufSha256: sha256.convert(bytes).toString(),
      ggufDownloadUrl: 'https://example.test/fake.gguf',
      minimumRamBytes: 0,
      contextLength: 128,
    );

class _FakePathProvider extends PathProviderPlatform
    with MockPlatformInterfaceMixin {
  _FakePathProvider(this.root);
  final String root;

  @override
  Future<String?> getApplicationSupportPath() async => root;
}

void main() {
  late Directory tempDir;
  late Directory cacheDir;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('bitnet_verify_test_');
    PathProviderPlatform.instance = _FakePathProvider(tempDir.path);
    cacheDir = Directory(p.join(tempDir.path, 'bitnet_flutter_ai'));
  });

  tearDown(() async {
    await tempDir.delete(recursive: true);
  });

  File stampFile(BitNetModelInfo m) =>
      File(p.join(cacheDir.path, '${m.ggufFileName}.verified'));
  File modelFile(BitNetModelInfo m) =>
      File(p.join(cacheDir.path, m.ggufFileName));

  /// Serves [bytes] once and records how many requests it saw.
  ({http.Client client, List<int> hits}) serving(List<int> bytes) {
    final hits = <int>[];
    final client = MockClient((req) async {
      hits.add(1);
      return http.Response.bytes(bytes, 200);
    });
    return (client: client, hits: hits);
  }

  group('verification stamp', () {
    test('a verified download writes a stamp and the next load skips the '
        're-download', () async {
      final bytes = List<int>.generate(2048, (i) => i % 251);
      final model = _modelForBytes(bytes);
      final server = serving(bytes);

      await ModelCache.ensureModel(model,
          onProgress: (_) {}, client: server.client);
      expect(stampFile(model).existsSync(), isTrue,
          reason: 'a passing hash check should record itself');
      expect(server.hits, hasLength(1));

      // Second load: file is present and stamped, so nothing is fetched again.
      await ModelCache.ensureModel(model,
          onProgress: (_) {}, client: server.client);
      expect(server.hits, hasLength(1));
    });

    test('a stamp for different bytes is rejected and the file re-verified',
        () async {
      final bytes = List<int>.generate(2048, (i) => i % 251);
      final model = _modelForBytes(bytes);
      final server = serving(bytes);
      await ModelCache.ensureModel(model,
          onProgress: (_) {}, client: server.client);

      // Corrupt the cached model behind the stamp's back. The stamp records
      // length + mtime, so it must no longer match.
      await modelFile(model).writeAsBytes(List<int>.filled(2048, 0));

      // Re-hashing now fails, the corrupt file is deleted, and a fresh copy is
      // downloaded — the stale stamp must not paper over the corruption.
      await ModelCache.ensureModel(model,
          onProgress: (_) {}, client: server.client);
      expect(server.hits, hasLength(2));
      expect(await modelFile(model).readAsBytes(), equals(Uint8List.fromList(bytes)));
    });

    test('clearModel removes the stamp along with the model', () async {
      final bytes = List<int>.generate(512, (i) => i % 251);
      final model = _modelForBytes(bytes);
      final server = serving(bytes);
      await ModelCache.ensureModel(model,
          onProgress: (_) {}, client: server.client);
      expect(stampFile(model).existsSync(), isTrue);

      await ModelCache.clearModel(model);
      expect(modelFile(model).existsSync(), isFalse);
      expect(stampFile(model).existsSync(), isFalse,
          reason: 'a stamp outliving its model would validate the next file '
              'that happens to land at the same path');
    });
  });

  group('progress reporting', () {
    test('is throttled well below one callback per chunk and still ends at 1.0',
        () async {
      // 400 KB delivered in 1 KB chunks: 400 chunks, but the throttle caps
      // callbacks at roughly 1/_progressEpsilon (200) plus the final 1.0.
      const chunkCount = 400;
      final bytes = List<int>.generate(chunkCount * 1024, (i) => i % 251);
      final model = _modelForBytes(bytes);

      final client = MockClient.streaming((req, _) async {
        Stream<List<int>> chunks() async* {
          for (var i = 0; i < chunkCount; i++) {
            yield bytes.sublist(i * 1024, (i + 1) * 1024);
          }
        }

        return http.StreamedResponse(chunks(), 200,
            contentLength: bytes.length);
      });

      final reported = <double>[];
      await ModelCache.ensureModel(model,
          onProgress: reported.add, client: client);

      expect(reported, isNotEmpty);
      expect(reported.length, lessThan(chunkCount),
          reason: 'a callback per network chunk drives a rebuild per chunk in '
              'a naive UI');
      expect(reported.last, 1.0, reason: 'progress must always land on 1.0');
      // Monotonic — a progress bar that jumps backwards looks broken.
      for (var i = 1; i < reported.length; i++) {
        expect(reported[i], greaterThanOrEqualTo(reported[i - 1]));
      }
    });
  });
}
