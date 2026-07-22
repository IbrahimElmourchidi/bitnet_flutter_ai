import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('BitNetModelInfo', () {
    test('bitnet2B4T has expected metadata', () {
      const model = BitNetCatalog.bitnet2B4T;

      expect(model.id, 'microsoft/bitnet-b1.58-2B-4T');
      expect(model.ggufFileName, 'ggml-model-i2_s.gguf');
      expect(model.contextLength, 4096);
      expect(model.minimumRamBytes, 3072 * 1024 * 1024);
      expect(model.ggufDownloadUrl, contains('huggingface.co'));
    });

    test('hasPinnedHash is true for the canonical 2B-4T model', () {
      expect(BitNetCatalog.bitnet2B4T.hasPinnedHash, isTrue);
      expect(BitNetCatalog.bitnet2B4T.ggufSha256.length, 64);
    });
  });

  group('BitNetCatalog', () {
    test('all contains the canonical 2B-4T model', () {
      expect(BitNetCatalog.all, contains(BitNetCatalog.bitnet2B4T));
      expect(BitNetCatalog.defaultModel, BitNetCatalog.bitnet2B4T);
    });

    test('byId returns the matching model and null otherwise', () {
      expect(
        BitNetCatalog.byId('microsoft/bitnet-b1.58-2B-4T'),
        BitNetCatalog.bitnet2B4T,
      );
      expect(BitNetCatalog.byId('does/not-exist'), isNull);
    });

    test('withCapability filters by capability tag', () {
      expect(
        BitNetCatalog.withCapability(BitNetCapability.summarization),
        contains(BitNetCatalog.bitnet2B4T),
      );
      // The larger Falcon3 models advertise code.
      expect(
        BitNetCatalog.withCapability(BitNetCapability.code),
        contains(BitNetCatalog.falcon3_7bInstruct),
      );
    });

    test('compatibleWith respects minimum and recommended RAM', () {
      // 1 GB → below every model's floor → no fit.
      expect(
        BitNetCatalog.compatibleWith(1 * 1024 * 1024 * 1024),
        isEmpty,
      );
      // 2 GB → the 2 GB-floor models fit (Falcon-E-1B, Bonsai 1.7B); nothing
      // with a higher floor does.
      final at2gb = BitNetCatalog.compatibleWith(2 * 1024 * 1024 * 1024);
      expect(at2gb, contains(BitNetCatalog.falconE_1bInstruct));
      expect(at2gb, contains(BitNetCatalog.bonsai1_7b));
      expect(at2gb, isNot(contains(BitNetCatalog.bitnet2B4T))); // 3 GB floor
    });

    test('compatibleWith prefers the largest model that fits', () {
      // 3.5 GB → meets 2B-4T's minimum.
      final low =
          BitNetCatalog.compatibleWith(3 * 1024 * 1024 * 1024 + 512 * 1024 * 1024);
      expect(low, contains(BitNetCatalog.bitnet2B4T));
      // 8 GB → every model's recommended RAM met → largest (10B) preferred.
      final high = BitNetCatalog.compatibleWith(8 * 1024 * 1024 * 1024);
      expect(high.first, BitNetCatalog.falcon3_10bInstruct);
    });

    test('every catalog entry has a unique on-disk filename', () {
      // ModelCache keys the cache path by ggufFileName in one shared dir, so a
      // duplicate would silently collide two models on disk.
      final names = BitNetCatalog.all.map((m) => m.ggufFileName).toList();
      expect(names.toSet().length, names.length);
    });

    test('every catalog entry has a pinned 64-char SHA-256 and a HF URL', () {
      for (final m in BitNetCatalog.all) {
        expect(m.hasPinnedHash, isTrue, reason: '${m.id} hash not pinned');
        expect(m.ggufSha256.length, 64, reason: '${m.id} bad hash length');
        expect(m.ggufDownloadUrl, contains('huggingface.co'),
            reason: '${m.id} bad URL');
        expect(m.minimumRamBytes, lessThanOrEqualTo(m.recommendedRamBytes),
            reason: '${m.id} min > recommended');
      }
    });
  });

  group('DeviceInspector threshold', () {
    test('meetsMinimumRam is false at 2 GB', () async {
      const inspector = _FakeInspector(2 * 1024 * 1024 * 1024);
      expect(await inspector.meetsMinimumRam(), isFalse);
    });

    test('meetsMinimumRam is true at 4 GB', () async {
      const inspector = _FakeInspector(4 * 1024 * 1024 * 1024);
      expect(await inspector.meetsMinimumRam(), isTrue);
    });

    test('meetsMinimumRam is false just below threshold', () async {
      const inspector = _FakeInspector(3072 * 1024 * 1024 - 1);
      expect(await inspector.meetsMinimumRam(), isFalse);
    });

    test('meetsMinimumRam is true at exact threshold', () async {
      const inspector = _FakeInspector(3072 * 1024 * 1024);
      expect(await inspector.meetsMinimumRam(), isTrue);
    });
  });

  group('BitNetException hierarchy', () {
    test('BitNetUnsupportedDeviceException message contains RAM value', () {
      const e = BitNetUnsupportedDeviceException(2048 * 1024 * 1024);
      expect(e.message, contains('2048 MB'));
    });

    test('BitNetHashMismatchException carries both hashes', () {
      final e = BitNetHashMismatchException('expected123', 'actual456');
      expect(e.expectedHash, 'expected123');
      expect(e.actualHash, 'actual456');
      expect(e.message, contains('expected123'));
      expect(e.message, contains('actual456'));
    });

    test('BitNetNotLoadedException has descriptive message', () {
      const e = BitNetNotLoadedException();
      expect(e.message, contains('load()'));
    });

    test('BitNetDownloadException carries HTTP status', () {
      const e = BitNetDownloadException('Server error', httpStatusCode: 503);
      expect(e.httpStatusCode, 503);
      expect(e.toString(), contains('503'));
    });
  });
}

/// Test double for [DeviceInspector] with a fixed RAM value.
class _FakeInspector extends DeviceInspector {
  final int _bytes;
  const _FakeInspector(this._bytes);

  @override
  Future<int> totalPhysicalRamBytes() async => _bytes;
}
