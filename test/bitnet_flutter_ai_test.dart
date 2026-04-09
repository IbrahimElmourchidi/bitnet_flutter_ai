import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('BitNetModel', () {
    test('bitnet2B4T has expected metadata', () {
      const model = BitNetModel.bitnet2B4T;

      expect(model.id, 'microsoft/bitnet-b1.58-2B-4T');
      expect(model.ggufFileName, 'ggml-model-i2_s.gguf');
      expect(model.contextLength, 4096);
      expect(model.minimumRamBytes, 3072 * 1024 * 1024);
      expect(model.ggufDownloadUrl, contains('huggingface.co'));
    });

    test('hasPinnedHash returns false for placeholder', () {
      expect(BitNetModel.bitnet2B4T.hasPinnedHash, isFalse);
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
