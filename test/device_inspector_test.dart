import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('DeviceInspector.meetsMinimumRam', () {
    test('returns false when RAM is 2 GB', () async {
      const inspector = _FakeInspector(2 * 1024 * 1024 * 1024);
      expect(await inspector.meetsMinimumRam(), isFalse);
    });

    test('returns true when RAM is 4 GB', () async {
      const inspector = _FakeInspector(4 * 1024 * 1024 * 1024);
      expect(await inspector.meetsMinimumRam(), isTrue);
    });

    test('returns false one byte below the 3072 MB threshold', () async {
      const inspector = _FakeInspector(3072 * 1024 * 1024 - 1);
      expect(await inspector.meetsMinimumRam(), isFalse);
    });

    test('returns true at exactly the 3072 MB threshold', () async {
      const inspector = _FakeInspector(3072 * 1024 * 1024);
      expect(await inspector.meetsMinimumRam(), isTrue);
    });

    test('returns false when RAM is 0 (fail-closed)', () async {
      const inspector = _FakeInspector(0);
      expect(await inspector.meetsMinimumRam(), isFalse);
    });
  });

  group('DeviceInspector.totalPhysicalRamBytes', () {
    test('returns the value provided by the implementation', () async {
      const expected = 8 * 1024 * 1024 * 1024;
      const inspector = _FakeInspector(expected);
      expect(await inspector.totalPhysicalRamBytes(), expected);
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
