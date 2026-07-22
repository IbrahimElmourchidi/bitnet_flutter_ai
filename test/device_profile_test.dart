import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('DeviceProfile', () {
    test('canRun and isComfortableFor honour RAM thresholds', () {
      const lowRam = DeviceProfile(
        os: 'android',
        abi: 'androidarm64',
        cpuCores: 8,
        totalRamBytes: 2 * 1024 * 1024 * 1024,
      );
      expect(lowRam.canRun(BitNetCatalog.bitnet2B4T), isFalse);
      expect(lowRam.isComfortableFor(BitNetCatalog.bitnet2B4T), isFalse);

      const okRam = DeviceProfile(
        os: 'android',
        abi: 'androidarm64',
        cpuCores: 8,
        totalRamBytes: 6 * 1024 * 1024 * 1024,
      );
      expect(okRam.canRun(BitNetCatalog.bitnet2B4T), isTrue);
    });

    test('estimateTokensPerSecond returns positive, plausible values', () {
      const desktopX64 = DeviceProfile(
        os: 'linux',
        abi: 'linuxx64',
        cpuCores: 8,
        totalRamBytes: 16 * 1024 * 1024 * 1024,
      );
      final est = desktopX64.estimateTokensPerSecond(BitNetCatalog.bitnet2B4T);
      // Reference anchor was 12.5 tok/s; same class device should be near it.
      expect(est, greaterThan(8.0));
      expect(est, lessThan(20.0));
    });

    test('mobile arm64 is slower than desktop x64 for the same cores', () {
      const desktop = DeviceProfile(
        os: 'linux',
        abi: 'linuxx64',
        cpuCores: 8,
        totalRamBytes: 8 * 1024 * 1024 * 1024,
      );
      const mobile = DeviceProfile(
        os: 'android',
        abi: 'androidarm64',
        cpuCores: 8,
        totalRamBytes: 8 * 1024 * 1024 * 1024,
      );
      expect(
        mobile.estimateTokensPerSecond(BitNetCatalog.bitnet2B4T),
        lessThan(desktop.estimateTokensPerSecond(BitNetCatalog.bitnet2B4T)),
      );
    });

    test('estimate is clamped to a positive minimum even on weak devices', () {
      const tinyArm = DeviceProfile(
        os: 'android',
        abi: 'androidarm64',
        cpuCores: 2,
        totalRamBytes: 3 * 1024 * 1024 * 1024,
      );
      final est = tinyArm.estimateTokensPerSecond(BitNetCatalog.bitnet2B4T);
      expect(est, greaterThanOrEqualTo(0.5));
    });
  });
}
