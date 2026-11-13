// On-device checks for the native interop and the inference stack.
//
//   flutter test integration_test/on_device_test.dart -d <device>
//
// The inference test needs the BitNet 2B-4T GGUF already in the model cache
// (it is skipped otherwise). To pre-seed it without a download:
//   adb push ggml-model-i2_s.gguf /data/local/tmp/
//   adb shell run-as <applicationId> sh -c \
//     'mkdir -p no_backup/bitnet_flutter_ai && \
//      cp /data/local/tmp/ggml-model-i2_s.gguf no_backup/bitnet_flutter_ai/'
import 'dart:io';
import 'dart:isolate';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('DeviceInspector.health() reads real values', (_) async {
    final h = await DeviceInspector.instance.health();
    // ignore: avoid_print
    print('health: $h');
    expect(h.totalRamBytes, greaterThan(512 << 20));
    if (Platform.isAndroid || Platform.isIOS) {
      expect(h.availableRamBytes, isNotNull);
      expect(h.availableRamBytes!, lessThanOrEqualTo(h.totalRamBytes));
      expect(h.lowPowerMode, isNotNull);
    }
  });

  testWidgets('device queries work from a background isolate', (_) async {
    // Impossible with the old MethodChannel; fine with direct interop.
    final total = await Isolate.run(
        () => DeviceInspector.instance.totalPhysicalRamBytes());
    expect(total, greaterThan(512 << 20));
  });

  testWidgets('models are stored outside device backups', (_) async {
    final path = await ModelCache.modelPath(BitNetCatalog.defaultModel);
    // ignore: avoid_print
    print('model path: $path');
    if (Platform.isAndroid) expect(path, contains('/no_backup/'));
  });

  testWidgets('loads the model and extracts a transaction', (_) async {
    if (!await ModelCache.isCached(BitNetCatalog.defaultModel)) {
      markTestSkipped('model not in cache');
      return;
    }
    final engine = BitNetEngine(model: BitNetCatalog.defaultModel);
    await engine.load();
    final session = BitNetSession(engine: engine);
    final sw = Stopwatch()..start();
    final r = await session.extract(
      'I spent 250 EGP at the supermarket',
      shape: const JsonShape({
        'amount': JsonField.number(),
        'currency': JsonField.enumOf(['EGP', 'USD']),
        'category': JsonField.enumOf(['groceries', 'transport', 'other']),
      }),
      examples: const [
        ExtractExample('paid 40 dollars for an uber',
            {'amount': 40, 'currency': 'USD', 'category': 'transport'}),
      ],
    );
    // ignore: avoid_print
    print('extract: $r in ${sw.elapsedMilliseconds}ms');
    expect(r.value['amount'], 250.0);
    await engine.dispose();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
