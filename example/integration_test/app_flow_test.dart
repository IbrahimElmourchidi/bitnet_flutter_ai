// Drives the real example app: load the model from the Device tab, then run
// an expense extraction on the Extract tab and check the rendered result.
//
// Needs the BitNet 2B-4T GGUF already in the model cache (see the seeding
// note in on_device_test.dart); skipped otherwise.
import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:bitnet_flutter_ai_example/main.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:material_ui/material_ui.dart';

Future<void> _pumpUntil(WidgetTester tester, Finder f,
    {Duration timeout = const Duration(minutes: 5)}) async {
  final end = DateTime.now().add(timeout);
  while (DateTime.now().isBefore(end)) {
    await tester.pump(const Duration(milliseconds: 500));
    if (f.evaluate().isNotEmpty) return;
  }
  throw TestFailure('Timed out waiting for $f');
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('load → extract shows a parsed expense with confidence',
      (tester) async {
    if (!await ModelCache.isCached(BitNetCatalog.defaultModel)) {
      markTestSkipped('model not in cache');
      return;
    }
    await tester.pumpWidget(const BitNetExampleApp());
    await _pumpUntil(tester, find.text('Device health'));
    await _pumpUntil(tester, find.text('Available RAM'));

    await tester.dragUntilVisible(
        find.text('Load model'), find.byType(ListView).first, const Offset(0, -300));
    await tester.tap(find.text('Load model'));
    await _pumpUntil(tester, find.text('Engine ready — switch tabs to use it.'));

    await tester.tap(find.text('Extract'));
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(FilledButton, 'Extract'));
    await _pumpUntil(tester, find.text('Overall confidence'));

    expect(find.text('250.0'), findsOneWidget);
    expect(find.text('EGP'), findsWidgets);
    // ignore: avoid_print
    print('SCREENSHOT_NOW');
    await tester.pump(const Duration(seconds: 1));
    await tester.runAsync(() => Future<void>.delayed(const Duration(seconds: 15)));
  }, timeout: const Timeout(Duration(minutes: 15)));
}
