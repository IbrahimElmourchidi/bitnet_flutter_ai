// End-to-end test against a real GGUF and the real native library, through
// the full stack: BitNetSession → NativeEngine isolate → dart:ffi → bn_* C ABI.
//
// Skipped unless BITNET_MODEL points at a GGUF. Linux example (bitnet backend):
//
//   BITNET_MODEL=~/.local/share/<app>/bitnet_flutter_ai/ggml-model-i2_s.gguf \
//   LD_LIBRARY_PATH=native/build-linux:../bitnet.cpp/stage/lib:\
//   ../bitnet.cpp/build-linux/3rdparty/llama.cpp/ggml/src \
//     flutter test test/real_model_test.dart
//
// For a PrismML (Bonsai) GGUF also set BITNET_BACKEND=prism and put
// native/build-linux-prism on LD_LIBRARY_PATH.
@TestOn('vm')
@Tags(['real-model'])
library;

import 'dart:io';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:bitnet_flutter_ai/src/engine/engine_native.dart';
import 'package:bitnet_flutter_ai/src/ffi/native_library.dart';
import 'package:flutter_test/flutter_test.dart';

final _modelPath = Platform.environment['BITNET_MODEL'];
final _prism = Platform.environment['BITNET_BACKEND'] == 'prism';

const _transaction = JsonShape({
  'amount': JsonField.number(description: 'Amount of money spent'),
  'currency': JsonField.enumOf(['EGP', 'USD', 'EUR'],
      description: 'Assume EGP when not stated'),
  'vendor': JsonField.string(nullable: true, description: 'Shop, brand or payee'),
  'category': JsonField.enumOf(
      ['food', 'groceries', 'transport', 'utilities', 'shopping', 'other']),
});

const _examples = [
  ExtractExample('paid 40 dollars for an uber',
      {'amount': 40, 'currency': 'USD', 'vendor': 'Uber', 'category': 'transport'}),
  ExtractExample('electricity bill 150',
      {'amount': 150, 'currency': 'EGP', 'vendor': null, 'category': 'utilities'}),
];

void main() {
  late NativeEngine engine;
  late BitNetSession session;

  setUpAll(() async {
    if (_modelPath == null) return;
    final model = _prism ? BitNetCatalog.bonsai1_7b : BitNetCatalog.defaultModel;
    engine = NativeEngine.withBackend(
      model,
      backendFactory: NativeLibrary.factory,
      resolveModelPath: (_, _) async => _modelPath!,
    );
    await engine.load();
    session = BitNetSession(engine: engine);
  });

  tearDownAll(() async {
    if (_modelPath != null) await engine.dispose();
  });

  final skip = _modelPath == null ? 'set BITNET_MODEL to run' : null;

  test('extracts Fino transactions as valid JSON with confidence', () async {
    final sw = Stopwatch()..start();
    final first = await session.extract('I spent 250 EGP at the supermarket',
        shape: _transaction,
        instructions: "Extract the expense from the user's sentence.",
        examples: _examples);
    final tFirst = sw.elapsedMilliseconds;
    printOnFailure('first: $first in ${tFirst}ms');
    expect(first.value['amount'], 250.0);
    expect(first.value['currency'], 'EGP');
    expect(first.confidence, inExclusiveRange(0, 1.0001));

    sw.reset();
    final second = await session.extract('Paid 75 for a taxi',
        shape: _transaction,
        instructions: "Extract the expense from the user's sentence.",
        examples: _examples);
    final tSecond = sw.elapsedMilliseconds;
    // ignore: avoid_print
    print('extract #1 ${tFirst}ms  $first\n'
        'extract #2 ${tSecond}ms (prefix reused)  $second');
    expect(second.value['amount'], 75.0);
    expect(second.value['category'], 'transport');
    expect(second.fieldConfidence.keys, _transaction.fields.keys);
  }, skip: skip, timeout: const Timeout(Duration(minutes: 5)));

  test('cancelGeneration stops a long generation promptly', () async {
    final sw = Stopwatch()..start();
    final text = session
        .assist('Write a long essay about budgeting.',
            options: const GenerationOptions(maxNewTokens: 2000))
        .then((t) => t);
    await Future<void>.delayed(const Duration(seconds: 2));
    final cancelAt = sw.elapsedMilliseconds;
    await session.stop();
    await text;
    final stopMs = sw.elapsedMilliseconds - cancelAt;
    // ignore: avoid_print
    print('stopped ${stopMs}ms after cancel');
    expect(stopMs, lessThan(3000));
  }, skip: skip, timeout: const Timeout(Duration(minutes: 5)));

  test('chat produces a reply and keeps alternating history', () async {
    session.setSystemPrompt('You are a concise personal finance assistant.');
    final reply = await session.chat('Give me one tip to save money.').join();
    // ignore: avoid_print
    print('chat: $reply');
    expect(reply.trim(), isNotEmpty);
    expect(session.history.map((t) => t.role),
        [ChatRole.user, ChatRole.assistant]);
  }, skip: skip, timeout: const Timeout(Duration(minutes: 5)));
}
