import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/fake_engine.dart';

void main() {
  group('BitNetSession history is always settled', () {
    test('cancelling mid-stream commits the partial reply, not a dangling user '
        'turn', () async {
      final engine = FakeEngine(tokens: ['He', 'llo', ' wor', 'ld'], perToken: const Duration(milliseconds: 20));
      final session = BitNetSession(engine: engine);

      final seen = StringBuffer();
      final sub = session.chat('Hi').listen(seen.write);
      // Let a token or two through, then walk away like a disposed widget.
      await Future<void>.delayed(const Duration(milliseconds: 30));
      await sub.cancel();
      await Future<void>.delayed(const Duration(milliseconds: 30));

      expect(engine.generatorTornDown, isTrue,
          reason: 'cancelling the stream must tear the engine generator down');
      expect(engine.ranToCompletion, isFalse,
          reason: 'the engine should have stopped early, not decoded on');
      expect(session.history, hasLength(2),
          reason: 'a user turn with no assistant turn corrupts every later '
              'prompt');
      expect(session.history[0].role, ChatRole.user);
      expect(session.history[1].role, ChatRole.assistant);
      expect(session.history[1].content, isNotEmpty);
      expect(session.history[1].content, seen.toString().trim());
    });

    test('cancelling before any token rolls the user turn back', () async {
      final engine = FakeEngine(tokens: ['a', 'b', 'c'], perToken: const Duration(milliseconds: 20));
      final session = BitNetSession(engine: engine);

      final sub = session.chat('Hi').listen((_) {});
      await sub.cancel();
      await Future<void>.delayed(const Duration(milliseconds: 5));

      expect(session.history, isEmpty,
          reason: 'an empty reply should leave no half-turn behind');
    });

    test('an engine error mid-reply surfaces on the stream and rolls the '
        'user turn back', () async {
      final session =
          BitNetSession(engine: FakeEngine(tokens: ['a', 'b', 'c'], failAfter: 1));

      await expectLater(session.chat('Hi').toList(),
          throwsA(isA<BitNetInferenceException>()));
      expect(session.history, isEmpty);
    });

    test('completing normally after a cancelled turn keeps the history '
        'alternating', () async {
      final engine = FakeEngine(tokens: ['x', 'y', 'z'], perToken: const Duration(milliseconds: 20));
      final session = BitNetSession(engine: engine);

      final sub = session.chat('first').listen((_) {});
      await Future<void>.delayed(const Duration(milliseconds: 5));
      await sub.cancel();
      await Future<void>.delayed(const Duration(milliseconds: 5));

      await session.chat('second').toList();

      for (var i = 0; i < session.history.length; i++) {
        expect(
          session.history[i].role,
          i.isEven ? ChatRole.user : ChatRole.assistant,
          reason: 'history must strictly alternate user/assistant',
        );
      }
    });
  });

  group('BitNetSession history trimming', () {
    test('never leaves the window opening on an assistant turn', () async {
      // maxHistoryTurns is odd, so a naive count-based trim lands mid-pair and
      // leaves a leading assistant turn.
      final engine = FakeEngine(tokens: ['ok']);
      final session = BitNetSession(engine: engine, maxHistoryTurns: 3);

      for (var i = 0; i < 5; i++) {
        await session.chat('turn $i').toList();
        expect(session.history.first.role, ChatRole.user,
            reason: 'a prompt starting with a bare assistant reply is out of '
                'distribution for every template');
      }
      expect(session.history.length, lessThanOrEqualTo(3));
    });
  });
}
