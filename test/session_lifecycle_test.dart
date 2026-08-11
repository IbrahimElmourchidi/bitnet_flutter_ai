import 'dart:async';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

/// Engine that streams one token per event-loop turn, so a test can walk away
/// mid-stream the way a disposed widget does.
///
/// [generatorTornDown] / [ranToCompletion] record how the stream ended: an
/// `async*` body unwinds through its `finally` when the subscription is
/// cancelled, which is exactly how the real engines learn to stop (NativeEngine
/// turns that into a `_CmdCancel`, WebEngine into a `_cancelRequested` flag).
class _SlowEngine implements BitNetEngine {
  final List<String> tokens;

  /// Wall-clock cost of a token. Real decode is tens of milliseconds per token,
  /// and the cancellation tests need to land *between* two of them.
  final Duration perToken;
  var generatorTornDown = false;
  var ranToCompletion = false;

  _SlowEngine(this.tokens, {this.perToken = const Duration(milliseconds: 20)});

  @override
  bool get isLoaded => true;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {}

  @override
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  }) async* {
    generatorTornDown = false;
    ranToCompletion = false;
    try {
      for (final t in tokens) {
        await Future<void>.delayed(perToken);
        yield t;
      }
      ranToCompletion = true;
    } finally {
      generatorTornDown = true;
    }
  }

  @override
  Future<void> cancelGeneration() async {}

  @override
  Future<void> dispose() async {}
}

/// Engine that rejects [generate] synchronously, mimicking the busy/not-loaded
/// guards in the real engines.
class _ThrowingEngine implements BitNetEngine {
  @override
  bool get isLoaded => true;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {}

  @override
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  }) =>
      throw const BitNetBusyException();

  @override
  Future<void> cancelGeneration() async {}

  @override
  Future<void> dispose() async {}
}

void main() {
  group('BitNetSession history is always settled', () {
    test('cancelling mid-stream commits the partial reply, not a dangling user '
        'turn', () async {
      final engine = _SlowEngine(['He', 'llo', ' wor', 'ld']);
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
      final engine = _SlowEngine(['a', 'b', 'c']);
      final session = BitNetSession(engine: engine);

      final sub = session.chat('Hi').listen((_) {});
      await sub.cancel();
      await Future<void>.delayed(const Duration(milliseconds: 5));

      expect(session.history, isEmpty,
          reason: 'an empty reply should leave no half-turn behind');
    });

    test('a synchronous generate() rejection rolls the user turn back', () {
      final session = BitNetSession(engine: _ThrowingEngine());

      expect(() => session.chat('Hi'), throwsA(isA<BitNetBusyException>()));
      expect(session.history, isEmpty);
    });

    test('completing normally after a cancelled turn keeps the history '
        'alternating', () async {
      final engine = _SlowEngine(['x', 'y', 'z']);
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
      final engine = _SlowEngine(['ok'], perToken: Duration.zero);
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
