import 'dart:async';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

/// Tiny fake [BitNetEngine] that echoes a scripted reply back as a stream
/// of single-character tokens. The most recent prompt it saw is recorded
/// so tests can assert on prompt formatting.
class _FakeEngine implements BitNetEngine {
  String reply;
  String? lastPrompt;
  int? lastMaxNewTokens;
  bool _loaded = true;

  _FakeEngine({this.reply = 'ok'});

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {
    _loaded = true;
  }

  @override
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  }) async* {
    lastPrompt = prompt;
    lastMaxNewTokens = maxNewTokens;
    for (final ch in reply.split('')) {
      yield ch;
    }
  }

  @override
  Future<void> cancelGeneration() async {}

  @override
  Future<void> dispose() async {
    _loaded = false;
  }
}

void main() {
  group('BitNetSession', () {
    test('chat appends user + assistant turns to history', () async {
      final engine = _FakeEngine(reply: 'Hi there');
      final session = BitNetSession(engine: engine);
      final pieces = await session.chat('Hello').toList();
      expect(pieces.join(), 'Hi there');
      expect(session.history, hasLength(2));
      expect(session.history[0].role, ChatRole.user);
      expect(session.history[0].content, 'Hello');
      expect(session.history[1].role, ChatRole.assistant);
      expect(session.history[1].content, 'Hi there');
    });

    test('system prompt is included in formatted prompt', () async {
      final engine = _FakeEngine();
      final session = BitNetSession(engine: engine);
      session.setSystemPrompt('You are helpful.');
      await session.chat('Hi').toList();
      expect(engine.lastPrompt, contains('System: You are helpful.'));
      expect(engine.lastPrompt, contains('<|eot_id|>User: Hi<|eot_id|>'));
      expect(engine.lastPrompt, startsWith('<|begin_of_text|>'));
      expect(engine.lastPrompt, endsWith('Assistant: '));
    });

    test('history is trimmed to maxHistoryTurns', () async {
      final engine = _FakeEngine(reply: 'r');
      final session = BitNetSession(engine: engine, maxHistoryTurns: 4);
      for (var i = 0; i < 6; i++) {
        await session.chat('msg $i').toList();
      }
      expect(session.history.length, lessThanOrEqualTo(4));
    });

    test('summarize does not mutate history', () async {
      final engine = _FakeEngine(reply: 'a short summary');
      final session = BitNetSession(engine: engine);
      final out = await session.summarize('long text here');
      expect(out, 'a short summary');
      expect(session.history, isEmpty);
      expect(engine.lastPrompt, contains('Summarise'));
    });

    test('summarize style picks the right instruction', () async {
      final engine = _FakeEngine(reply: '-');
      final session = BitNetSession(engine: engine);
      await session.summarize('text', style: 'bullets');
      expect(engine.lastPrompt, contains('bullet list'));
      await session.summarize('text', style: 'tldr');
      expect(engine.lastPrompt, contains('TL;DR'));
    });

    test('assist with context injects retrieved snippets', () async {
      final engine = _FakeEngine(reply: 'answer');
      final session = BitNetSession(engine: engine);
      final out = await session.assist(
        'What is BitNet?',
        context: 'BitNet is a 1.58-bit LLM.',
      );
      expect(out, 'answer');
      expect(engine.lastPrompt, contains('### Context'));
      expect(engine.lastPrompt, contains('1.58-bit LLM'));
      expect(engine.lastPrompt, contains('### Question'));
    });

    test('assist without context falls back to plain QA template', () async {
      final engine = _FakeEngine(reply: 'a');
      final session = BitNetSession(engine: engine);
      await session.assist('hello?');
      expect(engine.lastPrompt, isNot(contains('### Context')));
      expect(engine.lastPrompt, contains('### Question'));
    });

    test('throws BitNetNotLoadedException when engine is not loaded', () {
      final engine = _FakeEngine();
      engine.dispose();
      final session = BitNetSession(engine: engine);
      expect(
        () => session.chat('hi'),
        throwsA(isA<BitNetNotLoadedException>()),
      );
      expect(
        () async => session.summarize('x'),
        throwsA(isA<BitNetNotLoadedException>()),
      );
    });

    test('clearHistory wipes turns but keeps system prompt', () async {
      final engine = _FakeEngine();
      final session = BitNetSession(engine: engine);
      session.setSystemPrompt('S');
      await session.chat('hi').toList();
      session.clearHistory();
      expect(session.history, isEmpty);
      await session.chat('again').toList();
      expect(engine.lastPrompt, contains('S'));
    });
  });

  group('BitNetRag', () {
    test('ask retrieves relevant context and forwards to assist', () async {
      final engine = _FakeEngine(reply: 'final answer');
      final session = BitNetSession(engine: engine);
      final source = InMemoryKnowledgeSource(documents: [
        (source: 'kb', text: 'BitNet is a 1.58-bit LLM. Flutter is a UI toolkit.'),
      ]);
      final rag = BitNetRag(session: session, source: source);
      final res = await rag.ask('What is BitNet?');
      expect(res.answer, 'final answer');
      expect(res.citations, isNotEmpty);
      expect(engine.lastPrompt, contains('### Context'));
      expect(engine.lastPrompt, contains('BitNet'));
    });
  });
}
