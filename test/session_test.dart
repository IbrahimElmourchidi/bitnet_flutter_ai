
import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/fake_engine.dart';

void main() {
  group('BitNetSession', () {
    test('chat appends user + assistant turns to history', () async {
      final engine = FakeEngine(reply: 'Hi there');
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
      final engine = FakeEngine();
      final session =
          BitNetSession(engine: engine, systemPrompt: 'You are helpful.');
      await session.chat('Hi').toList();
      // bitnetLabeled template: the system prompt is folded into the first
      // Human turn, and the prompt ends on an open assistant cue. No
      // <|begin_of_text|> literal (llama.cpp prepends BOS via addBos).
      expect(engine.lastPrompt, contains('Human: You are helpful.'));
      expect(engine.lastPrompt, contains('Hi'));
      expect(engine.lastPrompt, endsWith('BITNETAssistant: '));
    });

    test('hallucinated turn-opener in reply is stripped before entering history',
        () async {
      // Simulate the reported failure: the engine streams a reply that runs on
      // into a fabricated user turn. Only the real answer must land in history,
      // so the next turn's prompt is not poisoned.
      final engine = FakeEngine(
        reply: 'Hello! I am BITNETAssistant.\n\n'
            'User: What are the best ways to reduce stress?\n'
            'ANSWER: Take a walk.',
      );
      final session = BitNetSession(engine: engine);
      await session.chat('hi').toList();
      expect(session.history.last.role, ChatRole.assistant);
      expect(session.history.last.content, 'Hello! I am BITNETAssistant.');
      expect(session.history.last.content, isNot(contains('User:')));
      expect(session.history.last.content, isNot(contains('ANSWER:')));
    });

    test('history is trimmed to maxHistoryTurns', () async {
      final engine = FakeEngine(reply: 'r');
      final session = BitNetSession(engine: engine, maxHistoryTurns: 4);
      for (var i = 0; i < 6; i++) {
        await session.chat('msg $i').toList();
      }
      expect(session.history.length, lessThanOrEqualTo(4));
    });

    test('summarize does not mutate history', () async {
      final engine = FakeEngine(reply: 'a short summary');
      final session = BitNetSession(engine: engine);
      final out = await session.summarize('long text here');
      expect(out, 'a short summary');
      expect(session.history, isEmpty);
      expect(engine.lastPrompt, contains('Summarise'));
    });

    test('summarize style picks the right instruction', () async {
      final engine = FakeEngine(reply: '-');
      final session = BitNetSession(engine: engine);
      await session.summarize('text', style: SummaryStyle.bullets);
      expect(engine.lastPrompt, contains('bullet list'));
      await session.summarize('text', style: SummaryStyle.tldr);
      expect(engine.lastPrompt, contains('TL;DR'));
    });

    test('assist with context injects retrieved snippets', () async {
      final engine = FakeEngine(reply: 'answer');
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
      final engine = FakeEngine(reply: 'a');
      final session = BitNetSession(engine: engine);
      await session.assist('hello?');
      expect(engine.lastPrompt, isNot(contains('### Context')));
      expect(engine.lastPrompt, contains('### Question'));
    });

    test('throws BitNetNotLoadedException when engine is not loaded', () {
      final engine = FakeEngine();
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
      final engine = FakeEngine();
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
      final engine = FakeEngine(reply: 'final answer');
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
