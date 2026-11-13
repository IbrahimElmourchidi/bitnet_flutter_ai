import 'dart:math' as math;

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/fake_engine.dart';

const _tx = JsonShape({
  'amount': JsonField.number(description: 'Amount paid'),
  'currency': JsonField.enumOf(['EGP', 'USD']),
  'vendor': JsonField.string(nullable: true, description: 'Shop or payee'),
  'category': JsonField.enumOf(['groceries', 'transport', 'other']),
  'date': JsonField.date(nullable: true),
});

void main() {
  group('JsonShape.toGbnf', () {
    final g = _tx.toGbnf();

    test('root lists every key in order, compact', () {
      final root = g.split('\n').first;
      expect(root, startsWith('root ::= "{"'));
      final order = ['amount', 'currency', 'vendor', 'category', 'date']
          .map((k) => root.indexOf('"\\"$k\\""'))
          .toList();
      expect(order, everyElement(greaterThan(0)));
      expect(order, orderedEquals([...order]..sort()));
      expect(root, endsWith('"}"'));
    });

    test('nullable fields accept null; others do not', () {
      expect(g, contains('f2 ::= str | "null"'));
      expect(g, contains('f4 ::= date | "null"'));
      expect(g, contains('f0 ::= num\n'));
    });

    test('enum values are quoted JSON string literals', () {
      expect(g, contains(r'f1 ::= ("\"EGP\"" | "\"USD\"")'));
    });

    test('uses flat repetition (bounded forms are slow in llama.cpp)', () {
      expect(g, contains(r'str ::= "\"" chr* "\""'));
      expect(g, contains('num ::= "-"? [0-9]+ ("." [0-9]+)?'));
    });

    test('literals escape quotes, backslashes and control characters', () {
      const s = JsonShape({'a"b\\c': JsonField.enumOf(['x\ny'])});
      final gg = s.toGbnf();
      expect(gg, contains(r'"\"a\\\"b\\\\c\""'));
      expect(gg, contains(r'"\"x\\ny\""'));
    });

    test('empty shapes and enums are rejected', () {
      expect(() => const JsonShape({}).toGbnf(), throwsArgumentError);
      expect(() => const JsonShape({'e': JsonField.enumOf([])}).toGbnf(),
          throwsArgumentError);
    });
  });

  group('JsonShape.parse', () {
    test('per-field confidence is the least likely token of that value', () {
      const raw =
          '{"amount":250,"currency":"EGP","vendor":null,"category":"groceries","date":null}';
      // One event per field chunk; the category value is the uncertain one.
      final events = [
        const TokenEvent('{"amount":', [-0.01]),
        const TokenEvent('250', [-0.05]),
        const TokenEvent(',"currency":', [-0.01]),
        const TokenEvent('"EGP"', [-0.1]),
        const TokenEvent(',"vendor":', [-0.01]),
        const TokenEvent('null', [-0.2]),
        const TokenEvent(',"category":', [-0.01]),
        const TokenEvent('"groceries"', [-2.0, -0.01]),
        const TokenEvent(',"date":', [-0.01]),
        const TokenEvent('null', [-0.3]),
        const TokenEvent('}', [-0.01]),
      ];
      final r = _tx.parse(raw, events);
      expect(r.value, {
        'amount': 250.0,
        'currency': 'EGP',
        'vendor': null,
        'category': 'groceries',
        'date': null,
      });
      expect(r.value['amount'], isA<double>());
      expect(r.fieldConfidence['amount'], closeTo(math.exp(-0.05), 1e-9));
      expect(r.fieldConfidence['category'], closeTo(math.exp(-2.0), 1e-9));
      expect(r.confidence, closeTo(math.exp(-2.0), 1e-9));
      expect(r.uncertainFields(), ['category']);
      expect(r.isConfident(0.1), isTrue);
    });

    test('truncated output is a BitNetExtractionException with the raw text',
        () {
      expect(
        () => _tx.parse('{"amount":25', const []),
        throwsA(isA<BitNetExtractionException>()
            .having((e) => e.raw, 'raw', '{"amount":25')),
      );
    });
  });

  group('BitNetSession.extract', () {
    test('runs greedy + grammar and builds a cache-friendly prompt', () async {
      final engine = FakeEngine(
        tokens: [
          '{"amount":',
          '75',
          ',"currency":"USD","vendor":"Uber","category":"transport","date":',
          '"2026-11-12"',
          '}',
        ],
        logprob: -0.05,
      );
      final session = BitNetSession(engine: engine);
      final r = await session.extract(
        'Paid 75 dollars for an uber yesterday',
        shape: _tx,
        instructions: "Extract the expense from the user's sentence.",
        examples: const [
          ExtractExample('electricity bill 150', {
            'amount': 150,
            'currency': 'EGP',
            'vendor': null,
            'category': 'other',
            'date': null,
          }),
        ],
        today: DateTime(2026, 11, 13),
      );

      expect(r.value['vendor'], 'Uber');
      expect(r.value['date'], '2026-11-12');
      expect(r.confidence, closeTo(math.exp(-0.05), 1e-9));

      final o = engine.lastOptions!;
      expect(o.isGreedy, isTrue);
      expect(o.repeatPenalty, 1);
      expect(o.grammar, _tx.toGbnf());

      final p = engine.lastPrompt!;
      expect(p, contains("Extract the expense from the user's sentence."));
      expect(p, contains('- category: one of "groceries", "transport", "other"'));
      expect(p, contains('Text: electricity bill 150'));
      expect(p, contains('JSON: {"amount":150,"currency":"EGP"'));
      expect(p, contains("Today's date: 2026-11-13."));
      // The variable part comes last, so the native KV cache can reuse the
      // whole instruction/examples prefix across extractions.
      final userAt = p.indexOf('Paid 75 dollars');
      expect(userAt, greaterThan(p.indexOf("Today's date")));
    });

    test('does not touch chat history', () async {
      final engine = FakeEngine(
          tokens: ['{"amount":1,"currency":"EGP","vendor":null,"category":"other","date":null}']);
      final session = BitNetSession(engine: engine);
      await session.extract('x', shape: _tx);
      expect(session.history, isEmpty);
    });
  });

  group('BitNetSession chat token budget', () {
    test('drops the oldest turns when the prompt would overflow', () async {
      // 1 token per character; context 4096, reply reserve 512 → 3584 budget.
      final engine = FakeEngine(reply: 'ok', tokenCounter: (t) => t.length);
      final session = BitNetSession(engine: engine);
      final big = 'x' * 1500;
      await session.chat('$big 1').toList();
      await session.chat('$big 2').toList();
      // Third turn: all three messages (~4500 tokens) cannot fit.
      await session.chat('$big 3').toList();
      expect(engine.lastPrompt!.length, lessThanOrEqualTo(3584));
      expect(session.history.first.role, ChatRole.user);
      expect(session.history.first.content, endsWith(' 2'));
      expect(session.history.last.role, ChatRole.assistant);
    });

    test('a single message larger than the context is a typed error',
        () async {
      final engine = FakeEngine(reply: 'ok', tokenCounter: (t) => t.length);
      final session = BitNetSession(engine: engine);
      await expectLater(session.chat('y' * 5000).toList(),
          throwsA(isA<BitNetPromptTooLongException>()));
      expect(session.history, isEmpty);
    });

    test('concurrent chat calls run in order with complete history', () async {
      final engine =
          FakeEngine(reply: 'r', perToken: const Duration(milliseconds: 2));
      final session = BitNetSession(engine: engine);
      await Future.wait([
        session.chat('one').toList(),
        session.chat('two').toList(),
      ]);
      expect(session.history.map((t) => t.content), ['one', 'r', 'two', 'r']);
      // The second prompt already contained the first exchange.
      expect(engine.prompts.last, contains('one'));
    });
  });
}
