import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('TextChunker', () {
    test('returns no chunks for empty text', () {
      const chunker = TextChunker();
      expect(chunker.chunk('', source: 's'), isEmpty);
      expect(chunker.chunk('   \n  \n', source: 's'), isEmpty);
    });

    test('splits long text into multiple chunks bounded by maxChars', () {
      const chunker = TextChunker(maxChars: 100, overlapChars: 10);
      final text = List.generate(
        20,
        (i) => 'Sentence number $i talks about thing $i.',
      ).join(' ');
      final chunks = chunker.chunk(text, source: 'long.txt');
      expect(chunks.length, greaterThan(1));
      for (final c in chunks) {
        expect(c.text.length, lessThanOrEqualTo(120)); // wiggle for overlap
      }
      expect(chunks.first.id, 'long.txt#0');
      expect(chunks.last.source, 'long.txt');
    });

    test('preserves paragraph boundaries when text is short', () {
      const chunker = TextChunker(maxChars: 1000);
      final chunks = chunker.chunk(
        'First paragraph.\n\nSecond paragraph.\n\nThird.',
        source: 'a',
      );
      expect(chunks, hasLength(1));
      expect(chunks.first.text, contains('First'));
      expect(chunks.first.text, contains('Third'));
    });
  });

  group('Bm25Retriever', () {
    const docs = [
      KnowledgeChunk(
        id: 'a#0',
        source: 'a',
        chunkIndex: 0,
        text: 'BitNet is a 1.58-bit large language model from Microsoft.',
      ),
      KnowledgeChunk(
        id: 'b#0',
        source: 'b',
        chunkIndex: 0,
        text: 'Flutter is a UI toolkit by Google for building cross-platform apps.',
      ),
      KnowledgeChunk(
        id: 'c#0',
        source: 'c',
        chunkIndex: 0,
        text: 'Quantization reduces model size by lowering numeric precision.',
      ),
      KnowledgeChunk(
        id: 'd#0',
        source: 'd',
        chunkIndex: 0,
        text: 'BitNet uses ternary weights to enable efficient on-device inference.',
      ),
    ];

    test('ranks the most relevant chunk first for a clear query', () {
      final r = Bm25Retriever(docs);
      final hits = r.query('What is BitNet?', k: 4);
      expect(hits, isNotEmpty);
      expect(hits.first.chunk.source, anyOf('a', 'd'));
      // BitNet docs should outrank the Flutter doc for a BitNet query.
      final flutterScore = hits
          .firstWhere(
            (h) => h.chunk.source == 'b',
            orElse: () => const RetrievedChunk(
              KnowledgeChunk(id: 'b#0', source: 'b', chunkIndex: 0, text: ''),
              0,
            ),
          )
          .score;
      expect(hits.first.score, greaterThan(flutterScore));
    });

    test('returns empty list when query has no matching tokens', () {
      final r = Bm25Retriever(docs);
      expect(r.query('zxqv plover gnar', k: 4), isEmpty);
    });

    test('respects topK cap', () {
      final r = Bm25Retriever(docs);
      final hits = r.query('BitNet model quantization', k: 2);
      expect(hits.length, lessThanOrEqualTo(2));
    });

    test('handles empty corpus gracefully', () {
      final r = Bm25Retriever(const []);
      expect(r.query('anything', k: 5), isEmpty);
    });
  });

  group('InMemoryKnowledgeSource', () {
    test('chunks all supplied documents and caches the result', () async {
      final src = InMemoryKnowledgeSource(
        documents: [
          (source: 'doc1', text: 'Hello world. This is doc one.'),
          (source: 'doc2', text: 'Another short document.'),
        ],
      );
      final first = await src.chunks();
      final second = await src.chunks();
      expect(first, hasLength(2));
      expect(identical(first, second), isTrue);
      expect(first.map((c) => c.source), containsAll(['doc1', 'doc2']));
    });
  });
}
