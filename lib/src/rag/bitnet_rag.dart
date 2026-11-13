import 'dart:async';

import '../engine/generation_options.dart';
import '../session/bitnet_session.dart';
import 'bm25.dart';
import 'knowledge_source.dart';

/// High-level Retrieval-Augmented Generation helper.
///
/// Wraps a [KnowledgeSource] with a [Bm25Retriever] and a [BitNetSession];
/// callers ask questions and the retriever quietly injects the top-K
/// matching chunks into [BitNetSession.assist].
///
/// ```dart
/// final rag = BitNetRag(
///   session: BitNetSession(engine: engine),
///   source: AssetKnowledgeSource(assetPaths: ['knowledge/faq.md']),
/// );
/// await rag.warmUp();          // load + index
/// final answer = await rag.ask('What is BitNet?');
/// ```
class BitNetRag {
  final BitNetSession session;
  final KnowledgeSource source;

  /// Number of top-scoring chunks injected into the prompt.
  final int topK;

  /// Hard cap on the total characters of context handed to the model.
  /// Retrieved chunks past this limit are dropped (lowest-scoring first).
  final int maxContextChars;

  Bm25Retriever? _retriever;

  BitNetRag({
    required this.session,
    required this.source,
    this.topK = 4,
    this.maxContextChars = 2400,
  });

  /// Loads the knowledge source and builds the BM25 index. Safe to call
  /// multiple times; subsequent calls are no-ops.
  Future<void> warmUp() async {
    if (_retriever != null) return;
    final chunks = await source.chunks();
    _retriever = Bm25Retriever(chunks);
  }

  /// Returns the retrieved chunks for [query] (without invoking the model).
  Future<List<RetrievedChunk>> retrieve(String query, {int? k}) async {
    await warmUp();
    return _retriever!.query(query, k: k ?? topK);
  }

  /// Retrieves relevant chunks and asks the session to answer [question].
  /// The retrieved context is also returned for UI / citation purposes.
  Future<RagAnswer> ask(
    String question, {
    GenerationOptions? options,
    int? k,
  }) async {
    final hits = await retrieve(question, k: k);
    final context = _packContext(hits, maxContextChars);
    final answer = await session.assist(
      question,
      context: context.isEmpty ? null : context,
      options: options,
    );
    return RagAnswer(answer: answer, citations: hits);
  }

  /// Interrupts an in-flight [ask] / [BitNetSession.assist]. The pending
  /// Future resolves with whatever the model produced before stopping.
  Future<void> stop() => session.stop();

  static String _packContext(List<RetrievedChunk> hits, int maxChars) {
    final buf = StringBuffer();
    for (final h in hits) {
      final block = '[${h.chunk.source}#${h.chunk.chunkIndex}]\n${h.chunk.text}';
      if (buf.length + block.length + 2 > maxChars) {
        if (buf.isEmpty) {
          // Even one chunk exceeds the budget — truncate it.
          buf.write(block.substring(0, maxChars));
        }
        break;
      }
      if (buf.isNotEmpty) buf.writeln();
      buf.writeln(block);
    }
    return buf.toString().trim();
  }
}

/// Result returned by [BitNetRag.ask].
class RagAnswer {
  final String answer;
  final List<RetrievedChunk> citations;
  const RagAnswer({required this.answer, required this.citations});
}
