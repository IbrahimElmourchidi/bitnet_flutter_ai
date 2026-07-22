/// A single retrievable chunk of text plus its metadata.
class KnowledgeChunk {
  /// Stable id (`'<source>#<index>'`) usable for caching / deduplication.
  final String id;

  /// The raw text the model will see.
  final String text;

  /// Free-form source label (e.g. asset path, URL, file name).
  final String source;

  /// 0-based index of this chunk within [source].
  final int chunkIndex;

  const KnowledgeChunk({
    required this.id,
    required this.text,
    required this.source,
    required this.chunkIndex,
  });
}

/// A retrieved chunk together with a relevance [score].
///
/// Higher scores mean more relevant; the absolute scale is retriever-specific
/// (BM25 returns roughly `0..30`).
class RetrievedChunk {
  final KnowledgeChunk chunk;
  final double score;
  const RetrievedChunk(this.chunk, this.score);

  @override
  String toString() =>
      'RetrievedChunk(${chunk.source}#${chunk.chunkIndex}, ${score.toStringAsFixed(2)})';
}

/// A source of retrievable text chunks.
///
/// Implementations are responsible for loading the raw text and splitting
/// it into [KnowledgeChunk]s. The result is cached after the first call.
abstract class KnowledgeSource {
  /// Loads (and chunks, if needed) the source. Safe to call repeatedly —
  /// implementations should memoise the result.
  Future<List<KnowledgeChunk>> chunks();
}
