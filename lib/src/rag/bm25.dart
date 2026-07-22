import 'dart:math' as math;

import 'knowledge_source.dart';

/// Tiny, dependency-free BM25 retriever (BM25-Okapi).
///
/// Index time: tokenise each chunk, build inverse-document-frequency
/// weights and document-length statistics. Query time: tokenise the query
/// and score every indexed document with the standard Okapi formula:
///
///     score(D, Q) = Σ idf(qᵢ) · (tf(qᵢ, D) · (k1 + 1)) /
///                   (tf(qᵢ, D) + k1 · (1 - b + b · |D| / avgdl))
///
/// This is good enough for the small (≤ a few thousand chunks) on-device
/// knowledge bases shipped with a Flutter app, and it has zero runtime
/// dependencies. For larger corpora prefer a vector DB.
class Bm25Retriever {
  final double k1;
  final double b;

  /// Build the index from a list of [chunks]. Tokenisation is built-in
  /// (lowercase + ASCII word characters); supply a custom [tokenize]
  /// function to handle non-English text.
  Bm25Retriever(
    List<KnowledgeChunk> chunks, {
    this.k1 = 1.5,
    this.b = 0.75,
    List<String> Function(String)? tokenize,
  })  : _tokenize = tokenize ?? _defaultTokenize,
        _chunks = List.unmodifiable(chunks) {
    _build();
  }

  final List<String> Function(String) _tokenize;
  final List<KnowledgeChunk> _chunks;

  /// term -> list of (docIndex, termFrequency)
  final Map<String, List<_Posting>> _postings = {};
  late final List<int> _docLengths;
  late final double _avgDocLength;
  late final Map<String, double> _idf;

  void _build() {
    final n = _chunks.length;
    _docLengths = List<int>.filled(n, 0);
    final df = <String, int>{};

    for (var d = 0; d < n; d++) {
      final tokens = _tokenize(_chunks[d].text);
      _docLengths[d] = tokens.length;

      final tf = <String, int>{};
      for (final t in tokens) {
        tf[t] = (tf[t] ?? 0) + 1;
      }
      tf.forEach((term, freq) {
        (_postings[term] ??= <_Posting>[]).add(_Posting(d, freq));
        df[term] = (df[term] ?? 0) + 1;
      });
    }

    _avgDocLength = n == 0
        ? 0
        : _docLengths.fold<int>(0, (a, b) => a + b) / n;

    _idf = <String, double>{};
    df.forEach((term, dfTerm) {
      // BM25-Okapi IDF (Robertson-Sparck-Jones), capped at 0 to avoid the
      // negative weight that pure Okapi assigns to terms appearing in
      // > 50% of documents.
      final raw = math.log((n - dfTerm + 0.5) / (dfTerm + 0.5) + 1);
      _idf[term] = raw < 0 ? 0 : raw;
    });
  }

  /// Returns the top-[k] chunks ranked by BM25 score, descending.
  /// Chunks with a score of 0 are omitted.
  List<RetrievedChunk> query(String text, {int k = 4}) {
    if (_chunks.isEmpty || k <= 0) return const [];

    final qTokens = _tokenize(text);
    if (qTokens.isEmpty) return const [];

    final scores = List<double>.filled(_chunks.length, 0);
    for (final term in qTokens) {
      final postings = _postings[term];
      if (postings == null) continue;
      final idf = _idf[term] ?? 0;
      if (idf == 0) continue;
      for (final p in postings) {
        final dl = _docLengths[p.docId];
        final norm = 1 - b + b * (_avgDocLength == 0 ? 1 : dl / _avgDocLength);
        final tfPart = (p.tf * (k1 + 1)) / (p.tf + k1 * norm);
        scores[p.docId] += idf * tfPart;
      }
    }

    // Build (docId, score) pairs, drop zeros, sort by score desc.
    final ranked = <_Hit>[];
    for (var d = 0; d < scores.length; d++) {
      if (scores[d] > 0) ranked.add(_Hit(d, scores[d]));
    }
    ranked.sort((a, b) => b.score.compareTo(a.score));
    final top = ranked.take(k);
    return [
      for (final h in top) RetrievedChunk(_chunks[h.docId], h.score),
    ];
  }

  static List<String> _defaultTokenize(String text) {
    return RegExp(r"[a-z0-9]+(?:'[a-z]+)?")
        .allMatches(text.toLowerCase())
        .map((m) => m.group(0)!)
        .where((t) => !_stopwords.contains(t))
        .toList();
  }

  /// Common English stop words. Kept minimal so the retriever stays useful
  /// for technical text where words like "use" / "set" carry meaning.
  static const Set<String> _stopwords = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'have', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'this', 'to', 'was', 'were', 'will', 'with',
  };
}

class _Posting {
  final int docId;
  final int tf;
  const _Posting(this.docId, this.tf);
}

class _Hit {
  final int docId;
  final double score;
  const _Hit(this.docId, this.score);
}
