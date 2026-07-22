import 'knowledge_source.dart';

/// Splits raw text into roughly fixed-size chunks suitable for retrieval.
///
/// The chunker prefers paragraph boundaries (blank lines), then sentence
/// boundaries (`.`, `!`, `?`), and only falls back to character-level cuts
/// when individual sentences are longer than [maxChars].
///
/// [overlapChars] controls how many characters of the previous chunk are
/// repeated at the start of the next — this preserves cross-boundary
/// context for queries that span paragraph breaks.
class TextChunker {
  final int maxChars;
  final int overlapChars;

  const TextChunker({this.maxChars = 800, this.overlapChars = 100})
      : assert(maxChars > 0),
        assert(overlapChars >= 0 && overlapChars < maxChars);

  /// Splits [text] into [KnowledgeChunk]s tagged with [source].
  List<KnowledgeChunk> chunk(String text, {required String source}) {
    final pieces = _splitParagraphs(text);
    final chunks = <KnowledgeChunk>[];
    final buf = StringBuffer();
    var idx = 0;

    void flush() {
      final body = buf.toString().trim();
      if (body.isEmpty) return;
      chunks.add(KnowledgeChunk(
        id: '$source#$idx',
        text: body,
        source: source,
        chunkIndex: idx,
      ));
      idx++;
      // Carry overlap forward so the next chunk has context.
      final tail = body.length > overlapChars
          ? body.substring(body.length - overlapChars)
          : body;
      buf
        ..clear()
        ..write(tail);
    }

    for (final piece in pieces) {
      // If a single piece exceeds maxChars, hard-split it on sentences.
      final fragments = piece.length > maxChars
          ? _splitSentences(piece, maxChars)
          : <String>[piece];
      for (final frag in fragments) {
        if (buf.length + frag.length + 1 > maxChars && buf.isNotEmpty) {
          flush();
        }
        if (buf.isNotEmpty) buf.write('\n\n');
        buf.write(frag);
      }
    }
    flush();
    return chunks;
  }

  static List<String> _splitParagraphs(String text) =>
      text.split(RegExp(r'\n\s*\n')).map((p) => p.trim()).where((p) => p.isNotEmpty).toList();

  static List<String> _splitSentences(String text, int maxChars) {
    final sentences = text.split(RegExp(r'(?<=[.!?])\s+'));
    final out = <String>[];
    final buf = StringBuffer();
    for (final s in sentences) {
      if (s.length > maxChars) {
        // Hard-cut very long sentences at maxChars boundaries.
        for (var i = 0; i < s.length; i += maxChars) {
          out.add(s.substring(i, (i + maxChars).clamp(0, s.length)));
        }
        continue;
      }
      if (buf.length + s.length + 1 > maxChars && buf.isNotEmpty) {
        out.add(buf.toString().trim());
        buf.clear();
      }
      if (buf.isNotEmpty) buf.write(' ');
      buf.write(s);
    }
    if (buf.isNotEmpty) out.add(buf.toString().trim());
    return out;
  }
}
