import 'package:flutter/services.dart' show rootBundle;

import 'knowledge_source.dart';
import 'text_chunker.dart';

/// A [KnowledgeSource] backed by Flutter `assets`. Each asset path is
/// loaded as UTF-8 text and split with [TextChunker].
///
/// Example:
/// ```dart
/// final src = AssetKnowledgeSource(
///   assetPaths: ['knowledge/bitnet_faq.md', 'knowledge/release_notes.txt'],
/// );
/// final chunks = await src.chunks();
/// ```
///
/// Make sure each path is also declared under `flutter.assets:` in the
/// host app's `pubspec.yaml`.
class AssetKnowledgeSource implements KnowledgeSource {
  final List<String> assetPaths;
  final TextChunker chunker;

  /// Override the default loader (useful in tests).
  final Future<String> Function(String path)? loader;

  List<KnowledgeChunk>? _cached;

  AssetKnowledgeSource({
    required this.assetPaths,
    TextChunker? chunker,
    this.loader,
  }) : chunker = chunker ?? const TextChunker();

  @override
  Future<List<KnowledgeChunk>> chunks() async {
    if (_cached != null) return _cached!;
    final out = <KnowledgeChunk>[];
    final load = loader ?? rootBundle.loadString;
    for (final path in assetPaths) {
      final text = await load(path);
      out.addAll(chunker.chunk(text, source: path));
    }
    _cached = List.unmodifiable(out);
    return _cached!;
  }
}

/// In-memory [KnowledgeSource] for callers that already have plain strings
/// (downloaded docs, user-pasted text, generated content, …).
class InMemoryKnowledgeSource implements KnowledgeSource {
  final List<({String source, String text})> documents;
  final TextChunker chunker;
  List<KnowledgeChunk>? _cached;

  InMemoryKnowledgeSource({
    required this.documents,
    TextChunker? chunker,
  }) : chunker = chunker ?? const TextChunker();

  @override
  Future<List<KnowledgeChunk>> chunks() async {
    if (_cached != null) return _cached!;
    final out = <KnowledgeChunk>[];
    for (final d in documents) {
      out.addAll(chunker.chunk(d.text, source: d.source));
    }
    _cached = List.unmodifiable(out);
    return _cached!;
  }
}
