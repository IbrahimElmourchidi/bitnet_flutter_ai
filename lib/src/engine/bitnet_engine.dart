import 'engine_native.dart'
    if (dart.library.html) 'engine_web.dart';

import '../model/bitnet_model_info.dart';

/// Marker strings that signal the end of an assistant turn for BitNet's chat
/// template. The native layer renders special tokens as literal text
/// (llama.cpp `token_to_piece` with `special=true`), so these arrive on the
/// Dart side as plain strings. The model ends each turn with `<|eot_id|>`
/// (id 128009), but its GGUF registers only `<|end_of_text|>` (id 128001) as
/// EOS — so `llama_token_is_eog` never fires on `<|eot_id|>` and cannot stop
/// the turn. The engines detect these markers to terminate the stream cleanly.
const List<String> kBitNetTurnStopMarkers = ['<|eot_id|>', '<|end_of_text|>'];

/// Whether [piece] is an end-of-turn marker that should terminate generation
/// without being surfaced to the caller.
///
/// Only matches a marker that arrives as a single, whole piece. Markers are
/// frequently split across several BPE pieces (e.g. `<|end`, `_of`, `_text|>`),
/// which this cannot catch — use [TurnStopDetector] for streaming detection.
bool isBitNetTurnStop(String piece) =>
    kBitNetTurnStopMarkers.contains(piece.trim());

/// Streaming detector for end-of-turn markers that may be split across several
/// token pieces.
///
/// The native layer renders `<|eot_id|>` / `<|end_of_text|>` as literal text,
/// but a marker rarely arrives as one piece — it is usually spread over
/// several BPE pieces. Feeding pieces individually to [isBitNetTurnStop]
/// therefore never matches, so the marker text leaks into the reply and
/// generation runs on into hallucinated turns.
///
/// [feed] buffers just enough trailing text to recognise a marker that is
/// still being assembled: it returns the portion that is safe to emit now and
/// holds back any suffix that could still become the start of a marker. When a
/// marker completes, [stopped] is set and everything up to (but not including)
/// the marker is returned. Call [flush] when generation ends for any other
/// reason (EOS, max tokens, cancellation) to release the held-back tail.
class TurnStopDetector {
  final List<String> _markers;
  final int _maxMarkerLen;
  String _buf = '';

  /// Set once a complete marker has been seen; no further text should be
  /// requested after this becomes true.
  bool stopped = false;

  TurnStopDetector([List<String> markers = kBitNetTurnStopMarkers])
      : _markers = markers,
        _maxMarkerLen =
            markers.fold(0, (m, s) => s.length > m ? s.length : m);

  /// Feeds one token [piece] and returns the text that is now safe to emit.
  String feed(String piece) {
    if (stopped) return '';
    _buf += piece;

    // If a full marker is present, emit everything before its first
    // occurrence and stop.
    var cut = -1;
    for (final m in _markers) {
      final idx = _buf.indexOf(m);
      if (idx != -1 && (cut == -1 || idx < cut)) {
        cut = idx;
      }
    }
    if (cut != -1) {
      final emit = _buf.substring(0, cut);
      _buf = '';
      stopped = true;
      return emit;
    }

    // No complete marker yet. Hold back the shortest trailing slice that could
    // still grow into a marker (a suffix of _buf that is a prefix of a marker);
    // emit the rest.
    final hold = _pendingSuffixLen();
    if (hold == 0) {
      final emit = _buf;
      _buf = '';
      return emit;
    }
    final emit = _buf.substring(0, _buf.length - hold);
    _buf = _buf.substring(_buf.length - hold);
    return emit;
  }

  /// Returns any buffered text that was held back and clears the buffer. Call
  /// when the stream ends without a marker (EOS, token cap, cancellation).
  String flush() {
    final rest = _buf;
    _buf = '';
    return rest;
  }

  /// Length of the longest suffix of [_buf] that is a (proper or full) prefix
  /// of some marker — i.e. the tail that must be withheld until more text
  /// arrives. Bounded by the longest marker length.
  int _pendingSuffixLen() {
    final maxLook =
        _buf.length < _maxMarkerLen ? _buf.length : _maxMarkerLen;
    for (var len = maxLook; len > 0; len--) {
      final suffix = _buf.substring(_buf.length - len);
      for (final m in _markers) {
        if (m.startsWith(suffix)) return len;
      }
    }
    return 0;
  }
}

/// The public interface for driving BitNet inference.
///
/// Usage:
/// ```dart
/// final engine = BitNetEngine();
/// await engine.load(onProgress: (p) => print('${(p*100).toInt()}%'));
/// await for (final token in engine.generate('Hello')) {
///   stdout.write(token);
/// }
/// await engine.dispose();
/// ```
abstract interface class BitNetEngine {
  /// Downloads (if needed), verifies, and loads [model] into memory.
  ///
  /// [onProgress] receives values in [0.0, 1.0] during the download phase.
  /// Throws [BitNetUnsupportedDeviceException] if RAM < 3072 MB.
  /// Throws [BitNetInitException] if the native context cannot be created.
  Future<void> load({void Function(double progress)? onProgress});

  /// Whether the engine has been successfully loaded and is ready to generate.
  bool get isLoaded;

  /// Streams token pieces for [prompt] until EOS or [maxNewTokens] is reached.
  ///
  /// Throws [BitNetNotLoadedException] if called before [load].
  Stream<String> generate(
    String prompt, {
    int maxNewTokens = 512,
    bool addBos = true,
  });

  /// Signals the current generation to stop at the next token boundary.
  Future<void> cancelGeneration();

  /// Frees native resources and kills the inference isolate / WASM context.
  Future<void> dispose();

  /// Creates an engine for [model] using the correct implementation for the
  /// current platform (native isolate or WASM).
  factory BitNetEngine({BitNetModelInfo model = BitNetCatalog.defaultModel}) =>
      createEngine(model: model);
}
