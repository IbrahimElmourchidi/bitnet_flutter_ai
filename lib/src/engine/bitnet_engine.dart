import 'dart:convert';
import 'dart:typed_data';

// Guard on dart.library.js_interop, NOT dart.library.html: `dart:html` does not
// exist under dart2wasm, so an html-based guard selects engine_native.dart for
// `flutter build web --wasm` and the build dies with "'dart:ffi' can't be
// imported when compiling to Wasm". `dart:js_interop` is the only library
// present on both dart2js and dart2wasm and absent on the VM, so it is the one
// correct discriminator for "is this a web target".
import 'engine_native.dart'
    if (dart.library.js_interop) 'engine_web.dart';

import '../model/bitnet_model_info.dart';

/// Marker strings that signal the end of an assistant turn for BitNet's chat
/// template. The native layer renders special tokens as literal text
/// (llama.cpp `token_to_piece` with `special=true`), so these arrive on the
/// Dart side as plain strings. The model ends each turn with `<|eot_id|>`
/// (id 128009), but its GGUF registers only `<|end_of_text|>` (id 128001) as
/// EOS — so `llama_token_is_eog` never fires on `<|eot_id|>` and cannot stop
/// the turn. The engines detect these markers to terminate the stream cleanly.
const List<String> kBitNetTurnStopMarkers = ['<|eot_id|>', '<|end_of_text|>'];

/// Streaming detector for end-of-turn markers that may be split across several
/// token pieces.
///
/// The native layer renders `<|eot_id|>` / `<|end_of_text|>` as literal text,
/// but a marker rarely arrives as one piece — it is usually spread over
/// several BPE pieces. Comparing each piece against the marker list on its own
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

/// Incrementally decodes UTF-8 bytes that arrive one token piece at a time.
///
/// The native tokenizer emits raw byte fragments (llama.cpp's byte-fallback
/// vocab), so a single multi-byte character — an emoji, a curly quote, an
/// em-dash — is frequently split across two or three consecutive token pieces.
/// Decoding each piece on its own throws `FormatException: Unfinished UTF-8
/// octet sequence`.
///
/// [add] buffers any trailing bytes that form an incomplete UTF-8 sequence and
/// decodes only the complete code points, carrying the partial tail into the
/// next call. Call [flush] once the stream ends to surface any leftover bytes
/// (rendered as U+FFFD replacement characters if still incomplete).
class Utf8StreamDecoder {
  // allowMalformed so a genuinely invalid/incomplete tail at end-of-stream is
  // rendered as U+FFFD rather than throwing — we only ever hand it *complete*
  // sequences except in flush().
  static const _decoder = Utf8Decoder(allowMalformed: true);

  // Held-back bytes: a suffix that is the start of an as-yet-incomplete
  // multi-byte sequence. Never longer than 3 bytes (max UTF-8 tail).
  final List<int> _pending = [];

  /// Feeds the raw [bytes] of one token piece and returns the text that can be
  /// fully decoded now. Returns `''` when [bytes] only extends an incomplete
  /// sequence.
  String add(Uint8List bytes) {
    if (bytes.isEmpty && _pending.isEmpty) return '';

    final buf = _pending.isEmpty
        ? bytes
        : (Uint8List(_pending.length + bytes.length)
          ..setAll(0, _pending)
          ..setAll(_pending.length, bytes));
    _pending.clear();

    // Walk back from the end over at most 3 bytes to find where the last
    // (possibly incomplete) sequence begins; hold it back if truncated.
    final completeLen = _completePrefixLength(buf);
    if (completeLen < buf.length) {
      _pending.addAll(buf.sublist(completeLen));
    }
    if (completeLen == 0) return '';
    return _decoder.convert(buf, 0, completeLen);
  }

  /// Returns any buffered bytes decoded (malformed tail → U+FFFD) and clears
  /// the buffer. Call when the stream ends (EOS, token cap, cancellation).
  String flush() {
    if (_pending.isEmpty) return '';
    final rest = Uint8List.fromList(_pending);
    _pending.clear();
    return _decoder.convert(rest);
  }

  /// Number of leading bytes of [buf] that form only complete UTF-8 sequences.
  /// Inspects at most the final 3 bytes: if the last sequence is truncated,
  /// returns the offset of its lead byte, otherwise [buf].length.
  static int _completePrefixLength(Uint8List buf) {
    // Scan back to the most recent lead byte (not a 10xxxxxx continuation).
    var i = buf.length - 1;
    final lo = buf.length - 3 < 0 ? 0 : buf.length - 3;
    while (i >= lo && (buf[i] & 0xC0) == 0x80) {
      i--;
    }
    if (i < 0 || i < lo && (buf[i] & 0xC0) == 0x80) {
      // No lead byte within reach — treat everything as complete and let the
      // malformed decoder handle it rather than buffering unboundedly.
      return buf.length;
    }
    final lead = buf[i];
    final expected = _sequenceLength(lead);
    // expected == 0 → not a valid lead (stray continuation / invalid byte);
    // let the decoder emit U+FFFD for it, so treat as complete.
    if (expected == 0) return buf.length;
    final have = buf.length - i;
    // Full sequence present → all of buf is complete. Truncated → hold from i.
    return have >= expected ? buf.length : i;
  }

  /// Expected total byte length of a UTF-8 sequence from its [lead] byte, or 0
  /// if [lead] is not a valid lead byte.
  static int _sequenceLength(int lead) {
    if (lead & 0x80 == 0x00) return 1; // 0xxxxxxx
    if (lead & 0xE0 == 0xC0) return 2; // 110xxxxx
    if (lead & 0xF0 == 0xE0) return 3; // 1110xxxx
    if (lead & 0xF8 == 0xF0) return 4; // 11110xxx
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
