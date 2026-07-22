import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

/// Feeds every piece through the detector and returns the concatenation of all
/// emitted text plus any flushed tail, mirroring how the engines drive it.
String runStream(List<String> pieces, {bool flushAtEnd = true}) {
  final d = TurnStopDetector();
  final buf = StringBuffer();
  for (final p in pieces) {
    buf.write(d.feed(p));
    if (d.stopped) return buf.toString();
  }
  if (flushAtEnd) buf.write(d.flush());
  return buf.toString();
}

void main() {
  group('TurnStopDetector', () {
    test('marker arriving as a single piece stops and is not emitted', () {
      final d = TurnStopDetector();
      expect(d.feed('Hello there'), 'Hello there');
      expect(d.feed('<|eot_id|>'), '');
      expect(d.stopped, isTrue);
    });

    test('marker split across two pieces is caught and stripped', () {
      final out = runStream(['Hi', '<|eot', '_id|>']);
      expect(out, 'Hi');
    });

    test('marker split across three pieces is caught and stripped', () {
      final out = runStream(['Answer: 42', '<|end', '_of', '_text|>extra']);
      // Text after a completed marker is discarded (generation stops).
      expect(out, 'Answer: 42');
    });

    test('marker glued to the end of real text is split correctly', () {
      final out = runStream(['The end.<|eot_id|>']);
      expect(out, 'The end.');
    });

    test('a false prefix that never completes is eventually emitted', () {
      // "<|end of story" shares a prefix with "<|end_of_text|>" but diverges.
      final out = runStream(['I said ', '<|end', ' of story']);
      expect(out, 'I said <|end of story');
    });

    test('lone "<|" prefix at stream end is flushed, not swallowed', () {
      final out = runStream(['almost', '<|']);
      expect(out, 'almost<|');
    });

    test('no marker: text passes through unchanged', () {
      final out = runStream(['one ', 'two ', 'three']);
      expect(out, 'one two three');
    });

    test('flush releases held-back tail when stream ends without a marker', () {
      final d = TurnStopDetector();
      // "<|eot_id" is a proper prefix of "<|eot_id|>" so it is held back.
      expect(d.feed('done <|eot_id'), 'done ');
      expect(d.stopped, isFalse);
      expect(d.flush(), '<|eot_id');
    });

    test('feed after stopped returns empty', () {
      final d = TurnStopDetector();
      d.feed('x<|eot_id|>');
      expect(d.stopped, isTrue);
      expect(d.feed('more'), '');
    });

    test('emits progressively as safe text accumulates', () {
      final d = TurnStopDetector();
      // Plain text with no marker-prefix suffix is emitted immediately.
      expect(d.feed('plain text '), 'plain text ');
      // A trailing marker-prefix is held back...
      expect(d.feed('and <|eot'), 'and ');
      // ...then released as non-marker once it diverges.
      expect(d.feed(' not a marker'), '<|eot not a marker');
    });
  });
}
