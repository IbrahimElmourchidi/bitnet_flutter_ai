import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:bitnet_flutter_ai/src/engine/bitnet_engine.dart';
import 'package:bitnet_flutter_ai/src/session/chat_templates.dart';
import 'package:flutter_test/flutter_test.dart';

/// Feeds [pieces] through a detector seeded with the bitnetLabeled template's
/// real stop markers (the ones the engine actually uses) and returns the
/// concatenation of emitted text, stopping early if a marker completes.
String runBitnet(List<String> pieces) {
  final markers = chatTemplateFor(ChatTemplateFamily.bitnetLabeled).stopMarkers;
  final d = TurnStopDetector(markers);
  final buf = StringBuffer();
  for (final p in pieces) {
    buf.write(d.feed(p));
    if (d.stopped) return buf.toString();
  }
  buf.write(d.flush());
  return buf.toString();
}

void main() {
  final markers = chatTemplateFor(ChatTemplateFamily.bitnetLabeled).stopMarkers;

  group('bitnetLabeled stop markers catch hallucinated turn openers', () {
    test('fabricated "User:" opener stops generation', () {
      // Reproduces the reported log: the reply runs on into a fake user turn.
      final out = runBitnet([
        'Hello! I am BITNETAssistant.',
        '\n\n',
        'User:',
        ' What are the best ways to reduce stress?',
      ]);
      expect(out, 'Hello! I am BITNETAssistant.');
    });

    test('fabricated "ANSWER:" opener stops generation', () {
      // Marker '\nANSWER:' is line-anchored, so the trailing '\n' of "Sure.\n"
      // is consumed as part of the marker and only "Sure." is emitted.
      final out = runBitnet(['Sure.\n', 'ANSWER:', ' 42']);
      expect(out, 'Sure.');
    });

    test('fabricated "Assistant:" opener stops generation', () {
      final out = runBitnet(['Reply body', '\n', 'Assistant:', ' more']);
      expect(out, 'Reply body');
    });

    test('opener split across pieces (User: as "Us","er:") is still caught', () {
      final out = runBitnet(['Answer.', '\n\n', 'Us', 'er:', ' next q']);
      expect(out, 'Answer.');
    });

    test('the word "user" mid-sentence is NOT truncated (line-anchored)', () {
      // A legit reply mentioning "user" must survive — the markers are anchored
      // to a fresh line/block, not a bare substring.
      final out = runBitnet(['The user should relax and breathe deeply.']);
      expect(out, 'The user should relax and breathe deeply.');
    });

    test('EOS token <|end_of_text|> still stops', () {
      final out = runBitnet(['Done.', '<|end_of_text|>']);
      expect(out, 'Done.');
    });

    test('<|eot_id|> (the turn-end token this GGUF actually uses) stops', () {
      final out = runBitnet(['Done.', '<|eot', '_id|>']);
      expect(out, 'Done.');
    });
  });

  group('marker set sanity', () {
    test('includes the openers seen hallucinated in the wild', () {
      expect(markers, containsAll(<String>['\n\nUser:', '\n\nANSWER:']));
    });
  });
}
