import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:bitnet_flutter_ai/src/session/chat_templates.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final turns = [
    const ChatTurn(ChatRole.user, 'hi'),
  ];

  group('chat templates', () {
    test('every ChatTemplateFamily resolves to a spec with stop markers', () {
      for (final family in ChatTemplateFamily.values) {
        final spec = chatTemplateFor(family);
        expect(spec.stopMarkers, isNotEmpty, reason: '$family has no markers');
        final out = spec.format(turns, 'You are helpful.');
        expect(out, contains('hi'), reason: '$family dropped the user turn');
      }
    });

    test('bitnetLabeled uses Human:/BITNETAssistant: labels, system folded in',
        () {
      final out = chatTemplateFor(ChatTemplateFamily.bitnetLabeled)
          .format(turns, 'You are helpful.');
      // No <|begin_of_text|> literal (llama.cpp prepends BOS via addBos). The
      // system prompt is folded into the first Human turn, and the prompt ends
      // on an open assistant cue. See _formatBitnetLabeled.
      expect(out, isNot(contains('<|begin_of_text|>')));
      expect(out, startsWith('Human: You are helpful.'));
      expect(out, contains('hi'));
      expect(out, endsWith('BITNETAssistant: '));
      // The completed Human turn MUST be closed with <|eot_id|>, not a blank
      // line — with \n\n the model never emits EOS and rambles into emoji filler
      // (verified against the real weights). This is the delimiter it was tuned
      // on. Regression guard for that fix.
      expect(out, contains('<|eot_id|>BITNETAssistant: '));
      expect(out, isNot(contains('hi\n\nBITNETAssistant')));
    });

    test('bitnetLabeled closes every completed turn (multi-turn) with <|eot_id|>',
        () {
      final multi = [
        const ChatTurn(ChatRole.user, 'first'),
        const ChatTurn(ChatRole.assistant, 'reply one'),
        const ChatTurn(ChatRole.user, 'second'),
      ];
      final out =
          chatTemplateFor(ChatTemplateFamily.bitnetLabeled).format(multi, null);
      // Two completed turns (user1, assistant1) + open user2 cue → exactly three
      // <|eot_id|> delimiters: after "first", after "reply one", after "second".
      expect('<|eot_id|>'.allMatches(out).length, 3);
      expect(out, contains('first<|eot_id|>BITNETAssistant: reply one<|eot_id|>'));
      expect(out, endsWith('second<|eot_id|>BITNETAssistant: '));
    });

    test('falconTagged uses <|role|> tags, open <|assistant|> cue, no BOS', () {
      final out = chatTemplateFor(ChatTemplateFamily.falconTagged)
          .format(turns, 'You are helpful.');
      expect(out, isNot(contains('<|begin_of_text|>')));
      expect(out, contains('<|system|>\nYou are helpful.\n'));
      expect(out, contains('<|user|>\nhi\n'));
      expect(out, endsWith('<|assistant|>\n'));
      expect(chatTemplateFor(ChatTemplateFamily.falconTagged).stopMarkers,
          contains('<|endoftext|>'));
    });

    test('chatml uses <|im_start|>/<|im_end|> and open assistant cue', () {
      final out = chatTemplateFor(ChatTemplateFamily.chatml)
          .format(turns, 'You are helpful.');
      expect(out, contains('<|im_start|>system\nYou are helpful.<|im_end|>\n'));
      expect(out, contains('<|im_start|>user\nhi<|im_end|>\n'));
      expect(out, endsWith('<|im_start|>assistant\n'));
      expect(chatTemplateFor(ChatTemplateFamily.chatml).stopMarkers,
          contains('<|im_end|>'));
    });

    test('chatmlQwen3 prefills an empty think block', () {
      final out = chatTemplateFor(ChatTemplateFamily.chatmlQwen3)
          .format(turns, null);
      expect(out, contains('<|im_start|>user\nhi<|im_end|>\n'));
      expect(out, endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'));
    });

    test('empty system prompt is omitted', () {
      final out =
          chatTemplateFor(ChatTemplateFamily.falconTagged).format(turns, '');
      expect(out, isNot(contains('<|system|>')));
    });
  });
}
