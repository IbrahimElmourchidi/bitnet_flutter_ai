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

    test('bitnetLabeled keeps the verbatim BitNet format', () {
      final out = chatTemplateFor(ChatTemplateFamily.bitnetLabeled)
          .format(turns, 'You are helpful.');
      expect(out, startsWith('<|begin_of_text|>'));
      expect(out, contains('System: You are helpful.<|eot_id|>'));
      expect(out, contains('User: hi<|eot_id|>'));
      expect(out, endsWith('Assistant: '));
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
