import '../model/bitnet_model_info.dart' show ChatTemplateFamily;
import 'bitnet_session.dart' show ChatRole, ChatTurn;

/// Resolves a model's [ChatTemplateFamily] to the concrete prompt formatter and
/// end-of-turn stop markers it was fine-tuned on.
///
/// Feeding a model the wrong template produces degenerate output (worst case a
/// runaway stream of a single token — see the BitNet 2B-4T notes in
/// [BitNetSession]), so [BitNetSession] dispatches through [kChatTemplates]
/// keyed by `model.chatTemplate` instead of hardcoding one format.
class ChatTemplateSpec {
  /// Builds the full prompt string (ending on an open assistant cue) from the
  /// running [ChatTurn] history and an optional system prompt.
  final String Function(List<ChatTurn> turns, String? systemPrompt) format;

  /// Literal strings that mark the end of an assistant turn. The native layer
  /// renders special tokens as text, so these arrive on the Dart side as plain
  /// strings and are matched by `TurnStopDetector` even when split across
  /// several BPE pieces.
  final List<String> stopMarkers;

  const ChatTemplateSpec({required this.format, required this.stopMarkers});
}

String _roleWord(ChatRole r) => switch (r) {
      ChatRole.system => 'System',
      ChatRole.user => 'User',
      ChatRole.assistant => 'Assistant',
    };

String _roleTag(ChatRole r) => switch (r) {
      ChatRole.system => 'system',
      ChatRole.user => 'user',
      ChatRole.assistant => 'assistant',
    };

// ---------------------------------------------------------------------------
// BitNet b1.58 2B-4T — `System: `/`User: `/`Assistant: ` word labels, no
// separators, `<|eot_id|>` terminators. Kept verbatim from the original
// BitNetSession._formatChatPrompt (this format is verified working; the model
// was SFT'd on it and the standard Llama-3 header template is out-of-distn).
// The leading `<|begin_of_text|>` is intentional — the native bridge sniffs it
// to suppress a duplicate BOS.
// ---------------------------------------------------------------------------
String _formatBitnetLabeled(List<ChatTurn> turns, String? systemPrompt) {
  final buf = StringBuffer('<|begin_of_text|>');
  if (systemPrompt != null && systemPrompt.trim().isNotEmpty) {
    buf
      ..write('System: ')
      ..write(systemPrompt.trim())
      ..write('<|eot_id|>');
  }
  for (final t in turns) {
    buf
      ..write('${_roleWord(t.role)}: ')
      ..write(t.content.trim())
      ..write('<|eot_id|>');
  }
  buf.write('Assistant: ');
  return buf.toString();
}

// ---------------------------------------------------------------------------
// Falcon3 Instruct — `<|system|>` / `<|user|>` / `<|assistant|>` tags, each on
// its own line, content newline-terminated, open `<|assistant|>\n` cue. No BOS
// literal: the engine passes addBos=true and llama.cpp adds the model's
// configured BOS (add_special) — including a literal here would double it.
// eos/turn-end token: `<|endoftext|>`.
// ---------------------------------------------------------------------------
String _formatFalconTagged(List<ChatTurn> turns, String? systemPrompt) {
  final buf = StringBuffer();
  if (systemPrompt != null && systemPrompt.trim().isNotEmpty) {
    buf.write('<|system|>\n${systemPrompt.trim()}\n');
  }
  for (final t in turns) {
    buf.write('<|${_roleTag(t.role)}|>\n${t.content.trim()}\n');
  }
  buf.write('<|assistant|>\n');
  return buf.toString();
}

// ---------------------------------------------------------------------------
// Standard ChatML (Falcon-E) — `<|im_start|>role\n{content}<|im_end|>\n`, open
// `<|im_start|>assistant\n` cue. No BOS literal (see note above; Falcon-E's BOS
// is the unusual `>>ABSTRACT<<` special token, added by llama.cpp via
// add_special if the GGUF configures it). Turn-end `<|im_end|>`; eos
// `<|end_of_text|>` — both listed as stop markers.
// ---------------------------------------------------------------------------
String _formatChatml(List<ChatTurn> turns, String? systemPrompt) {
  final buf = StringBuffer();
  if (systemPrompt != null && systemPrompt.trim().isNotEmpty) {
    buf.write('<|im_start|>system\n${systemPrompt.trim()}<|im_end|>\n');
  }
  for (final t in turns) {
    buf.write('<|im_start|>${_roleTag(t.role)}\n${t.content.trim()}<|im_end|>\n');
  }
  buf.write('<|im_start|>assistant\n');
  return buf.toString();
}

// ---------------------------------------------------------------------------
// Qwen3-derived ChatML (PrismML Ternary Bonsai) — identical to ChatML but the
// assistant cue prefills an empty `<think></think>` block so the (slow,
// on-device) model skips reasoning and answers directly, matching Qwen3's
// `enable_thinking=false` behaviour.
// ---------------------------------------------------------------------------
String _formatChatmlQwen3(List<ChatTurn> turns, String? systemPrompt) {
  final buf = StringBuffer();
  if (systemPrompt != null && systemPrompt.trim().isNotEmpty) {
    buf.write('<|im_start|>system\n${systemPrompt.trim()}<|im_end|>\n');
  }
  for (final t in turns) {
    buf.write('<|im_start|>${_roleTag(t.role)}\n${t.content.trim()}<|im_end|>\n');
  }
  buf.write('<|im_start|>assistant\n<think>\n\n</think>\n\n');
  return buf.toString();
}

/// Concrete template + stop markers for every [ChatTemplateFamily].
const Map<ChatTemplateFamily, ChatTemplateSpec> kChatTemplates = {
  ChatTemplateFamily.bitnetLabeled: ChatTemplateSpec(
    format: _formatBitnetLabeled,
    stopMarkers: ['<|eot_id|>', '<|end_of_text|>'],
  ),
  ChatTemplateFamily.falconTagged: ChatTemplateSpec(
    format: _formatFalconTagged,
    stopMarkers: ['<|endoftext|>'],
  ),
  ChatTemplateFamily.chatml: ChatTemplateSpec(
    format: _formatChatml,
    stopMarkers: ['<|im_end|>', '<|end_of_text|>'],
  ),
  ChatTemplateFamily.chatmlQwen3: ChatTemplateSpec(
    format: _formatChatmlQwen3,
    stopMarkers: ['<|im_end|>'],
  ),
};

/// Convenience accessor: the [ChatTemplateSpec] for [family].
ChatTemplateSpec chatTemplateFor(ChatTemplateFamily family) =>
    kChatTemplates[family]!;
