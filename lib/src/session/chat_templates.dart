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

String _roleTag(ChatRole r) => switch (r) {
      ChatRole.system => 'system',
      ChatRole.user => 'user',
      ChatRole.assistant => 'assistant',
    };

// ---------------------------------------------------------------------------
// BitNet b1.58 2B-4T — `Human: ` / `BITNETAssistant: ` labels, with every
// COMPLETED turn terminated by the special `<|eot_id|>` token:
//   {bos}Human: {content}<|eot_id|>BITNETAssistant: {content}<|eot_id|>...
//
// CRITICAL — the turn separator must be `<|eot_id|>`, NOT a blank line. Verified
// empirically against the real 2B-4T weights via a host harness on the shipped
// GGUF (native EOS/token dump): with `\n\n` separators the model NEVER emits an
// end-of-turn token — it finishes the answer then rambles indefinitely into
// hashtag/emoji filler (runs to the token cap). With `<|eot_id|>` separators the
// model stops cleanly on `<|eot_id|>` after ~15-25 tokens with no filler. Both
// single-turn and multi-turn conversations were confirmed. This is because the
// model was fine-tuned with `<|eot_id|>` (id 128009) as the turn delimiter; the
// blank-line format is out-of-distribution and leaves the turn "open".
//
// The system prompt has no dedicated role: it is folded into the first Human
// turn, separated from the user's message by a blank line (that blank line is
// WITHIN a turn and is fine — only the turn DELIMITER must be `<|eot_id|>`).
//
// No `<|begin_of_text|>` literal: the engine passes addBos=true and the GGUF has
// add_bos_token=true, so llama.cpp prepends the BOS (128000). Writing one here
// would double it. `<|eot_id|>` IS written literally because it is a mid-prompt
// delimiter, not the auto-prepended BOS — llama.cpp tokenizes it as the single
// special token 128009 (parse_special=true).
// ---------------------------------------------------------------------------
String _formatBitnetLabeled(List<ChatTurn> turns, String? systemPrompt) {
  const eot = '<|eot_id|>';
  final buf = StringBuffer();
  final sys = systemPrompt?.trim();
  var prependedSystem = false;
  for (final t in turns) {
    switch (t.role) {
      case ChatRole.system:
        // Rare (system turns normally come via systemPrompt); treat as Human.
        buf.write('Human: ${t.content.trim()}$eot');
      case ChatRole.user:
        if (!prependedSystem && sys != null && sys.isNotEmpty) {
          buf.write('Human: $sys\n\n${t.content.trim()}${eot}BITNETAssistant: ');
          prependedSystem = true;
        } else {
          buf.write('Human: ${t.content.trim()}${eot}BITNETAssistant: ');
        }
      case ChatRole.assistant:
        // Prior assistant reply in multi-turn history — close its turn.
        buf.write('${t.content.trim()}$eot');
    }
  }
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
    // Assistant turns SHOULD end at the model's EOS (<|end_of_text|>), enforced
    // natively via llama_vocab_is_eog. But this GGUF ends turns with <|eot_id|>
    // while registering only <|end_of_text|> as EOS, so the native stop often
    // never fires and the model rambles on, FABRICATING the next turn — it
    // emits openers it saw in pretraining ("User:", "ANSWER:", "Assistant:")
    // even though we prompt with "Human:"/"BITNETAssistant:". Left unchecked,
    // that hallucinated tail is committed to history and poisons every later
    // turn (see BitNetSession._sanitizeReply).
    //
    // These markers are the text-level safety net. They are LINE-ANCHORED
    // (each starts with a newline, except the special tokens and bare "Human:")
    // so a reply that merely mentions the word "user" mid-sentence is NOT
    // truncated — only a turn-opener at the start of a fresh line/block stops
    // generation. Ordered longest-first is unnecessary (TurnStopDetector picks
    // the earliest match), but grouped by opener for readability.
    stopMarkers: [
      // Special / EOS tokens rendered as literal text by token_to_piece.
      '<|end_of_text|>',
      '<|eot_id|>',
      // The correct opener for THIS template.
      '\n\nHuman:', '\nHuman:', 'Human:',
      '\n\nBITNETAssistant:', '\nBITNETAssistant:',
      // Openers the model hallucinates from other chat formats it saw in
      // pretraining. Line-anchored to avoid false positives.
      '\n\nUser:', '\nUser:',
      '\n\nAssistant:', '\nAssistant:',
      '\n\nANSWER:', '\nANSWER:',
      '\n\nQuestion:', '\nQuestion:',
      '\n\nSystem:', '\nSystem:',
    ],
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
