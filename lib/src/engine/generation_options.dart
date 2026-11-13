import 'package:flutter/foundation.dart';

/// How a single generation is decoded.
///
/// Two presets cover almost every app:
///
/// * [GenerationOptions.new] (the default) — light stochastic sampling with a
///   repetition penalty. Right for chat, advice and summaries; small ternary
///   models loop under pure greedy decoding.
/// * [GenerationOptions.precise] — greedy, no penalty. Right for structured
///   output (pair it with a [grammar]) and for anything that must be
///   reproducible.
///
/// Instances are immutable and sendable across isolates.
@immutable
final class GenerationOptions {
  /// Upper bound on generated tokens. `0` means "until end-of-turn or the
  /// context window is full".
  final int maxNewTokens;

  /// Sampling temperature. `<= 0` selects greedy (argmax) decoding, in which
  /// case [topK], [topP] and [seed] are ignored.
  final double temperature;

  /// Keep only the `topK` most likely tokens. `<= 0` disables.
  final int topK;

  /// Nucleus sampling threshold. `>= 1` disables.
  final double topP;

  /// Penalty applied to recently generated tokens. `1.0` disables.
  final double repeatPenalty;

  /// How many recent tokens [repeatPenalty] looks back over.
  final int repeatLastN;

  /// RNG seed for stochastic sampling (same seed + same prompt → same text).
  final int seed;

  /// Optional GBNF grammar (root rule `root`) the output must match — see
  /// `JsonShape.toGbnf` for a ready-made JSON grammar. `null` = unconstrained.
  final String? grammar;

  /// Extra stop strings, in addition to the model template's end-of-turn
  /// markers. Generation ends before the first occurrence; the stop string
  /// itself is not emitted.
  final List<String> stop;

  /// Whether the tokenizer prepends the model's BOS token. Leave `true` for
  /// prompts built by the chat templates.
  final bool addBos;

  const GenerationOptions({
    this.maxNewTokens = 512,
    this.temperature = 0.7,
    this.topK = 40,
    this.topP = 0.95,
    this.repeatPenalty = 1.3,
    this.repeatLastN = 256,
    this.seed = 0x5eed5eed,
    this.grammar,
    this.stop = const [],
    this.addBos = true,
  });

  /// Greedy, deterministic decoding with no repetition penalty — for
  /// extraction, classification and other structured output.
  const GenerationOptions.precise({
    this.maxNewTokens = 256,
    this.grammar,
    this.stop = const [],
    this.addBos = true,
  })  : temperature = 0,
        topK = 0,
        topP = 1,
        repeatPenalty = 1,
        repeatLastN = 0,
        seed = 0;

  /// Whether decoding is greedy (deterministic).
  bool get isGreedy => temperature <= 0;

  GenerationOptions copyWith({
    int? maxNewTokens,
    double? temperature,
    int? topK,
    double? topP,
    double? repeatPenalty,
    int? repeatLastN,
    int? seed,
    String? grammar,
    bool clearGrammar = false,
    List<String>? stop,
    bool? addBos,
  }) =>
      GenerationOptions(
        maxNewTokens: maxNewTokens ?? this.maxNewTokens,
        temperature: temperature ?? this.temperature,
        topK: topK ?? this.topK,
        topP: topP ?? this.topP,
        repeatPenalty: repeatPenalty ?? this.repeatPenalty,
        repeatLastN: repeatLastN ?? this.repeatLastN,
        seed: seed ?? this.seed,
        grammar: clearGrammar ? null : (grammar ?? this.grammar),
        stop: stop ?? this.stop,
        addBos: addBos ?? this.addBos,
      );

  /// Whether two option sets configure the native sampler identically (the
  /// worker skips rebuilding it when they do).
  bool samplerEquals(GenerationOptions o) =>
      temperature == o.temperature &&
      topK == o.topK &&
      topP == o.topP &&
      repeatPenalty == o.repeatPenalty &&
      repeatLastN == o.repeatLastN &&
      seed == o.seed &&
      grammar == o.grammar;
}

/// One chunk of generated text plus the model's confidence in it.
///
/// [text] is always complete UTF-8 (a character split across tokens is held
/// back until it is whole), so one event may cover several tokens.
@immutable
final class TokenEvent {
  /// Decoded text of this chunk (never contains a stop marker).
  final String text;

  /// Natural-log probability the model assigned to each token in this chunk,
  /// measured on its raw distribution (before temperature, top-k/p or
  /// grammar). `exp(logprob)` is in (0, 1]; values near 0 mean the model was
  /// unsure. Empty when the backend does not report probabilities.
  final List<double> logprobs;

  const TokenEvent(this.text, [this.logprobs = const []]);

  @override
  String toString() => 'TokenEvent(${text.length}ch, ${logprobs.length} tok)';
}
