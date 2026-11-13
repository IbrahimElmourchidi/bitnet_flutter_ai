import 'dart:async';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';

/// Scriptable in-memory [BitNetEngine] for session-level tests.
///
/// Streams [tokens] (one [TokenEvent] each, with [logprob]) after an optional
/// [perToken] delay, records the last prompt/options it saw, and reports how
/// its generator ended — an `async*` body unwinds through `finally` when the
/// subscription is cancelled, which is how real engines learn to stop.
class FakeEngine implements BitNetEngine {
  List<String> tokens;
  double logprob;
  Duration perToken;

  /// Thrown as a stream error after this many tokens (null = never).
  int? failAfter;

  /// Token counting rule (default: 1 token per 4 characters, rounded up).
  int Function(String text) tokenCounter;

  String? lastPrompt;
  GenerationOptions? lastOptions;
  final List<String> prompts = [];
  bool generatorTornDown = false;
  bool ranToCompletion = false;
  bool _loaded = true;
  bool _cancel = false;

  @override
  BitNetModelInfo model;

  FakeEngine({
    String reply = 'ok',
    List<String>? tokens,
    this.logprob = -0.1,
    this.perToken = Duration.zero,
    this.failAfter,
    int Function(String)? tokenCounter,
    this.model = BitNetCatalog.defaultModel,
  })  : tokens = tokens ?? reply.split(''),
        tokenCounter = tokenCounter ?? ((t) => (t.length + 3) ~/ 4);

  set reply(String r) => tokens = r.split('');

  @override
  bool get isLoaded => _loaded;

  @override
  Future<void> load({void Function(double progress)? onProgress}) async {
    _loaded = true;
  }

  @override
  Stream<TokenEvent> generateTokens(
    String prompt, {
    GenerationOptions options = const GenerationOptions(),
  }) async* {
    if (!_loaded) throw const BitNetNotLoadedException();
    lastPrompt = prompt;
    lastOptions = options;
    prompts.add(prompt);
    generatorTornDown = false;
    ranToCompletion = false;
    _cancel = false;
    try {
      var i = 0;
      for (final t in tokens) {
        if (perToken > Duration.zero) {
          await Future<void>.delayed(perToken);
        } else {
          await Future<void>.value();
        }
        if (_cancel) return;
        if (failAfter != null && i == failAfter) {
          throw const BitNetInferenceException(message: 'scripted failure');
        }
        yield TokenEvent(t, [logprob]);
        i++;
      }
      ranToCompletion = true;
    } finally {
      generatorTornDown = true;
    }
  }

  @override
  Future<int> countTokens(String text, {bool addBos = false}) async =>
      tokenCounter(text) + (addBos ? 1 : 0);

  @override
  Future<void> cancelGeneration() async => _cancel = true;

  @override
  Future<void> dispose() async => _loaded = false;
}
