import 'dart:async';

import '../diagnostics/bitnet_log.dart';
import '../engine/bitnet_engine.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart';
import 'chat_templates.dart';

/// Role of a participant in a chat conversation.
enum ChatRole { system, user, assistant }

/// Single turn (one role + one body) in a [BitNetSession] history.
class ChatTurn {
  final ChatRole role;
  final String content;
  const ChatTurn(this.role, this.content);

  @override
  String toString() => '${role.name}: $content';
}

/// A stateful chat wrapper around a [BitNetEngine].
///
/// Maintains the running list of [ChatTurn]s, applies a simple
/// chat-formatted prompt template before each call to the underlying
/// engine, and exposes high-level helpers ([summarize], [assist]) that
/// are stateless and do not mutate the session history.
///
/// Typical usage:
/// ```dart
/// final engine = BitNetEngine();
/// await engine.load();
/// final session = BitNetSession(engine: engine);
/// session.setSystemPrompt('You are a friendly Flutter expert.');
/// await for (final piece in session.chat('What is BitNet?')) {
///   stdout.write(piece);
/// }
/// ```
class BitNetSession {
  final BitNetEngine engine;
  final BitNetModelInfo model;

  /// Maximum number of past turns (user + assistant pairs) kept in the
  /// prompt. Older turns are silently truncated to stay within the model's
  /// context window. Defaults to 12 turns ≈ 6 exchanges.
  final int maxHistoryTurns;

  /// Default token budget for [chat] / [assist] / [summarize] when the
  /// caller does not pass `maxNewTokens` explicitly.
  final int defaultMaxNewTokens;

  final List<ChatTurn> _history = <ChatTurn>[];
  String? _systemPrompt;

  BitNetSession({
    required this.engine,
    BitNetModelInfo? model,
    this.maxHistoryTurns = 12,
    this.defaultMaxNewTokens = 512,
  }) : model = model ?? BitNetCatalog.defaultModel;

  /// Read-only view of the running conversation.
  List<ChatTurn> get history => List.unmodifiable(_history);

  /// Replaces the system prompt prepended to every turn. Pass `null` to
  /// remove it.
  void setSystemPrompt(String? prompt) {
    _systemPrompt = prompt;
  }

  /// Wipes the conversation history (keeps the system prompt).
  void clearHistory() => _history.clear();

  /// Streams the assistant's reply to [userMessage] piece-by-piece while
  /// also appending both turns to [history] when the stream completes.
  Stream<String> chat(
    String userMessage, {
    int? maxNewTokens,
  }) {
    if (!engine.isLoaded) {
      throw const BitNetNotLoadedException();
    }
    final controller = StreamController<String>();
    final buffer = StringBuffer();

    // Snapshot turns sent to the model — append the user turn _now_ so it
    // appears in the formatted prompt, but don't commit to history until
    // we have a full assistant reply.
    _history.add(ChatTurn(ChatRole.user, userMessage));
    _trimHistory();
    final prompt = _formatChatPrompt(_history, _systemPrompt);
    BitNetLog.log('session',
        'chat() user=${userMessage.length}ch history=${_history.length} '
        'prompt=${prompt.length}ch maxNewTokens=${maxNewTokens ?? defaultMaxNewTokens}');
    BitNetLog.log('session.prompt', '\n$prompt');

    // The history must reach exactly one of two end states, exactly once:
    // either the assistant reply is committed, or the user turn we optimistically
    // appended above is rolled back. Leaving a user turn with no assistant reply
    // renders every later prompt as two consecutive user turns, which is
    // malformed under every template and never recovers.
    var settled = false;

    void commitReply() {
      if (settled) return;
      settled = true;
      // Defensive: if a hallucinated turn-opener slipped past the streaming
      // TurnStopDetector (e.g. a marker split oddly across pieces, or an
      // opener variant not in the list), strip it before it enters history.
      // Without this, one leaked "User: …/ANSWER: …" tail is fed back on every
      // subsequent turn and derails the whole conversation.
      final reply = _sanitizeReply(buffer.toString());
      BitNetLog.log('session',
          'chat() done reply=${reply.length}ch: "${_truncate(reply, 200)}"');
      _history.add(ChatTurn(ChatRole.assistant, reply));
      _trimHistory();
    }

    void rollbackUserTurn() {
      if (settled) return;
      settled = true;
      if (_history.isNotEmpty && _history.last.role == ChatRole.user) {
        _history.removeLast();
      }
    }

    late StreamSubscription<String> sub;
    try {
      sub = engine
          .generate(prompt, maxNewTokens: maxNewTokens ?? defaultMaxNewTokens)
          .listen(
        (piece) {
          buffer.write(piece);
          controller.add(piece);
        },
        onError: (Object err, StackTrace st) {
          BitNetLog.log('session', 'chat() error: $err');
          // Roll back the user turn on hard failures so the next call doesn't
          // ship a half-broken history.
          rollbackUserTurn();
          controller.addError(err, st);
          controller.close();
        },
        onDone: () {
          commitReply();
          controller.close();
        },
        cancelOnError: true,
      );
    } catch (_) {
      // generate() rejects synchronously when the engine is busy or unloaded,
      // and the user turn is already in `_history` by then. Undo it before the
      // error escapes, or the next call renders two consecutive user turns.
      rollbackUserTurn();
      rethrow;
    }

    // Fires when the consumer walks away mid-stream (widget disposed, tab
    // switched, `await for` broken out of) — onDone never runs in that case, so
    // the history has to be settled here too. Cancelling `sub` propagates down
    // to the engine, which stops the worker.
    controller.onCancel = () async {
      await sub.cancel();
      if (buffer.isEmpty) {
        rollbackUserTurn();
      } else {
        BitNetLog.log('session', 'chat() cancelled — committing partial reply');
        commitReply();
      }
    };
    return controller.stream;
  }

  /// Stateless summarisation helper — does not touch [history].
  ///
  /// [style] picks one of: `'paragraph'` (default), `'bullets'`, `'tldr'`.
  Future<String> summarize(
    String text, {
    String style = 'paragraph',
    int? maxNewTokens,
  }) async {
    if (!engine.isLoaded) {
      throw const BitNetNotLoadedException();
    }
    final instruction = switch (style) {
      'bullets' =>
        'Summarise the following text as a concise bullet list (3-7 bullets):',
      'tldr' =>
        'Write a one-sentence TL;DR of the following text:',
      _ => 'Summarise the following text in a single short paragraph:',
    };
    final prompt = _formatInstructionPrompt(
      instruction: instruction,
      input: text,
      systemPrompt: _systemPrompt,
    );
    return _collect(engine.generate(
      prompt,
      maxNewTokens: maxNewTokens ?? defaultMaxNewTokens,
    ));
  }

  /// Stateless assist/QA helper — answers [question] using optional
  /// supporting [context] (e.g. retrieved RAG snippets). Does not touch
  /// [history].
  Future<String> assist(
    String question, {
    String? context,
    int? maxNewTokens,
  }) async {
    if (!engine.isLoaded) {
      throw const BitNetNotLoadedException();
    }
    final body = StringBuffer();
    if (context != null && context.trim().isNotEmpty) {
      body
        ..writeln('Use the following context to answer the question. '
            'If the answer is not in the context, say you do not know.')
        ..writeln()
        ..writeln('### Context')
        ..writeln(context.trim())
        ..writeln()
        ..writeln('### Question')
        ..writeln(question.trim());
    } else {
      body
        ..writeln('Answer the following question concisely and accurately.')
        ..writeln()
        ..writeln('### Question')
        ..writeln(question.trim());
    }
    final prompt = _formatInstructionPrompt(
      instruction: body.toString(),
      input: '',
      systemPrompt: _systemPrompt,
    );
    return _collect(engine.generate(
      prompt,
      maxNewTokens: maxNewTokens ?? defaultMaxNewTokens,
    ));
  }

  /// Disposes the underlying engine. Subsequent calls throw.
  Future<void> dispose() async {
    _history.clear();
    await engine.dispose();
  }

  /// Interrupts the in-flight generation (if any) for [chat], [summarize]
  /// or [assist]. The active stream / Future resolves with whatever was
  /// generated up to the point of cancellation.
  ///
  /// Safe to call when nothing is in-flight — it is a no-op in that case.
  Future<void> stop() async {
    BitNetLog.log('session', 'stop() requested');
    await engine.cancelGeneration();
  }

  // ---------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------

  void _trimHistory() {
    if (_history.length > maxHistoryTurns) {
      _history.removeRange(0, _history.length - maxHistoryTurns);
    }
    // Trimming cuts on a raw entry count, so it can leave the window opening on
    // an assistant turn. Every template then renders a prompt that starts with a
    // bare model reply with no user turn before it — out of distribution, and
    // for `bitnetLabeled` it emits content before the first `Human:` label.
    // Drop those orphans so the window always starts on a user turn.
    while (_history.isNotEmpty && _history.first.role == ChatRole.assistant) {
      _history.removeAt(0);
    }
  }

  static Future<String> _collect(Stream<String> s) async {
    final buf = StringBuffer();
    await for (final piece in s) {
      buf.write(piece);
    }
    return buf.toString().trim();
  }

  static String _truncate(String s, int n) =>
      s.length <= n ? s : '${s.substring(0, n)}…';

  /// Formats a multi-turn chat using the template family declared by [model]
  /// (`model.chatTemplate`), resolved via [chatTemplateFor] in
  /// `chat_templates.dart`.
  ///
  /// Each model family was supervised-fine-tuned on a specific format (BitNet's
  /// `Human:`/`BITNETAssistant:` labels, Falcon3's `<|user|>` tags, ChatML, …);
  /// feeding the wrong one is out-of-distribution and produces wrong or
  /// degenerate output. The per-family formatters live in [kChatTemplates] and
  /// mirror each model's embedded `tokenizer.chat_template`; they are hardcoded
  /// (rather than read via `bn_meta`) so the Dart side controls stop markers and
  /// history assembly, but they must MATCH the GGUF's template — the BitNet
  /// 2B-4T one is `Human: … BITNETAssistant: …`, not the old `System:/User:`
  /// labels (which gave canned, wrong answers).
  String _formatChatPrompt(List<ChatTurn> turns, String? systemPrompt) =>
      chatTemplateFor(model.chatTemplate).format(turns, systemPrompt);

  /// Truncates [reply] at the first hallucinated turn-opener and trims
  /// surrounding whitespace, so a fabricated next turn never enters history.
  ///
  /// The streaming [TurnStopDetector] normally strips these live, but this is a
  /// belt-and-braces pass on the fully-assembled reply: it uses the SAME stop
  /// markers as the detector (from the model's [ChatTemplateSpec]) and cuts at
  /// the earliest one found anywhere in the text. Idempotent and cheap.
  String _sanitizeReply(String reply) {
    final markers = chatTemplateFor(model.chatTemplate).stopMarkers;
    var cut = -1;
    for (final m in markers) {
      final idx = reply.indexOf(m);
      if (idx != -1 && (cut == -1 || idx < cut)) cut = idx;
    }
    final trimmed = cut == -1 ? reply : reply.substring(0, cut);
    return trimmed.trim();
  }

  String _formatInstructionPrompt({
    required String instruction,
    required String input,
    String? systemPrompt,
  }) {
    final body = StringBuffer()..write(instruction.trim());
    if (input.trim().isNotEmpty) {
      body
        ..write('\n\n')
        ..write(input.trim());
    }
    return _formatChatPrompt(
      [ChatTurn(ChatRole.user, body.toString())],
      systemPrompt,
    );
  }
}
