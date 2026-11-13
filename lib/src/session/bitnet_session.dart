import 'dart:async';
import 'dart:convert';

import '../diagnostics/bitnet_log.dart';
import '../engine/bitnet_engine.dart';
import '../engine/generation_options.dart';
import '../exceptions/bitnet_exception.dart';
import '../model/bitnet_model_info.dart';
import 'chat_templates.dart';
import 'json_shape.dart';

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

/// Output style for [BitNetSession.summarize].
enum SummaryStyle {
  /// One short paragraph.
  paragraph,

  /// 3–7 concise bullets.
  bullets,

  /// A single sentence.
  tldr,
}

/// High-level, prompt-template-aware API over a [BitNetEngine].
///
/// * [chat] — multi-turn conversation with automatic history management.
/// * [extract] — structured JSON extraction with per-field confidence
///   (grammar-constrained, so the output always parses).
/// * [summarize] / [assist] — stateless one-shot helpers.
///
/// ```dart
/// final engine = BitNetEngine(model: BitNetCatalog.defaultModel);
/// await engine.load();
/// final session = BitNetSession(engine: engine,
///     systemPrompt: 'You are a friendly finance assistant.');
/// await for (final text in session.chat('How can I save more?')) {
///   stdout.write(text);
/// }
/// ```
///
/// The prompt template comes from `engine.model`, so it always matches the
/// loaded weights. Several sessions may share one engine (e.g. an extractor
/// and an advisor chat); their generations are queued, never interleaved. A
/// session does not own its engine — dispose the engine yourself.
class BitNetSession {
  final BitNetEngine engine;

  /// Hard cap on remembered turns (user + assistant entries). Within the cap,
  /// [chat] also drops the oldest turns whenever the prompt would not leave
  /// room for the reply in the model's context window.
  final int maxHistoryTurns;

  /// Decoding defaults for [chat], [summarize] and [assist]. ([extract]
  /// always decodes greedily.)
  final GenerationOptions options;

  final List<ChatTurn> _history = <ChatTurn>[];
  String? _systemPrompt;

  /// Chat turns run strictly in order: a turn's prompt is built only once the
  /// previous turn has settled, so its reply is part of the history.
  Future<void> _chatChain = Future<void>.value();

  BitNetSession({
    required this.engine,
    String? systemPrompt,
    this.maxHistoryTurns = 12,
    this.options = const GenerationOptions(),
  }) : _systemPrompt = systemPrompt; // ignore: prefer_initializing_formals

  /// The model this session formats prompts for (always `engine.model`).
  BitNetModelInfo get model => engine.model;

  ChatTemplateSpec get _template => chatTemplateFor(model.chatTemplate);

  /// Read-only view of the running conversation.
  List<ChatTurn> get history => List.unmodifiable(_history);

  /// The system prompt prepended to every [chat] turn.
  String? get systemPrompt => _systemPrompt;

  /// Replaces the system prompt. Pass `null` to remove it.
  void setSystemPrompt(String? prompt) => _systemPrompt = prompt;

  /// Wipes the conversation history (keeps the system prompt).
  void clearHistory() => _history.clear();

  // -------------------------------------------------------------------- chat

  /// Streams the assistant's reply to [message].
  ///
  /// Both turns are appended to [history] once the reply completes. If the
  /// stream is cancelled mid-reply, the partial reply is kept; if it fails
  /// (or produced nothing), the user turn is rolled back so the history never
  /// holds an unanswered message.
  ///
  /// Throws [BitNetNotLoadedException] synchronously if the engine is not
  /// loaded; emits [BitNetPromptTooLongException] if [message] alone does not
  /// fit in the context window.
  Stream<String> chat(String message, {GenerationOptions? options}) {
    if (!engine.isLoaded) throw const BitNetNotLoadedException();
    final o = options ?? this.options;
    final out = StreamController<String>();
    final reply = StringBuffer();
    StreamSubscription<TokenEvent>? sub;
    var consumerGone = false;
    var settled = false;
    final finished = Completer<void>();

    void settle({required bool keepReply}) {
      if (settled) return;
      settled = true;
      if (keepReply && reply.isNotEmpty) {
        final text = _sanitizeReply(reply.toString());
        BitNetLog.log('session', 'chat reply ${text.length}ch');
        _history.add(ChatTurn(ChatRole.assistant, text));
        _trimToCap();
      } else if (_history.isNotEmpty && _history.last.role == ChatRole.user) {
        _history.removeLast();
      }
      if (!finished.isCompleted) finished.complete();
    }

    out.onCancel = () async {
      consumerGone = true;
      // Cancelling the engine subscription stops the model (native: at once).
      await sub?.cancel();
      settle(keepReply: true);
    };

    _chatChain = _chatChain.then((_) async {
      if (consumerGone) return;
      _history.add(ChatTurn(ChatRole.user, message));
      try {
        final prompt = await _fitChatPrompt(o);
        if (consumerGone) {
          settle(keepReply: false);
          return;
        }
        sub = engine.generateTokens(prompt, options: o).listen(
          (e) {
            reply.write(e.text);
            if (!consumerGone) out.add(e.text);
          },
          onError: (Object e, StackTrace st) {
            settle(keepReply: false);
            if (!consumerGone) out.addError(e, st);
          },
          onDone: () => settle(keepReply: true),
          cancelOnError: true,
        );
        await finished.future;
      } catch (e, st) {
        settle(keepReply: false);
        if (!consumerGone) out.addError(e, st);
      } finally {
        await out.close();
      }
    });
    return out.stream;
  }

  /// Formats the history into a prompt that leaves room for the reply,
  /// dropping the oldest turns as needed.
  Future<String> _fitChatPrompt(GenerationOptions o) async {
    _trimToCap();
    final reserve = o.maxNewTokens == 0 ? 256 : o.maxNewTokens;
    final budget = model.contextLength - reserve;
    while (true) {
      final prompt = _template.format(_history, _systemPrompt);
      final n = await engine.countTokens(prompt, addBos: o.addBos);
      if (n <= budget) return prompt;
      if (_history.length <= 1) {
        throw BitNetPromptTooLongException(
            'message needs $n tokens; ${budget < 0 ? 0 : budget} available '
            '(context ${model.contextLength} minus $reserve reserved for the reply)');
      }
      BitNetLog.log('session', 'prompt $n tokens > budget $budget — dropping oldest turn');
      _history.removeAt(0);
      _dropLeadingAssistant();
    }
  }

  void _trimToCap() {
    if (_history.length > maxHistoryTurns) {
      _history.removeRange(0, _history.length - maxHistoryTurns);
    }
    _dropLeadingAssistant();
  }

  /// A window must open on a user turn: every template renders a leading bare
  /// assistant reply as out-of-distribution text.
  void _dropLeadingAssistant() {
    while (_history.isNotEmpty && _history.first.role == ChatRole.assistant) {
      _history.removeAt(0);
    }
  }

  /// Cuts [reply] at the first end-of-turn / hallucinated turn-opener marker
  /// (belt and braces over the streaming detector) and trims it.
  String _sanitizeReply(String reply) {
    var cut = -1;
    for (final m in _template.stopMarkers) {
      final idx = reply.indexOf(m);
      if (idx != -1 && (cut == -1 || idx < cut)) cut = idx;
    }
    return (cut == -1 ? reply : reply.substring(0, cut)).trim();
  }

  // ----------------------------------------------------------------- extract

  /// Extracts a JSON object of the given [shape] from [text].
  ///
  /// Decoding is greedy and grammar-constrained, so the result always has
  /// exactly [shape]'s keys and types. [ExtractResult.confidence] tells you
  /// how sure the model was — e.g. ask the user to confirm below 0.5.
  ///
  /// * [instructions] — what the task is ("Extract the expense from the
  ///   user's sentence"). The field list is appended automatically.
  /// * [examples] — 2–3 worked examples; the biggest accuracy lever.
  /// * [today] — adds today's date so relative dates ("yesterday") resolve.
  ///
  /// Everything except [text] forms a stable prompt prefix that the native
  /// engine keeps in its KV cache, so repeated extractions with the same
  /// shape/instructions/examples only process the new text.
  ///
  /// Throws [BitNetExtractionException] if generation stopped before the
  /// object was complete (e.g. [maxNewTokens] too small, or cancelled).
  Future<ExtractResult> extract(
    String text, {
    required JsonShape shape,
    String? instructions,
    List<ExtractExample> examples = const [],
    DateTime? today,
    int maxNewTokens = 192,
  }) async {
    if (!engine.isLoaded) throw const BitNetNotLoadedException();
    final system = StringBuffer()
      ..writeln((instructions ?? 'Extract structured data from the text.').trim())
      ..writeln()
      ..writeln('Reply with only a JSON object with these fields:')
      ..writeln(shape.describe())
      ..writeln('Use null for anything the text does not state.');
    if (examples.isNotEmpty) {
      system
        ..writeln()
        ..writeln('Examples:');
      for (final ex in examples) {
        system
          ..writeln('Text: ${ex.text.trim()}')
          ..writeln('JSON: ${jsonEncode(ex.output)}');
      }
    }
    if (today != null) {
      system
        ..writeln()
        ..write("Today's date: ${_isoDate(today)}.");
    }
    final prompt = _template.format(
      [ChatTurn(ChatRole.user, text.trim())],
      system.toString().trim(),
    );

    final events = <TokenEvent>[];
    await for (final e in engine.generateTokens(
      prompt,
      options: GenerationOptions.precise(
          maxNewTokens: maxNewTokens, grammar: shape.toGbnf()),
    )) {
      events.add(e);
    }
    final raw = events.map((e) => e.text).join().trim();
    final result = shape.parse(raw, events);
    BitNetLog.log('session', 'extract → $result');
    return result;
  }

  static String _isoDate(DateTime d) =>
      '${d.year.toString().padLeft(4, '0')}-${d.month.toString().padLeft(2, '0')}'
      '-${d.day.toString().padLeft(2, '0')}';

  // --------------------------------------------------------- one-shot helpers

  /// Summarises [text] (does not touch [history]).
  Future<String> summarize(
    String text, {
    SummaryStyle style = SummaryStyle.paragraph,
    GenerationOptions? options,
  }) {
    final instruction = switch (style) {
      SummaryStyle.bullets =>
        'Summarise the following text as a concise bullet list (3-7 bullets):',
      SummaryStyle.tldr => 'Write a one-sentence TL;DR of the following text:',
      SummaryStyle.paragraph =>
        'Summarise the following text in a single short paragraph:',
    };
    return _oneShot('$instruction\n\n${text.trim()}', options);
  }

  /// Answers [question], optionally grounded in [context] (e.g. retrieved
  /// RAG snippets). Does not touch [history].
  Future<String> assist(
    String question, {
    String? context,
    GenerationOptions? options,
  }) {
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
        ..write(question.trim());
    } else {
      body
        ..writeln('Answer the following question concisely and accurately.')
        ..writeln()
        ..writeln('### Question')
        ..write(question.trim());
    }
    return _oneShot(body.toString(), options);
  }

  Future<String> _oneShot(String userText, GenerationOptions? options) async {
    if (!engine.isLoaded) throw const BitNetNotLoadedException();
    final prompt = _template.format(
        [ChatTurn(ChatRole.user, userText)], _systemPrompt);
    final buf = StringBuffer();
    await for (final e
        in engine.generateTokens(prompt, options: options ?? this.options)) {
      buf.write(e.text);
    }
    return _sanitizeReply(buf.toString());
  }

  /// Stops the running generation (chat, extract, summarize or assist). Its
  /// stream / Future completes with what was produced so far. No-op if idle.
  Future<void> stop() => engine.cancelGeneration();
}
