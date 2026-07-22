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

    late StreamSubscription<String> sub;
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
        if (_history.isNotEmpty && _history.last.role == ChatRole.user) {
          _history.removeLast();
        }
        controller.addError(err, st);
        controller.close();
      },
      onDone: () {
        BitNetLog.log('session',
            'chat() done reply=${buffer.length}ch: "${_truncate(buffer.toString(), 200)}"');
        _history.add(ChatTurn(ChatRole.assistant, buffer.toString()));
        _trimHistory();
        controller.close();
      },
      cancelOnError: true,
    );

    controller.onCancel = () => sub.cancel();
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
    if (_history.length <= maxHistoryTurns) return;
    final overflow = _history.length - maxHistoryTurns;
    _history.removeRange(0, overflow);
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
  /// `System:`/`User:`/`Assistant:` labels, Falcon3's `<|user|>` tags, ChatML,
  /// …); feeding the wrong one is out-of-distribution and produces degenerate
  /// output (a runaway stream of `!`). The per-family formatters live in
  /// [kChatTemplates]. NOTE: the template embedded in a GGUF's
  /// `tokenizer.chat_template` metadata can be corrupt (BitNet 2B-4T's emits
  /// `Human: … BITNETAssistant: …`), so formats are hardcoded per family — do
  /// NOT drive them from `bn_meta('tokenizer.chat_template')`.
  String _formatChatPrompt(List<ChatTurn> turns, String? systemPrompt) =>
      chatTemplateFor(model.chatTemplate).format(turns, systemPrompt);

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
