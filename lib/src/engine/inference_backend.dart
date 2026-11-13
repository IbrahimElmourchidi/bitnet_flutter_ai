import 'dart:typed_data';

import '../model/bitnet_model_info.dart' show EngineBackend;
import 'generation_options.dart';

/// Synchronous, single-threaded view of one native inference context — the
/// surface the inference isolate drives.
///
/// Production uses `NativeLibrary` (dart:ffi over the bn_* C ABI). Tests
/// substitute a fake via `NativeEngine.withBackend`, which is how the isolate
/// protocol (queueing, cancellation, teardown) is exercised without a model.
abstract interface class InferenceBackend {
  /// Loads the model and creates the context. Throws `BitNetInitException`.
  void init(String modelPath, {required int nCtx, required int nThreads});

  /// Registers the address of a shared `int32` cancellation flag (0 = none).
  /// While it holds a non-zero value, [feedPrompt] returns
  /// [PromptStatus.aborted] and [nextToken] returns `null`.
  void setAbortFlag(int address);

  /// Configures the sampler (and grammar) for the next generation.
  /// Throws `BitNetInferenceException` for an invalid grammar.
  void setSampling(GenerationOptions options);

  /// Tokenizes and prefills [prompt] (reusing any shared KV prefix).
  /// Throws `BitNetPromptTooLongException` / `BitNetInferenceException`.
  PromptStatus feedPrompt(String prompt, {required bool addBos});

  /// Produces the next token, or `null` at end-of-turn / context full / abort.
  NextToken? nextToken();

  /// Number of tokens [text] tokenizes to.
  int tokenCount(String text, {required bool addBos});

  /// Leading prompt tokens the last [feedPrompt] reused from the KV cache.
  int get lastPromptReused;

  /// Frees the context. Safe to call more than once.
  void free();
}

/// Outcome of [InferenceBackend.feedPrompt].
enum PromptStatus { ok, aborted }

/// One generated token: its raw UTF-8 bytes (possibly a fragment of a
/// multi-byte character) and the model's log-probability for it.
final class NextToken {
  final Uint8List bytes;
  final double logprob;
  const NextToken(this.bytes, this.logprob);
}

/// Creates the backend inside the inference isolate. Must be a top-level or
/// static function so it can be sent to `Isolate.spawn`.
typedef InferenceBackendFactory = InferenceBackend Function(
    EngineBackend backend);
