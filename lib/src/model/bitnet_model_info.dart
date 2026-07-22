import 'package:flutter/foundation.dart';

/// High-level capability tag describing what a model is good at.
///
/// Used by callers (and the catalog filters) to pick a sensible default for
/// a given workload. A single model usually advertises several capabilities.
enum BitNetCapability {
  /// Open-ended chat / instruction following.
  chat,

  /// Short-form summarisation (paragraph / bullet list).
  summarization,

  /// Question answering grounded in supplied context (RAG-friendly).
  questionAnswering,

  /// Generic English text generation / completion.
  textGeneration,

  /// Code-related completion, explanation or refactoring.
  code,
}

/// On-disk quantisation format of the GGUF weights.
///
/// BitNet uses ternary weights packaged as `i2_s` (2-bit signed packed) or
/// `tl1` / `tl2` (lookup-table optimised variants). Other GGUF formats are
/// listed for forward compatibility — callers should not assume only one
/// value is ever supported.
enum BitNetQuantization {
  /// Native BitNet 2-bit signed packed weights (`I2_S`).
  i2s,

  /// 8-bit on-disk format (`TL1`).
  tl1,

  /// 8-bit on-disk format (`TL2`).
  tl2,

  /// Generic 4-bit GGUF (`Q4_K_M` and friends), reserved for future models.
  q4km,

  /// Ternary weights in PrismML's `Q2_0` (g128) format — a distinct ggml
  /// tensor type that only loads with PrismML's llama.cpp fork, not the
  /// bitnet.cpp stack. Paired with [EngineBackend.prismQ2].
  q2_0,

  /// Catch-all for non-BitNet GGUFs.
  other,
}

/// The chat prompt / stop-marker convention a model was fine-tuned on.
///
/// The wrong template for a model produces degenerate output (in the worst
/// case a runaway stream of a single token), so every [BitNetModelInfo]
/// declares the family its GGUF expects. Resolved to a concrete
/// `ChatTemplateSpec` by `kChatTemplates` in `session/chat_templates.dart`.
enum ChatTemplateFamily {
  /// BitNet b1.58 2B-4T: `System: `/`User: `/`Assistant: ` word labels with
  /// `<|eot_id|>` turn terminators. The original (and default) template.
  bitnetLabeled,

  /// Falcon3 Instruct: `<|system|>` / `<|user|>` / `<|assistant|>` tags,
  /// newline-separated, `<|endoftext|>` terminator.
  falconTagged,

  /// Falcon-E Instruct: standard ChatML (`<|im_start|>`/`<|im_end|>`).
  chatml,

  /// Qwen3-derived (PrismML Ternary Bonsai): ChatML with an empty
  /// `<think></think>` prefill to suppress slow reasoning blocks.
  chatmlQwen3,
}

/// Which native inference stack a model requires.
///
/// The two stacks use incompatible ggml ABIs and cannot share libraries, so
/// each model routes to the correct backend at load time (see `NativeLibrary`
/// and `engine_native.dart`).
enum EngineBackend {
  /// Microsoft bitnet.cpp — I2_S / TL1 / TL2 ternary kernels.
  /// `libbitnet_bridge.so` (and its `libllama.so`/`libggml.so`).
  bitnetCpp,

  /// PrismML llama.cpp fork — Q2_0 (g128) ternary kernels.
  /// Self-contained `libprism_bridge.so`.
  prismQ2,
}

/// Immutable descriptor for a downloadable GGUF model artifact.
///
/// Instances are intentionally `const` — the canonical catalog lives in
/// [BitNetCatalog]. Application code should rarely build its own
/// `BitNetModelInfo`; instead pick from `BitNetCatalog.all` or look up by
/// id with `BitNetCatalog.byId(...)`.
@immutable
final class BitNetModelInfo {
  /// Unique model identifier, typically the HuggingFace repo slug.
  final String id;

  /// Human-readable display name (suitable for a model picker UI).
  final String displayName;

  /// Marketing / family label, e.g. `'BitNet b1.58'`.
  final String family;

  /// Parameter count in billions (e.g. `2.0` for a 2B-parameter model).
  final double paramsBillion;

  /// On-disk quantisation format.
  final BitNetQuantization quantization;

  /// SPDX license identifier for the weights (e.g. `'MIT'`, `'Apache-2.0'`).
  final String license;

  /// Capability tags describing what the model is best at.
  final Set<BitNetCapability> capabilities;

  /// Filename used when storing the model on-device.
  final String ggufFileName;

  /// Approximate compressed file size in bytes. Used for progress reporting.
  final int ggufSizeBytes;

  /// Expected SHA-256 hex digest of the GGUF file.
  ///
  /// If this starts with `'PLACEHOLDER'`, verification is skipped and a
  /// warning is logged on first download.
  final String ggufSha256;

  /// Direct download URL for the GGUF file.
  final String ggufDownloadUrl;

  /// Hard floor for physical RAM, in bytes — below this the runtime will
  /// refuse to load the model.
  final int minimumRamBytes;

  /// Recommended physical RAM for a comfortable user experience, in bytes.
  /// Devices below this still load but will likely thrash on long contexts.
  final int recommendedRamBytes;

  /// Context window this package configures for the model, in tokens.
  ///
  /// Note: the native layer allocates the full KV cache at this size on load,
  /// so entries deliberately cap this well below a model's native maximum to
  /// bound mobile RAM use (conversation history is trimmed separately).
  final int contextLength;

  /// The chat template / stop-marker convention this model was tuned on.
  final ChatTemplateFamily chatTemplate;

  /// Which native inference stack this model requires.
  final EngineBackend engineBackend;

  /// Returns true when [ggufSha256] is a real hash and verification should run.
  bool get hasPinnedHash => !ggufSha256.startsWith('PLACEHOLDER');

  const BitNetModelInfo._({
    required this.id,
    required this.displayName,
    required this.family,
    required this.paramsBillion,
    required this.quantization,
    required this.license,
    required this.capabilities,
    required this.ggufFileName,
    required this.ggufSizeBytes,
    required this.ggufSha256,
    required this.ggufDownloadUrl,
    required this.minimumRamBytes,
    required this.recommendedRamBytes,
    required this.contextLength,
    this.chatTemplate = ChatTemplateFamily.bitnetLabeled,
    this.engineBackend = EngineBackend.bitnetCpp,
  });

  /// Test-only constructor for building [BitNetModelInfo] instances with
  /// custom metadata in unit tests. Production code must use
  /// [BitNetCatalog].
  @visibleForTesting
  const BitNetModelInfo.forTest({
    required this.id,
    required this.displayName,
    this.family = 'BitNet b1.58',
    this.paramsBillion = 2.0,
    this.quantization = BitNetQuantization.i2s,
    this.license = 'MIT',
    this.capabilities = const {
      BitNetCapability.chat,
      BitNetCapability.textGeneration,
    },
    required this.ggufFileName,
    required this.ggufSizeBytes,
    required this.ggufSha256,
    required this.ggufDownloadUrl,
    required this.minimumRamBytes,
    int? recommendedRamBytes,
    required this.contextLength,
    this.chatTemplate = ChatTemplateFamily.bitnetLabeled,
    this.engineBackend = EngineBackend.bitnetCpp,
  }) : recommendedRamBytes = recommendedRamBytes ?? minimumRamBytes;

  @override
  String toString() =>
      'BitNetModelInfo($displayName, ${paramsBillion}B, ${quantization.name})';
}

/// Catalog of curated models known to ship cleanly with the bridge binaries
/// in this package.
///
/// The catalog is intentionally read-only. Adding a model means landing a PR
/// against this package so that the supported set is auditable and pinned by
/// SHA-256.
final class BitNetCatalog {
  BitNetCatalog._();

  /// Microsoft BitNet b1.58 2B-4T, I2_S GGUF quantisation.
  ///
  /// Source: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
  static const BitNetModelInfo bitnet2B4T = BitNetModelInfo._(
    id: 'microsoft/bitnet-b1.58-2B-4T',
    displayName: 'BitNet b1.58 2B-4T',
    family: 'BitNet b1.58',
    paramsBillion: 2.0,
    quantization: BitNetQuantization.i2s,
    license: 'MIT',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
    },
    ggufFileName: 'ggml-model-i2_s.gguf',
    ggufSizeBytes: 1187801280,
    // Pinned SHA-256 of the official upload (see CHANGELOG).
    ggufSha256:
        '4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162',
    ggufDownloadUrl:
        'https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 3 * 1024 * 1024 * 1024, // 3 GB hard floor
    recommendedRamBytes: 4 * 1024 * 1024 * 1024, // 4 GB recommended
    contextLength: 4096,
  );

  // ---------------------------------------------------------------------------
  // TII Falcon3 Instruct (1.58-bit). Pre-built I2_S GGUFs with GGUF
  // architecture "llama" — load through the same bitnet.cpp native stack as
  // 2B-4T with no rebuild. Chat template: <|system|>/<|user|>/<|assistant|>.
  // Source: https://huggingface.co/tiiuae/Falcon3-<n>-Instruct-1.58bit-GGUF
  // contextLength capped at 4096 (native max up to 32768) to bound KV cache.
  // ---------------------------------------------------------------------------

  static const BitNetModelInfo falcon3_1bInstruct = BitNetModelInfo._(
    id: 'tiiuae/Falcon3-1B-Instruct-1.58bit',
    displayName: 'Falcon3 1B Instruct (1.58-bit)',
    family: 'Falcon3',
    paramsBillion: 1.67,
    quantization: BitNetQuantization.i2s,
    license: 'falcon-llm-license',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
    },
    ggufFileName: 'falcon3-1b-instruct-i2_s.gguf',
    ggufSizeBytes: 1361904672,
    ggufSha256:
        '0ecef8ad9bcb1b7d3b73bac7b0237daf2faa962b60cbbaf62c86ff51a39444b4',
    ggufDownloadUrl:
        'https://huggingface.co/tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 3 * 1024 * 1024 * 1024,
    recommendedRamBytes: 4 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.falconTagged,
  );

  static const BitNetModelInfo falcon3_3bInstruct = BitNetModelInfo._(
    id: 'tiiuae/Falcon3-3B-Instruct-1.58bit',
    displayName: 'Falcon3 3B Instruct (1.58-bit)',
    family: 'Falcon3',
    paramsBillion: 3.23,
    quantization: BitNetQuantization.i2s,
    license: 'falcon-llm-license',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
    },
    ggufFileName: 'falcon3-3b-instruct-i2_s.gguf',
    ggufSizeBytes: 2221465632,
    ggufSha256:
        '2ee9723dc1abcc53f231ef1637bdd7c1ec1dbaf132c2c59873100cfb48b41455',
    ggufDownloadUrl:
        'https://huggingface.co/tiiuae/Falcon3-3B-Instruct-1.58bit-GGUF/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 4 * 1024 * 1024 * 1024,
    recommendedRamBytes: 5 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.falconTagged,
  );

  static const BitNetModelInfo falcon3_7bInstruct = BitNetModelInfo._(
    id: 'tiiuae/Falcon3-7B-Instruct-1.58bit',
    displayName: 'Falcon3 7B Instruct (1.58-bit)',
    family: 'Falcon3',
    paramsBillion: 7.46,
    quantization: BitNetQuantization.i2s,
    license: 'falcon-llm-license',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
      BitNetCapability.code,
    },
    ggufFileName: 'falcon3-7b-instruct-i2_s.gguf',
    ggufSizeBytes: 3278680768,
    ggufSha256:
        '612ab67d4c5fb77d9f810eb521eb4a477dae46df1a85ca501018490e2dac35c6',
    ggufDownloadUrl:
        'https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 5 * 1024 * 1024 * 1024,
    recommendedRamBytes: 6 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.falconTagged,
  );

  static const BitNetModelInfo falcon3_10bInstruct = BitNetModelInfo._(
    id: 'tiiuae/Falcon3-10B-Instruct-1.58bit',
    displayName: 'Falcon3 10B Instruct (1.58-bit)',
    family: 'Falcon3',
    paramsBillion: 10.31,
    quantization: BitNetQuantization.i2s,
    license: 'falcon-llm-license',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
      BitNetCapability.code,
    },
    ggufFileName: 'falcon3-10b-instruct-i2_s.gguf',
    ggufSizeBytes: 3991393696,
    ggufSha256:
        'e37945ee82693a6541b5fa5484f0e24787c04a9ce95e6e377f68a6b15f139c1f',
    ggufDownloadUrl:
        'https://huggingface.co/tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 6 * 1024 * 1024 * 1024,
    recommendedRamBytes: 7 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.falconTagged,
  );

  // ---------------------------------------------------------------------------
  // TII Falcon-E Instruct (1.58-bit). Pre-built I2_S GGUFs, GGUF architecture
  // "llama". Chat template: ChatML (<|im_start|>/<|im_end|>). BOS token is the
  // literal ">>ABSTRACT<<" special token (see chat_templates.dart).
  // Source: https://huggingface.co/tiiuae/Falcon-E-<n>-Instruct-GGUF
  // ---------------------------------------------------------------------------

  static const BitNetModelInfo falconE_1bInstruct = BitNetModelInfo._(
    id: 'tiiuae/Falcon-E-1B-Instruct',
    displayName: 'Falcon-E 1B Instruct',
    family: 'Falcon-E',
    paramsBillion: 1.72,
    quantization: BitNetQuantization.i2s,
    license: 'falcon-llm-license',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
    },
    ggufFileName: 'falcon-e-1b-instruct-i2_s.gguf',
    ggufSizeBytes: 666324256,
    ggufSha256:
        'feb7478007e916d26bb807cb1a01cc45ac16f197e355d8c30aed25e550ecd73b',
    ggufDownloadUrl:
        'https://huggingface.co/tiiuae/Falcon-E-1B-Instruct-GGUF/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 2 * 1024 * 1024 * 1024,
    recommendedRamBytes: 3 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.chatml,
  );

  static const BitNetModelInfo falconE_3bInstruct = BitNetModelInfo._(
    id: 'tiiuae/Falcon-E-3B-Instruct',
    displayName: 'Falcon-E 3B Instruct',
    family: 'Falcon-E',
    paramsBillion: 3.05,
    quantization: BitNetQuantization.i2s,
    license: 'falcon-llm-license',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
    },
    ggufFileName: 'falcon-e-3b-instruct-i2_s.gguf',
    ggufSizeBytes: 999908608,
    ggufSha256:
        'acef6896311c5d0713d80e4c7f7bc2ffa1fa183d905e2ca9236545372f434255',
    ggufDownloadUrl:
        'https://huggingface.co/tiiuae/Falcon-E-3B-Instruct-GGUF/resolve/main/ggml-model-i2_s.gguf',
    minimumRamBytes: 3 * 1024 * 1024 * 1024,
    recommendedRamBytes: 4 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.chatml,
  );

  // ---------------------------------------------------------------------------
  // PrismML "Ternary Bonsai" (Qwen3-derived, Q2_0 g128). Requires the SEPARATE
  // PrismML llama.cpp backend (libprism_bridge.so) — a different ggml ABI than
  // bitnet.cpp. Chat template: Qwen3 ChatML with think-suppression.
  // Source: https://huggingface.co/collections/prism-ml/ternary-bonsai
  // Not available on web (no WASM build of the prism fork yet).
  // ---------------------------------------------------------------------------

  static const BitNetModelInfo bonsai1_7b = BitNetModelInfo._(
    id: 'prism-ml/Ternary-Bonsai-1.7B',
    displayName: 'Ternary Bonsai 1.7B',
    family: 'Ternary Bonsai',
    paramsBillion: 1.72,
    quantization: BitNetQuantization.q2_0,
    license: 'Apache-2.0',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
    },
    ggufFileName: 'ternary-bonsai-1.7b-q2_0.gguf',
    ggufSizeBytes: 463290464,
    ggufSha256:
        'd97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a',
    ggufDownloadUrl:
        'https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf',
    minimumRamBytes: 2 * 1024 * 1024 * 1024,
    recommendedRamBytes: 3 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.chatmlQwen3,
    engineBackend: EngineBackend.prismQ2,
  );

  static const BitNetModelInfo bonsai4b = BitNetModelInfo._(
    id: 'prism-ml/Ternary-Bonsai-4B',
    displayName: 'Ternary Bonsai 4B',
    family: 'Ternary Bonsai',
    paramsBillion: 4.0,
    quantization: BitNetQuantization.q2_0,
    license: 'Apache-2.0',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
      BitNetCapability.code,
    },
    ggufFileName: 'ternary-bonsai-4b-q2_0.gguf',
    ggufSizeBytes: 1074969344,
    ggufSha256:
        '4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b',
    ggufDownloadUrl:
        'https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/resolve/main/Ternary-Bonsai-4B-Q2_0.gguf',
    minimumRamBytes: 3 * 1024 * 1024 * 1024,
    recommendedRamBytes: 4 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.chatmlQwen3,
    engineBackend: EngineBackend.prismQ2,
  );

  static const BitNetModelInfo bonsai8b = BitNetModelInfo._(
    id: 'prism-ml/Ternary-Bonsai-8B',
    displayName: 'Ternary Bonsai 8B',
    family: 'Ternary Bonsai',
    paramsBillion: 8.19,
    quantization: BitNetQuantization.q2_0,
    license: 'Apache-2.0',
    capabilities: {
      BitNetCapability.chat,
      BitNetCapability.summarization,
      BitNetCapability.questionAnswering,
      BitNetCapability.textGeneration,
      BitNetCapability.code,
    },
    ggufFileName: 'ternary-bonsai-8b-q2_0.gguf',
    ggufSizeBytes: 2182184672,
    ggufSha256:
        '3c8d70470a5d97e5a2b9410ddd899cb740116591462626c60cb2fead6448f60b',
    ggufDownloadUrl:
        'https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf',
    minimumRamBytes: 5 * 1024 * 1024 * 1024,
    recommendedRamBytes: 6 * 1024 * 1024 * 1024,
    contextLength: 4096,
    chatTemplate: ChatTemplateFamily.chatmlQwen3,
    engineBackend: EngineBackend.prismQ2,
  );

  /// All curated models known to this package, in priority order.
  static const List<BitNetModelInfo> all = <BitNetModelInfo>[
    bitnet2B4T,
    falconE_1bInstruct,
    falcon3_1bInstruct,
    bonsai1_7b,
    falconE_3bInstruct,
    falcon3_3bInstruct,
    bonsai4b,
    falcon3_7bInstruct,
    bonsai8b,
    falcon3_10bInstruct,
  ];

  /// Default model surfaced when callers do not pick one explicitly.
  static const BitNetModelInfo defaultModel = bitnet2B4T;

  /// Looks up a model by [id] (HuggingFace-style slug). Returns `null` when
  /// no entry matches.
  static BitNetModelInfo? byId(String id) {
    for (final m in all) {
      if (m.id == id) return m;
    }
    return null;
  }

  /// Returns every catalog entry that advertises [capability].
  static List<BitNetModelInfo> withCapability(BitNetCapability capability) =>
      all.where((m) => m.capabilities.contains(capability)).toList();

  /// Returns the catalog entries that fit within [availableRamBytes],
  /// preferring those whose [BitNetModelInfo.recommendedRamBytes] is also met.
  ///
  /// Models are returned ordered: best-fit (recommended RAM met) first, then
  /// minimum-RAM-only fits, sorted by descending parameter count within each
  /// group so that a more capable model is preferred when it fits.
  static List<BitNetModelInfo> compatibleWith(int availableRamBytes) {
    final recommended = <BitNetModelInfo>[];
    final minimumOnly = <BitNetModelInfo>[];
    for (final m in all) {
      if (availableRamBytes >= m.recommendedRamBytes) {
        recommended.add(m);
      } else if (availableRamBytes >= m.minimumRamBytes) {
        minimumOnly.add(m);
      }
    }
    int byParams(BitNetModelInfo a, BitNetModelInfo b) =>
        b.paramsBillion.compareTo(a.paramsBillion);
    recommended.sort(byParams);
    minimumOnly.sort(byParams);
    return [...recommended, ...minimumOnly];
  }
}
