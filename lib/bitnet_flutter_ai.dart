/// bitnet_flutter_ai — run Microsoft BitNet b1.58 2B-4T locally on device.
///
/// Quick start:
/// ```dart
/// import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
///
/// final engine = BitNetEngine(model: BitNetCatalog.defaultModel);
/// await engine.load(onProgress: (p) => print('${(p * 100).toInt()}%'));
/// final session = BitNetSession(engine: engine);
/// await for (final text in session.chat('Explain ternary quantization')) {
///   stdout.write(text);
/// }
/// await engine.dispose();
/// ```
library;

export 'src/cache/model_cache.dart';
export 'src/device/device_inspector.dart';
export 'src/device/device_profile.dart';
export 'src/diagnostics/bitnet_log.dart';
export 'src/engine/bitnet_engine.dart'
    hide TurnStopDetector, Utf8StreamDecoder, kBitNetTurnStopMarkers;
export 'src/engine/generation_options.dart';
export 'src/exceptions/bitnet_exception.dart';
export 'src/model/bitnet_model_info.dart';
export 'src/rag/asset_knowledge_source.dart';
export 'src/rag/bitnet_rag.dart';
export 'src/rag/bm25.dart';
export 'src/rag/knowledge_source.dart';
export 'src/rag/text_chunker.dart';
export 'src/session/bitnet_session.dart';
export 'src/session/json_shape.dart';
