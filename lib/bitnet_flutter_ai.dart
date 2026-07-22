/// bitnet_flutter_ai — run Microsoft BitNet b1.58 2B-4T locally on device.
///
/// Quick start:
/// ```dart
/// import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
///
/// final engine = BitNetEngine();
/// await engine.load(onProgress: (p) => print('${(p * 100).toInt()}%'));
/// await for (final token in engine.generate('Explain ternary quantization')) {
///   stdout.write(token);
/// }
/// await engine.dispose();
/// ```
library bitnet_flutter_ai;

export 'src/cache/model_cache.dart';
export 'src/device/device_inspector.dart';
export 'src/device/device_profile.dart';
export 'src/diagnostics/bitnet_log.dart';
export 'src/engine/bitnet_engine.dart';
export 'src/exceptions/bitnet_exception.dart';
export 'src/model/bitnet_model_info.dart';
export 'src/rag/asset_knowledge_source.dart';
export 'src/rag/bitnet_rag.dart';
export 'src/rag/bm25.dart';
export 'src/rag/knowledge_source.dart';
export 'src/rag/text_chunker.dart';
export 'src/session/bitnet_session.dart';
