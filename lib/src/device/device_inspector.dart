library;

// dart.library.js_interop (not dart.library.html) — see the note in
// engine/bitnet_engine.dart: `dart:html` is absent under dart2wasm, so an
// html-based guard picks the dart:io implementation on wasm web builds.
import 'device_health.dart';
import 'device_inspector_impl.dart'
    if (dart.library.js_interop) 'device_inspector_web.dart';

export 'device_health.dart';

/// Compatibility check threshold: 3072 MB.
const int kMinRamBytes = 3072 * 1024 * 1024;

/// Reads device resources (RAM, temperature, power mode) on every platform.
///
/// On Android and iOS this uses direct native interop (JNIgen / Objective-C
/// via `package:objective_c`) instead of a platform channel, so it works
/// from any isolate — including a background isolate deciding whether to run
/// a scheduled insight job.
abstract class DeviceInspector {
  const DeviceInspector();

  /// Returns total physical RAM in bytes.
  Future<int> totalPhysicalRamBytes();

  /// Current memory / thermal / power snapshot. Cheap; call it right before
  /// starting heavy inference.
  Future<DeviceHealth> health() async =>
      DeviceHealth(totalRamBytes: await totalPhysicalRamBytes());

  /// Returns true when [totalPhysicalRamBytes] >= 3072 MB.
  Future<bool> meetsMinimumRam() async =>
      (await totalPhysicalRamBytes()) >= kMinRamBytes;

  /// The singleton for the current platform.
  // ignore: prefer_const_declarations
  static final DeviceInspector instance = deviceInspectorInstance;
}
