library;

import 'device_inspector_impl.dart'
    if (dart.library.html) 'device_inspector_web.dart';

/// Compatibility check threshold: 3072 MB.
const int kMinRamBytes = 3072 * 1024 * 1024;

/// Abstracts physical RAM detection across all supported platforms.
abstract class DeviceInspector {
  const DeviceInspector();

  /// Returns total physical RAM in bytes.
  Future<int> totalPhysicalRamBytes();

  /// Returns true when [totalPhysicalRamBytes] >= 3072 MB.
  Future<bool> meetsMinimumRam() async =>
      (await totalPhysicalRamBytes()) >= kMinRamBytes;

  /// The singleton for the current platform.
  // ignore: prefer_const_declarations
  static final DeviceInspector instance = deviceInspectorInstance;
}
