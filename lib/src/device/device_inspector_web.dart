import 'dart:js_interop';

import 'device_inspector.dart';

/// Web implementation of [DeviceInspector] using `navigator.deviceMemory`.
///
/// `navigator.deviceMemory` is a coarse hint (0.25 / 0.5 / 1 / 2 / 4 / 8 GB).
/// Safari does not expose it — we fail closed (return 0) in that case.
/// A return value of 4 GB is required to clear the 3 GB threshold,
/// since the value 2 could represent anywhere from ~1.5 to ~2.9 GB.
class WebDeviceInspector extends DeviceInspector {
  const WebDeviceInspector();

  @override
  Future<int> totalPhysicalRamBytes() async {
    final gb = _deviceMemory;
    if (gb == null) return 0; // Safari — fail closed
    return (gb * 1024 * 1024 * 1024).toInt();
  }
}

@JS('navigator.deviceMemory')
external JSNumber? get _deviceMemoryJS;

double? get _deviceMemory {
  final v = _deviceMemoryJS;
  if (v == null) return null;
  return v.toDartDouble;
}

/// Singleton exposed via [DeviceInspector.instance].
const DeviceInspector deviceInspectorInstance = WebDeviceInspector();
