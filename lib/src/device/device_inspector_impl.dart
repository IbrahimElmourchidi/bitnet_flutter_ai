import 'dart:io';

import 'package:flutter/services.dart';
import 'package:system_info2/system_info2.dart';

import 'device_inspector.dart';

/// Native (non-Web) implementation of [DeviceInspector].
///
/// - Android / iOS: queries total physical RAM via a [MethodChannel] call
///   into the platform plugin (ActivityManager / ProcessInfo).
/// - macOS / Linux / Windows: queries via system_info2.
class NativeDeviceInspector extends DeviceInspector {
  const NativeDeviceInspector();

  static const _channel = MethodChannel('bitnet_flutter_ai');

  @override
  Future<int> totalPhysicalRamBytes() async {
    if (Platform.isAndroid || Platform.isIOS) {
      final bytes = await _channel.invokeMethod<int>('getTotalRam');
      return bytes ?? 0;
    }
    return SysInfo.getTotalPhysicalMemory();
  }
}

/// Singleton exposed via [DeviceInspector.instance].
const DeviceInspector deviceInspectorInstance = NativeDeviceInspector();
