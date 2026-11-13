import 'dart:io';

import 'package:system_info2/system_info2.dart';

import 'android/android_device.dart';
import 'apple/apple_device.dart';
import 'device_inspector.dart';

/// Native (non-web) [DeviceInspector].
///
/// - Android: JNIgen bindings to `ActivityManager` / `PowerManager`.
/// - iOS / macOS: Foundation `NSProcessInfo` via `package:objective_c`.
/// - Linux / Windows: `system_info2`.
///
/// No platform channel is involved, so this works from any isolate.
class NativeDeviceInspector extends DeviceInspector {
  const NativeDeviceInspector();

  @override
  Future<int> totalPhysicalRamBytes() async {
    if (Platform.isAndroid) return AndroidDevice.totalRamBytes();
    if (Platform.isIOS || Platform.isMacOS) return AppleDevice.totalRamBytes();
    return SysInfo.getTotalPhysicalMemory();
  }

  @override
  Future<DeviceHealth> health() async {
    if (Platform.isAndroid) return AndroidDevice.health();
    if (Platform.isIOS || Platform.isMacOS) return AppleDevice.health();
    return DeviceHealth(
      totalRamBytes: SysInfo.getTotalPhysicalMemory(),
      availableRamBytes: SysInfo.getFreePhysicalMemory(),
    );
  }
}

/// Singleton exposed via [DeviceInspector.instance].
const DeviceInspector deviceInspectorInstance = NativeDeviceInspector();
