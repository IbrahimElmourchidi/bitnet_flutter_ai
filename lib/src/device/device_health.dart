import 'package:flutter/foundation.dart';

/// Coarse device temperature, normalised across platforms (iOS
/// `ProcessInfo.ThermalState`, Android `PowerManager` thermal status).
enum ThermalState {
  /// Not reported by this platform / OS version.
  unknown,

  /// No thermal pressure.
  nominal,

  /// Slightly elevated (Android: light / moderate).
  fair,

  /// High — the OS is throttling. Defer heavy inference if you can.
  serious,

  /// Critical — avoid inference entirely.
  critical,
}

/// Snapshot of the resources that matter for running an on-device LLM.
///
/// Fields a platform cannot report are `null` (or [ThermalState.unknown]).
@immutable
final class DeviceHealth {
  /// Total physical RAM, in bytes.
  final int totalRamBytes;

  /// RAM currently available to apps, in bytes (Android `availMem`, iOS
  /// `os_proc_available_memory` — the headroom before this process is
  /// killed, desktop free memory).
  final int? availableRamBytes;

  /// The OS reports the system as low on memory (Android only).
  final bool? lowMemory;

  /// Device temperature.
  final ThermalState thermal;

  /// Battery saver / Low Power Mode is on.
  final bool? lowPowerMode;

  const DeviceHealth({
    required this.totalRamBytes,
    this.availableRamBytes,
    this.lowMemory,
    this.thermal = ThermalState.unknown,
    this.lowPowerMode,
  });

  /// Whether this is a bad moment for a long generation: the device is hot,
  /// in power-saving mode, or short on memory. Apps can defer non-urgent work
  /// (e.g. background insights) while it is `true`.
  bool get shouldThrottle =>
      thermal == ThermalState.serious ||
      thermal == ThermalState.critical ||
      lowPowerMode == true ||
      lowMemory == true;

  /// Whether at least [bytes] are available right now (`true` when unknown).
  bool hasAvailable(int bytes) => (availableRamBytes ?? bytes) >= bytes;

  @override
  String toString() => 'DeviceHealth(total: ${totalRamBytes >> 20} MB, '
      'available: ${availableRamBytes == null ? '?' : '${availableRamBytes! >> 20} MB'}, '
      'lowMemory: $lowMemory, thermal: ${thermal.name}, lowPower: $lowPowerMode)';
}
