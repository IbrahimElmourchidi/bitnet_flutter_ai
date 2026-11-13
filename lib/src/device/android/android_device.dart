import 'package:jni/jni.dart';
import 'package:jni_flutter/jni_flutter.dart';

import '../device_health.dart';
import 'android_bindings.dart';

/// Android platform queries over JNIgen bindings (see `jnigen.yaml`).
///
/// Every JNI reference obtained here is released before returning.
abstract final class AndroidDevice {
  static Context _context() =>
      androidApplicationContext.as(Context.type, releaseOriginal: true);

  static DeviceHealth health() {
    final ctx = _context();
    final am = _service(ctx, Context.ACTIVITY_SERVICE)
        ?.as(ActivityManager.type, releaseOriginal: true);
    final pm = _service(ctx, Context.POWER_SERVICE)
        ?.as(PowerManager.type, releaseOriginal: true);
    ctx.release();
    try {
      var total = 0;
      int? avail;
      bool? low;
      if (am != null) {
        final info = ActivityManager$MemoryInfo();
        try {
          am.getMemoryInfo(info);
          total = info.totalMem;
          avail = info.availMem;
          low = info.lowMemory;
        } finally {
          info.release();
        }
      }
      var thermal = ThermalState.unknown;
      bool? powerSave;
      if (pm != null) {
        powerSave = _try(() => pm.isPowerSaveMode);
        // getCurrentThermalStatus is API 29+; older devices throw.
        final status = _try(() => pm.currentThermalStatus);
        if (status != null) thermal = _thermal(status);
      }
      return DeviceHealth(
        totalRamBytes: total,
        availableRamBytes: avail,
        lowMemory: low,
        thermal: thermal,
        lowPowerMode: powerSave,
      );
    } finally {
      am?.release();
      pm?.release();
    }
  }

  static int totalRamBytes() => health().totalRamBytes;

  /// `Context.getNoBackupFilesDir()`: app-private storage that Android Auto
  /// Backup never uploads. A >1 GB model in the regular files dir blows the
  /// 25 MB backup quota, which makes the app's whole backup fail.
  static String noBackupDir() {
    final ctx = _context();
    final dir = ctx.noBackupFilesDir;
    ctx.release();
    if (dir == null) throw StateError('getNoBackupFilesDir() returned null');
    try {
      final path = dir.absolutePath;
      if (path == null) throw StateError('getAbsolutePath() returned null');
      return path.toDartString(releaseOriginal: true);
    } finally {
      dir.release();
    }
  }

  static JObject? _service(Context ctx, JString? name) {
    if (name == null) return null;
    try {
      return ctx.getSystemService$1(name);
    } finally {
      name.release();
    }
  }

  static T? _try<T>(T Function() f) {
    try {
      return f();
    } catch (_) {
      return null;
    }
  }

  static ThermalState _thermal(int status) => switch (status) {
        PowerManager.THERMAL_STATUS_NONE => ThermalState.nominal,
        PowerManager.THERMAL_STATUS_LIGHT ||
        PowerManager.THERMAL_STATUS_MODERATE =>
          ThermalState.fair,
        PowerManager.THERMAL_STATUS_SEVERE => ThermalState.serious,
        _ when status > PowerManager.THERMAL_STATUS_SEVERE =>
          ThermalState.critical,
        _ => ThermalState.unknown,
      };
}
