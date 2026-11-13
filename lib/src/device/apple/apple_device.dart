// Foundation calls for iOS/macOS over package:objective_c — the runtime that
// FFIgen's Objective-C output (stable since Dart 3.14) is built on. These few
// selectors are written in exactly the form ffigen emits (typed objc_msgSend
// casts), because generating them needs the Apple SDK headers and the package
// is also developed on Linux. To bind more of Foundation, generate with
// ffigen's ObjC support on macOS instead of extending this by hand.
import 'dart:ffi' as ffi;

import 'package:objective_c/objective_c.dart' as objc;

import '../device_health.dart';

typedef _Obj = ffi.Pointer<objc.ObjCObjectImpl>;
typedef _Sel = ffi.Pointer<objc.ObjCSelector>;

final _clsProcessInfo = objc.getClass('NSProcessInfo');
final _selProcessInfo = objc.registerName('processInfo');
final _selPhysicalMemory = objc.registerName('physicalMemory');
final _selThermalState = objc.registerName('thermalState');
final _selLowPower = objc.registerName('isLowPowerModeEnabled');
final _selSetResource = objc.registerName('setResourceValue:forKey:error:');

final _msgObj = objc.msgSendPointer
    .cast<ffi.NativeFunction<_Obj Function(_Obj, _Sel)>>()
    .asFunction<_Obj Function(_Obj, _Sel)>();
final _msgU64 = objc.msgSendPointer
    .cast<ffi.NativeFunction<ffi.UnsignedLongLong Function(_Obj, _Sel)>>()
    .asFunction<int Function(_Obj, _Sel)>();
final _msgLong = objc.msgSendPointer
    .cast<ffi.NativeFunction<ffi.Long Function(_Obj, _Sel)>>()
    .asFunction<int Function(_Obj, _Sel)>();
final _msgBool = objc.msgSendPointer
    .cast<ffi.NativeFunction<ffi.Bool Function(_Obj, _Sel)>>()
    .asFunction<bool Function(_Obj, _Sel)>();
final _msgSetResource = objc.msgSendPointer
    .cast<
        ffi.NativeFunction<
            ffi.Bool Function(_Obj, _Sel, _Obj, _Obj, ffi.Pointer<_Obj>)>>()
    .asFunction<bool Function(_Obj, _Sel, _Obj, _Obj, ffi.Pointer<_Obj>)>();

/// `size_t os_proc_available_memory(void)` (libSystem, iOS 13+): bytes this
/// process can still allocate before jetsam kills it.
int Function()? _osProcAvailableMemory() {
  try {
    return ffi.DynamicLibrary.process()
        .lookupFunction<ffi.Size Function(), int Function()>(
            'os_proc_available_memory');
  } catch (_) {
    return null; // macOS / older iOS
  }
}

abstract final class AppleDevice {
  static _Obj get _processInfo => _msgObj(_clsProcessInfo, _selProcessInfo);

  static int totalRamBytes() => _msgU64(_processInfo, _selPhysicalMemory);

  static DeviceHealth health() {
    final info = _processInfo; // singleton; no ownership to manage
    final avail = _osProcAvailableMemory()?.call();
    return DeviceHealth(
      totalRamBytes: _msgU64(info, _selPhysicalMemory),
      availableRamBytes: (avail == null || avail == 0) ? null : avail,
      thermal: switch (_msgLong(info, _selThermalState)) {
        0 => ThermalState.nominal,
        1 => ThermalState.fair,
        2 => ThermalState.serious,
        3 => ThermalState.critical,
        _ => ThermalState.unknown,
      },
      lowPowerMode: _msgBool(info, _selLowPower),
    );
  }

  /// Sets `NSURLIsExcludedFromBackupKey` on [path] so iCloud / iTunes backups
  /// skip it (Apple's storage guideline for re-downloadable data). On a
  /// directory it covers everything inside. Returns whether it succeeded.
  static bool excludeFromBackup(String path) {
    final url = objc.NSURL.fileURLWithPath(objc.NSString(path));
    final yes = objc.NSNumber.alloc().initWithBool(true);
    final key = _backupKey();
    return _msgSetResource(url.ref.pointer, _selSetResource, yes.ref.pointer,
        key.ref.pointer, ffi.nullptr);
  }

  static objc.NSString _backupKey() {
    try {
      final sym = ffi.DynamicLibrary.process()
          .lookup<_Obj>('NSURLIsExcludedFromBackupKey');
      return objc.NSString.fromPointer(sym.value, retain: true, release: true);
    } catch (_) {
      // The constant's value is its own name.
      return objc.NSString('NSURLIsExcludedFromBackupKey');
    }
  }
}
