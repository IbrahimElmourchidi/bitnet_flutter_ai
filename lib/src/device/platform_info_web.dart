import 'dart:js_interop';

/// Web platform detection.
///
/// `dart:io` and `dart:ffi` are unavailable when compiling to JS/WASM, so this
/// reads the browser's `navigator` instead of `Platform` / `Abi`. Selected over
/// [platform_info_impl.dart] by the conditional import in `device_profile.dart`.
({String os, int cores, String abi}) detectPlatformInfo() => (
      os: 'web',
      // navigator.hardwareConcurrency is widely supported; fall back to a
      // conservative 4 when the browser (or a privacy setting) hides it.
      cores: _hardwareConcurrency ?? 4,
      abi: 'wasm',
    );

@JS('navigator.hardwareConcurrency')
external JSNumber? get _hardwareConcurrencyJS;

int? get _hardwareConcurrency {
  final v = _hardwareConcurrencyJS;
  if (v == null) return null;
  return v.toDartDouble.toInt();
}
