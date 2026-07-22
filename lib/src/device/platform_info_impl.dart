import 'dart:ffi' show Abi;
import 'dart:io' show Platform;

/// Native platform detection via `dart:io` + `dart:ffi`.
///
/// The companion web implementation lives in `platform_info_web.dart`; the two
/// are swapped by the conditional import in `device_profile.dart` so that
/// neither `dart:ffi` nor `dart:io` is referenced when compiling for the web
/// (both libraries are unavailable there and would fail to compile).
({String os, int cores, String abi}) detectPlatformInfo() {
  String abi;
  try {
    abi = Abi.current().toString().toLowerCase();
  } catch (_) {
    // Abi.current() throws UnsupportedError on ABIs the SDK doesn't know.
    abi = 'unknown';
  }
  return (
    os: Platform.operatingSystem,
    cores: Platform.numberOfProcessors,
    abi: abi,
  );
}
