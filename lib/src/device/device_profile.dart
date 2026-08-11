import 'dart:math' as math;

import 'device_inspector.dart';
// Swaps dart:io/dart:ffi platform detection for a browser-based one on web, so
// this file (reachable from the example's Device tab) compiles for the web.
// dart.library.js_interop (not dart.library.html) — see the note in
// engine/bitnet_engine.dart: `dart:html` is absent under dart2wasm.
import 'platform_info_impl.dart'
    if (dart.library.js_interop) 'platform_info_web.dart';
import '../model/bitnet_model_info.dart';

/// Snapshot of the current device's hardware capabilities, used to decide
/// which model to load and to forecast inference speed before a user
/// commits to a multi-hundred-megabyte download.
///
/// Construct via [DeviceProfile.current]; do not instantiate directly.
class DeviceProfile {
  /// Operating system identifier (`'android'`, `'ios'`, `'linux'`, `'macos'`,
  /// `'windows'`, `'web'`, or `'unknown'`).
  final String os;

  /// Lowercase ABI name as reported by [Abi.current], e.g.
  /// `'androidarm64'`, `'iosarm64'`, `'linuxx64'`. On web this is `'wasm'`.
  final String abi;

  /// Number of logical CPU cores reported by the OS.
  final int cpuCores;

  /// Total physical RAM in bytes (0 on web, where it cannot be measured).
  final int totalRamBytes;

  const DeviceProfile({
    required this.os,
    required this.abi,
    required this.cpuCores,
    required this.totalRamBytes,
  });

  /// Captures the current device's profile by combining the platform-specific
  /// [DeviceInspector] with [detectPlatformInfo] (native: `dart:io` + `dart:ffi`
  /// `Abi`; web: the browser `navigator`).
  static Future<DeviceProfile> current() async {
    final ram = await DeviceInspector.instance.totalPhysicalRamBytes();
    final info = detectPlatformInfo();
    return DeviceProfile(
      os: info.os,
      abi: info.abi,
      cpuCores: info.cores,
      totalRamBytes: ram,
    );
  }

  /// True when the device clears [BitNetModelInfo.minimumRamBytes] for [model].
  bool canRun(BitNetModelInfo model) =>
      totalRamBytes >= model.minimumRamBytes;

  /// True when the device clears the recommended RAM threshold and the
  /// estimated decode rate is at least [minTokensPerSecond] tok/s
  /// (default 4 tok/s — comfortably faster than typical reading speed).
  bool isComfortableFor(
    BitNetModelInfo model, {
    double minTokensPerSecond = 4.0,
  }) {
    if (totalRamBytes < model.recommendedRamBytes) return false;
    return estimateTokensPerSecond(model) >= minTokensPerSecond;
  }

  /// Heuristic estimate of decoded tokens-per-second for [model] on this
  /// device. The model is intentionally simple — see the comments inside the
  /// implementation for the calibration data.
  ///
  /// **Do not** use the return value as a contract; treat it as a best-effort
  /// hint suitable for "expect roughly N tok/s" UI copy.
  double estimateTokensPerSecond(BitNetModelInfo model) {
    // Calibration constants. The single measured anchor is BitNet b1.58 2B-4T
    // (I2_S, ~2.0B params) on a Ryzen 7 5700U laptop: 8 logical cores, x86_64,
    // observed ~12.5 tok/s during single-stream decode (Phase 0 Linux test).
    const refTokensPerSecond = 12.5; // tok/s on the reference device
    const refCores = 8;
    const refParamsB = 2.0;

    // 1. Scale by core count. Decode is single-stream so adding cores helps
    //    sub-linearly; sqrt(cores/ref) is a decent first-order approximation
    //    for memory-bound matvec workloads like ternary BitNet.
    final effectiveCores = cpuCores.clamp(2, 16).toDouble();
    final coreScale = math.sqrt(effectiveCores / refCores);

    // 2. Scale inversely with parameter count. BitNet's matvec is roughly
    //    linear in parameters per token at fixed quantisation.
    final paramScale = refParamsB / model.paramsBillion;

    // 3. Per-architecture multiplier. Mobile big.LITTLE configurations don't
    //    sustain the same throughput per core as a desktop x86_64 core, and
    //    web (WASM SIMD) is roughly half that of native.
    final archScale = switch (_archFamily()) {
      _Arch.x64 => 1.0,
      _Arch.arm64Mobile => 0.55, // android/ios arm64 — calibrate later
      _Arch.arm64Desktop => 0.85, // apple silicon, linux arm64 desktops
      _Arch.wasm => 0.4,
      _Arch.unknown => 0.6,
    };

    // 4. Quantisation multiplier — TL1/TL2 lookup-table kernels are slightly
    //    faster than I2_S on the architectures they target.
    final quantScale = switch (model.quantization) {
      BitNetQuantization.i2s => 1.0,
      BitNetQuantization.tl1 => 1.1,
      BitNetQuantization.tl2 => 1.05,
      BitNetQuantization.q2_0 => 0.9, // PrismML ternary; uncalibrated, ≈ i2s
      BitNetQuantization.q4km => 0.7, // not BitNet — generic GGUF baseline
      BitNetQuantization.other => 0.6,
    };

    // 5. RAM-pressure penalty: if the device is just at the minimum, the OS
    //    will swap aggressively and decode rate collapses. Apply a smooth
    //    penalty between minimumRamBytes (×0.6) and recommendedRamBytes (×1.0).
    final ramScale = _ramPressureScale(
      totalRamBytes,
      min: model.minimumRamBytes,
      recommended: model.recommendedRamBytes,
    );

    final est = refTokensPerSecond *
        coreScale *
        paramScale *
        archScale *
        quantScale *
        ramScale;

    // Floor at 0.5 tok/s so callers always get something positive when the
    // device technically meets the minimum; cap at 60 tok/s — beyond that
    // the heuristic is meaningless and a real measurement is needed.
    return est.clamp(0.5, 60.0);
  }

  _Arch _archFamily() {
    if (abi.contains('wasm')) return _Arch.wasm;
    if (abi.contains('x64')) return _Arch.x64;
    if (abi.contains('arm64') || abi.contains('aarch64')) {
      if (os == 'android' || os == 'ios') return _Arch.arm64Mobile;
      return _Arch.arm64Desktop;
    }
    return _Arch.unknown;
  }

  @override
  String toString() => 'DeviceProfile($os/$abi, ${cpuCores}c, '
      '${(totalRamBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB)';
}

enum _Arch { x64, arm64Mobile, arm64Desktop, wasm, unknown }

double _ramPressureScale(int total, {required int min, required int recommended}) {
  if (total >= recommended) return 1.0;
  if (total <= min) return 0.6;
  final t = (total - min) / (recommended - min);
  return 0.6 + 0.4 * t;
}
