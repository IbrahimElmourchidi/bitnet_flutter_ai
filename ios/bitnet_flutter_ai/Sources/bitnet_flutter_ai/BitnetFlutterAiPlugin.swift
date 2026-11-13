import Flutter
import UIKit

/// Registration shell only.
///
/// This target exists to link the vendored xcframeworks (see Package.swift /
/// the podspec). The inference engine is reached from Dart through dart:ffi,
/// and device queries (physical memory, thermal state, Low Power Mode,
/// backup exclusion) through Objective-C interop via package:objective_c —
/// see lib/src/device/apple/. There is no platform channel, so those calls
/// also work from background isolates.
public class BitnetFlutterAiPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {}
}
