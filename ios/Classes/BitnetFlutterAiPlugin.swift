import Flutter
import UIKit

public class BitnetFlutterAiPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "bitnet_flutter_ai",
            binaryMessenger: registrar.messenger()
        )
        let instance = BitnetFlutterAiPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getTotalRam":
            // ProcessInfo.physicalMemory returns UInt64 (bytes).
            // Cast to Int64 so it maps cleanly to Dart's int.
            result(Int64(bitPattern: ProcessInfo.processInfo.physicalMemory))
        default:
            result(FlutterMethodNotImplemented)
        }
    }
}
