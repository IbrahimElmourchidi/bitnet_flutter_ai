// swift-tools-version: 5.9
// Swift Package Manager manifest for the bitnet_flutter_ai iOS plugin.
//
// This mirrors ios/bitnet_flutter_ai.podspec so the plugin builds under BOTH
// toolchains: CocoaPods (legacy) and Swift Package Manager (Flutter's newer
// default). Both consume the SAME sources under Sources/bitnet_flutter_ai/ and
// the SAME two xcframeworks under ../Frameworks/ — see the podspec for the full
// rationale on why there are two separate backends.
import PackageDescription

let package = Package(
    name: "bitnet_flutter_ai",
    platforms: [
        .iOS("14.0"),
    ],
    products: [
        .library(name: "bitnet-flutter-ai", targets: ["bitnet_flutter_ai"]),
    ],
    dependencies: [
        // Flutter generates this sibling package (symlinking the Flutter.xcframework
        // as a binary target) at build time, relative to this Package.swift. It
        // provides the `Flutter` module that BitnetFlutterAiPlugin.swift imports.
        .package(name: "FlutterFramework", path: "../FlutterFramework"),
    ],
    targets: [
        // The plugin's own sources (Swift plugin registrant + the ObjC
        // keep-alive shim) plus the two vendored xcframeworks.
        .target(
            name: "bitnet_flutter_ai",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework"),
                "bitnet_bridge",
                "prism_bridge",
            ],
            // Sources/bitnet_flutter_ai/ holds BitnetFlutterAiPlugin.swift and
            // bitnet_force_link.m. The empty include/bitnet_flutter_ai/ dir
            // exists so headerSearchPath resolves (no public ObjC headers are
            // exported — the bn_* symbols are reached only via Dart FFI).
            cSettings: [
                .headerSearchPath("include/bitnet_flutter_ai"),
            ],
            linkerSettings: [
                // Match the podspec: link the C++ stdlib for the bridge.
                // Accelerate/BLAS is intentionally NOT linked (CPU-only build;
                // ggml's BLAS backend mishandles the I2_S quant layout).
                //
                // No -force_load / -ObjC needed: the bitnet bn_* symbols live in
                // the STATIC bitnet_bridge archive and are reached only via Dart
                // FFI (dlsym), so nothing references them at link time.
                // bitnet_force_link.m (compiled into THIS target, not the vendored
                // archive) takes their addresses inside a __attribute__((used))
                // array, forcing the linker to pull the archive's objects into
                // the plugin framework — the same keep-alive strategy the podspec
                // documents and has proven under CocoaPods.
                .linkedLibrary("c++"),
            ]
        ),

        // STATIC archive (libbitnet_bridge.a) merging bitnet.cpp's llama/ggml.
        // Linked into the plugin framework; bn_* retained by the shim above and
        // resolved at runtime via DynamicLibrary.process().
        .binaryTarget(
            name: "bitnet_bridge",
            path: "../Frameworks/bitnet_bridge.xcframework"
        ),

        // Self-contained DYNAMIC framework (PrismML fork linked static inside)
        // exporting only the seven bn_*. Kept SEPARATE from bitnet because both
        // export the same bn_* names with incompatible ggml copies; Dart opens
        // it by path (DynamicLibrary.open('prism_bridge.framework/prism_bridge')).
        .binaryTarget(
            name: "prism_bridge",
            path: "../Frameworks/prism_bridge.xcframework"
        ),
    ]
)
