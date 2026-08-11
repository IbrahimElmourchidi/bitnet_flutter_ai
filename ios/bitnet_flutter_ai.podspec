Pod::Spec.new do |s|
  s.name             = 'bitnet_flutter_ai'
  s.version          = '0.1.0-beta.1'
  s.summary          = 'Run Microsoft BitNet b1.58 2B-4T locally on iOS.'
  s.description      = <<-DESC
    A Flutter plugin that runs BitNet b1.58 (I2_S) and PrismML Q2_0 models
    on-device via a llama.cpp C bridge and Dart FFI. Requires two pre-built
    xcframeworks (device + simulator) in ios/Frameworks/: bitnet_bridge
    (tool/build_ios.sh) and prism_bridge (tool/build_ios_prism.sh).
  DESC
  s.homepage         = 'https://github.com/IbrahimElmourchidi/bitnet_flutter_ai'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'Utanium' => 'hello@utanium.org' }
  s.source           = { :path => '.' }

  # Sources live under the Swift Package Manager layout so CocoaPods and SPM
  # build from ONE copy (see bitnet_flutter_ai/Package.swift). No public ObjC
  # headers are exported: the bn_* symbols are reached only via Dart FFI.
  s.source_files     = 'bitnet_flutter_ai/Sources/bitnet_flutter_ai/**/*.{swift,m,h}'

  s.platform         = :ios, '14.0'

  # Two pre-built xcframeworks (device arm64 + simulator arm64/x86_64):
  #  - bitnet_bridge.xcframework (tool/build_ios.sh): STATIC archive merging
  #    libbitnet_bridge.a + static llama/ggml. Linked into the plugin's dynamic
  #    framework; the bn_* symbols are retained by the keep-alive references in
  #    Sources/bitnet_flutter_ai/bitnet_force_link.m (see note below). Resolved at runtime via
  #    DynamicLibrary.process().
  #  - prism_bridge.xcframework (tool/build_ios_prism.sh): a self-contained
  #    DYNAMIC framework (PrismML llama.cpp fork linked static inside) exporting
  #    only the seven bn_*. It is a SEPARATE framework — the two backends export
  #    the same bn_* names and carry incompatible ggml copies, so prism cannot be
  #    statically merged with bitnet. Dart opens it by path:
  #    DynamicLibrary.open('prism_bridge.framework/prism_bridge').
  s.vendored_frameworks = [
    'Frameworks/bitnet_bridge.xcframework',
    'Frameworks/prism_bridge.xcframework',
  ]

  s.dependency 'Flutter'

  # Link the C++ stdlib for the bridge. Accelerate/BLAS is intentionally NOT
  # linked: the bitnet iOS build is CPU-only (GGML_BLAS=OFF), matching Android,
  # because ggml's BLAS backend mishandles the I2_S quant layout during prefill.
  s.libraries        = 'c++'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE'              => 'YES',
    'OTHER_LDFLAGS'               => '-lc++',
  }

  # NOTE: the bitnet bn_* symbols are reached only via Dart FFI
  # (DynamicLibrary.process() → dlsym), so nothing calls them at link time and
  # the linker would dead-strip the vendored static libbitnet_bridge.a. The
  # keep-alive references in Sources/bitnet_flutter_ai/bitnet_force_link.m take the address of each
  # bn_* function, forcing the linker to retain the archive's objects inside the
  # plugin's dynamic framework — no -force_load flag or Podfile hook needed
  # (that approach breaks clean builds, since Xcode validates the force_load path
  # before the Copy-XCFrameworks phase stages it). The prism backend does NOT
  # need this: it is its own dynamic framework whose -exported_symbols_list keeps
  # its bn_* alive.
  s.swift_version    = '5.0'
end
