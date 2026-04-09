Pod::Spec.new do |s|
  s.name             = 'bitnet_flutter_ai'
  s.version          = '0.1.0-beta.1'
  s.summary          = 'Run Microsoft BitNet b1.58 2B-4T locally on iOS.'
  s.description      = <<-DESC
    A Flutter plugin that runs Microsoft BitNet b1.58 2B-4T on-device via a
    llama.cpp C bridge and Dart FFI. Requires the pre-built libbitnet_bridge.a
    static library to be placed in ios/Frameworks/ before building.
  DESC
  s.homepage         = 'https://github.com/IbrahimElmourchidi/bitnet_flutter_ai'
  s.license          = { :type => 'MIT', :file => '../LICENSE' }
  s.author           = { 'Utanium' => 'hello@utanium.org' }
  s.source           = { :path => '.' }

  s.source_files     = 'Classes/**/*'
  s.public_header_files = 'Classes/**/*.h'

  # Minimum iOS version — arm64 only (no simulator for on-device inference).
  s.platform         = :ios, '14.0'

  # Static library produced by the CMake/Xcode build of native/bitnet_bridge.cc.
  # Place the compiled libbitnet_bridge.a in ios/Frameworks/ before pod install.
  s.vendored_libraries = 'Frameworks/libbitnet_bridge.a'

  s.dependency 'Flutter'

  # Required for C++ standard library linkage.
  s.libraries        = 'c++'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE'              => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64 i386',
    'OTHER_LDFLAGS'               => '-lc++',
  }
  s.swift_version    = '5.0'
end
