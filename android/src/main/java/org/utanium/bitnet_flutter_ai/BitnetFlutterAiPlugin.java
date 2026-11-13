package org.utanium.bitnet_flutter_ai;

import androidx.annotation.NonNull;
import io.flutter.embedding.engine.plugins.FlutterPlugin;

/**
 * Registration shell only.
 *
 * <p>This Android module exists to package the prebuilt native libraries
 * (src/main/jniLibs). Everything else is reached from Dart directly: the
 * inference engine through dart:ffi (bn_* C ABI), and device queries (memory,
 * thermal status, power-save mode, no-backup storage) through JNIgen bindings
 * — see lib/src/device/android/. There is no platform channel, so those calls
 * also work from background isolates.
 *
 * <p>Written in Java so the module needs no Kotlin toolchain and builds the
 * same under AGP 8 and AGP 9 (built-in Kotlin) host apps.
 */
public class BitnetFlutterAiPlugin implements FlutterPlugin {
    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {}

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {}
}
