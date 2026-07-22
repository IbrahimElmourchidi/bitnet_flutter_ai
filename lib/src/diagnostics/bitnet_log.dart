import 'dart:developer' as developer;

/// Lightweight debug logger for the bitnet_flutter_ai package.
///
/// Disabled by default. Turn on from app code with:
/// ```dart
/// BitNetLog.enabled = true;
/// ```
/// Output is routed through `dart:developer` so it shows up in the
/// Flutter console (`flutter run`) and `adb logcat` under tag `bitnet`.
class BitNetLog {
  /// Master switch. When `false`, all log calls are no-ops.
  static bool enabled = false;

  /// Optional sink — if set, log lines are also forwarded to this callback
  /// so apps can pipe them into an in-app debug pane.
  static void Function(String line)? sink;

  static void log(String tag, String message) {
    if (!enabled) return;
    final line = '[bitnet:$tag] $message';
    developer.log(line, name: 'bitnet');
    sink?.call(line);
  }
}
