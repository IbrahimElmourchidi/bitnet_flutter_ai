import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../device/android/android_device.dart';
import '../device/apple/apple_device.dart';
import '../diagnostics/bitnet_log.dart';

const _folder = 'bitnet_flutter_ai';

bool _prepared = false;

/// Where downloaded models live — always somewhere device backups skip, so a
/// >1 GB re-downloadable file never lands in the user's iCloud / Google
/// backup (on Android it would also exceed Auto Backup's 25 MB quota and make
/// the app's entire backup fail).
///
/// * Android: `Context.getNoBackupFilesDir()/bitnet_flutter_ai` (JNIgen).
///   Models from older versions (in the files dir) are moved there once.
/// * iOS / macOS: `<Application Support>/bitnet_flutter_ai`, flagged with
///   `NSURLIsExcludedFromBackupKey` (Objective-C interop).
/// * Elsewhere: `<Application Support>/bitnet_flutter_ai`.
Future<Directory> modelCacheDirectory() async {
  final support = await getApplicationSupportDirectory();
  final legacy = Directory(p.join(support.path, _folder));
  if (!Platform.isAndroid) {
    if (!_prepared && (Platform.isIOS || Platform.isMacOS)) {
      await legacy.create(recursive: true);
      if (!AppleDevice.excludeFromBackup(legacy.path)) {
        BitNetLog.log('cache', 'could not exclude ${legacy.path} from backup');
      }
      _prepared = true;
    }
    return legacy;
  }

  final dir = Directory(p.join(AndroidDevice.noBackupDir(), _folder));
  if (!_prepared) {
    await _migrate(legacy, dir);
    _prepared = true;
  }
  return dir;
}

/// One-time move of models downloaded by older package versions. Same
/// filesystem (app-private storage), so renames are instant.
Future<void> _migrate(Directory from, Directory to) async {
  try {
    if (!from.existsSync()) return;
    if (!to.existsSync()) {
      await to.parent.create(recursive: true);
      await from.rename(to.path);
      BitNetLog.log('cache', 'moved model cache to no-backup storage');
      return;
    }
    for (final e in from.listSync().whereType<File>()) {
      final target = File(p.join(to.path, p.basename(e.path)));
      if (!target.existsSync()) await e.rename(target.path);
    }
    if (from.listSync().isEmpty) await from.delete();
  } catch (e) {
    // Never block a load on a migration hiccup; the model is re-downloaded.
    BitNetLog.log('cache', 'model cache migration skipped: $e');
  }
}
