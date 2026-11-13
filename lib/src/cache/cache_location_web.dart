import 'dart:io';

/// Web never uses [ModelCache] (the WASM engine provisions models through the
/// browser Cache API), so there is no on-disk location.
Future<Directory> modelCacheDirectory() =>
    throw UnsupportedError('ModelCache is not available on the web');
