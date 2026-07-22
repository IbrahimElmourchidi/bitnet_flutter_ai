// ignore_for_file: avoid_print
//
// End-to-end Dart-only integration test for the native FFI surface.
//
// This bypasses the Flutter `BitNetEngine` (and its path_provider /
// isolate machinery) and exercises [NativeLibrary] directly against an
// already-downloaded GGUF file on disk. It is the Dart equivalent of
// the C smoke_test.c.
//
// Run from the package root:
//
//   LD_LIBRARY_PATH=\
//     native/build-linux:\
//     ../bitnet.cpp/build-linux/3rdparty/llama.cpp/src:\
//     ../bitnet.cpp/build-linux/3rdparty/llama.cpp/ggml/src \
//     dart run tool/dart_smoke.dart \
//       ../bitnet.cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
//       "Hello, who are you?"

import 'dart:io';

import 'package:bitnet_flutter_ai/src/ffi/native_library.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln('Usage: dart_smoke.dart <model.gguf> [prompt] [maxTokens]');
    exit(64);
  }
  final modelPath = args[0];
  final prompt = args.length > 1 ? args[1] : 'Hello, who are you?';
  final maxNew = args.length > 2 ? int.parse(args[2]) : 32;

  if (!File(modelPath).existsSync()) {
    stderr.writeln('Model file not found: $modelPath');
    exit(2);
  }

  stderr.writeln('[dart_smoke] opening native library...');
  final lib = NativeLibrary.open();

  stderr.writeln('[dart_smoke] init context...');
  final t0 = DateTime.now();
  lib.initContext(modelPath, nCtx: 512, nThreads: 4);
  final loadMs = DateTime.now().difference(t0).inMilliseconds;
  stderr.writeln('[dart_smoke] loaded in ${loadMs}ms, n_ctx=${lib.contextSize()}');

  stderr.writeln('[dart_smoke] feed prompt: $prompt');
  lib.feedPrompt(prompt, addBos: true);

  stderr.writeln('[dart_smoke] streaming up to $maxNew tokens:');
  stdout.write('RESPONSE: ');
  final t1 = DateTime.now();
  var produced = 0;
  for (var i = 0; i < maxNew; i++) {
    final piece = lib.nextToken();
    if (piece == null) {
      stderr.writeln('\n[dart_smoke] EOS at token $i');
      break;
    }
    stdout.write(piece);
    produced++;
  }
  stdout.writeln();
  final genMs = DateTime.now().difference(t1).inMilliseconds;
  final tps = produced > 0 ? (produced * 1000.0 / genMs).toStringAsFixed(1) : '0';
  stderr.writeln('[dart_smoke] generated $produced tokens in ${genMs}ms (~$tps tok/s)');

  lib.freeContext();
  stderr.writeln('[dart_smoke] OK');
}
