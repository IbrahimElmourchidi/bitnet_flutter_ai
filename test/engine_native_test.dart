// Exercises NativeEngine's isolate protocol (queueing, cancellation, abort
// flag, teardown, worker death) against a scripted InferenceBackend running in
// a real spawned isolate. No model or native library is needed.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:io' show sleep;
import 'dart:isolate';
import 'dart:typed_data';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:bitnet_flutter_ai/src/engine/engine_native.dart';
import 'package:bitnet_flutter_ai/src/engine/inference_backend.dart';
import 'package:flutter_test/flutter_test.dart';

/// Behaviour is selected by the model path the engine passes to init():
///   /ok        8 tokens per prompt, 5 ms each (blocking, like FFI)
///   /slowfill  prefill takes ~2 s unless the abort flag is raised
///   /crash     the isolate dies on the 3rd token
///   /badinit   init() throws
///   /utf8      emits "é" split across two tokens, then "|"
class _ScriptedBackend implements InferenceBackend {
  String _mode = '';
  Pointer<Int32>? _flag;
  String _label = '';
  int _i = 0;

  bool get _aborted => _flag != null && _flag!.value != 0;

  @override
  void init(String modelPath, {required int nCtx, required int nThreads}) {
    if (modelPath == '/badinit') {
      throw const BitNetInitException(message: 'scripted init failure');
    }
    _mode = modelPath;
  }

  @override
  void setAbortFlag(int address) =>
      _flag = address == 0 ? null : Pointer<Int32>.fromAddress(address);

  @override
  void setSampling(GenerationOptions options) {
    if (options.grammar == 'invalid') {
      throw const BitNetInferenceException(message: 'bad grammar');
    }
  }

  @override
  PromptStatus feedPrompt(String prompt, {required bool addBos}) {
    if (_mode == '/slowfill') {
      for (var k = 0; k < 400; k++) {
        if (_aborted) return PromptStatus.aborted;
        sleep(const Duration(milliseconds: 5));
      }
    }
    _label = prompt;
    _i = 0;
    return PromptStatus.ok;
  }

  @override
  NextToken? nextToken() {
    if (_aborted) return null;
    if (_mode == '/utf8') {
      final bytes = utf8.encode('é|'); // [0xC3, 0xA9, 0x7C]
      if (_i >= 3) return null;
      final out = switch (_i) {
        0 => Uint8List.fromList([bytes[0]]),
        1 => Uint8List.fromList([bytes[1]]),
        _ => Uint8List.fromList([bytes[2]]),
      };
      _i++;
      return NextToken(out, -0.5);
    }
    if (_mode == '/crash' && _i == 2) Isolate.current.kill();
    if (_i >= 8) return null;
    sleep(const Duration(milliseconds: 5));
    // Each token names the prompt it belongs to: a generation that consumed
    // another prompt's tokens shows up as a gap / foreign label.
    return NextToken(Uint8List.fromList(utf8.encode('$_label:${_i++}|')), -0.25);
  }

  @override
  int tokenCount(String text, {required bool addBos}) =>
      text.length + (addBos ? 1 : 0);

  @override
  int get lastPromptReused => 0;

  @override
  void free() {}
}

InferenceBackend _scripted(EngineBackend _) => _ScriptedBackend();

const _model = BitNetCatalog.defaultModel;

NativeEngine _engine(String mode, {void Function()? onResolve}) =>
    NativeEngine.withBackend(
      _model,
      backendFactory: _scripted,
      resolveModelPath: (_, _) async {
        onResolve?.call();
        return mode;
      },
    );

String _expected(String prompt) =>
    List.generate(8, (i) => '$prompt:$i|').join();

void main() {
  group('NativeEngine', () {
    test('generates, reports logprobs, and counts tokens', () async {
      final e = _engine('/ok');
      await e.load();
      final events = await e.generateTokens('p').toList();
      expect(events.map((x) => x.text).join(), _expected('p'));
      expect(events.expand((x) => x.logprobs), everyElement(-0.25));
      expect(await e.countTokens('abcd', addBos: true), 5);
      await e.dispose();
    });

    test('cancel-then-immediately-generate never interleaves on the context',
        () async {
      // Regression: the worker used to start the second decode loop while the
      // first was parked mid-generation, so both pulled tokens from one
      // context and the second reply lost tokens to the first.
      final e = _engine('/ok');
      await e.load();
      for (var round = 0; round < 5; round++) {
        final first = e.generate('a$round').listen((_) {});
        await Future<void>.delayed(const Duration(milliseconds: 12));
        await first.cancel();
        final second = await e.generate('b$round').join();
        expect(second, _expected('b$round'), reason: 'round $round');
      }
      await e.dispose();
    });

    test('concurrent generations are queued in call order', () async {
      final e = _engine('/ok');
      await e.load();
      final results = await Future.wait([
        e.generate('x').join(),
        e.generate('y').join(),
        e.generate('z').join(),
      ]);
      expect(results, [_expected('x'), _expected('y'), _expected('z')]);
      await e.dispose();
    });

    test('cancelGeneration interrupts a long prefill via the shared flag',
        () async {
      final e = _engine('/slowfill');
      await e.load();
      final sw = Stopwatch()..start();
      final done = e.generate('long').join();
      await Future<void>.delayed(const Duration(milliseconds: 50));
      await e.cancelGeneration();
      expect(await done, isEmpty);
      // The prefill alone would take ~2 s; the abort flag stops it at once.
      expect(sw.elapsedMilliseconds, lessThan(800));
      // The flag is reset for the next generation: it runs to completion.
      expect(await e.generate('q').join(), _expected('q'));
      await e.dispose();
    });

    test('UTF-8 split across tokens is reassembled', () async {
      final e = _engine('/utf8');
      await e.load();
      final events = await e.generateTokens('u').toList();
      expect(events.map((x) => x.text).join(), 'é|');
      // Both bytes of "é" are attributed to the chunk that completed it.
      expect(events.first.logprobs, hasLength(2));
      await e.dispose();
    });

    test('options.stop ends the stream before the stop string', () async {
      final e = _engine('/ok');
      await e.load();
      final out = await e
          .generate('s', options: const GenerationOptions(stop: [':3|']))
          .join();
      expect(out, 's:0|s:1|s:2|s');
      await e.dispose();
    });

    test('an invalid grammar surfaces as BitNetInferenceException', () async {
      final e = _engine('/ok');
      await e.load();
      await expectLater(
        e.generate('g', options: const GenerationOptions(grammar: 'invalid')).join(),
        throwsA(isA<BitNetInferenceException>()),
      );
      // ...and the next generation is unaffected.
      expect(await e.generate('h').join(), _expected('h'));
      await e.dispose();
    });

    test('concurrent load() calls share one load', () async {
      var resolves = 0;
      final e = _engine('/ok', onResolve: () => resolves++);
      await Future.wait([e.load(), e.load(), e.load()]);
      expect(resolves, 1);
      expect(e.isLoaded, isTrue);
      await e.dispose();
    });

    test('dispose() during load() aborts the load cleanly', () async {
      final e = _engine('/ok');
      final loading =
          expectLater(e.load(), throwsA(isA<BitNetIsolateException>()));
      await e.dispose();
      await loading;
      expect(e.isLoaded, isFalse);
      expect(() => e.generate('x'), throwsA(isA<BitNetNotLoadedException>()));
    });

    test('init failure is reported and load() can be retried', () async {
      final e = _engine('/badinit');
      await expectLater(e.load(), throwsA(isA<BitNetInitException>()));
      expect(e.isLoaded, isFalse);
      await e.dispose();
    });

    test('a dying worker fails the stream instead of hanging it', () async {
      final e = _engine('/crash');
      await e.load();
      await expectLater(
        e.generate('c').join().timeout(const Duration(seconds: 5)),
        throwsA(isA<BitNetIsolateException>()),
      );
      expect(e.isLoaded, isFalse);
      await e.dispose();
    });

    test('dispose() completes queued generations with NotLoaded', () async {
      final e = _engine('/ok');
      await e.load();
      final a = e.generate('a').join();
      final b = expectLater(
          e.generate('b').join(), throwsA(isA<BitNetNotLoadedException>()));
      await Future<void>.delayed(const Duration(milliseconds: 8));
      await e.dispose();
      await a; // cut short, completes normally
      await b;
    });
  });
}
