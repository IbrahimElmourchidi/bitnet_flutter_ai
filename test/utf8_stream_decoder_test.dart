import 'dart:convert';
import 'dart:typed_data';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter_test/flutter_test.dart';

/// Splits the UTF-8 encoding of [text] into byte-slices at the given [cuts]
/// (offsets into the byte stream) and feeds each slice as a separate piece,
/// mirroring how the tokenizer emits fragments that can split a character.
List<Uint8List> sliceUtf8(String text, List<int> cuts) {
  final bytes = utf8.encode(text);
  final slices = <Uint8List>[];
  var start = 0;
  for (final cut in [...cuts, bytes.length]) {
    slices.add(Uint8List.fromList(bytes.sublist(start, cut)));
    start = cut;
  }
  return slices;
}

/// Feeds every byte-slice through the decoder and returns the full decoded
/// text (emitted + flushed), mirroring how the native engine drives it.
String runDecode(List<Uint8List> pieces) {
  final d = Utf8StreamDecoder();
  final buf = StringBuffer();
  for (final p in pieces) {
    buf.write(d.add(p));
  }
  buf.write(d.flush());
  return buf.toString();
}

void main() {
  group('Utf8StreamDecoder', () {
    test('pure ASCII passes through unchanged, one piece per byte', () {
      expect(runDecode(sliceUtf8('Hello', [1, 2, 3, 4])), 'Hello');
    });

    test('3-byte char (em-dash) split across two pieces reassembles', () {
      // '—' (U+2014) encodes as E2 80 94. This is the exact case in the bug
      // report: a piece ending mid-character used to throw.
      const text = 'a—b';
      final bytes = utf8.encode(text); // 61 E2 80 94 62
      // Cut in the MIDDLE of the em-dash (after its first byte).
      expect(runDecode(sliceUtf8(text, [2])), text);
      expect(bytes.length, 5); // sanity: the em-dash really is 3 bytes
    });

    test('4-byte emoji split across three pieces reassembles', () {
      // '😀' (U+1F600) encodes as F0 9F 98 80.
      const text = 'hi😀!';
      // Feed one byte of the emoji at a time.
      final bytes = utf8.encode(text);
      final emojiStart = utf8.encode('hi').length;
      expect(
        runDecode(sliceUtf8(text, [emojiStart + 1, emojiStart + 2, emojiStart + 3])),
        text,
      );
      expect(bytes.length, 2 + 4 + 1);
    });

    test('add() returns only complete prefix, holds the partial tail', () {
      final d = Utf8StreamDecoder();
      // 'a' + first byte of em-dash.
      final r1 = d.add(Uint8List.fromList([0x61, 0xE2]));
      expect(r1, 'a'); // the lone 0xE2 is held back
      // remaining two bytes of the em-dash.
      final r2 = d.add(Uint8List.fromList([0x80, 0x94]));
      expect(r2, '—');
      expect(d.flush(), '');
    });

    test('empty pieces are tolerated', () {
      final d = Utf8StreamDecoder();
      expect(d.add(Uint8List(0)), '');
      expect(d.add(Uint8List.fromList(utf8.encode('ok'))), 'ok');
      expect(d.flush(), '');
    });

    test('truncated sequence at end-of-stream flushes as U+FFFD', () {
      final d = Utf8StreamDecoder();
      // Lone lead byte of a 3-byte sequence, never completed.
      expect(d.add(Uint8List.fromList([0xE2])), '');
      final flushed = d.flush();
      expect(flushed, isNotEmpty);
      expect(flushed.codeUnits.first, 0xFFFD);
    });

    test('multi-char string sliced at every byte boundary round-trips', () {
      // Mix of 1-, 2-, 3-, and 4-byte characters.
      const text = 'A é 你 😀 z';
      final bytes = utf8.encode(text);
      final everyByte = [for (var i = 1; i < bytes.length; i++) i];
      expect(runDecode(sliceUtf8(text, everyByte)), text);
    });
  });
}
