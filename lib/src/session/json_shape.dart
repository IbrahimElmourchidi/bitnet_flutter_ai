import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import '../engine/generation_options.dart';
import '../exceptions/bitnet_exception.dart';

enum _Kind { string, number, integer, boolean, enumeration, date }

/// One field of a [JsonShape].
///
/// [description] is shown to the model (it is the field's instruction), so
/// write it for the model: "Total amount paid, without currency symbol".
@immutable
final class JsonField {
  final _Kind _kind;

  /// Instruction for the model describing what goes in this field.
  final String? description;

  /// Whether the model may answer `null` (use it for anything the input text
  /// might not mention — forcing a value invites hallucination).
  final bool nullable;

  /// Allowed values for [JsonField.enumOf].
  final List<String> values;

  /// Free text.
  const JsonField.string({this.description, this.nullable = false})
      : _kind = _Kind.string,
        values = const [];

  /// A JSON number (integer or decimal).
  const JsonField.number({this.description, this.nullable = false})
      : _kind = _Kind.number,
        values = const [];

  /// A JSON integer.
  const JsonField.integer({this.description, this.nullable = false})
      : _kind = _Kind.integer,
        values = const [];

  /// `true` / `false`.
  const JsonField.boolean({this.description, this.nullable = false})
      : _kind = _Kind.boolean,
        values = const [];

  /// Exactly one of [values] (e.g. a category list).
  const JsonField.enumOf(this.values, {this.description, this.nullable = false})
      : _kind = _Kind.enumeration;

  /// An ISO date string, `YYYY-MM-DD`.
  const JsonField.date({this.description, this.nullable = false})
      : _kind = _Kind.date,
        values = const [];

  String get _typeLabel {
    final base = switch (_kind) {
      _Kind.string => 'text',
      _Kind.number => 'number',
      _Kind.integer => 'integer',
      _Kind.boolean => 'true or false',
      _Kind.enumeration => 'one of ${values.map(jsonEncode).join(', ')}',
      _Kind.date => 'date as "YYYY-MM-DD"',
    };
    return nullable ? '$base, or null' : base;
  }
}

/// The JSON object a structured extraction must produce: an ordered set of
/// named [JsonField]s.
///
/// ```dart
/// const transaction = JsonShape({
///   'amount': JsonField.number(description: 'Amount paid'),
///   'currency': JsonField.enumOf(['EGP', 'USD', 'EUR']),
///   'vendor': JsonField.string(nullable: true, description: 'Shop or payee'),
///   'category': JsonField.enumOf(['food', 'transport', 'utilities', 'other']),
///   'date': JsonField.date(nullable: true),
/// });
/// ```
///
/// [toGbnf] compiles it to a grammar that makes it impossible for the model
/// to emit anything but an object of exactly this shape (keys in this order,
/// compact, no extra fields).
@immutable
final class JsonShape {
  final Map<String, JsonField> fields;

  const JsonShape(this.fields);

  /// GBNF grammar (root rule `root`) accepting exactly this object, compactly
  /// serialised with keys in declaration order.
  String toGbnf() {
    if (fields.isEmpty) throw ArgumentError('JsonShape needs at least one field');
    final rules = <String>[];
    final root = StringBuffer('root ::= "{"');
    var i = 0;
    for (final MapEntry(key: name, value: f) in fields.entries) {
      if (i > 0) root.write(' ","');
      root.write(' ${_lit(jsonEncode(name))} ":" f$i');
      final v = switch (f._kind) {
        _Kind.string => 'str',
        _Kind.number => 'num',
        _Kind.integer => 'int',
        _Kind.boolean => '("true" | "false")',
        _Kind.enumeration => '(${f.values.map((v) => _lit(jsonEncode(v))).join(' | ')})',
        _Kind.date => 'date',
      };
      if (f._kind == _Kind.enumeration && f.values.isEmpty) {
        throw ArgumentError('JsonField.enumOf("$name") needs at least one value');
      }
      rules.add('f$i ::= ${f.nullable ? '$v | "null"' : v}');
      i++;
    }
    root.write(' "}"');
    return [
      root.toString(),
      ...rules,
      // Flat `*` / `+` repetition on purpose: llama.cpp expands bounded
      // forms (x{0,n}, nested optionals) into deep rule chains, measured at
      // ~2x the per-token grammar cost. maxNewTokens bounds any runaway.
      'num ::= "-"? [0-9]+ ("." [0-9]+)?',
      'int ::= "-"? [0-9]+',
      r'str ::= "\"" chr* "\""',
      'date ::= "\\"" [12] [0-9] [0-9] [0-9] "-" [01] [0-9] "-" [0-3] [0-9] "\\""',
      r'chr ::= [^"\\\x7F\x00-\x1F] | "\\" ["\\/bfnrt]',
    ].join('\n');
  }

  /// Plain-language description of the fields, used in the prompt.
  String describe() {
    final b = StringBuffer();
    for (final MapEntry(key: name, value: f) in fields.entries) {
      b.write('- $name: ${f._typeLabel}');
      final d = f.description?.trim();
      if (d != null && d.isNotEmpty) b.write('. $d');
      b.writeln();
    }
    return b.toString().trimRight();
  }

  /// GBNF string literal for [text].
  static String _lit(String text) {
    final b = StringBuffer('"');
    for (final c in text.runes) {
      if (c == 0x22) {
        b.write(r'\"');
      } else if (c == 0x5C) {
        b.write(r'\\');
      } else if (c < 0x20) {
        b.write('\\x${c.toRadixString(16).padLeft(2, '0')}');
      } else {
        b.writeCharCode(c);
      }
    }
    b.write('"');
    return b.toString();
  }

  /// Parses [raw] model output into a map with exactly this shape's keys.
  ///
  /// [events] (with their per-token log-probabilities) are used to score
  /// each field. Throws [BitNetExtractionException] for malformed output.
  ExtractResult parse(String raw, List<TokenEvent> events) {
    final Object? decoded;
    try {
      decoded = jsonDecode(raw);
    } on FormatException catch (e) {
      throw BitNetExtractionException('Model output is not complete JSON: ${e.message}',
          raw: raw);
    }
    if (decoded is! Map<String, Object?>) {
      throw BitNetExtractionException('Model output is not a JSON object', raw: raw);
    }
    final value = <String, Object?>{};
    for (final MapEntry(key: name, value: f) in fields.entries) {
      if (!decoded.containsKey(name)) {
        throw BitNetExtractionException('Missing field "$name"', raw: raw);
      }
      final v = decoded[name];
      value[name] = (v is int && f._kind == _Kind.number) ? v.toDouble() : v;
    }

    // Character span of each event in `raw`, to attribute token log-probs
    // to the field values they produced.
    final spans = <(int, int, List<double>)>[];
    var pos = 0;
    for (final e in events) {
      spans.add((pos, pos + e.text.length, e.logprobs));
      pos += e.text.length;
    }
    final fieldConfidence = <String, double>{};
    var searchFrom = 0;
    final names = fields.keys.toList();
    for (var k = 0; k < names.length; k++) {
      final keyText = '${jsonEncode(names[k])}:';
      final keyAt = raw.indexOf(keyText, searchFrom);
      if (keyAt < 0) continue;
      final start = keyAt + keyText.length;
      var end = raw.lastIndexOf('}');
      if (k + 1 < names.length) {
        final next = raw.indexOf(',${jsonEncode(names[k + 1])}:', start);
        if (next >= 0) end = next;
      }
      searchFrom = end;
      var minLp = 0.0;
      var seen = false;
      for (final (a, b, lps) in spans) {
        if (b <= start || a >= end) continue;
        for (final lp in lps) {
          if (!seen || lp < minLp) minLp = lp;
          seen = true;
        }
      }
      fieldConfidence[names[k]] = seen ? math.exp(minLp) : 1.0;
    }
    final confidence = fieldConfidence.values.fold(1.0, math.min);
    return ExtractResult(
      value: value,
      confidence: confidence,
      fieldConfidence: fieldConfidence,
      raw: raw,
    );
  }
}

/// A worked example for `BitNetSession.extract`: an input text and the JSON
/// object it should produce. Two or three good examples matter more than any
/// instruction wording for small on-device models.
@immutable
final class ExtractExample {
  final String text;
  final Map<String, Object?> output;
  const ExtractExample(this.text, this.output);
}

/// Result of `BitNetSession.extract`.
@immutable
final class ExtractResult {
  /// The extracted object; keys exactly match the [JsonShape]. Numbers from
  /// [JsonField.number] are `double`, [JsonField.integer] are `int`,
  /// dates are `"YYYY-MM-DD"` strings, absent values are `null`.
  final Map<String, Object?> value;

  /// How sure the model was, in (0, 1]: the confidence of its least certain
  /// field. Use it to ask the user to confirm instead of saving silently.
  final double confidence;

  /// Per-field confidence: the probability the model gave the least likely
  /// token it emitted for that field's value (on its raw, unconstrained
  /// distribution — a low value means it would rather have said something
  /// else).
  final Map<String, double> fieldConfidence;

  /// The raw JSON text the model produced.
  final String raw;

  const ExtractResult({
    required this.value,
    required this.confidence,
    required this.fieldConfidence,
    required this.raw,
  });

  /// Whether every field's confidence is at least [threshold].
  bool isConfident([double threshold = 0.5]) => confidence >= threshold;

  /// Fields whose confidence is below [threshold] — the ones to highlight
  /// for the user to check.
  List<String> uncertainFields([double threshold = 0.5]) => [
        for (final MapEntry(:key, :value) in fieldConfidence.entries)
          if (value < threshold) key,
      ];

  @override
  String toString() =>
      'ExtractResult($value, confidence: ${confidence.toStringAsFixed(2)})';
}
