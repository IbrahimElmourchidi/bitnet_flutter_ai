// AUTO-GENERATED FILE. DO NOT EDIT.
// Run: dart run ffigen --config ffigen.yaml
// ignore_for_file: type=lint, always_use_package_imports, camel_case_types, non_constant_identifier_names

import 'dart:ffi';

// ---------------------------------------------------------------------------
// Opaque struct
// ---------------------------------------------------------------------------

/// Opaque handle to a BitNet inference context.
/// Never instantiated in Dart — only ever held as [Pointer<BnContext>].
final class BnContext extends Opaque {}

// ---------------------------------------------------------------------------
// C function typedefs (native side)
// ---------------------------------------------------------------------------

typedef _BnInitNative = Pointer<BnContext> Function(
  Pointer<Char> modelPath,
  Int32 nCtx,
  Int32 nThreads,
  Pointer<Pointer<Char>> outErr,
);

typedef _BnFreeNative = Void Function(Pointer<BnContext> ctx);

typedef _BnFreeStringNative = Void Function(Pointer<Char> str);

typedef _BnPromptNative = Int32 Function(
  Pointer<BnContext> ctx,
  Pointer<Char> prompt,
  Int32 addBos,
  Pointer<Pointer<Char>> outErr,
);

typedef _BnNextTokenNative = Pointer<Char> Function(
  Pointer<BnContext> ctx,
  Pointer<Int32> outDone,
  Pointer<Pointer<Char>> outErr,
);

typedef _BnContextSizeNative = Int32 Function(Pointer<BnContext> ctx);

typedef _BnMetaNative = Pointer<Char> Function(
  Pointer<BnContext> ctx,
  Pointer<Char> key,
);

// ---------------------------------------------------------------------------
// Dart-side function typedefs
// ---------------------------------------------------------------------------

typedef _BnInitDart = Pointer<BnContext> Function(
  Pointer<Char> modelPath,
  int nCtx,
  int nThreads,
  Pointer<Pointer<Char>> outErr,
);

typedef _BnFreeDart = void Function(Pointer<BnContext> ctx);

typedef _BnFreeStringDart = void Function(Pointer<Char> str);

typedef _BnPromptDart = int Function(
  Pointer<BnContext> ctx,
  Pointer<Char> prompt,
  int addBos,
  Pointer<Pointer<Char>> outErr,
);

typedef _BnNextTokenDart = Pointer<Char> Function(
  Pointer<BnContext> ctx,
  Pointer<Int32> outDone,
  Pointer<Pointer<Char>> outErr,
);

typedef _BnContextSizeDart = int Function(Pointer<BnContext> ctx);

typedef _BnMetaDart = Pointer<Char> Function(
  Pointer<BnContext> ctx,
  Pointer<Char> key,
);

// ---------------------------------------------------------------------------
// Bindings class
// ---------------------------------------------------------------------------

/// Raw FFI bindings to bitnet_bridge (mirrors ffigen output shape).
///
/// Consumers should use [NativeLibrary] instead of this class directly.
/// [NativeLibrary] manages pointer lifetimes and converts errors to exceptions.
class BitNetBindings {
  final DynamicLibrary _lib;

  BitNetBindings(this._lib);

  // Lazily resolved function pointers

  late final _BnInitDart bn_init = _lib
      .lookupFunction<_BnInitNative, _BnInitDart>('bn_init');

  late final _BnFreeDart bn_free = _lib
      .lookupFunction<_BnFreeNative, _BnFreeDart>('bn_free');

  late final _BnFreeStringDart bn_free_string = _lib
      .lookupFunction<_BnFreeStringNative, _BnFreeStringDart>('bn_free_string');

  late final _BnPromptDart bn_prompt = _lib
      .lookupFunction<_BnPromptNative, _BnPromptDart>('bn_prompt');

  late final _BnNextTokenDart bn_next_token = _lib
      .lookupFunction<_BnNextTokenNative, _BnNextTokenDart>('bn_next_token');

  late final _BnContextSizeDart bn_context_size = _lib
      .lookupFunction<_BnContextSizeNative, _BnContextSizeDart>('bn_context_size');

  late final _BnMetaDart bn_meta = _lib
      .lookupFunction<_BnMetaNative, _BnMetaDart>('bn_meta');
}
