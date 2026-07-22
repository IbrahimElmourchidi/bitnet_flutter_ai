# BitNet b1.58 2B-4T — Frequently Asked Questions

## What is BitNet?

BitNet is a 1.58-bit large language model architecture from Microsoft. Each
weight is constrained to one of three values: -1, 0, or +1. This ternary
quantisation lets the entire 2-billion-parameter model fit in roughly 1.1 GB
on disk, and it runs efficiently on CPU without a GPU.

## Why does BitNet matter on mobile?

Traditional 2B-parameter LLMs require 4-8 GB of RAM and dedicated accelerators
to be useful. BitNet's ternary weights and lookup-table kernels (TL1 on ARM,
TL2 on x86) bring inference down to roughly 3 GB of RAM and 5-15 tokens per
second on a modern phone. That makes truly offline, private LLMs practical
inside a Flutter app.

## What is the context window?

BitNet 2B-4T was trained with a 4096-token context window. You can feed up to
that many tokens of prompt + history; longer conversations are silently
truncated.

## How big is the download?

The official I2_S GGUF artifact is about 1.13 GB. The package downloads it
once on first run and caches it in the app's documents directory; subsequent
launches start instantly.

## Which Android devices are supported?

Any 64-bit Android device (arm64-v8a) running Android 7 (API 24) or later
with at least 3 GB of physical RAM. The plugin also bundles x86_64 binaries
so the model can be developed against an Android emulator on Intel/AMD dev
machines. 32-bit ABIs (armeabi-v7a, x86) are not supported.

## What about iOS?

iOS 13 and above on any 64-bit Apple device. Apple Silicon Macs running
Catalyst are also supported through the iOS xcframework.

## Is the model output reliable?

BitNet 2B-4T is a small base/instruct hybrid. Use it for short summaries,
straightforward question answering, and on-device chat. For complex
multi-step reasoning a larger hosted model is still the right choice.
