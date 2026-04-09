import 'dart:async';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const BitNetExampleApp());
}

class BitNetExampleApp extends StatelessWidget {
  const BitNetExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BitNet Flutter AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6750A4)),
        useMaterial3: true,
      ),
      home: const ChatPage(),
    );
  }
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _engine = BitNetEngine();
  final _promptController = TextEditingController();
  final _scrollController = ScrollController();

  _EngineState _engineState = _EngineState.idle;
  double _downloadProgress = 0;
  String _response = '';
  bool _generating = false;
  StreamSubscription<String>? _sub;

  @override
  void initState() {
    super.initState();
    _loadEngine();
  }

  Future<void> _loadEngine() async {
    setState(() => _engineState = _EngineState.loading);
    try {
      await _engine.load(
        onProgress: (p) => setState(() => _downloadProgress = p),
      );
      setState(() => _engineState = _EngineState.ready);
    } on BitNetUnsupportedDeviceException catch (e) {
      _showError(
        'Not enough RAM',
        '${e.availableRamBytes ~/ (1024 * 1024)} MB available — 3072 MB required.',
      );
      setState(() => _engineState = _EngineState.error);
    } on BitNetDownloadException catch (e) {
      _showError('Download failed', e.message);
      setState(() => _engineState = _EngineState.error);
    } on BitNetException catch (e) {
      _showError('Engine error', e.message);
      setState(() => _engineState = _EngineState.error);
    }
  }

  Future<void> _send() async {
    final prompt = _promptController.text.trim();
    if (prompt.isEmpty || _generating) return;

    setState(() {
      _response = '';
      _generating = true;
    });

    _sub = _engine
        .generate(prompt, maxNewTokens: 512)
        .listen(
          (token) {
            setState(() => _response += token);
            _scrollToBottom();
          },
          onError: (Object e) {
            _showError('Generation error', e.toString());
            setState(() => _generating = false);
          },
          onDone: () => setState(() => _generating = false),
        );
  }

  Future<void> _cancel() async {
    await _sub?.cancel();
    await _engine.cancelGeneration();
    setState(() => _generating = false);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 80),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showError(String title, String body) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$title: $body'),
        backgroundColor: Theme.of(context).colorScheme.error,
        duration: const Duration(seconds: 6),
      ),
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _engine.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('BitNet 2B-4T'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: _StatusChip(state: _engineState),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(child: _buildBody()),
          _buildInput(),
        ],
      ),
    );
  }

  Widget _buildBody() {
    switch (_engineState) {
      case _EngineState.idle:
        return const SizedBox.shrink();

      case _EngineState.loading:
        return Center(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Loading model…',
                  style: TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 16),
                if (_downloadProgress > 0) ...[
                  LinearProgressIndicator(value: _downloadProgress),
                  const SizedBox(height: 8),
                  Text('${(_downloadProgress * 100).toInt()}%'),
                ] else
                  const CircularProgressIndicator(),
              ],
            ),
          ),
        );

      case _EngineState.error:
        return Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 12),
              const Text('Failed to load engine.'),
              const SizedBox(height: 12),
              FilledButton(
                onPressed: _loadEngine,
                child: const Text('Retry'),
              ),
            ],
          ),
        );

      case _EngineState.ready:
        return _response.isEmpty
            ? Center(
                child: Text(
                  'Model ready.\nType a prompt below.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Theme.of(context).hintColor),
                ),
              )
            : SingleChildScrollView(
                controller: _scrollController,
                padding: const EdgeInsets.all(16),
                child: SelectableText(
                  _response,
                  style: const TextStyle(fontSize: 15, height: 1.5),
                ),
              );
    }
  }

  Widget _buildInput() {
    final ready = _engineState == _EngineState.ready;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _promptController,
                enabled: ready && !_generating,
                decoration: InputDecoration(
                  hintText: ready ? 'Enter a prompt…' : 'Loading…',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                ),
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _send(),
                maxLines: null,
              ),
            ),
            const SizedBox(width: 8),
            _generating
                ? IconButton.filled(
                    onPressed: _cancel,
                    icon: const Icon(Icons.stop),
                    tooltip: 'Stop generation',
                  )
                : IconButton.filled(
                    onPressed: ready ? _send : null,
                    icon: const Icon(Icons.send),
                    tooltip: 'Send',
                  ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

enum _EngineState { idle, loading, ready, error }

class _StatusChip extends StatelessWidget {
  final _EngineState state;
  const _StatusChip({required this.state});

  @override
  Widget build(BuildContext context) {
    final (label, color) = switch (state) {
      _EngineState.idle => ('Idle', Colors.grey),
      _EngineState.loading => ('Loading', Colors.orange),
      _EngineState.ready => ('Ready', Colors.green),
      _EngineState.error => ('Error', Colors.red),
    };
    return Chip(
      label: Text(label, style: const TextStyle(fontSize: 12)),
      backgroundColor: color.withAlpha(40),
      side: BorderSide(color: color),
      padding: EdgeInsets.zero,
      visualDensity: VisualDensity.compact,
    );
  }
}
