import 'dart:async';

import 'package:bitnet_flutter_ai/bitnet_flutter_ai.dart';
import 'package:flutter/material.dart';

void main() {
  BitNetLog.enabled = true;
  BitNetLog.sink = _LogBuffer.instance.add;
  runApp(const BitNetExampleApp());
}

/// In-memory ring buffer for the Logs tab.
class _LogBuffer extends ChangeNotifier {
  static final instance = _LogBuffer();
  final List<String> lines = [];
  void add(String line) {
    lines.add(line);
    if (lines.length > 500) lines.removeRange(0, lines.length - 500);
    notifyListeners();
  }

  void clear() {
    lines.clear();
    notifyListeners();
  }
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
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  BitNetSession? _session;
  BitNetRag? _rag;
  DeviceProfile? _profile;
  BitNetModelInfo _selectedModel = BitNetCatalog.defaultModel;
  List<BitNetModelInfo> _compatibleModels = const [];

  _LoadState _state = _LoadState.idle;
  double _progress = 0;
  String? _errorMsg;

  @override
  void initState() {
    super.initState();
    _profileDevice();
  }

  Future<void> _profileDevice() async {
    final profile = await DeviceProfile.current();
    setState(() {
      _profile = profile;
      _compatibleModels =
          BitNetCatalog.compatibleWith(profile.totalRamBytes);
    });
  }

  Future<void> _loadEngine() async {
    setState(() {
      _state = _LoadState.loading;
      _errorMsg = null;
      _progress = 0;
    });
    final engine = BitNetEngine(model: _selectedModel);
    try {
      await engine.load(onProgress: (p) => setState(() => _progress = p));
      final session = BitNetSession(engine: engine, model: _selectedModel)
        ..setSystemPrompt(
            'You are a helpful, concise assistant running fully on-device.');
      final rag = BitNetRag(
        session: session,
        source: AssetKnowledgeSource(
          assetPaths: const ['assets/knowledge/bitnet_faq.md'],
        ),
      );
      await rag.warmUp();
      setState(() {
        _session = session;
        _rag = rag;
        _state = _LoadState.ready;
      });
    } on BitNetException catch (e) {
      await engine.dispose();
      setState(() {
        _state = _LoadState.error;
        _errorMsg = e.message;
      });
    }
  }

  @override
  void dispose() {
    _session?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 5,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('BitNet Flutter AI'),
          actions: [
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: _StatusChip(state: _state),
            ),
          ],
          bottom: const TabBar(tabs: [
            Tab(text: 'Device'),
            Tab(text: 'Chat'),
            Tab(text: 'Summarise'),
            Tab(text: 'Ask (RAG)'),
            Tab(text: 'Logs'),
          ]),
        ),
        body: TabBarView(children: [
          _DeviceTab(
            profile: _profile,
            compatibleModels: _compatibleModels,
            selectedModel: _selectedModel,
            onSelectModel: (m) => setState(() => _selectedModel = m),
            state: _state,
            progress: _progress,
            errorMsg: _errorMsg,
            onLoad: _loadEngine,
          ),
          _ChatTab(session: _session),
          _SummariseTab(session: _session),
          _RagTab(rag: _rag),
          const _LogsTab(),
        ]),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Device tab
// ---------------------------------------------------------------------------

class _DeviceTab extends StatelessWidget {
  final DeviceProfile? profile;
  final List<BitNetModelInfo> compatibleModels;
  final BitNetModelInfo selectedModel;
  final ValueChanged<BitNetModelInfo> onSelectModel;
  final _LoadState state;
  final double progress;
  final String? errorMsg;
  final VoidCallback onLoad;

  const _DeviceTab({
    required this.profile,
    required this.compatibleModels,
    required this.selectedModel,
    required this.onSelectModel,
    required this.state,
    required this.progress,
    required this.errorMsg,
    required this.onLoad,
  });

  @override
  Widget build(BuildContext context) {
    if (profile == null) {
      return const Center(child: CircularProgressIndicator());
    }
    final tokps = profile!.estimateTokensPerSecond(selectedModel);
    final canRun = profile!.canRun(selectedModel);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _Card(
          title: 'Device profile',
          children: [
            _Kv('OS / ABI', '${profile!.os} / ${profile!.abi}'),
            _Kv('CPU cores', '${profile!.cpuCores}'),
            _Kv(
              'RAM',
              '${(profile!.totalRamBytes / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB',
            ),
          ],
        ),
        const SizedBox(height: 12),
        _Card(
          title: 'Available models',
          children: [
            for (final m in BitNetCatalog.all)
              RadioListTile<BitNetModelInfo>(
                value: m,
                groupValue: selectedModel,
                onChanged: state == _LoadState.loading
                    ? null
                    : (v) => onSelectModel(v!),
                title: Text(m.displayName),
                subtitle: Text(
                  '${m.paramsBillion}B • ${m.quantization.name.toUpperCase()} • '
                  '${(m.ggufSizeBytes / (1024 * 1024)).toStringAsFixed(0)} MB '
                  '• ${compatibleModels.contains(m) ? "compatible" : "needs more RAM"}',
                  style: const TextStyle(fontSize: 12),
                ),
              ),
          ],
        ),
        const SizedBox(height: 12),
        _Card(
          title: 'Selected: ${selectedModel.displayName}',
          children: [
            _Kv('Compatible', canRun ? 'yes' : 'no'),
            _Kv('Estimated speed', '~${tokps.toStringAsFixed(1)} tok/s'),
            _Kv(
              'Context window',
              '${selectedModel.contextLength} tokens',
            ),
            _Kv(
              'Capabilities',
              selectedModel.capabilities.map((c) => c.name).join(', '),
            ),
          ],
        ),
        const SizedBox(height: 16),
        if (state == _LoadState.loading) ...[
          LinearProgressIndicator(value: progress > 0 ? progress : null),
          const SizedBox(height: 8),
          if (progress > 0)
            // Covers both phases: model download, then (on web) the WASM
            // model-load progress reported during bn_init.
            Center(child: Text('Loading… ${(progress * 100).toInt()}%'))
          else
            const Center(child: Text('Initialising…')),
        ] else if (state == _LoadState.error) ...[
          Text(
            errorMsg ?? 'Failed to load engine.',
            style: TextStyle(color: Theme.of(context).colorScheme.error),
          ),
          const SizedBox(height: 8),
          FilledButton(onPressed: onLoad, child: const Text('Retry')),
        ] else if (state == _LoadState.ready)
          const Center(child: Text('Engine ready — switch tabs to use it.'))
        else
          FilledButton(
            onPressed: canRun ? onLoad : null,
            child: const Text('Load model'),
          ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Chat tab
// ---------------------------------------------------------------------------

class _ChatTab extends StatefulWidget {
  final BitNetSession? session;
  const _ChatTab({required this.session});

  @override
  State<_ChatTab> createState() => _ChatTabState();
}

class _ChatTabState extends State<_ChatTab> {
  final _ctrl = TextEditingController();
  final _scroll = ScrollController();
  final List<({String role, String body})> _ui = [];
  String _streaming = '';
  StreamSubscription<String>? _sub;
  bool _busy = false;

  @override
  void dispose() {
    _sub?.cancel();
    _ctrl.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final session = widget.session;
    if (session == null || _busy) return;
    final msg = _ctrl.text.trim();
    if (msg.isEmpty) return;
    _ctrl.clear();
    setState(() {
      _ui.add((role: 'user', body: msg));
      _streaming = '';
      _busy = true;
    });
    _sub = session.chat(msg).listen(
      (piece) => setState(() => _streaming += piece),
      onDone: () => setState(() {
        _ui.add((role: 'assistant', body: _streaming));
        _streaming = '';
        _busy = false;
      }),
      onError: (Object e) => setState(() {
        _ui.add((role: 'assistant', body: 'Error: $e'));
        _streaming = '';
        _busy = false;
      }),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (widget.session == null) return const _NotReady();
    return Column(children: [
      Expanded(
        child: ListView.builder(
          controller: _scroll,
          padding: const EdgeInsets.all(12),
          itemCount: _ui.length + (_streaming.isNotEmpty ? 1 : 0),
          itemBuilder: (ctx, i) {
            if (i == _ui.length) {
              return _Bubble(role: 'assistant', body: _streaming);
            }
            return _Bubble(role: _ui[i].role, body: _ui[i].body);
          },
        ),
      ),
      SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Row(children: [
            Expanded(
              child: TextField(
                controller: _ctrl,
                enabled: !_busy,
                decoration: const InputDecoration(
                  hintText: 'Message…',
                  border: OutlineInputBorder(),
                ),
                onSubmitted: (_) => _send(),
              ),
            ),
            const SizedBox(width: 4),
            if (_busy)
              IconButton.filled(
                onPressed: () => widget.session?.stop(),
                icon: const Icon(Icons.stop),
                tooltip: 'Stop',
              )
            else
              IconButton.filled(
                onPressed: _send,
                icon: const Icon(Icons.send),
              ),
          ]),
        ),
      ),
    ]);
  }
}

// ---------------------------------------------------------------------------
// Summarise tab
// ---------------------------------------------------------------------------

class _SummariseTab extends StatefulWidget {
  final BitNetSession? session;
  const _SummariseTab({required this.session});

  @override
  State<_SummariseTab> createState() => _SummariseTabState();
}

class _SummariseTabState extends State<_SummariseTab> {
  final _ctrl = TextEditingController(
    text:
        'BitNet b1.58 is a 1.58-bit quantised LLM architecture from Microsoft. '
        'It constrains every weight to -1, 0, or +1, which dramatically reduces '
        'model size and lets the network run on CPU without a GPU. The 2B-4T '
        'variant fits in roughly 1.1 GB on disk.',
  );
  String _style = 'paragraph';
  String _output = '';
  bool _busy = false;

  Future<void> _run() async {
    final session = widget.session;
    if (session == null) return;
    setState(() {
      _busy = true;
      _output = '';
    });
    try {
      final out = await session.summarize(_ctrl.text, style: _style);
      setState(() => _output = out);
    } catch (e) {
      setState(() => _output = 'Error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.session == null) return const _NotReady();
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        TextField(
          controller: _ctrl,
          maxLines: 8,
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            labelText: 'Text to summarise',
          ),
        ),
        const SizedBox(height: 12),
        SegmentedButton<String>(
          segments: const [
            ButtonSegment(value: 'paragraph', label: Text('Paragraph')),
            ButtonSegment(value: 'bullets', label: Text('Bullets')),
            ButtonSegment(value: 'tldr', label: Text('TL;DR')),
          ],
          selected: {_style},
          onSelectionChanged: (s) => setState(() => _style = s.first),
        ),
        const SizedBox(height: 12),
        Row(children: [
          Expanded(
            child: FilledButton(
              onPressed: _busy ? null : _run,
              child: Text(_busy ? 'Working…' : 'Summarise'),
            ),
          ),
          if (_busy) ...[
            const SizedBox(width: 8),
            OutlinedButton.icon(
              onPressed: () => widget.session?.stop(),
              icon: const Icon(Icons.stop),
              label: const Text('Stop'),
            ),
          ],
        ]),
        const SizedBox(height: 16),
        if (_output.isNotEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: SelectableText(_output),
            ),
          ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// RAG tab
// ---------------------------------------------------------------------------

class _RagTab extends StatefulWidget {
  final BitNetRag? rag;
  const _RagTab({required this.rag});

  @override
  State<_RagTab> createState() => _RagTabState();
}

class _RagTabState extends State<_RagTab> {
  final _ctrl = TextEditingController(text: 'How big is the BitNet download?');
  String _answer = '';
  List<RetrievedChunk> _citations = const [];
  bool _busy = false;

  Future<void> _ask() async {
    final rag = widget.rag;
    if (rag == null) return;
    setState(() {
      _busy = true;
      _answer = '';
      _citations = const [];
    });
    try {
      final res = await rag.ask(_ctrl.text);
      setState(() {
        _answer = res.answer;
        _citations = res.citations;
      });
    } catch (e) {
      setState(() => _answer = 'Error: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.rag == null) return const _NotReady();
    return ListView(
      padding: const EdgeInsets.all(12),
      children: [
        Text(
          'Asks the model with retrieved context from '
          'assets/knowledge/bitnet_faq.md.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _ctrl,
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            labelText: 'Question',
          ),
          onSubmitted: (_) => _ask(),
        ),
        const SizedBox(height: 12),
        Row(children: [
          Expanded(
            child: FilledButton(
              onPressed: _busy ? null : _ask,
              child: Text(_busy ? 'Thinking…' : 'Ask'),
            ),
          ),
          if (_busy) ...[
            const SizedBox(width: 8),
            OutlinedButton.icon(
              onPressed: () => widget.rag?.stop(),
              icon: const Icon(Icons.stop),
              label: const Text('Stop'),
            ),
          ],
        ]),
        const SizedBox(height: 16),
        if (_answer.isNotEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: SelectableText(_answer),
            ),
          ),
        if (_citations.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text('Citations',
              style: Theme.of(context).textTheme.titleSmall),
          for (final c in _citations)
            ListTile(
              dense: true,
              title: Text('${c.chunk.source}#${c.chunk.chunkIndex}',
                  style: const TextStyle(fontFamily: 'monospace')),
              subtitle: Text(
                c.chunk.text,
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
              trailing: Text('score ${c.score.toStringAsFixed(1)}'),
            ),
        ],
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Logs tab
// ---------------------------------------------------------------------------

class _LogsTab extends StatefulWidget {
  const _LogsTab();
  @override
  State<_LogsTab> createState() => _LogsTabState();
}

class _LogsTabState extends State<_LogsTab> {
  @override
  void initState() {
    super.initState();
    _LogBuffer.instance.addListener(_onChange);
  }

  @override
  void dispose() {
    _LogBuffer.instance.removeListener(_onChange);
    super.dispose();
  }

  void _onChange() => setState(() {});

  @override
  Widget build(BuildContext context) {
    final lines = _LogBuffer.instance.lines;
    return Column(children: [
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(children: [
          Expanded(
            child: Text('${lines.length} log lines',
                style: Theme.of(context).textTheme.bodySmall),
          ),
          TextButton.icon(
            onPressed: _LogBuffer.instance.clear,
            icon: const Icon(Icons.clear_all, size: 18),
            label: const Text('Clear'),
          ),
        ]),
      ),
      const Divider(height: 1),
      Expanded(
        child: ListView.builder(
          reverse: true,
          itemCount: lines.length,
          itemBuilder: (ctx, i) {
            final line = lines[lines.length - 1 - i];
            return Padding(
              padding: const EdgeInsets.symmetric(
                  horizontal: 12, vertical: 2),
              child: SelectableText(
                line,
                style: const TextStyle(
                    fontFamily: 'monospace', fontSize: 11),
              ),
            );
          },
        ),
      ),
    ]);
  }
}

// ---------------------------------------------------------------------------
// Shared widgets
// ---------------------------------------------------------------------------

enum _LoadState { idle, loading, ready, error }

class _StatusChip extends StatelessWidget {
  final _LoadState state;
  const _StatusChip({required this.state});

  @override
  Widget build(BuildContext context) {
    final (label, color) = switch (state) {
      _LoadState.idle => ('Idle', Colors.grey),
      _LoadState.loading => ('Loading', Colors.orange),
      _LoadState.ready => ('Ready', Colors.green),
      _LoadState.error => ('Error', Colors.red),
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

class _NotReady extends StatelessWidget {
  const _NotReady();
  @override
  Widget build(BuildContext context) => const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'Engine not loaded yet.\nGo to the Device tab and tap "Load model".',
            textAlign: TextAlign.center,
          ),
        ),
      );
}

class _Card extends StatelessWidget {
  final String title;
  final List<Widget> children;
  const _Card({required this.title, required this.children});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            ...children,
          ],
        ),
      ),
    );
  }
}

class _Kv extends StatelessWidget {
  final String k;
  final String v;
  const _Kv(this.k, this.v);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(children: [
          SizedBox(width: 130, child: Text(k)),
          Expanded(
            child: Text(v, style: const TextStyle(fontFamily: 'monospace')),
          ),
        ]),
      );
}

class _Bubble extends StatelessWidget {
  final String role;
  final String body;
  const _Bubble({required this.role, required this.body});
  @override
  Widget build(BuildContext context) {
    final isUser = role == 'user';
    final bg = isUser
        ? Theme.of(context).colorScheme.primaryContainer
        : Theme.of(context).colorScheme.surfaceContainerHighest;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(10),
        constraints: const BoxConstraints(maxWidth: 480),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(12),
        ),
        child: SelectableText(body),
      ),
    );
  }
}
