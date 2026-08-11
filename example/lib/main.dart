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
  // The model currently loaded into _session, or null if none. Distinct from
  // _selectedModel so the UI can detect when the selection has diverged and
  // offer to (re)load — without this, switching models after one is loaded
  // silently does nothing.
  BitNetModelInfo? _loadedModel;
  List<BitNetModelInfo> _compatibleModels = const [];
  // Which catalog models are already downloaded, keyed by model id. Drives the
  // "Cached / Needs download" badge in the model picker.
  Map<String, bool> _cachedStatus = const {};

  _LoadState _state = _LoadState.idle;
  double _progress = 0;
  String? _errorMsg;

  @override
  void initState() {
    super.initState();
    _profileDevice();
    _refreshCacheStatus();
  }

  Future<void> _profileDevice() async {
    final profile = await DeviceProfile.current();
    setState(() {
      _profile = profile;
      _compatibleModels =
          BitNetCatalog.compatibleWith(profile.totalRamBytes);
    });
  }

  /// Refreshes which catalog models are already downloaded on disk. Called on
  /// startup and after a load completes (a fresh download may have landed).
  Future<void> _refreshCacheStatus() async {
    final status = await ModelCache.cachedStatus(BitNetCatalog.all);
    if (!mounted) return;
    setState(() => _cachedStatus = status);
  }

  Future<void> _loadEngine() async {
    // Tear down any previously loaded model first: its engine owns a native
    // context + inference isolate that must be released before loading another
    // (two full models won't fit in RAM, and the old _session would keep
    // serving the other tabs otherwise).
    final previous = _session;
    setState(() {
      _state = _LoadState.loading;
      _errorMsg = null;
      _progress = 0;
      _session = null;
      _rag = null;
      _loadedModel = null;
    });
    await previous?.dispose();

    final target = _selectedModel;
    final engine = BitNetEngine(model: target);
    try {
      await engine.load(onProgress: (p) {
        // The model is ~1 GB; the user can leave this screen mid-download, so
        // every callback that outlives the widget has to be dropped.
        if (!mounted) return;
        setState(() => _progress = p);
      });
      final session = BitNetSession(engine: engine, model: target)
        ..setSystemPrompt(
            'You are a helpful, concise assistant running fully on-device.');
      final rag = BitNetRag(
        session: session,
        source: AssetKnowledgeSource(
          assetPaths: const ['assets/knowledge/bitnet_faq.md'],
        ),
      );
      await rag.warmUp();
      if (!mounted) {
        // Widget went away while the model was loading — nothing will ever
        // dispose this session, so release the native context here.
        await session.dispose();
        return;
      }
      setState(() {
        _session = session;
        _rag = rag;
        _loadedModel = target;
        _state = _LoadState.ready;
      });
      // A download may have just completed — reflect it in the picker badges.
      _refreshCacheStatus();
    } catch (e) {
      // Catches everything, not just BitNetException: a MissingPluginException
      // from the RAM channel or any other stray error used to escape here and
      // strand the UI on "Loading" with no message and no way to retry.
      await engine.dispose();
      if (!mounted) return;
      setState(() {
        _state = _LoadState.error;
        _errorMsg = e is BitNetException ? e.message : e.toString();
      });
    }
  }

  @override
  void dispose() {
    // Fire-and-forget: dispose() is synchronous, but the session teardown must
    // still run so the native context (and its ~1 GB of weights) is freed.
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
            cachedStatus: _cachedStatus,
            selectedModel: _selectedModel,
            loadedModel: _loadedModel,
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
  /// Downloaded-on-disk status keyed by model id. A missing key is treated as
  /// not-yet-known (cache scan still in flight).
  final Map<String, bool> cachedStatus;
  final BitNetModelInfo selectedModel;
  final BitNetModelInfo? loadedModel;
  final ValueChanged<BitNetModelInfo> onSelectModel;
  final _LoadState state;
  final double progress;
  final String? errorMsg;
  final VoidCallback onLoad;

  const _DeviceTab({
    required this.profile,
    required this.compatibleModels,
    required this.cachedStatus,
    required this.selectedModel,
    required this.loadedModel,
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
            // A plain ListTile rather than RadioListTile: the latter's
            // groupValue/onChanged pair is deprecated in favour of a RadioGroup
            // ancestor, which needs Flutter 3.32 — above this package's
            // declared ">=3.24.0" floor.
            for (final m in BitNetCatalog.all)
              ListTile(
                onTap: state == _LoadState.loading
                    ? null
                    : () => onSelectModel(m),
                enabled: state != _LoadState.loading,
                leading: Icon(
                  m == selectedModel
                      ? Icons.radio_button_checked
                      : Icons.radio_button_unchecked,
                  color: m == selectedModel
                      ? Theme.of(context).colorScheme.primary
                      : null,
                ),
                title: Text(m.displayName),
                subtitle: Text(
                  '${m.paramsBillion}B • ${m.quantization.name.toUpperCase()} • '
                  '${(m.ggufSizeBytes / (1024 * 1024)).toStringAsFixed(0)} MB '
                  '• ${compatibleModels.contains(m) ? "compatible" : "needs more RAM"}',
                  style: const TextStyle(fontSize: 12),
                ),
                trailing: _CacheBadge(cached: cachedStatus[m.id]),
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
        ] else if (state == _LoadState.ready && selectedModel == loadedModel)
          const Center(child: Text('Engine ready — switch tabs to use it.'))
        else if (state == _LoadState.ready)
          // A different model is selected than the one loaded — offer to switch.
          Column(
            children: [
              Text(
                '${loadedModel?.displayName ?? "A model"} is loaded. '
                'Load ${selectedModel.displayName} instead?',
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 13),
              ),
              const SizedBox(height: 8),
              FilledButton(
                onPressed: canRun ? onLoad : null,
                child: Text('Switch to ${selectedModel.displayName}'),
              ),
            ],
          )
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

class _ChatTabState extends State<_ChatTab>
    with AutomaticKeepAliveClientMixin {
  final _ctrl = TextEditingController();
  final _scroll = ScrollController();
  final List<({String role, String body})> _ui = [];

  /// Live reply text. Held in a [ValueNotifier] rather than in State because a
  /// `setState` per token rebuilt every bubble in the list — on the same CPU
  /// that is busy decoding the model.
  final _streaming = ValueNotifier<String>('');
  StreamSubscription<String>? _sub;
  bool _busy = false;

  /// TabBarView only keeps adjacent pages alive, so without this the whole
  /// transcript is thrown away as soon as the user visits a non-neighbouring
  /// tab and comes back.
  @override
  bool get wantKeepAlive => true;

  @override
  void dispose() {
    _sub?.cancel();
    _ctrl.dispose();
    _scroll.dispose();
    _streaming.dispose();
    super.dispose();
  }

  /// Keeps the newest text in view while streaming, unless the user has
  /// scrolled up to read back — then leave their position alone.
  void _followStream() {
    if (!_scroll.hasClients) return;
    final pos = _scroll.position;
    if (pos.maxScrollExtent - pos.pixels > 120) return;
    pos.jumpTo(pos.maxScrollExtent);
  }

  Future<void> _send() async {
    final session = widget.session;
    if (session == null || _busy) return;
    final msg = _ctrl.text.trim();
    if (msg.isEmpty) return;
    _ctrl.clear();
    _streaming.value = '';
    setState(() {
      _ui.add((role: 'user', body: msg));
      _busy = true;
    });

    void finish(String body) {
      _sub = null;
      _streaming.value = '';
      if (!mounted) return;
      setState(() {
        _ui.add((role: 'assistant', body: body));
        _busy = false;
      });
    }

    try {
      _sub = session.chat(msg).listen(
        (piece) {
          _streaming.value += piece;
          WidgetsBinding.instance.addPostFrameCallback((_) => _followStream());
        },
        onDone: () => finish(_streaming.value),
        onError: (Object e) => finish('Error: $e'),
      );
    } catch (e) {
      // chat() rejects synchronously when the engine is busy — the Summarise
      // and Ask tabs drive the same session, so this is reachable without
      // touching this tab twice.
      finish('Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // required by AutomaticKeepAliveClientMixin
    if (widget.session == null) return const _NotReady();
    return Column(children: [
      Expanded(
        child: ListView.builder(
          controller: _scroll,
          padding: const EdgeInsets.all(12),
          itemCount: _ui.length + (_busy ? 1 : 0),
          itemBuilder: (ctx, i) {
            if (i == _ui.length) {
              // Only this bubble rebuilds per token.
              return ValueListenableBuilder<String>(
                valueListenable: _streaming,
                builder: (_, text, __) =>
                    _Bubble(role: 'assistant', body: text),
              );
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

class _SummariseTabState extends State<_SummariseTab>
    with AutomaticKeepAliveClientMixin {
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

  /// See the note on _ChatTabState.wantKeepAlive.
  @override
  bool get wantKeepAlive => true;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Future<void> _run() async {
    final session = widget.session;
    if (session == null || _busy) return;
    setState(() {
      _busy = true;
      _output = '';
    });
    try {
      final out = await session.summarize(_ctrl.text, style: _style);
      if (!mounted) return;
      setState(() => _output = out);
    } catch (e) {
      if (!mounted) return;
      setState(() => _output = 'Error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // required by AutomaticKeepAliveClientMixin
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

class _RagTabState extends State<_RagTab> with AutomaticKeepAliveClientMixin {
  final _ctrl = TextEditingController(text: 'How big is the BitNet download?');
  String _answer = '';
  List<RetrievedChunk> _citations = const [];
  bool _busy = false;

  /// See the note on _ChatTabState.wantKeepAlive.
  @override
  bool get wantKeepAlive => true;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Future<void> _ask() async {
    final rag = widget.rag;
    if (rag == null || _busy) return;
    setState(() {
      _busy = true;
      _answer = '';
      _citations = const [];
    });
    try {
      final res = await rag.ask(_ctrl.text);
      if (!mounted) return;
      setState(() {
        _answer = res.answer;
        _citations = res.citations;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _answer = 'Error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context); // required by AutomaticKeepAliveClientMixin
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

/// Small trailing badge indicating whether a model's GGUF is already on disk.
/// [cached] null → the cache scan is still running (shows a spinner).
class _CacheBadge extends StatelessWidget {
  final bool? cached;
  const _CacheBadge({required this.cached});

  @override
  Widget build(BuildContext context) {
    if (cached == null) {
      return const SizedBox(
        width: 16,
        height: 16,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    }
    final (label, icon, color) = cached!
        ? ('Cached', Icons.check_circle, Colors.green)
        : ('Download', Icons.download, Colors.blueGrey);
    return Chip(
      avatar: Icon(icon, size: 16, color: color),
      label: Text(label, style: const TextStyle(fontSize: 11)),
      backgroundColor: color.withAlpha(30),
      side: BorderSide(color: color),
      padding: EdgeInsets.zero,
      visualDensity: VisualDensity.compact,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
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
