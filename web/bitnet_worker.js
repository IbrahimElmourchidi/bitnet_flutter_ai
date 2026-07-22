/**
 * bitnet_worker.js — runs the BitNet WASM module inside a Web Worker so model
 * loading and token generation never block the browser's main (UI) thread.
 * This mirrors the dedicated Dart isolate used by the native engine.
 *
 * Protocol (postMessage): the main thread (bitnet_glue.js) sends
 *   { op, id, ... } and this worker replies with
 *   { type: 'result', id, value } | { type: 'error', id, error }.
 * Download progress is pushed as { type: 'progress', frac }. When the module
 * has instantiated, the worker emits { type: 'ready' } once.
 *
 * A plain Web Worker (not a SharedArrayBuffer/pthread worker) needs no COOP/
 * COEP headers — the module is single-threaded.
 */
/* global importScripts, createBitNetModule, postMessage, caches, fetch */
'use strict';

importScripts('bitnet_bridge.js'); // defines createBitNetModule

var CACHE_NAME = 'bitnet-models-v1';
var Module = null;

createBitNetModule().then(function (m) {
  Module = m;
  postMessage({ type: 'ready' });
}).catch(function (e) {
  postMessage({ type: 'fatal', error: String(e) });
});

function mkdirTree(FS, dir) {
  var parts = dir.split('/').filter(Boolean);
  var path = '';
  for (var i = 0; i < parts.length; i++) {
    path += '/' + parts[i];
    try { FS.mkdir(path); } catch (e) { /* exists */ }
  }
}

// Download `url` (Cache-API persisted so refreshes don't re-download) and
// stream it into the module's in-memory FS at `fsPath`. `cache.add` streams to
// the disk-backed Cache, so we never hold the whole ~1.1 GB in a JS buffer.
async function provisionModel(url, fsPath, sizeBytes) {
  var FS = Module.FS;
  try {
    if (FS.stat(fsPath).size === sizeBytes && sizeBytes > 0) {
      postMessage({ type: 'progress', frac: 1 });
      return;
    }
  } catch (e) { /* not present */ }

  var slash = fsPath.lastIndexOf('/');
  if (slash > 0) mkdirTree(FS, fsPath.substring(0, slash));

  var response = null;
  var cache = null;
  if (self.caches) {
    try {
      cache = await caches.open(CACHE_NAME);
      response = await cache.match(url);
      if (!response) {
        await cache.add(url);
        response = await cache.match(url);
      }
    } catch (e) {
      response = null;
    }
  }
  if (!response) {
    response = await fetch(url, { credentials: 'same-origin' });
    if (!response.ok) throw new Error('model download failed: HTTP ' + response.status);
  }

  var total = sizeBytes || Number(response.headers.get('content-length')) || 0;
  var received = 0;
  var offset = 0;
  var stream = FS.open(fsPath, 'w');
  try {
    var reader = response.body.getReader();
    for (;;) {
      var step = await reader.read();
      if (step.done) break;
      var data = step.value;
      FS.write(stream, data, 0, data.length, offset);
      offset += data.length;
      received += data.length;
      if (total) postMessage({ type: 'progress', frac: Math.min(1, received / total) });
    }
  } finally {
    FS.close(stream);
  }
  postMessage({ type: 'progress', frac: 1 });
}

onmessage = async function (ev) {
  var msg = ev.data;
  var id = msg.id;
  try {
    if (!Module) throw new Error('WASM module not ready');
    switch (msg.op) {
      case 'provision':
        await provisionModel(msg.url, msg.fsPath, msg.sizeBytes);
        postMessage({ type: 'result', id: id, value: null });
        break;
      case 'init': {
        var h = Module.bn_init(msg.fsPath, msg.nCtx, msg.nThreads);
        // Reclaim the in-memory GGUF now that tensors are in llama's buffers.
        try { Module.FS.unlink(msg.fsPath); } catch (e) { /* ignore */ }
        postMessage({ type: 'result', id: id, value: (h === null || h === undefined) ? null : h });
        break;
      }
      case 'prompt': {
        var rc = Module.bn_prompt(msg.ctx, msg.prompt, msg.addBos);
        postMessage({ type: 'result', id: id, value: rc });
        break;
      }
      case 'next': {
        var r = Module.bn_next_token(msg.ctx);
        postMessage({ type: 'result', id: id, value: { done: r.done, token: r.token } });
        break;
      }
      case 'free':
        Module.bn_free(msg.ctx);
        postMessage({ type: 'result', id: id, value: null });
        break;
      default:
        throw new Error('unknown op: ' + msg.op);
    }
  } catch (e) {
    postMessage({ type: 'error', id: id, error: String(e) });
  }
};
