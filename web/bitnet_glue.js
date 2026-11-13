/**
 * bitnet_glue.js — main-thread proxy over the BitNet Web Worker.
 *
 * Spawns bitnet_worker.js (which owns the WASM module) and exposes an async
 * `window.BitNetWasm` whose methods return Promises, resolved when the worker
 * replies. Because the module lives in the worker, model loading and token
 * generation never block the UI thread.
 *
 * index.html only needs this one script:
 *   <script src="bitnet_glue.js"></script>
 * (bitnet_worker.js loads bitnet_bridge.js + .wasm itself via importScripts.)
 *
 * API on window.BitNetWasm (all async):
 *   provisionModel(url, fsPath, sizeBytes, onProgress) -> Promise<void>
 *   bn_init(fsPath, nCtx, nThreads)                    -> Promise<handle|null>
 *   bn_set_sampling(handle, temp, topK, topP, repeat, lastN, seed, grammar)
 *                                                      -> Promise<{ok, error}>
 *   bn_prompt(handle, prompt, addBos)                  -> Promise<{rc, error}>
 *   bn_next_token(handle)          -> Promise<{done, bytes: Uint8Array|null, logprob, error}>
 *   bn_token_count(handle, text, addBos)               -> Promise<int>
 *   bn_free(handle)                                    -> Promise<void>
 *
 * window.BitNetWasmReady resolves to the same object once the worker is up.
 */
(function () {
  'use strict';

  var worker;
  try {
    worker = new Worker('bitnet_worker.js');
  } catch (e) {
    console.error('[bitnet] failed to spawn worker:', e);
    return;
  }

  var nextId = 1;
  var pending = new Map(); // id -> { resolve, reject }
  var progressCb = null;

  var resolveReady;
  window.BitNetWasmReady = new Promise(function (r) { resolveReady = r; });

  var api = {
    provisionModel: function (url, fsPath, sizeBytes, onProgress) {
      // Keep progressCb set past download completion: model-load progress
      // (emitted during bn_init, below) reuses the same callback so the UI
      // keeps moving instead of freezing at "100%". Cleared when bn_init ends.
      progressCb = onProgress || null;
      return call('provision', { url: url, fsPath: fsPath, sizeBytes: sizeBytes });
    },
    bn_init: function (fsPath, nCtx, nThreads) {
      return call('init', { fsPath: fsPath, nCtx: nCtx, nThreads: nThreads })
        .then(function (v) { progressCb = null; return v; },
              function (e) { progressCb = null; throw e; });
    },
    bn_set_sampling: function (ctx, temperature, topK, topP, repeatPenalty, repeatLastN, seed, grammar) {
      return call('sampling', {
        ctx: ctx, temperature: temperature, topK: topK, topP: topP,
        repeatPenalty: repeatPenalty, repeatLastN: repeatLastN, seed: seed,
        grammar: grammar,
      });
    },
    bn_prompt: function (ctx, prompt, addBos) {
      return call('prompt', { ctx: ctx, prompt: prompt, addBos: addBos });
    },
    bn_token_count: function (ctx, text, addBos) {
      return call('count', { ctx: ctx, text: text, addBos: addBos });
    },
    bn_next_token: function (ctx) {
      return call('next', { ctx: ctx });
    },
    bn_free: function (ctx) {
      return call('free', { ctx: ctx });
    },
  };

  worker.onmessage = function (ev) {
    var m = ev.data;
    switch (m.type) {
      case 'ready':
        window.BitNetWasm = api;
        resolveReady(api);
        console.log('[bitnet] worker ready; window.BitNetWasm defined');
        break;
      case 'progress':        // download progress (worker -> here)
        if (progressCb) progressCb(m.frac);
        break;
      case 'load_progress':   // model tensor-load progress (WASM -> main thread)
        if (progressCb) progressCb(m.frac);
        break;
      case 'result': {
        var p = pending.get(m.id);
        if (p) { pending.delete(m.id); p.resolve(m.value); }
        break;
      }
      case 'error': {
        var pe = pending.get(m.id);
        if (pe) { pending.delete(m.id); pe.reject(new Error(m.error)); }
        break;
      }
      case 'fatal':
        console.error('[bitnet] worker fatal:', m.error);
        break;
      default:
        break;
    }
  };

  worker.onerror = function (e) {
    console.error('[bitnet] worker error:', e.message || e);
  };

  function call(op, extra) {
    var id = nextId++;
    return new Promise(function (resolve, reject) {
      pending.set(id, { resolve: resolve, reject: reject });
      var payload = { op: op, id: id };
      for (var k in extra) { if (Object.prototype.hasOwnProperty.call(extra, k)) payload[k] = extra[k]; }
      worker.postMessage(payload);
    });
  }
})();
