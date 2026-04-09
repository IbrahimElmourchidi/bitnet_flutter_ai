/**
 * coi_service_worker.js
 *
 * Cross-Origin Isolation (COI) service worker shim.
 *
 * Injects the headers required for SharedArrayBuffer on browsers that do not
 * allow setting them at the server level (e.g. GitHub Pages, Firebase Hosting):
 *
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 *
 * SETUP — add this snippet to your web/index.html BEFORE the Flutter loader:
 *
 *   <script>
 *     if (typeof SharedArrayBuffer === 'undefined') {
 *       const reloaded = sessionStorage.getItem('coi-reload');
 *       if (!reloaded) {
 *         sessionStorage.setItem('coi-reload', '1');
 *         navigator.serviceWorker.register('/coi_service_worker.js')
 *           .then(() => location.reload());
 *       } else {
 *         sessionStorage.removeItem('coi-reload');
 *       }
 *     }
 *   </script>
 *
 * The worker registers, the page reloads once, and from the second load
 * onward the intercepted responses carry the COOP/COEP headers, making
 * SharedArrayBuffer available for the WASM pthreads runtime.
 */

const CACHE_NAME = 'coi-v1';

// Install: take control immediately without waiting for old SW to expire.
self.addEventListener('install', () => self.skipWaiting());

// Activate: claim all clients so the very next navigation is intercepted.
self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// Fetch: clone every response and inject the COI headers.
self.addEventListener('fetch', (event) => {
  // Only intercept same-origin requests; pass cross-origin requests through.
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    fetch(event.request).then((response) => {
      // Don't modify opaque responses (cross-origin no-cors).
      if (response.type === 'opaque') return response;

      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }).catch(() => fetch(event.request))
  );
});
