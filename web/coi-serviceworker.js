// coi-serviceworker — enables cross-origin isolation (and therefore
// SharedArrayBuffer + wasm threads) on static hosts that cannot set COOP/COEP
// response headers, such as GitHub Pages. It registers a service worker that
// re-serves every response with the isolation headers, then reloads the page
// once so the document loads cross-origin-isolated.
//
// Adapted from gzuidhof/coi-serviceworker (MIT). The threaded solver and the
// viz's parallel solve light up when this succeeds; without it (or on a browser
// that blocks the SW) the harnesses fall back to the serial path.
//
// Must be loaded as a CLASSIC script — `<script src="coi-serviceworker.js"></script>` —
// so `document.currentScript` resolves (module scripts don't have it).

if (typeof window === "undefined") {
  // ---- service worker context ----
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener("fetch", (event) => {
    const req = event.request;
    if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.status === 0) return res; // opaque (no-cors) — can't modify
          const headers = new Headers(res.headers);
          headers.set("Cross-Origin-Embedder-Policy", "require-corp");
          headers.set("Cross-Origin-Opener-Policy", "same-origin");
          // Mark every (incl. cross-origin, e.g. basemap tiles) resource loadable
          // under require-corp. Tiles are fetched CORS, so the body is readable.
          headers.set("Cross-Origin-Resource-Policy", "cross-origin");
          return new Response(res.body, {
            status: res.status,
            statusText: res.statusText,
            headers,
          });
        })
        .catch((e) => {
          // Let the failure surface as a normal network error rather than
          // resolving respondWith() to undefined.
          console.error(e);
          throw e;
        })
    );
  });
} else {
  // ---- page context: register the SW, reload once when it takes control ----
  (() => {
    if (window.crossOriginIsolated !== false) return; // already isolated
    if (!window.isSecureContext) return; // SW needs HTTPS or localhost
    const src = document.currentScript && document.currentScript.src;
    if (!src || !navigator.serviceWorker) return;
    navigator.serviceWorker.register(src).then(
      (reg) => {
        reg.addEventListener("updatefound", () => window.location.reload());
        if (reg.active && !navigator.serviceWorker.controller) window.location.reload();
      },
      (err) => console.error("coi-serviceworker registration failed:", err)
    );
  })();
}
