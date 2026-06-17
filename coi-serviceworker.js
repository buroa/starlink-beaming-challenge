// coi-serviceworker — enables cross-origin isolation (and therefore
// SharedArrayBuffer + wasm threads) on static hosts that cannot set COOP/COEP
// response headers, such as GitHub Pages. It registers a service worker that
// re-serves every response with the isolation headers, then reloads the page
// once so the document loads cross-origin-isolated. Locally, `serve --config
// serve.json` sets the headers directly and this is a harmless no-op.
//
// Adapted from gzuidhof/coi-serviceworker (MIT). Each app copies it into its
// build (webpack CopyPlugin) and loads it as a CLASSIC script in index.html —
// `<script src="coi-serviceworker.js"></script>` — so `document.currentScript`
// resolves (module scripts don't have it).

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
          // Make cross-origin resources (e.g. basemap tiles, fetched CORS) loadable
          // under require-corp.
          headers.set("Cross-Origin-Resource-Policy", "cross-origin");
          return new Response(res.body, {
            status: res.status,
            statusText: res.statusText,
            headers,
          });
        })
        .catch((e) => {
          // Surface as a normal network error rather than resolving to undefined.
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
