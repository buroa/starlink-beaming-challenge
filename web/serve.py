#!/usr/bin/env python3
"""Static file server with cross-origin isolation headers.

The threaded WASM build uses a SharedArrayBuffer, which the browser only hands
out to a cross-origin-isolated page — i.e. one served with COOP + COEP. Plain
`python3 -m http.server` does not set those, so the worker pool init fails. This
server adds them and serves the repo root, so both /web/ and /test_cases/ are
reachable.

    python3 web/serve.py [port]   # default 8000  → http://localhost:8000/web/

The serial build does not need any of this; a vanilla static host works.
"""
import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        # Cross-origin isolation → enables SharedArrayBuffer for the thread pool.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"serving repo root (COOP/COEP on) → http://localhost:{PORT}/web/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
