"""Serve a local precomputed datastore to Neuroglancer, with CORS.

Neuroglancer runs in a browser on someone else's origin, so a plain
`python -m http.server` is refused: the browser blocks the cross-origin read
before the server ever sees a problem. This adds the two headers that make it
work, and answers the preflight.

    python tools/serve_precomputed.py            # serves ./precomputed on 8000
    python tools/serve_precomputed.py --port 9000 --root somewhere/else

It prints a Neuroglancer link with the exported layers already loaded.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
import urllib.parse


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static files, plus the headers a cross-origin viewer needs."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Length")
        # The datastore is rewritten often; a cached `info` is a confusing way
        # to spend an afternoon.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def log_message(self, fmt, *a):
        if self.path.endswith("info") or "404" in (fmt % a):
            super().log_message(fmt, *a)


def neuroglancer_url(root: str, port: int) -> str:
    """A link with every exported layer already in it."""
    layers = []
    for name in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, name, "info")):
            layers.append(
                {
                    "type": "segmentation",
                    "source": f"precomputed://http://localhost:{port}/{name}",
                    "name": name,
                    "visible": len(layers) == 0,
                }
            )
    if not layers:
        return ""
    state = {"layers": layers, "layout": "3d"}
    return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(
        json.dumps(state, separators=(",", ":")), safe=""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="precomputed")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"no such directory: {root}\nRun tools/export_precomputed.py first.")
        return 1

    handler = functools.partial(CORSRequestHandler, directory=root)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {root} on http://localhost:{args.port} (CORS open)\n")
        for name in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, name, "info")):
                print(f"  precomputed://http://localhost:{args.port}/{name}")
        url = neuroglancer_url(root, args.port)
        if url:
            print(f"\nor open both at once:\n\n{url}\n")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
