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
        # Neuroglancer asks for <mesh>/info to find out whether a mesh source is
        # the multi-resolution format. There is no such file here, and the 404
        # is how it learns to read the legacy single-resolution one instead --
        # so it is the protocol working, not a fault. Creating the file to
        # silence it would switch Neuroglancer into multi-resolution mode and
        # break the meshes.
        if self.path.endswith("/mesh/info"):
            return
        if self.path.endswith("info") or "404" in (fmt % a):
            super().log_message(fmt, *a)


# The BossDB mirror of the MICrONS minnie65 EM, which is what the cutouts here
# were taken from. Public over HTTPS, so the browser can read it without
# credentials. Note this is *not* interchangeable with other MICrONS-adjacent
# imagery: v1dd, for instance, is a different animal at 4.85 x 4.85 x 45 nm and
# would not line up with anything here.
IMAGERY = (
    "precomputed://https://bossdb-open-data.s3.amazonaws.com"
    "/iarpa_microns/minnie/minnie65/em"
)


def neuroglancer_url(root: str, port: int, imagery: str = IMAGERY) -> str:
    """A link with the imagery and every exported layer already in it."""
    layers = []
    if imagery:
        layers.append({"type": "image", "source": imagery, "name": "em"})
    for name in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, name, "info")):
            layers.append(
                {
                    "type": "segmentation",
                    "source": f"precomputed://http://localhost:{port}/{name}",
                    "name": name,
                    # One segmentation on at a time, so toggling compares them
                    # in place rather than stacking two sets of meshes.
                    "visible": not any(la["type"] == "segmentation" for la in layers),
                }
            )
    if not any(la["type"] == "segmentation" for la in layers):
        return ""
    state = {"layers": layers, "layout": "4panel"}
    return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(
        json.dumps(state, separators=(",", ":")), safe=""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="precomputed")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--imagery",
        default=IMAGERY,
        help="EM layer to show underneath; empty string for none",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"no such directory: {root}\nRun tools/export_precomputed.py first.")
        return 1

    # The request log goes to stderr unbuffered while these notes go to stdout,
    # so without this the banner disappears the moment the output is piped and
    # only the log survives.
    sys.stdout.reconfigure(line_buffering=True)

    handler = functools.partial(CORSRequestHandler, directory=root)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {root} on http://localhost:{args.port} (CORS open)\n")
        print(
            "  (a 404 on <layer>/mesh/info is expected: it is how Neuroglancer\n"
            "   detects that these are legacy single-resolution meshes)\n"
        )
        for name in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, name, "info")):
                print(f"  precomputed://http://localhost:{args.port}/{name}")
        url = neuroglancer_url(root, args.port, args.imagery)
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
