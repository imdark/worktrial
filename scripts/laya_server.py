"""Serve laya-vision on localhost, so teleop_sim can ask it typed questions.

laya-vision cannot live in the project's venv: it needs transformers >= 5.3,
and lerobot 0.3.2 pins transformers < 4.52. So it runs in its own venv, set up
by scripts/setup_laya_vision.sh, and teleop_sim talks to it over HTTP
(perception/laya_vision.py). The client needs nothing beyond the standard
library.

    third_party/laya-vision/.venv/bin/python scripts/laya_server.py

Endpoints, JSON in and out, bound to 127.0.0.1 only:

    GET  /health   -> {"ok", "model", "revision", "device"}
    POST /predict  {"image": <base64 JPEG/PNG>, "note": str | null,
                    "questions": {name: {"type": "noul"|"choice"|"score", ...}}}
                -> laya's predict() result, plus "server_latency_s"

The checkpoint is pinned: the model card warns the weights can change under
an unpinned id. Weights are CC BY-NC-SA 4.0 -- non-commercial use only.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ID = "thaitea/laya-vision"
REVISION = "8b318c99d7ad3ce19c24369263463882eada9d1e"  # 2026-09-24


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    import laya
    from PIL import Image

    started = time.perf_counter()
    agent = laya.load_vlm(args.model, revision=args.revision, device=args.device)
    device = str(getattr(agent, "device", args.device))
    # The first calls on MPS compile kernels; take that hit before serving.
    warm = {"q": {"type": "noul", "instructions": "Is there an object in this image?"}}
    for _ in range(3):
        agent.predict({"image": Image.new("RGB", (64, 64))}, warm)
    print(f"[laya] {args.model}@{args.revision[:8]} on {device}, "
          f"ready in {time.perf_counter() - started:.1f}s")

    lock = threading.Lock()  # one model, one request at a time

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
            if self.path != "/health":
                return self._send(404, {"error": f"no route {self.path}"})
            self._send(200, {"ok": True, "model": args.model, "revision": args.revision,
                             "device": device})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                return self._send(404, {"error": f"no route {self.path}"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                image = Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB")
                state = {"image": image}
                if req.get("note"):
                    state["note"] = req["note"]
                t = time.perf_counter()
                with lock:
                    result = agent.predict(state, req["questions"])
                result["server_latency_s"] = round(time.perf_counter() - t, 4)
            except Exception as exc:  # report, don't crash the server
                return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            self._send(200, result)

        def log_message(self, fmt: str, *fargs) -> None:
            pass  # teleop_sim logs every call; per-request stderr lines are noise

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
