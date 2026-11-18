"""Persistent RK3588 HTTP action-chunk inference service (NumPy + RKNNLite)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from runtime import INPUT_NAMES, RKNNPolicy

ROOT = Path(__file__).resolve().parents[1]
SHAPES = [(1, 18), (1, 3, 128, 128), (1, 3, 128, 128)]
MAX_BODY = 2 * 1024 * 1024


def decode_inputs(body: bytes, content_type: str):
    if content_type.startswith("application/json"):
        payload = json.loads(body)
        values = [payload[name] for name in INPUT_NAMES]
    else:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > MAX_BODY:
                raise ValueError("Uncompressed NPZ is too large")
        with np.load(io.BytesIO(body), allow_pickle=False) as payload:
            values = [payload[name].copy() for name in INPUT_NAMES]
    inputs = []
    for name, value, shape in zip(INPUT_NAMES, values, SHAPES, strict=True):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: NaN/Inf is not allowed")
        if name.startswith("observation.images.") and (array.min() < 0 or array.max() > 1):
            raise ValueError(f"{name}: expected RGB float32 in [0,1], before normalization")
        inputs.append(np.ascontiguousarray(array))
    return inputs


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, policy, model):
        self.policy = policy
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.inferences = 0
        with model.open("rb") as stream:
            self.model_sha256 = (
                hashlib.file_digest(stream, "sha256").hexdigest()
                if hasattr(hashlib, "file_digest")
                else hashlib.sha256(stream.read()).hexdigest()
            )
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: InferenceServer

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def respond(self, status, value):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in {"/healthz", "/"}:
            self.respond(404, {"error": "not found"})
            return
        self.respond(
            200,
            {
                "status": "ready",
                "backend": "RKNNLite / RK3588 NPU",
                "model_sha256": self.server.model_sha256,
                "uptime_seconds": time.monotonic() - self.server.started,
                "inferences": self.server.inferences,
                "inputs": dict(zip(INPUT_NAMES, SHAPES, strict=True)),
                "image_format": "RGB NCHW float32 [0,1]; do not apply normalization",
                "output_shape": [1, 50, 4],
                "n_action_steps": 10,
            },
        )

    def do_POST(self):
        if self.path != "/infer":
            self.respond(404, {"error": "not found"})
            return
        started = time.perf_counter()
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                self.respond(413, {"error": "Body must be between 1 byte and 2 MiB"})
                return
            inputs = decode_inputs(self.rfile.read(size), self.headers.get("Content-Type", ""))
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            self.respond(400, {"error": str(error)})
            return
        try:
            with self.server.lock:
                infer_start = time.perf_counter()
                actions = self.server.policy._infer(inputs).reshape(1, 50, 4)
                inference_ms = (time.perf_counter() - infer_start) * 1000
                if not np.isfinite(actions).all():
                    raise RuntimeError("RKNN output contains NaN/Inf")
                self.server.inferences += 1
            self.respond(
                200,
                {
                    "action": actions.tolist(),
                    "shape": list(actions.shape),
                    "inference_ms": inference_ms,
                    "server_ms": (time.perf_counter() - started) * 1000,
                    "n_action_steps": 10,
                },
            )
        except Exception:
            logging.exception("RKNN inference failed")
            self.respond(500, {"error": "RKNN inference failed; see service log"})

    def log_message(self, fmt, *args):
        logging.info("%s %s", self.client_address[0], fmt % args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "act_franka_rk3588_fp16.rknn")
    parser.add_argument("--warmup-sample", type=Path, help="Optional real observation NPZ for warmup")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = RKNNPolicy(args.model)
    server = None
    try:
        if args.warmup_sample is not None:
            warmup = decode_inputs(args.warmup_sample.read_bytes(), "application/octet-stream")
        else:
            # Synthetic inputs only initialize the runtime; never execute their actions.
            warmup = [np.zeros(shape, dtype=np.float32) for shape in SHAPES]
        for _ in range(3):
            actions = policy._infer(warmup)
            if not np.isfinite(actions).all():
                raise RuntimeError("Warmup output contains NaN/Inf")
        server = InferenceServer((args.host, args.port), policy, args.model)
        logging.info("Ready at http://%s:%d (model loaded once)", args.host, args.port)
        server.serve_forever(poll_interval=0.5)
    finally:
        if server is not None:
            server.server_close()
        policy.close()


if __name__ == "__main__":
    main()
