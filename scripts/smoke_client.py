"""Check the HTTP interface with synthetic inputs; never execute returned actions."""

import argparse
import io
import json
import time
import urllib.request

import numpy as np
from client import infer
from runtime import INPUT_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18088")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url.rstrip("/") + "/healthz", timeout=10) as response:
        health = json.load(response)
    print(json.dumps(health, indent=2))
    payload = io.BytesIO()
    values = [np.zeros((1, 18), np.float32)] + [np.zeros((1, 3, 128, 128), np.float32) for _ in range(2)]
    np.savez(payload, **dict(zip(INPUT_NAMES, values, strict=True)))
    start = time.perf_counter()
    result = infer(args.url, payload.getvalue())
    action = np.asarray(result["action"])
    if action.shape != (1, 50, 4) or not np.isfinite(action).all():
        raise RuntimeError("Invalid response")
    print(
        json.dumps(
            {
                "shape": list(action.shape),
                "inference_ms": result["inference_ms"],
                "http_roundtrip_ms": (time.perf_counter() - start) * 1000,
                "note": "Synthetic interface check only; actions were not executed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
