"""Call the board HTTP service using a saved sample; report precision and timings."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def infer(url, payload):
    request = urllib.request.Request(
        url.rstrip("/") + "/infer", data=payload, headers={"Content-Type": "application/octet-stream"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "board_http.json")
    args = parser.parse_args()
    if not args.all_samples and args.sample is None:
        parser.error("Provide --sample INPUT.npz or --all-samples after exporting validation samples")
    report = {"url": args.url, "samples": {}, "timing": []}
    with urllib.request.urlopen(args.url.rstrip("/") + "/healthz", timeout=10) as response:
        report["health"] = json.load(response)
    paths = sorted((ROOT / "samples").glob("*.npz")) if args.all_samples else [args.sample]
    for path in paths:
        payload = path.read_bytes()
        with np.load(path) as sample:
            expected = sample["action"]
        errors = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = infer(args.url, payload)
            roundtrip_ms = (time.perf_counter() - started) * 1000
            actual = np.asarray(result["action"], dtype=np.float32)
            if actual.shape != (1, 50, 4) or not np.isfinite(actual).all():
                raise RuntimeError(f"Invalid model output: {path.name}")
            errors.append(np.abs(actual - expected).reshape(-1, 4).max(0))
            report["timing"].append(
                {"sample": path.stem, "inference_ms": result["inference_ms"], "roundtrip_ms": roundtrip_ms}
            )
        report["samples"][path.stem] = {"per_action_dim_max_abs": np.max(errors, axis=0).tolist()}
    report["per_action_dim_max_abs"] = np.max(
        [row["per_action_dim_max_abs"] for row in report["samples"].values()], axis=0
    ).tolist()
    for field in ["inference_ms", "roundtrip_ms"]:
        values = [row[field] for row in report["timing"]]
        report[field] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "min": min(values),
            "max": max(values),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"timing", "samples"}}, indent=2))


if __name__ == "__main__":
    main()
