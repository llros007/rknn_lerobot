"""Compile the static ONNX graph to a target-specific FP16 RKNN model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(ret, stage):
    if ret != 0:
        raise RuntimeError(f"RKNN {stage} failed: {ret}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="rk3588", choices=["rk3588", "rk3576", "rk3568", "rk3566"])
    parser.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--verify", action="store_true", help="Compare the build's PC simulator to PyTorch samples"
    )
    parser.add_argument("--eval-episodes", type=int, default=0, help="Run MuJoCo with the RKNN PC simulator")
    args = parser.parse_args()
    from importlib.metadata import version

    from rknn.api import RKNN

    work = ROOT / "tmp" / f"build_{args.target}"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    output = ROOT / "models" / f"act_franka_{args.target}_fp16.rknn"
    rknn = RKNN(verbose=True, verbose_file=str(ROOT / "logs" / f"rknn_{args.target}_verbose.log"))
    try:
        check(
            rknn.config(
                target_platform=args.target, float_dtype="float16", optimization_level=args.optimization_level
            ),
            "config",
        )
        check(rknn.load_onnx(model=str(ROOT / "onnx" / "act.onnx")), "load_onnx")
        check(rknn.build(do_quantization=False), "build")
        check(rknn.export_rknn(str(output)), "export_rknn")
        # Read actual compiler metadata: RKNN's native image layout may be NHWC
        # even though the source ONNX interface is NCHW.
        metadata = getattr(rknn.rknn_base, "inputs_meta", {})
        attrs = metadata.get("attrs", {})
        tensors = []
        for name, attr in attrs.items():
            if isinstance(attr, dict):
                tensors.append(
                    {
                        "name": name,
                        **{k: attr[k] for k in ["idx", "shape", "dtype", "layout", "is_output"] if k in attr},
                    }
                )
        (ROOT / "models" / f"act_franka_{args.target}_fp16.abi.json").write_text(
            json.dumps(tensors, indent=2, default=str)
        )
        if args.verify or args.eval_episodes:
            import numpy as np

            check(rknn.init_runtime(), "init_runtime (PC simulator)")
            input_names = json.loads((ROOT / "manifest.json").read_text())["input_order"]
            report = {"backend": "RKNN Toolkit2 PC simulator (not board hardware)", "samples": {}}
            for path in sorted((ROOT / "samples").glob("*.npz")):
                sample = np.load(path)
                inputs = [sample[key] for key in input_names]
                start = time.perf_counter()
                outputs = rknn.inference(inputs=inputs, data_format=["nchw"] * len(inputs))
                if outputs is None:
                    raise RuntimeError(f"RKNN simulator inference failed: {path.name}")
                action = outputs[0].reshape(sample["action"].shape)
                if not np.isfinite(action).all():
                    raise RuntimeError(f"Nonfinite RKNN output: {path.name}")
                diff = np.abs(action - sample["action"])
                report["samples"][path.stem] = {
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                    "per_action_dim_max_abs": diff.reshape(-1, 4).max(0).tolist(),
                    "simulator_seconds": time.perf_counter() - start,
                }
                np.save(ROOT / "samples" / f"{path.stem}_rknn.npy", action)
            report["per_action_dim_max_abs"] = np.max(
                [row["per_action_dim_max_abs"] for row in report["samples"].values()], axis=0
            ).tolist()
            (ROOT / "reports" / f"rknn_{args.target}_validation.json").write_text(
                json.dumps(report, indent=2)
            )
            print("RKNN simulator error:", report["per_action_dim_max_abs"], flush=True)
            if args.eval_episodes:
                from eval_mujoco import evaluate

                def infer(inputs):
                    outputs = rknn.inference(inputs=inputs, data_format=["nchw"] * len(inputs))
                    if outputs is None:
                        raise RuntimeError("RKNN simulator inference failed")
                    return outputs[0]

                result = evaluate(infer, args.eval_episodes, ROOT / "reports" / "rknn_simulator_rollout.mp4")
                (ROOT / "reports" / f"rknn_{args.target}_mujoco.json").write_text(
                    json.dumps(result, indent=2)
                )
    finally:
        rknn.release()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    with output.open("rb") as model_file:
        model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    manifest["rknn"] = {
        "path": str(output.relative_to(ROOT)),
        "target": args.target,
        "precision": "float16",
        "toolkit_version": version("rknn-toolkit2"),
        "optimization_level": args.optimization_level,
        "board_validation": "not yet run",
        "sha256": model_sha256,
        "abi": str(output.with_suffix(".abi.json").relative_to(ROOT)),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"RKNN exported: {output} ({output.stat().st_size / 1024**2:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
